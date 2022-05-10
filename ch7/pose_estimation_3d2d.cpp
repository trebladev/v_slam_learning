//
// Created by xuan on 22-5-4.
//
#include "Timing.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Eigen/Core"
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/solver.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "sophus/se3.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
void bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
);
// BA by gauss-newton
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
);

int main(int argc,char **argv)
{
    Mat img_1 = imread("../ch7/1.png",CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("../ch7/2.png",CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "can not load images");
    // find Fast Feature points,compute ORB description of each point and get matches
    vector<KeyPoint> keypoints_1,keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    cout<<"find all "<<matches.size()<<"match points"<<endl;

    Mat d1 = imread("../ch7/1_depth.png",CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2d> pts_2d;
    for(DMatch m:matches){
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if(d==0)
            continue;
        float  dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt,K );
        pts_3d.push_back(Point3f(p1.x*dd,p1.y*dd,dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);

    }
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    Sophus::SE3d pose_gn;
    TIMER_START(gauss_newton);
    bundleAdjustmentGaussNewton(pts_3d_eigen,pts_2d_eigen,K,pose_gn);
    TIMER_END(gauss_newton);

    Sophus::SE3d pose_g2o;
    TIMER_START(PNPG2O);
    bundleAdjustmentG2O(pts_3d_eigen,pts_2d_eigen,K,pose_g2o);
    TIMER_END(PNPG2O);
    return 0;

}



void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
)
{
    typedef Eigen::Matrix<double,6,1> Vector6d;
    const int iterations = 10;
    double cost = 0,lastCost = 0;
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    for(int iter = 0;iter<iterations;iter++)
    {
        Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
        Vector6d b = Vector6d ::Zero();

        cost = 0;
        //compute cost
        for(int i=0;i<points_3d.size();i++)
        {
            Eigen::Vector3d pc = pose*points_3d[i];
            double inv_z = 1.0/pc[2];
            double inv_z2 = inv_z*inv_z;
            Eigen::Vector2d proj(fx*pc[0]/pc[2]+cx,fy*pc[1]/pc[2]+cy);

            Eigen::Vector2d e = points_2d[i] - proj;
            cost += e.squaredNorm();
            Eigen::Matrix<double,2,6> J;
            J << -fx*inv_z,
            0,
            fx*pc[0]*inv_z2,
            fx*pc[0]*pc[1]*inv_z2,
            -fx-fx*pc[0]*pc[0]*inv_z2,
            fx*pc[1]*inv_z,
            0,
            -fy*inv_z,
            fy*pc[1]*inv_z2,
            fy+fy*pc[1]*pc[1]*inv_z2,
            -fy*pc[0]*pc[1]*inv_z2,
            -fy*pc[0]*inv_z;

            H +=J.transpose() *J;   //compute H
            b = -J.transpose()*e;   //compute b(g)
        }
        Vector6d dx;
        dx = H.ldlt().solve(b);
        if(iter >0 && cost >=lastCost){
            cout<<"cost:"<<cost<<",last cost:"<<lastCost<<endl;
            break;
        }

        pose = Sophus::SE3d::exp(dx) *pose;
        lastCost = cost;
        if(dx.norm() <1e-6){
            break;
        }
    }
    cout<<"pose by g-n: \n"<<pose.matrix()<<endl;
}

class VertexPose : public g2o::BaseVertex<6,Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override{
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override{
        Eigen::Matrix<double,6,1> update_eigen;
        update_eigen << update[0],update[1],update[2],update[3],update[4],update[4],update[5];
        _estimate = Sophus::SE3d::exp(update_eigen)*_estimate;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
};

///edges used in g2o ba
class EdgeProjection : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos,const Eigen::Matrix3d &K) : _pos3d(pos),_K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K *(T*_pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override{
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T*_pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
                << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
                0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
        ){
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType; /// ?
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // type of linear solver

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
            );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    VertexPose *vertex_pose = new VertexPose();
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
            K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
            K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    int index = 1;
    for(size_t i=0;i<points_2d.size();++i)
    {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d,K_eigen);
        edge->setId(index);
        edge->setVertex(0,vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    TIMER_START(G2O);
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    TIMER_END(G2O);
    pose = vertex_pose->estimate();

}