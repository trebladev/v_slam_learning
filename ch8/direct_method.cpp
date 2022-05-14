//
// Created by xuan on 22-5-14.
//
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include "pangolin/pangolin.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <Timing.h>

using namespace std;

typedef vector<Eigen::Vector2d,Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// paths
string left_file = "../ch8/left.png";
string disparity_file = "../ch8/disparity.png";
boost::format fmt_others("../ch8/%06d.png");    // other files

typedef Eigen::Matrix<double,6,6> Matrix6d;
typedef Eigen::Matrix<double,2,6> Matrix26d;
typedef Eigen::Matrix<double,6,1> Vector6d;

/// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const VecVector2d &px_ref_,
            const vector<double> depth_ref_,
            Sophus::SE3d &T21_
            ) : img1(img1_),img2(img2_),px_ref(px_ref_),depth_ref(depth_ref_),T21(T21_){
        projection = VecVector2d(px_ref.size(),Eigen::Vector2d(0,0));
    }

    /// accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    /// get hessian matrix
    Matrix6d hessian() const {return H;}

    /// get bias
    Vector6d bias() const {return b;}

    /// get total cost
    double cost_func() const {return cost;}

    /// get projected points
    VecVector2d projected_points() const {return projection;}

    /// reset h,b,cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }


private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
);

int main(int argc,char* argv[]){

    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int i = 1; i < 6; i++) {  // 1~10
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        // try single layer by uncomment this line
        DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        //DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }
    return 0;
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    //parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start;i<range.end;i++){

        //compute the projection in the second image
        Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0]-cx)/fx,(px_ref[i][1]-cy)/fy,1);
        Eigen::Vector3d point_cur = T21*point_ref;
        if(point_cur[2] < 0)           //depth invalid
            continue;

        float u = fx*point_cur[0] / point_cur[2] +cx,v = fy*point_cur[1] / point_cur[2]+cy;
        if(u<half_patch_size || u>img2.cols-half_patch_size || v<half_patch_size || v>img2.rows-half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u,v);
        double X = point_cur[0],Y = point_cur[1],Z = point_cur[2],Z2 = Z*Z,Z_inv = 1.0/Z,Z2_inv = Z_inv*Z_inv;
        cnt_good++;

        // compute error and jacobian
        for(int x= -half_patch_size;x<=half_patch_size;x++){
            for(int y= -half_patch_size;y<=half_patch_size;y++){
                double error = GetPixelValue(img1,px_ref[i][0]+x,px_ref[i][1]+y)-
                        GetPixelValue(img2,u+x,v+y);

                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));
                //total jacobian
                Vector6d J = -1.0*(J_img_pixel.transpose()*J_pixel_xi).transpose();

                hessian += J*J.transpose();
                bias += -error*J;
                cost_tmp += error*error;
            }
        }
    }
    if (cnt_good){
        //set hessian,bias cost
        unique_lock<mutex> lck(hessian_mutex);
        H +=hessian;
        b += bias;
        cost += cost_tmp/cnt_good;
    }
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const vector<double> depth_ref,
        Sophus::SE3d &T21
        ){

    const int iterations = 10;
    double cost = 0,lastcost = 0;
    TIMER_START(single_layer);

    JacobianAccumulator jaco_accu(img1,img2,px_ref,depth_ref,T21);

    for (int iter = 0;iter < iterations;iter++){
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        //solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) *T21;
        cost = jaco_accu.cost_func();

        if(std::isnan(update[0])){
            cout << "update is nan" <<endl;
            break;
        }
        if(iter >0 && cost>lastcost)
        {
            cout<<"cost increased:"<<cost <<","<<lastcost<<endl;
            break;
        }
        if(update.norm()<1e-3){
            break;
        }
        lastcost = cost;
        cout<<"iterarion:"<<iter<<",cost" <<cost<<endl;
    }

    cout<<"T21 = \n"<<T21.matrix()<<endl;
    TIMER_END(single_layer);

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}