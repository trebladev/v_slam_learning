//
// Created by xuan on 22-4-29.
//
#include "Timing.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img_1,const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches
        );

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat &R,Mat &t
        );

Point2d pixel2cam(const Point2d &p,const Mat &K);

int main(int argc,char **argv){
    //load image
    Mat img_1 = imread("../ch7/1.png",CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("../ch7/2.png",CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "can not load images");

    //find Fast Feature points andompute ORB description for each point then get matches
    vector<KeyPoint> keypoints_1,keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    cout<<"find all "<<matches.size()<<"match points"<<endl;

    Mat R,t;
    //use Polar constraints to compute movement of camera(E,t)
    pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);
    Mat t_x =
            (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);
    cout<<"t^R="<<endl<<t_x*R<<endl;
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }

    return 0;
}
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- ?????????
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- ?????????:?????? Oriented FAST ????????????
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- ?????????:???????????????????????? BRIEF ?????????
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- ?????????:?????????????????????BRIEF?????????????????????????????? Hamming ??????
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- ?????????:??????????????????
    double min_dist = 10000, max_dist = 0;

    //??????????????????????????????????????????????????????, ????????????????????????????????????????????????????????????
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //?????????????????????????????????????????????????????????,?????????????????????.????????????????????????????????????,?????????????????????30????????????.
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

void pose_estimation_2d2d(std::vector<KeyPoint> keypoint_1,
                          std::vector<KeyPoint> keypoint_2,
                          std::vector<DMatch> matches,
                          Mat &R,Mat &t)
{
    //CAM inside matrix
    Mat K = (Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);

    //transform keypoint to point2f in vector
    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i=0;i<(int) matches.size();i++){
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }

    //compute fundamental matrix
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1,points2,FM_8POINT);

    //compute essential_matrix
    Point2d principal_point(325.1,249.7);
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1,points2,focal_length,principal_point);

    //recover pose from essential
    recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);
    cout<<"R is"<<endl<<R<<endl;
    cout<<"t is"<<endl<<t<<endl;
}