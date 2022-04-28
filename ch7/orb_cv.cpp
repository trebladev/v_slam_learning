#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Timing.h>
#include "opencv2/imgcodecs/legacy/constants_c.h"


using namespace std;
using namespace cv;

int main(int argc,char **argv)
{
    Mat img_1 = imread("../ch7/1.png",CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread("../ch7/2.png",CV_LOAD_IMAGE_COLOR);
    assert(img_1.data !=nullptr && img_2.data !=nullptr);

    std::vector<KeyPoint> keypoint1,keypoint2;
    Mat descriptors_1,descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    TIMER_START(DETECTOR_FAST);
    detector->detect(img_1,keypoint1);
    detector->detect(img_2,keypoint2);

    descriptor->compute(img_1,keypoint1,descriptors_1);
    descriptor->compute(img_2,keypoint2,descriptors_2);

    TIMER_END(DETECTOR_FAST);

    Mat outimg1;
    drawKeypoints(img_1,keypoint1,outimg1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
    imshow("ORB features",outimg1);

    vector<DMatch> matches;
    TIMER_START(MATCH);
    matcher->match(descriptors_1,descriptors_2,matches);
    TIMER_END(MATCH);

    auto min_max = minmax_element(matches.begin(),matches.end(),[](const DMatch &m1,const DMatch &m2){return m1.distance<m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n",max_dist);
    printf("-- Min dist : %f \n",min_dist);

    std::vector<DMatch> good_matches;
    for(int i=0;i<descriptors_1.rows;i++)
    {
        if(matches[i].distance <= max(2*min_dist,30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1,keypoint1,img_2,keypoint2,matches,img_match);
    drawMatches(img_1,keypoint1,img_2,keypoint2,good_matches,img_goodmatch);
    imshow("all mathches",img_match);
    imshow("good mathces",img_goodmatch);

    waitKey(0);

    return 0;

}

