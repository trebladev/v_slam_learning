#include <iostream>
#include <chrono>

using namespace std;

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui.hpp>

int main(int argc,char **argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);

    if (image.data = nullptr)
    {
        cerr<<"file"<<argv[1]<<"do not exist"<<endl;
        return 0;
    }

    cout<<"weith ="<<image.cols<<"high ="<<image.rows<<"number of channels="<<image.channels()<<endl;
    cv::imshow("image",image);
    cv::waitKey();

    if(image.type()!=CV_8UC1 && image.type()!=CV_8UC3)
    {
        cout<<"please input a colorful image or gray image"<<endl;
        return 0;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y=0;y<image.rows;y++)
    {
        unsigned char *row_ptr =image.ptr<unsigned char>(y);
        for(size_t x=0;x<image.cols;x++)
        {
            unsigned char *data_ptr = &row_ptr[x*image.channels()];
        }
        for(int c=0; c !=image.channels();c++)
        {
            unsigned char data = data_ptr[c];
        }
    }

    return 0;
}