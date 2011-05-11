

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "siftfast.h"
#include <time.h>
#include <cstdio>
#include <funcs.cpp>


using namespace cv;
using namespace std;


vector<Mat> read_conts_from_file(string fname)
{
    FileStorage fs(fname, FileStorage::READ);
    vector<Mat> v;
    int i = 0;
    while(true)
    {
        stringstream ss;
        ss << "cont" << i;
        Mat tmp;
        fs[ss.str()] >> tmp;
        if(tmp.empty())
            break;
        v.push_back(tmp);
        i++;
    }
    return v;
}

int n_bins = 600;
float range[] = {-10e+26, +10e+26};
const float *ranges[] = { range };
int channels[] = {0, 1};

int main()
{


    vector<Mat> v1 = read_conts_from_file("/home/dedan/obj_att/out/cont0.yml");
    vector<Mat> v2 = read_conts_from_file("/home/dedan/obj_att/out/cont2.yml");

    Mat res(v1.size(), v2.size(), CV_32FC1);

    for(int i; i < v1.size(); i++)
    {
        for(int j; j < v2.size(); j++)
        {
            res.at<float>(i,j) = matchShapes(v1[i], v2[j], CV_CONTOURS_MATCH_I1, 0);
        }
    }

    cout << v1.size() << " " << v2.size() << endl;
    cout << res << endl;
    res.convertTo(res, CV_8UC1);
    cout << res << endl;
    Mat hist;
    calcHist(&res, 1, channels, Mat(), hist, 1, &n_bins, ranges, true, false);

//    cout << hist << endl;
//    fs["cont2"] >> test;
//    cout << test << endl;

//    Mat test1;
//    fs["cont3"] >> test1;
//    cout << test1 << endl;

//    double bla = cv::matchShapes(test, test1, CV_CONTOURS_MATCH_I1, 0);
//    cout << bla << endl;

    return 0;
}
