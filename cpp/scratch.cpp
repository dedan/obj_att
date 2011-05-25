

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "siftfast.h"
#include <time.h>
#include <cstdio>
#include <funcs.cpp>
#include <algorithm>


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

void simple_draw_cont(Mat im, Mat cont)
{
    for(int i = 0; i < cont.rows; i++)
    {
        im.at<char>(cont.at<int>(i,0), cont.at<int>(i,0)) = 255;
    }
}

vector<vector<Point> > mat2vec(vector<Mat> in)
{
    vector<vector<Point> > out;
    for(int i = 0; i < in.size(); i++)
    {
        vector<Point> tmp;
        Mat tmp_mat = in[i];
        for(int j = 0; j < tmp_mat.rows; j++)
        {
            tmp.push_back(Point(in[i].at<int>(j,0), in[i].at<int>(j,1)));
        }
        out.push_back(tmp);
    }
}


bool too_small(vector<Point> in)
{
    return in.size() < 5;
}

int n_bins = 600;
float range[] = {-10e+26, +10e+26};
const float *ranges[] = { range };
int channels[] = {0, 1};

int main()
{


    vector<vector<Point> > v1 = mat2vec(read_conts_from_file("/home/dedan/obj_att/out/cont0.yml"));
    vector<vector<Point> > v2 = mat2vec(read_conts_from_file("/home/dedan/obj_att/out/cont2.yml"));
    cout << v1[0][0] << endl;
    cout << v1[0][1] << endl;
    // filter out small contours
    v1.erase(remove_if(v1.begin(), v2.end(), too_small), v1.end());
    v2.erase(remove_if(v2.begin(), v2.end(), too_small), v2.end());
    cout << "here" << endl;

    if(v2.size() < v1.size())
    {
        vector<vector<Point> > tmp = v1;
        v1 = v2;
        v2 = tmp;
    }

    cout << v1.size() << endl;
    cout << v2.size() << endl;

    for(int i = 0; i < v1.size(); i++)
    {
        int i_max, v_max = 0;
        for(int j = 0; j < v2.size(); j++)
        {
            double match = matchShapes(v1[i], v2[j], CV_CONTOURS_MATCH_I1, 0);
            if(match > v_max)
            {
                v_max = match;
                i_max = j;
            }
        }
        Mat im1 = Mat::zeros(480, 640, CV_8UC1);
        drawContours(im1, v1, i, Scalar(255), CV_FILLED, 8);
//            simple_draw_cont(im1, v1[i]);
        imshow("bla", im1);
        waitKey(0);

    }

    return 0;
}
