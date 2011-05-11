
#include <iostream>
#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <boost/format.hpp>
#include "siftfast.h"



using namespace std;
using namespace cv;

void save_as_text(string fname, Mat data)
{
    FILE* fp = fopen(fname.c_str(), "w");

    for (int i = 0 ; i < data.rows ; i++ ){
        for (int j = 0 ; j < data.cols ; j++ ){
            fprintf(fp, "%i\t", data.at<ushort>(i,j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}



string create_timestamp()
{
    time_t ltime;
    struct tm *Tm;

    ltime=time(NULL);
    Tm=localtime(&ltime);

    boost::format fmter("%d%d%d_%d%d%d");
    fmter % Tm->tm_mday
          % (Tm->tm_mon+1)
          % (Tm->tm_year+1900)
          % Tm->tm_hour
          % Tm->tm_min
          % Tm->tm_sec;
    return fmter.str();
}


// extract image patch for sift feature computation
Image create_sift_patch(Mat image, Rect r)
{
    Image patch = CreateImage(r.height, r.width);
    for(int i = 0; i < r.height; ++i)
    {
        for(int j = 0; j < r.width; ++j)
        {

           patch->pixels[i*patch->stride+j] = image.at<uchar>(r.y + i, r.x + j) / 255.;
        }
    }
    return patch;
}


// print framerate etc to stdout
void print_camera_info(VideoCapture capture)
{
    // Print some avalible Kinect settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH    " << capture.get( CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT   " << capture.get( CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH    " << capture.get( CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS    " << capture.get( CV_CAP_PROP_FPS ) << endl;

    cout << "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH    " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT   " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS    " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;
}


// get a random color table for cluster coloring
vector<Scalar> get_rand_colors(int n_bins)
{
    vector<Scalar> table;
    for(int i = 0; i < n_bins; i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        table.push_back(Scalar((uchar)b, (uchar)g, (uchar)r));
    }
    return table;
}

