

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int imcount = 0;


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



int main()
{

    cout << "Kinect opening ..." << endl;
    VideoCapture capture( CV_CAP_OPENNI );
    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ); // default


    for(;;)
    {
        Mat depthMap;
        Mat image;
        const float scaleFactor = 0.05f;
        Mat show;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {

            //process the depth image
            if(capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP) )
            {
                depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow("depth image", show);

            }

            if(capture.retrieve( image, CV_CAP_OPENNI_GRAY_IMAGE) )
            {
                imshow("beide", 0.5 * show + 0.5 * image);
            }


            // process the rgb image
            if(capture.retrieve( image, CV_CAP_OPENNI_BGR_IMAGE) )
            {
                imshow( "rgb image", image);
            }



            char key = waitKey(20);
            if(key == 27) break;
            if(key == 's')
            {
                // set jpg quality to 100%
                std::vector<int> p;
                p.push_back(CV_IMWRITE_JPEG_QUALITY);
                p.push_back(100);

                stringstream fname;
                fname << "./images/img_depth" << imcount << ".txt";
                save_as_text(fname.str(), depthMap);

                fname.str("");
                fname << "./images/img_rgb" << imcount << ".jpg";
                imwrite(fname.str(), image, p);

                cout << "saved to img" << imcount << endl;
                imcount++;
            }
        }
    }

    return 0;
    cvDestroyAllWindows();
}

