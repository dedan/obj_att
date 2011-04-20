

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "siftfast.h"

using namespace cv;
using namespace std;


Image create_sift_patch(Mat image, Point p1, Point p2)
{
    assert(p2.y > p1.y);
    assert(p2.x > p1.x);
    int height = p2.y - p1.y;
    int width = p2.x - p1.x;
    Image patch = CreateImage(height, width);
    for(int r = 0; r < height; ++r)
    {
        for(int c = 0; c < width; ++c)
        {

           patch->pixels[r*patch->stride+c] = image.at<uchar>(p1.y + r, p1.x + c) / 255.;
        }
    }
    return patch;
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

    capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ ); // default

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

    for(;;)
    {
        Mat depthMap;
        Mat image;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if(capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show;
                depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if(capture.retrieve( image, CV_CAP_OPENNI_GRAY_IMAGE ) )
            {


                Point p1 = Point(10, 10);
                Point p2 = Point(100, 100);
                Image img_sift = create_sift_patch(image, p1, p2);
                Keypoint keypts = GetKeypoints(img_sift);
                Keypoint key = keypts;
                while(key)
                {
                    rectangle(image,
                                Point(p1.x + key->col-1 , p1.y + key->row-1),
                                Point(p1.x + key->col+1 , p1.y + key->row+1),
                                Scalar(255,0,0), 1);
                    key = key->next;
                }
                rectangle(image, p1, p2, Scalar(80, 0, 0), 1);

                FreeKeypoints(keypts);
                DestroyAllResources();
                imshow( "rgb image", image);

            }

            if( waitKey( 30 ) >= 0 )
                break;
        }
    }

    return 0;
    cvDestroyAllWindows();
}


// TODO depth map maske benutzen

