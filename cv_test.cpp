

#include "cv.h" //main OpenCV header
#include "highgui.h" //GUI header
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/kinect_grabber.h>
#include <stdio.h>
#include "siftfast.h"
#include "omp.h"

#define WIDTH 640
#define HEIGHT 480

class CvTest{
private:
    IplImage* img_depth;
    IplImage* img_rgb;
    pcl::Grabber* interface;
    int c;


public:
    CvTest() : interface(new pcl::OpenNIGrabber()),
    img_depth(cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_32F, 1)),
    img_rgb(cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3)),
    c(0)
    {}

    // cloud callback
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {


        Image img_sift = CreateImage(HEIGHT, WIDTH);
        Keypoint keypts;
        for(int r = 0; r < cloud->height; ++r)
        {
            for(int c = 0; c < cloud->width; ++c)
            {
                CvScalar s;
                s.val[0] = (*cloud)(c,r).z;
                cvSet2D(img_depth,r,c,s);

                float frgb = (*cloud)(c,r).rgb;
                int rgb = *reinterpret_cast<int*>(&frgb);
                s.val[2] = ((rgb >> 16) & 0xff);
                s.val[1] = ((rgb >> 8) & 0xff);
                s.val[0] = (rgb & 0xff);
                cvSet2D(img_rgb,r,c,s);


                img_sift->pixels[r*img_sift->stride+c] = (s.val[0] + s.val[1] + s.val[2]) / (3.*255);
            }
        }
        if(c % (1 * 10) == 0){
            # pragma omp parallel
            {
                keypts = GetKeypoints(img_sift);
                Keypoint key = keypts;
                while(key)
                {
                    cvRectangle(img_rgb,
                                cvPoint(key->col-1 ,key->row-1),
                                cvPoint(key->col+1 ,key->row+1),
                                cvScalar(255,0,0), 1);
                    key = key->next;
                }
                FreeKeypoints(keypts);
                DestroyAllResources();
            }

        }
        c++;
        cvShowImage("depth", img_depth);
        cvShowImage("rgb", img_rgb);
    }


    void run()
    {
        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
                boost::bind (&CvTest::cloud_cb_, this, _1);

        interface->registerCallback(f);

        cvNamedWindow("depth", 1);
        cvMoveWindow("depth", 100, 200);
        cvNamedWindow("rgb", 1);
        cvMoveWindow("rgb", 800, 200);

        interface->start();
        cvWaitKey(0);
        interface->stop();

        cvDestroyAllWindows();
        cvReleaseImage(&img_depth);
        cvReleaseImage(&img_rgb);
    }
};




int main() {
    CvTest v;
    v.run ();
    return 0;
}


// TODO pointcloud aufnehmen
// TODO den z wert sift skala quotienten mit in den 128 d sift feature vector
// TODO nans aus tiefenbild entfernen (erst mal schauen was da so drin steht
