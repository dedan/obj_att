


#include "cv.h" //main OpenCV header
#include "highgui.h" //GUI header
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/kinect_grabber.h>
#include <stdio.h>

#define WIDTH 640
#define HEIGHT 480


class CrashTest
{

private:
    IplImage* img_depth;
    IplImage* img_rgb;
    pcl::Grabber* interface;
    int c;

public:

    CrashTest() : interface(new pcl::OpenNIGrabber()),
    img_depth(cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_32F, 1)),
    img_rgb(cvCreateImage(cvSize(WIDTH, HEIGHT), IPL_DEPTH_8U, 3)),
    c(0)
    {}


    // callback
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {
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
            }
        }
        cvShowImage("depth", img_depth);
        cvShowImage("rgb", img_rgb);
    }


    // the function which is called to run the class
    void run ()
    {
        // interface to the NI (from here we read the pointcloud
        pcl::Grabber* interface = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
                boost::bind (&CrashTest::cloud_cb_, this, _1);

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


int main()
{
    CrashTest v;
    v.run ();
    return 0;
}
