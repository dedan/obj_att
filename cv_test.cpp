

#include "cv.h" //main OpenCV header
#include "highgui.h" //GUI header
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/kinect_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <Eigen/Geometry>
#include "pcl/range_image/range_image.h"
#include <pcl/features/range_image_border_extractor.h>
#include <stdio.h>


class CvTest{
private:
    IplImage* img_depth;
    IplImage* img_rgb;
    pcl::Grabber* interface;

public:
    CvTest() : interface(new pcl::OpenNIGrabber()),
               img_depth(cvCreateImage(cvSize(640,480),IPL_DEPTH_32F,1)),
               img_rgb(cvCreateImage(cvSize(640,480),IPL_DEPTH_8U,3))
               {}

    // cloud callback
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {

        for(int i = 0; i < cloud->width-1; ++i)
        {
            for(int j = 0; j < cloud->height-1; ++j)
            {
                CvScalar s;
                s.val[0] = (*cloud)(i,j).z;
                cvSet2D(img_depth,j,i,s);

                float frgb = (*cloud)(i,j).rgb;
                int rgb = *reinterpret_cast<int*>(&frgb);
                s.val[0] = ((rgb >> 16) & 0xff);
                s.val[1] = ((rgb >> 8) & 0xff);
                s.val[2] = (rgb & 0xff);
                cvSet2D(img_rgb,j,i,s);

             }
        }
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
