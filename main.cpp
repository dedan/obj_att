

/*
    help for writing this code came from the following sites:
    http://www.ros.org/wiki/pcl/Tutorials/Range%20image%20visualization
    http://www.ros.org/wiki/pcl/Tutorials/Range%20image%20creation#Tutorial_code
    http://pointclouds.org/documentation/tutorials/kinect_grabber.php
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/kinect_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <Eigen/Geometry>
#include "pcl/range_image/range_image.h"
#include <pcl/features/range_image_border_extractor.h>


class SimpleRangeViewer
{
  public:

    // constructor
    SimpleRangeViewer () : range_viewer("Range Viewer"){}


    // the function which is called regularly for visualization
    void range_cb_ (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
    {

        pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
        float angularResolution     = pcl::deg2rad(1.0f);
        float maxAngleWidth         = pcl::deg2rad(360.0f);
        float maxAngleHeight        = pcl::deg2rad(180.0f);
        Eigen::Affine3f sensorPose  = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        float noiseLevel            = 0.00;
        float minRange              = 0.0f;
        int borderSize              = 0;

        pcl::RangeImage range_image;
        range_image.createFromPointCloud(*cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                        sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
        pcl::RangeImageBorderExtractor border_extractor(&range_image);
        pcl::PointCloud<pcl::BorderDescription> border_descriptions;
        border_extractor.compute(border_descriptions);

        range_viewer.getRangeImageBordersWidget(range_image, -INFINITY, INFINITY, false,
                                                border_descriptions, "Range image with borders");
        range_viewer.setRangeImage(range_image, 0., 1000., true);
    }



    // the function which is called to run the class
    void run ()
    {

        // interface to the NI (from here we read the pointcloud
        pcl::Grabber* interface = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> g =
            boost::bind (&SimpleRangeViewer::range_cb_, this, _1);

        interface->registerCallback (g);
        interface->start();
        while(true)
        {
            pcl_visualization::ImageWidgetWX::spinOnce();  // process GUI events
            usleep(100000);
        }
        interface->stop ();
    }


    // class members
    pcl_visualization::RangeImageVisualizer range_viewer;
//    pcl_visualization::CloudViewer cloud_viewer;

};

int main ()
{
  SimpleRangeViewer v;
  v.run ();
  return 0;
}
