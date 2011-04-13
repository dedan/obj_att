

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
private:

    // class members
    pcl_visualization::RangeImageVisualizer range_viewer;
    pcl_visualization::CloudViewer cloud_viewer;
    pcl::RangeImage::CoordinateFrame coordinate_frame;
    float angularResolution;
    float maxAngleWidth;
    float maxAngleHeight;
    Eigen::Affine3f sensorPose;
    float noiseLevel;
    float minRange;
    int borderSize;

public:

    // constructor - initialization
    SimpleRangeViewer () :  cloud_viewer("cloud"),
                            range_viewer("Range Viewer"),
                            coordinate_frame(pcl::RangeImage::CAMERA_FRAME),
                            angularResolution(pcl::deg2rad(1.0f)),
                            maxAngleWidth(pcl::deg2rad(360.0f)),
                            maxAngleHeight(pcl::deg2rad(180.0f)),
                            sensorPose((Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f)),
                            noiseLevel(0.00),
                            minRange(0.0f),
                            borderSize(0){}


    // visualize the cloud
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {
        if (!cloud_viewer.wasStopped())
            cloud_viewer.showCloud(*cloud);
    }


    // visualize the depth image with borders
    void range_cb_ (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
    {
        pcl::RangeImage range_image;
        range_image.createFromPointCloud(*cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                         sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
        pcl::RangeImageBorderExtractor border_extractor(&range_image);
        pcl::PointCloud<pcl::BorderDescription> border_descriptions;
        border_extractor.compute(border_descriptions);

        range_viewer.visualizeBorders(range_image, -INFINITY, INFINITY, false, border_descriptions);
    }


    // the function which is called to run the class
    void run ()
    {
        // interface to the NI (from here we read the pointcloud
        pcl::Grabber* interface = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> g =
                boost::bind (&SimpleRangeViewer::range_cb_, this, _1);
        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
                boost::bind (&SimpleRangeViewer::cloud_cb_, this, _1);

        interface->registerCallback(g);
        interface->registerCallback(f);

        interface->start();
        while(!cloud_viewer.wasStopped() || range_viewer.isShown())
        {
            pcl_visualization::ImageWidgetWX::spinOnce();  // process GUI events
            usleep(100000);
        }
        interface->stop ();
    }
};


int main ()
{
    SimpleRangeViewer v;
    v.run ();
    return 0;
}
