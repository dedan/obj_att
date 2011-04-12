

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


class SimpleRangeViewer
{
  public:

    // constructor
    SimpleRangeViewer () : cloud_viewer("Cloud Viewer"),
                            range_viewer("Range Viewer"){}



    // the function which is called regularly for visualization
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {
        if (!cloud_viewer.wasStopped())
          cloud_viewer.showCloud(*cloud);
    }


    // the function which is called regularly for visualization
    void range_cb_ (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
    {

        pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
        float angularResolution     =   1.0f * (M_PI/180.0f); //   1.0 degree in rad
        float maxAngleWidth         = 360.0f * (M_PI/180.0f); // 360.0 degree in rad
        float maxAngleHeight        = 180.0f * (M_PI/180.0f); // 180.0 degree in rad
        Eigen::Affine3f sensorPose  = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
        float noiseLevel            = 0.00;
        float minRange              = 0.0f;
        int borderSize              = 0;

        pcl::RangeImage rangeImage;
        rangeImage.createFromPointCloud(*cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                        sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
        range_viewer.setRangeImage(rangeImage, 0., 1000., true);
    }



    // the function which is called to run the class
    void run ()
    {

        // interface to the NI (from here we read the pointcloud
        pcl::Grabber* interface = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
            boost::bind (&SimpleRangeViewer::cloud_cb_, this, _1);

        boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> g =
            boost::bind (&SimpleRangeViewer::range_cb_, this, _1);


      boost::signals2::connection c = interface->registerCallback (f);
      boost::signals2::connection d = interface->registerCallback (g);

      
      interface->start ();      
      while(!cloud_viewer.wasStopped() || range_viewer.isShown())
      {
          pcl_visualization::ImageWidgetWX::spinOnce();  // process GUI events
          usleep(100000);
      }
      interface->stop ();
    }


    // class members
    pcl_visualization::RangeImageVisualizer range_viewer;
    pcl_visualization::CloudViewer cloud_viewer;

};

int main ()
{
  SimpleRangeViewer v;
  v.run ();
  return 0;
}
