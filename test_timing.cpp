


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

    int c;
    float d;
    SimpleRangeViewer(float del)
    {
        c = 0;
        d = del;
    }


    // callback
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud)
    {
        usleep(d * 1000.);
        c++;
        if(c % 30 == 0){
            cout << " bla bla" << endl;
        }
    }


    // the function which is called to run the class
    void run ()
    {
        // interface to the NI (from here we read the pointcloud
        pcl::Grabber* interface = new pcl::OpenNIGrabber();

        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr&)> f =
                boost::bind (&SimpleRangeViewer::cloud_cb_, this, _1);

        interface->registerCallback(f);

        interface->start();
        while(true)
        {
            usleep(1000000);
        }

    }
};


int main(int argc, char* argv[])
{
    SimpleRangeViewer v(atof(argv[1]));
    v.run ();
    return 0;
}
