


#include "cv.h" //main OpenCV header
#include "highgui.h" //GUI header
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/kinect_grabber.h>
#include <stdio.h>
#include "siftfast.h"


int main(){

    IplImage* lena = cvLoadImage("lena.tif", 0);
    Image img_sift = CreateImage(lena->height, lena->width);
    for(int r = 0; r < lena->height; ++r)
    {
        for(int c = 0; c < lena->width; ++c)
        {

            CvScalar s = cvGet2D(lena,c,r);
            img_sift->pixels[r*img_sift->stride+c] =
                    (s.val[0] + s.val[1] + s.val[2]) / (3.*255);
        }
    }
    Keypoint keypts = GetKeypoints(img_sift);
    Keypoint key = keypts;
    while(key)
    {
        cvRectangle(lena,
                    cvPoint(key->row-1 ,key->col-1),
                    cvPoint(key->row+1 ,key->col+1),
                    cvScalar(255,0,0), 1);
        key = key->next;
    }

    FreeKeypoints(keypts);
    DestroyAllResources();
    cvNamedWindow("lena", 1);
    cvMoveWindow("lena", 800, 200);

    cvShowImage("lena", lena);

    cvWaitKey(0);

    cvDestroyAllWindows();
    cvReleaseImage(&lena);


}

