

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "siftfast.h"

using namespace cv;
using namespace std;


// histogramm settings
int n_bins = 256;
float range[] = {1, 255};
const float *ranges[] = { range };
int channels[] = {0, 1};


// extract image patch for sift feature computation
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


// draw the histogram
Mat draw_hist(const Mat& hist, int y_max, int x_max)
{

    Mat hist_img = Mat::zeros(x_max, y_max, CV_8UC1);


    for(int i=0; i<hist.rows; i++)
    {

        float scaleX = 1;
        float scaleY = 1;
        float cur_value   = hist.at<float>(i);
        float next_value  = hist.at<float>(i+1);

        double max_hist = 0;
        minMaxLoc(hist, 0, &max_hist);

        Point pt1 = Point(i*scaleX, 64*scaleY);
        Point pt2 = Point(i*scaleX+scaleX, 64*scaleY);
        Point pt3 = Point(i*scaleX+scaleX, (64-next_value*64/max_hist)*scaleY);
        Point pt4 = Point(i*scaleX, (64-cur_value*64/max_hist)*scaleY);

        int numPts = 5;
        Point pts[] = {pt1, pt2, pt3, pt4, pt1};

        fillConvexPoly(hist_img, pts, numPts, Scalar(127));
    }
    return hist_img;
}


// print framerate etc to stdout
void print_camera_info(VideoCapture capture)
{
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
}


// get a random color table for cluster coloring
vector<Vec3b> get_color_table(int n_entries)
{
    vector<Vec3b> color_tab;
    for( int i = 0; i < n_bins; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        color_tab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    return color_tab;
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
    print_camera_info(capture);



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

            //process the depth image
            if(capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP) )
            {
                const float scaleFactor = 0.05f;
                Mat show;
                depthMap.convertTo( show, CV_8UC1, scaleFactor );


                Mat hist;
                calcHist(&show, 1, channels, Mat(), hist, 1, &n_bins, ranges, true, false);

                // TODO smooth histogram (cvSmooth

                Mat cluster = Mat::zeros(show.size(), CV_8UC1);

                // histogram clustering
                int start = 0;
                int end   = 0;
                int c     = 1;
                cout << hist << endl << endl;
                for(int i = 0; i < hist.rows; i++)
                {
                    float cur_value   = hist.at<float>(i);

                    if(cur_value < hist.at<float>(start))
                        start = i;

                    if(cur_value > hist.at<float>(end))
                        end = i;

                    // end of valley
                    if(cur_value < 0.4 * hist.at<float>(end))
                    {
                        Mat mask = show > hist.at<float>(start) & show < hist.at<float>(i);
                        imshow("mask", mask);
                        waitKey();

                        Mat val = Mat::ones(cluster.size(), CV_8UC1) * c & mask;

                        imshow("cluster", cluster);
                        bitwise_or(cluster, val, cluster);
                        imshow("cluster", cluster);
                        waitKey(0);
                        start = i;
                        end = i;
                        c++;
                    }
                }

                break;
                cluster = (cluster / c) * 255;
                imshow("cluster", cluster);

//                Mat hist_img = Mat::zeros(x_max, y_max, CV_8UC1);


//                for(int i=0; i<hist.rows; i++)
//                {

//                    float scaleX = 1;
//                    float scaleY = 1;
//                    float cur_value   = hist.at<float>(i);
//                    float next_value  = hist.at<float>(i+1);

//                    double max_hist = 0;
//                    minMaxLoc(hist, 0, &max_hist);

//                    Point pt1 = Point(i*scaleX, 64*scaleY);
//                    Point pt2 = Point(i*scaleX+scaleX, 64*scaleY);
//                    Point pt3 = Point(i*scaleX+scaleX, (64-next_value*64/max_hist)*scaleY);
//                    Point pt4 = Point(i*scaleX, (64-cur_value*64/max_hist)*scaleY);

//                    int numPts = 5;
//                    Point pts[] = {pt1, pt2, pt3, pt4, pt1};

//                    fillConvexPoly(hist_img, pts, numPts, Scalar(127));
//                }


























                Mat hist_img = draw_hist(hist, 256, 64);

                Rect hist_rect(0,0,256,64);
                Mat hist_roi(show, hist_rect);
                hist_img.copyTo(hist_roi);
                rectangle(show, hist_rect, Scalar(255));
                imshow("depth", show);




//                Mat cluster(show.size(), CV_8UC3);

//                // paint the watershed image
//                for( int i = 0; i < show.cols; i++ )
//                {
//                    for( int j = 0; j < show.rows; j++ )
//                    {
//                        int idx = labels.at<int>(i*show.rows+j);
//                        cluster.at<Vec3b>(j,i) = colorTab[idx];
//                    }
//                }

//                //cluster = cluster*0.5 + show*0.5;
//                imshow( "depth", cluster );
                imshow("depth", show);


                //break;

            }

//            // process the rgb image
//            if(capture.retrieve( image, CV_CAP_OPENNI_GRAY_IMAGE ) )
//            {


//                Point p1 = Point(10, 10);
//                Point p2 = Point(100, 100);
//                Image img_sift = create_sift_patch(image, p1, p2);
//                Keypoint keypts = GetKeypoints(img_sift);
//                Keypoint key = keypts;
//                while(key)
//                {
//                    rectangle(image,
//                                Point(p1.x + key->col-1 , p1.y + key->row-1),
//                                Point(p1.x + key->col+1 , p1.y + key->row+1),
//                                Scalar(255,0,0), 1);
//                    key = key->next;
//                }
//                rectangle(image, p1, p2, Scalar(80, 0, 0), 1);

//                FreeKeypoints(keypts);
//                DestroyAllResources();
//                imshow( "rgb image", image);

//            }

            if( waitKey( 30 ) >= 0 )
                break;
        }
    }

    return 0;
    cvDestroyAllWindows();
}

