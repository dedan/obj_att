

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "siftfast.h"
#include <time.h>
#include <cstdio>
#include <funcs.cpp>


using namespace cv;
using namespace std;

// constants
int n_bins = 600;
float range[] = {1, 6000};
const float *ranges[] = { range };
int channels[] = {0, 1};
int width = 640;
int height = 480;
int hist_height = 64;
const float scale_factor = 0.05f;
int n_colors = 100;
int max_dist = 4000;


class Obj{
public:
    Obj(RotatedRect _box, vector<Point> _cont)
    {
        box         = _box;
        cont        = _cont;
        inact_count = 0;
        id          = rand();
    }
    RotatedRect box;
    vector<Point> cont;
    int id;
    int inact_count;
    int size()
    {
        return box.boundingRect().area();
    }
    int compactness()
    {
        return -1;
    }
};


// extract image patch for sift feature computation
Image create_sift_patch(Mat image, Rect r)
{
    Image patch = CreateImage(r.height, r.width);
    for(int i = 0; i < r.height; ++i)
    {
        for(int j = 0; j < r.width; ++j)
        {

           patch->pixels[i*patch->stride+j] = image.at<uchar>(r.y + i, r.x + j) / 255.;
        }
    }
    return patch;
}

int main(int argc, char **argv)
{

    // parse commandline
    int k_width = 5;
    float perc  = 0.2;

    if(argc > 1)
    {
        k_width = atoi(argv[1]);
        perc    = atof(argv[2]);
    }

    //variables
    clock_t tstart, tend;
    Mat data;
    Mat hist;
    Mat image_raw;

    vector<Scalar> color_tab = get_rand_colors(n_colors);
    vector<Obj> objects;



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
        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {

            //process the depth image
            if(capture.retrieve( data, CV_CAP_OPENNI_DEPTH_MAP) )
            {

                // measure execution time
                tstart = clock();

                // compute and smoooth histogram
                calcHist(&data, 1, channels, Mat(), hist, 1, &n_bins, ranges, true, false);
                boxFilter(hist, hist, -1, Size(1, k_width));


                // histogram clustering
                int start = 0;
                int end   = 0;
                int c     = 1;
                Mat cluster = Mat::zeros(height, width, CV_8UC1);
                Mat contours = Mat::zeros(data.size(), CV_8UC3);
                Mat hist_img = Mat::zeros(hist_height, width, CV_8UC3);


                for(int i = 0; i < max_dist / 10; i++)
                {
                    float cur_value     = hist.at<float>(i);
                    float next_value    = hist.at<float>(i+1);

                    // remember value as long it rises
                    if(cur_value > hist.at<float>(end))
                        end = i;

                    // end of valley
                    if(cur_value < perc * hist.at<float>(end) &&
                       1.1 * cur_value < next_value)
                    {
                        // get all the pixels in the depth image belonging to a cluster
                        Mat mask = data > start * 10 & data < i * 10;
                        Mat val = Mat::ones(cluster.size(), CV_8UC1) * c & mask;
                        bitwise_or(cluster, val, cluster);

                        vector<vector<Point> > conts;
                        erode(mask, mask, getStructuringElement(MORPH_RECT, Size(10, 10)));


                        // find the contours (objects)
                        findContours(mask, conts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
                        for(std::vector<int>::size_type j = 0; j != conts.size(); j++)
                        {
                            // draw the rectangle
                            drawContours( contours, conts, j, color_tab[c], CV_FILLED, 8);
                            RotatedRect box = minAreaRect(Mat(conts[j]));
                            Point2f vtx[4];
                            box.points(vtx);
                            for( int k = 0; k < 4; k++ )
                                line(contours, vtx[k], vtx[(k+1)%4], Scalar(0, 255, 0), 1, CV_AA);

                            // and store them for tracking and sift

                            // forget about it if to small or not compact enough

                            // check whether a similar object is already in the list
                                // if yes, update the object
                                // if no, add it to the list




                            Obj tmp(box, conts[j]);
                            objects.push_back(tmp);
                        }

                        // start again
                        start = i;
                        end = i;
                        c++;
                    }

                    int n_points    = 5;
                    double max_hist = 0;
                    minMaxLoc(hist, 0, &max_hist);

                    Point pt1   = Point(i, hist_height);
                    Point pt2   = Point(i, hist_height);
                    Point pt3   = Point(i, (hist_height-next_value*hist_height/max_hist));
                    Point pt4   = Point(i, (hist_height-cur_value*hist_height/max_hist));
                    Point pts[] = {pt1, pt2, pt3, pt4, pt1};
                    fillConvexPoly(hist_img, pts, n_points, color_tab[c]);
                }

                tend = clock();
   //             cout << "execution time: " << (double)(tend-tstart)/CLOCKS_PER_SEC << endl;


                // ########### draw everything in a final image;
                Mat out(height + hist_height, width, CV_8UC3);

                // the cluster image
                Mat col_clust = Mat(cluster.size(), CV_8UC3);
                for( int i = 0; i < cluster.rows; i++ )
                {
                    for( int j = 0; j < cluster.cols; j++ )
                    {
                        int idx = cluster.at<uchar>(i,j);

                        col_clust.at<Vec3b>(i,j)[0] = color_tab[idx-1][0];
                        col_clust.at<Vec3b>(i,j)[1] = color_tab[idx-1][1];
                        col_clust.at<Vec3b>(i,j)[2] = color_tab[idx-1][2];
                    }
                }
                Mat outroi = out(Rect(0, hist_height, width, height));
                col_clust.copyTo(outroi);

                // the histogram
                Rect hist_rect(0,0,width, hist_height);
                outroi = out(hist_rect);
                hist_img.copyTo(outroi);

                imshow("main", out);

                imshow("conts", contours);



            }

            // process the rgb image
            Mat image;
            if(capture.retrieve( image_raw, CV_CAP_OPENNI_GRAY_IMAGE ) )
            {

                image = image_raw.clone();
                // for all the detected objects
                for(std::vector<int>::size_type i = 0; i != objects.size(); i++)
                {
                    Rect obj_frame = objects[i].box.boundingRect();
//                    Image img_sift = create_sift_patch(image, obj_frame);
//                    Keypoint keypts = GetKeypoints(img_sift);
//                    Keypoint key = keypts;
//                    while(key)
//                    {
//                        rectangle(image,
//                                    Point(obj_frame.x + key->col-1 , obj_frame.y + key->row-1),
//                                    Point(obj_frame.x + key->col+1 , obj_frame.y + key->row+1),
//                                    Scalar(255,0,0), 1);
//                        key = key->next;
//                    }
                    rectangle(image, obj_frame, Scalar(80, 0, 0), 1);

//                    FreeKeypoints(keypts);
//                    DestroyAllResources();

                }
                imshow( "rgb image", image);

            }

            cout << objects.size() << endl;

            char key = waitKey(30);
            if(key == 27) break;
            if(key == 's')
            {
                // set jpg quality to 100%
                std::vector<int> p;
                p.push_back(CV_IMWRITE_JPEG_QUALITY);
                p.push_back(100);

                stringstream fname;
                string ts = create_timestamp();
                fname << "./images/img_depth" << ts << ".txt";
                save_as_text(fname.str(), data);

                fname.str("");
                fname << "./images/img_rgb" << ts << ".jpg";
                imwrite(fname.str(), image_raw, p);

                cout << "saved to img" << ts << endl;
            }
        }
    }

    return 0;
    cvDestroyAllWindows();
}

