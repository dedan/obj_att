

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <time.h>

using namespace std;
using namespace cv;


// read in Matrix from textfile
Mat read_from_text(string fname)
{

    ifstream ifs(fname.c_str());
    vector<int> data;
    int n;
    while(ifs >> n)
    {
        data.push_back(n);
    }
    ifs.close();
    return Mat(data, true).reshape(1, 480);
}


// get a random color for cluster coloring
Scalar get_rand_col()
{
    int b = theRNG().uniform(0, 200);
    int g = theRNG().uniform(0, 255);
    int r = theRNG().uniform(0, 255);

    return Scalar((uchar)b, (uchar)g, (uchar)r);
}



int n_bins = 600;
float range[] = {1, 6000};
const float *ranges[] = { range };
int channels[] = {0, 1};
int width = 640;
int height = 480;
int hist_height = 64;
const float scale_factor = 0.05f;
int max_dist = 4000;


int main()
{

    clock_t tstart, tend;

    // read in the data
    Mat data = read_from_text("images/img_depth1.txt");
    data.convertTo(data, CV_16UC1);

    // measure execution time
    tstart = clock();

    // compute and smoooth histogram
    Mat hist;
    calcHist(&data, 1, channels, Mat(), hist, 1, &n_bins, ranges, true, false);
    boxFilter(hist, hist, -1, Size(1, 15));


    // histogram clustering
    int start = 0;
    int end   = 0;
    int c     = 1;

    Mat hist_img = Mat::zeros(hist_height, width, CV_8UC3);

    vector<Scalar> color_tab;
    color_tab.push_back(get_rand_col());
    Mat cluster = Mat::zeros(data.size(), CV_8UC1);
    Mat contours = Mat::zeros(data.size(), CV_8UC3);

    for(int i = 0; i < max_dist / 10; i++)
    {
        float cur_value     = hist.at<float>(i);
        float next_value    = hist.at<float>(i+1);

        // remember value as long it rises
        if(cur_value > hist.at<float>(end))
            end = i;

        // end of valley
        if(cur_value < 0.6 * hist.at<float>(end) &&
           1.1 * cur_value < next_value)
        {
            // get all the pixels in the depth image belonging to a cluster
            Mat mask = data > start * 10 & data < i * 10;
            Mat val = Mat::ones(cluster.size(), CV_8UC1) * c & mask;
            bitwise_or(cluster, val, cluster);

            vector<vector<Point> > conts;
            findContours(mask, conts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

            for(std::vector<int>::size_type j = 0; j != conts.size(); j++)
            {
                Scalar color( rand()&255, rand()&255, rand()&255 );
                drawContours( contours, conts, j, color, CV_FILLED, 8);
                cout << Mat(conts[j]) << endl;
                RotatedRect box = minAreaRect(Mat(conts[j]));
                Point2f vtx[4];
                box.points(vtx);
                for( int k = 0; k < 4; k++ )
                    line(contours, vtx[k], vtx[(k+1)%4], Scalar(0, 255, 0), 1, CV_AA);
            }




            // create a random color for the cluster and start again
            color_tab.push_back(get_rand_col());
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
        fillConvexPoly(hist_img, pts, n_points, color_tab.back());
    }

    tend = clock();
    cout << "execution time: " << (double)(tend-tstart)/CLOCKS_PER_SEC << endl;


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
//    rectangle(out, hist_rect, Scalar(255));
    imshow("main", out);

    imshow("conts", contours);

    waitKey(0);

    return 0;
}
