

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>


using namespace std;
using namespace cv;

Mat read_from_texta(string fname)
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





int n_bins = 600;
float range[] = {1, 8000};
const float *ranges[] = { range };
int channels[] = {0, 1};
int width = 640;
int height = 480;
int hist_height = 64;
const float scale_factor = 0.05f;


int main()
{

    // read in the data
    Mat data = read_from_texta("images/img_depth1.txt");
    data.convertTo(data, CV_16UC1);

    // compute histogram
    Mat hist;
    calcHist(&data, 1, channels, Mat(), hist, 1, &n_bins, ranges, true, false);


    // histogram clustering
    int start = 0;
    int end   = 0;
    int c     = 1;
    cout << hist << endl << endl;

    Mat hist_img = Mat::zeros(hist_height, width, CV_8UC1);

    for(int i = 0; i < hist.rows; i++)
    {
        float cur_value = hist.at<float>(i);
        float next_value  = hist.at<float>(i+1);


        if(cur_value > hist.at<float>(end))
            end = i;

        // end of valley
        if(cur_value < 0.4 * hist.at<float>(end))
        {
//            Mat mask = show > hist.at<float>(start) & show < hist.at<float>(i);
//            imshow("mask", mask);
//            waitKey();

//            Mat val = Mat::ones(cluster.size(), CV_8UC1) * c & mask;

//            imshow("cluster", cluster);
//            bitwise_or(cluster, val, cluster);
//            imshow("cluster", cluster);
//            waitKey(0);



            start = i;
            end = i;
            c++;
        }

        float scaleX = 1;
        float scaleY = 1;

        double max_hist = 0;
        minMaxLoc(hist, 0, &max_hist);

        Point pt1 = Point(i*scaleX, hist_height*scaleY);
        Point pt2 = Point(i*scaleX+scaleX, hist_height*scaleY);
        Point pt3 = Point(i*scaleX+scaleX, (hist_height-next_value*hist_height/max_hist)*scaleY);
        Point pt4 = Point(i*scaleX, (hist_height-cur_value*hist_height/max_hist)*scaleY);

        int numPts = 5;
        Point pts[] = {pt1, pt2, pt3, pt4, pt1};

        cout << (int)floor((start/width)*255) << endl;
        fillConvexPoly(hist_img, pts, numPts, Scalar((int)floor(((float)start/width)*255)));

    }





    // draw everything in a final image;
    Mat out(height + hist_height, width, CV_8UC1);
    Mat show;
    data.convertTo( show, CV_8UC1, scale_factor );
    Mat outroi = out(Rect(0, hist_height, width, height));
    show.copyTo(outroi);

    Rect hist_rect(0,0,width, hist_height);
    outroi = out(hist_rect);
    hist_img.copyTo(outroi);
    rectangle(out, hist_rect, Scalar(255));
    imshow("depth", out);
    waitKey(0);

    // TODO smooth histogram (cvSmooth

//    Mat cluster = Mat::zeros(show.size(), CV_8UC1);



//    break;
//    cluster = (cluster / c) * 255;
//    imshow("cluster", cluster);





//    Mat frame = Mat::ones(10,10,CV_8UC1);

//    randu(frame, Scalar(0), Scalar(256));
//    cout << frame << endl << endl;

//    Mat mask = frame <100 | frame > 200;
//    cout << mask << endl << endl;



//    Mat res;
//    frame = frame & mask;
// //   cout << frame << endl << endl;


//    Mat not_mask;
//    bitwise_not(mask, not_mask);
//   // cout << not_mask << endl << endl;

//    Mat val = Mat::ones(frame.size(), CV_8UC1) * 7 & not_mask;
//  //  cout << val << endl << endl;


//    bitwise_or(frame, val, res);
//    cout << res << endl;

    return 0;
}
