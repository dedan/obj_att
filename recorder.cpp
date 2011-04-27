

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;


int main()
{


    vector<Mat> v;
    const int n = 2;

    string names[n] = {"images/baboon.jpg", "images/fruits.jpg"}; //, "lena.tif"};
    for(int i =0; i < n; i++){
        Mat im = imread(names[i]);
        v.push_back(im);
    }


    for( vector<Mat>::iterator it = v.begin(); it != v.end(); it++ ){
        imshow("blub", *it);
    }
    waitKey(0);


    return 0;
}
