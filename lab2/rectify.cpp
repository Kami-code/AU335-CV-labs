//
// Created by baochen on 11/30/20.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void hello2() {
    Mat imgL = imread("/home/baochen/Pictures/left2.jpg");
    Mat imgR = imread("/home/baochen/Pictures/right2.jpg");
    std::vector <Point2f> tmpPts1, tmpPts2;

    tmpPts1.push_back(Point2f(635,153) );
    tmpPts1.push_back(Point2f(647,168) );
    tmpPts1.push_back(Point2f(609,602) );
    tmpPts1.push_back(Point2f(630,648) );
    tmpPts1.push_back(Point2f(963,649) );
    tmpPts1.push_back(Point2f(1011,714) );
    tmpPts1.push_back(Point2f(1198,688) );
    tmpPts1.push_back(Point2f(1365,700) );
    tmpPts1.push_back(Point2f(1401,105) );
    tmpPts1.push_back(Point2f(1385,119 ) );

    tmpPts2.push_back(Point2f(409,147) );
    tmpPts2.push_back(Point2f(423,163) );
    tmpPts2.push_back(Point2f(397,599) );
    tmpPts2.push_back(Point2f(415,646) );
    tmpPts2.push_back(Point2f(735,632) );
    tmpPts2.push_back(Point2f(773,699) );
    tmpPts2.push_back(Point2f(954,663) );
    tmpPts2.push_back(Point2f(1108,670) );
    tmpPts2.push_back(Point2f(1131,80) );
    tmpPts2.push_back(Point2f(1120,95) );


    Mat camK = (Mat_<float>(3, 3) <<  1.26806204e+05,0.00000000e+00,2.20188107e+03,0.00000000e+00,5.61470734e+04,1.99806318e+03,0.00000000e+00,0.00000000e+00,1.00000000e+00);
    cout << camK << endl;
    cv::Mat E = cv::findEssentialMat(tmpPts1, tmpPts2, camK, RANSAC);
    cout << "E = " << E << endl;
    cv::Mat R1, R2, t, R, P1,P2, Q;
    cv::decomposeEssentialMat(E, R1, R2, t);
    cout << "R1 = " << R1 << endl << " R2 = " << R2 << endl;
    R = R2.clone();
    t = -t.clone();
    Mat mapx, mapy;
    Mat recImgL, recImgR;
    Mat D = (Mat_<float>(1, 5) <<  -2.89682065e+02,9.37242996e+04,-1.06037273e-02,-7.16517920e-01,4.16254379e+04);
    cv::stereoRectify(camK, D, camK, D, imgL.size(), R, -R*t,  R1, R2, P1, P2, Q);
    cv::initUndistortRectifyMap(P1(cv::Rect(0, 0, 3, 3)), D, R1, P1(cv::Rect(0, 0, 3, 3)), imgL.size(), CV_32FC1, mapx, mapy);
    cv::remap(imgL, recImgL, mapx, mapy, INTER_LINEAR);
    cv::imwrite("/home/baochen/Pictures/recConyL.png", recImgL);

    cv::initUndistortRectifyMap(P2(cv::Rect(0, 0, 3, 3)), D, R2, P2(cv::Rect(0, 0, 3, 3)), imgL.size(), CV_32FC1, mapx, mapy);
    cv::remap(imgR, recImgR, mapx, mapy, INTER_LINEAR);
    cv::imwrite("/home/baochen/Pictures/recConyR.png", recImgR);

    return;
}
