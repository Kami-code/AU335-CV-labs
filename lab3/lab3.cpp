//
// Created by baochen on 12/2/20.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{

    Mat m_bi,m_bi_dilate,m_rect, m_rect2;
    m_rect = imread("/home/baochen/Pictures/picture.png");
    Size sizer = m_rect.size();
    resize(m_rect, m_rect, cv::Size(640, 480));
    m_rect2 = m_rect.clone();
    cvtColor(m_rect,m_bi,COLOR_BGR2GRAY);
    adaptiveThreshold(m_bi,m_bi,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY_INV,17,15); //适应性阈值二值化，基本上可以只得到单词的掩膜 


    Mat m_22 = m_bi.clone();  //对单词四方向膨胀，填充一般空洞。 
    dilate(m_22, m_22, Mat(), Point(-1, -1), 2);
    dilate(m_22, m_22, Mat(), Point(-1, -1), 2);
    dilate(m_22, m_22, Mat(), Point(-1, -1), 2);
    dilate(m_22, m_22, Mat(), Point(-1, -1), 2);
    dilate(m_22, m_22, Mat(), Point(-1, -1), 2);
    imshow("m_22",m_22);




    int tmp = m_bi.cols;
    Mat horizontalStructure_ = getStructuringElement(MORPH_RECT, Size(tmp,1)); //对图像横向膨胀，使得含单词的行为全黑，方便得到掩膜。 
    dilate(m_bi, m_bi_dilate, horizontalStructure_, Point(-1, -1));


    vector<Vec4f> plines;
    erode(m_bi_dilate, m_bi_dilate, Mat(), Point(-1, -1), 2);
    erode(m_bi_dilate, m_bi_dilate, Mat(), Point(-1, -1), 2);
    erode(m_bi_dilate, m_bi_dilate, Mat(), Point(-1, -1), 2);
    HoughLinesP(m_bi_dilate, plines, 1, CV_PI / 2, 300, 100.0, 1); //利用霍夫变换，找到直线。 
    cvtColor(m_bi_dilate,m_bi_dilate,COLOR_GRAY2BGR);
    Scalar color = Scalar(255, 255, 255);


    Mat mask = Mat::zeros(m_22.size(), CV_8UC1);
    cout << "line lengths = " <<plines.size() << endl;
    for (size_t i = 0; i < plines.size(); i++) {
        Vec4f hline = plines[i];
        line(mask, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
    }
    Mat mask2;
    mask.copyTo(mask2, m_22); //利用 开始得到的单词掩膜，得到与单词交集不为空的直线部分。 


    cvtColor(mask2,mask2, COLOR_GRAY2BGR);
    imshow("mask2",mask2);

    Mat img = mask2.clone(), img2;
    Mat mask5 = Mat::zeros(mask2.size(), CV_8UC1);

    Rect rect;
    rect.x = 50;
    rect.y = 310;
    rect.width = 100;
    rect.height = 30;//对腐蚀需要特判的区域，设置矩形掩模
    mask5(rect).setTo(255);
    mask2.copyTo(img2, mask5);
    Mat horizontalStructure_5 = getStructuringElement(MORPH_RECT, Size(1,5));
    erode(img2, img2, horizontalStructure_5, Point(-1, -1)); //纵向腐蚀，将覆盖线腐蚀为删除线 

    imshow("img2", img2);
    Mat mask4;
    Mat horizontalStructure_3 = getStructuringElement(MORPH_RECT, Size(1,11));
    erode(mask2, mask4, horizontalStructure_3, Point(-1, -1)); //纵向腐蚀，将覆盖线腐蚀为删除线 

    add(img2, mask4, mask4);

    add(mask4, m_rect, m_rect);
    imshow("mask4",mask4);
    imshow("m_rect",m_rect);


    Mat mask4not, m_rect3, ans;
    bitwise_not(mask4, mask4not);
    imshow("mask4not",mask4not);
    m_rect2.copyTo(m_rect3, mask4not);
    imshow("m_rect2",m_rect2);
    imshow("m_rect3",m_rect3);

    resize(m_rect3, ans, sizer);
    imshow("ans",ans);
    imwrite("/home/baochen/Pictures/ans.png",ans);
    waitKey(0);
	return 0;
}
