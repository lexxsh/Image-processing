#include <iostream>
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
const double PI = 3.14159265358979323846;
int KK[9][9] = { 0 };
double gaussian(int x, int y,double sigma) {
    return exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * PI * sigma * sigma);
}
void printkernel() {
    const int filterSize = 9;
    double filter[filterSize][filterSize];
    double sum = 0.0;
    int center = filterSize / 2; // 마스크의 중심 위치

    for (int x = 0; x < filterSize; ++x) {
        for (int y = 0; y < filterSize; ++y) {
            // 중심으로부터의 거리 계산
            int dx = x - center;
            int dy = y - center;
            filter[x][y] = gaussian(dx, dy, 1.0); // 표준 편차는 1로 설정
            sum += filter[x][y];
        }
    }

    // 필터 마스크 정규화 및 출력
    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            filter[i][j] /= sum;
            std::cout << filter[i][j] << " ";
            KK[i][j] = filter[i][j];
        }
        std::cout << std::endl;
    }
}


int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
    int sum = 0;
    int sumKernel = 0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
                sumKernel += kernel[i + 1][j + 1];
            }
        }
    }
    if (sumKernel != 0) { return sum / sumKernel; }
    else { return sum; }
}
int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height) {
    int sum = 0;
    int sumKernel = 0;
    for (int j = -4; j <= 4; j++) {
        for (int i = -4; i <= 4; i++) {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
                sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
                sumKernel += kernel[i + 4][j + 4];
            }
        }
    }
    if (sumKernel != 0) { return sum / sumKernel; }
    else { return sum; }
}
Mat myGaussianFilter(Mat srcImg) {
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[9][9] = {
        {1, 2, 4, 5, 6, 5, 4, 2, 1},
        {2, 5, 8, 12, 13, 12, 8, 5, 2},
        {4, 8, 15, 22, 25, 22, 15, 8, 4},
        {5, 12, 22, 31, 36, 31, 22, 12, 5},
        {6, 13, 25, 36, 40, 36, 25, 13, 6},
        {5, 12, 22, 31, 36, 31, 22, 12, 5},
        {4, 8, 15, 22, 25, 22, 15, 8, 4},
        {2, 5, 8, 12, 13, 12, 8, 5, 2},
        {1, 2, 4, 5, 6, 5, 4, 2, 1}
    };
    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dstData[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height);
        }
    }
    return dstImg;
}
Mat GetHistogram(Mat src) {
    Mat histogram;
    const int* channel_numbers = { 0 };
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = channel_range;
    int number_bins = 255;

    calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / number_bins);

    Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < number_bins; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
    }

    return histImage;
}

Mat SpreadSalts_pepers(Mat img, int num) {
    Mat dst_img = img;
    for (int n = 0; n < num; n++) {
        int x = rand() % dst_img.cols;
        int y = rand() % dst_img.rows;
        img.at<uchar>(y, x) = 255;
    }
    for (int n = 0; n < num; n++) {
        int x = rand() % dst_img.cols;
        int y = rand() % dst_img.rows;
        img.at<uchar>(y, x) = 0;
    }
    return dst_img;
}
Mat mySobelFilter(Mat srcImg) {

    int kernelX[3][3] = {
     {-2, -1, 0},
     {-1,  0, 1},
     { 0,  1, 2} // 45도 마스크
    };
    int kernelY[3][3] = {
        {0, 1, 2},
        { -1,  0,  1},
        { -2,  -1,  0} // 135도 마스크
    };
    Mat dstImg(srcImg.size(), CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    int width = srcImg.cols;
    int height = srcImg.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gx = abs(myKernelConv3x3(srcData, kernelX, x, y, width, height));
            int gy = abs(myKernelConv3x3(srcData, kernelY, x, y, width, height));
            dstData[y * width + x] = (gx + gy) / 2; 
        }
    }
    //for (int y = 0; y < height; y++) {    ////45도
    //    for (int x = 0; x < width; x++) {
    //        int gx = abs(myKernelConv3x3(srcData, kernelX, x, y, width, height));
    //        int gy = abs(myKernelConv3x3(srcData, kernelY, x, y, width, height));
    //        dstData[y * width + x] = gx;
    //    }
    //}
    //for (int y = 0; y < height; y++) {    ////135도
    //    for (int x = 0; x < width; x++) {
    //        int gx = abs(myKernelConv3x3(srcData, kernelX, x, y, width, height));
    //        int gy = abs(myKernelConv3x3(srcData, kernelY, x, y, width, height));
    //        dstData[y * width + x] = gy;
    //    }
    //}
    return dstImg;
}
int myKernelConv9x9_Color(uchar* arr, int kernel[][9], int x, int y, int width, int height, int color) {
    int sum = 0;
    int sumKernel = 0;
    //특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
    for (int j = -4; j <= 4; j++) {
        for (int i = -1; i <= 1; i++) {
            if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
                //영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
                sum += arr[(y + j) * width * 3 + (x + i) * 3 + color] * kernel[i + 4][j + 4];
                sumKernel += kernel[i + 4][j + 4];
            }
        }
    }
    if (sumKernel != 0) {
        return sum / sumKernel;
    }
    else {
        return sum;
    }
}

Mat myGaussianFilter_Color(Mat srcImg) {
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[9][9] = {
        {1, 2, 4, 5, 6, 5, 4, 2, 1},
        {2, 5, 8, 12, 13, 12, 8, 5, 2},
        {4, 8, 15, 22, 25, 22, 15, 8, 4},
        {5, 12, 22, 31, 36, 31, 22, 12, 5},
        {6, 13, 25, 36, 40, 36, 25, 13, 6},
        {5, 12, 22, 31, 36, 31, 22, 12, 5},
        {4, 8, 15, 22, 25, 22, 15, 8, 4},
        {2, 5, 8, 12, 13, 12, 8, 5, 2},
        {1, 2, 4, 5, 6, 5, 4, 2, 1}
    }; //9x9 형태의 gaussian 마스크 배열

    Mat dstImg(height, width, CV_8UC3/*CV_8uC1*/);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dstData[y * width * 3 + x * 3] = myKernelConv9x9_Color(srcData, kernel, x, y, width, height, 0);
            dstData[y * width * 3 + x * 3 + 1] = myKernelConv9x9_Color(srcData, kernel, x, y, width, height, 1);
            dstData[y * width * 3 + x * 3 + 2] = myKernelConv9x9_Color(srcData, kernel, x, y, width, height, 2);
        }
    }
    return dstImg;
}
Mat mySampling(Mat srcImg) {
    int width = srcImg.cols / 2;
    int height = srcImg.rows / 2;
    Mat dstImg(height, width, CV_8UC3);

    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            dstData[y * width * 3 + x * 3] = srcData[(y * 2) * (width * 2) * 3 + (x * 2) * 3];
            dstData[y * width * 3 + x * 3 + 1] = srcData[(y * 2) * (width * 2) * 3 + (x * 2) * 3 + 1];
            dstData[y * width * 3 + x * 3 + 2] = srcData[(y * 2) * (width * 2) * 3 + (x * 2) * 3 + 2];
        }
    }
    return dstImg;
}
vector<Mat> myGaussianPyramid(Mat srcImg, int levels) {
    vector<Mat> pyramid;
    pyramid.push_back(srcImg);

    for (int i = 0; i < levels; i++) {
        srcImg = mySampling(srcImg);
        srcImg = myGaussianFilter_Color(srcImg);
        pyramid.push_back(srcImg);
    }

    return pyramid;
}
void exGaussianPyramid(Mat srcImg) {
    vector<Mat> gaussianPyramid = myGaussianPyramid(srcImg, 5);

    // Display the pyramid images
    for (size_t i = 0; i < gaussianPyramid.size(); ++i) {
        imshow("Gaussian Pyramid Level " + to_string(i), gaussianPyramid[i]);
    }
}

vector<Mat> myLaplacianPyramid(Mat srcImg, int levels) {
    vector<Mat> pyramid;
    pyramid.push_back(srcImg);
    Mat dst_img;
    for (int i = 0; i < levels; i++) {
        if (i != levels - 1) {
            Mat Firstimg = srcImg;
            srcImg = mySampling(srcImg);
            srcImg = myGaussianFilter_Color(srcImg);
            Mat lowImg;
            resize(srcImg, lowImg, Firstimg.size());
            pyramid.push_back(Firstimg - lowImg + Scalar(128, 128, 128)); // Adding 128 to each channel
        } else {
            pyramid.push_back(srcImg);
        }
    }

    return pyramid;
}
void exLaplacianPyramid(Mat srcImg) {
    vector<Mat> VecLap = myLaplacianPyramid(srcImg, 4);
    Mat dst_img;
    reverse(VecLap.begin(), VecLap.end());
    for (int i = 0; i < VecLap.size(); i++) {
        imshow("Laplacian pyramid", VecLap[i]);
        waitKey(0);
    }
    for (int i = 0; i < VecLap.size(); i++) {

        if (i == 0) {
            dst_img = VecLap[i];
        }
        else {
            resize(dst_img, dst_img, VecLap[i].size());
            dst_img = dst_img + VecLap[i] - 128;
        }
        string fname = "recovered_image" + to_string(i) + ".png";
        imwrite(fname, dst_img);
        string fname2 = "lap_pyr_img" + to_string(i) + ".png";
        imwrite(fname2, VecLap[i]);
        imshow("Recovered_img", dst_img);
        waitKey(0);
    }
}
int main() {

    //printkernel();
    Mat src_img = imread("gear.jpg", 1);
    Mat dst_img, dst_img2, dst_img3, dst_img4, dst_img5, dst_img6;

   /* dst_img = myGaussianFilter(src_img);
    hconcat(src_img, dst_img, dst_img);
    imshow("Task #1", dst_img);*/

    //Mat histo_img = GetHistogram(src_img);
    //Mat histo_img2 = GetHistogram(dst_img);
    //hconcat(histo_img, histo_img2, dst_img2);
    //imshow("Task #1_2", dst_img2);


    //dst_img3 = SpreadSalts_pepers(src_img, 1000);
    //dst_img4 = myGaussianFilter(dst_img3);

    //hconcat(dst_img3, dst_img4, dst_img4);
    //imshow("Task 2", dst_img4);

    //dst_img5 = mySobelFilter(src_img);
    //imshow("Task 3", dst_img5);

    //exGaussianPyramid(src_img);

    exLaplacianPyramid(src_img);
    waitKey(0);
    destroyAllWindows();
}