#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 이미지와 노출 시간을 읽어오는 함수
void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
    int numImages = 4;
    static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
    times.assign(timesArray, timesArray + numImages);  // 노출 시간 설정
    static const char* filenames[] = { "11/im.jpg", "11/im1.jpg", "11/im2.jpg", "11/im3.jpg" };
    for (int i = 0; i < numImages; i++) {
        Mat im = imread(filenames[i]);  // 이미지를 읽어옴
        if (im.empty()) {
            cout << "Could not open or find the image: " << filenames[i] << endl;
            return;
        }
        images.push_back(im);  // 읽어온 이미지를 벡터에 추가
    }
}

// 이미지 크기를 조정하여 보여주는 함수
void showResized(const string& windowName, const Mat& img, int width = 500, int height = 700) {
    Mat resizedImg;
    resize(img, resizedImg, Size(width, height));  // 이미지 크기 조정
    namedWindow(windowName, WINDOW_AUTOSIZE);  // 윈도우 생성
    imshow(windowName, resizedImg);  // 이미지 보여줌
}

Mat GetHistogram(Mat src) {
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = src;
    }

    Mat histogram;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    calcHist(&gray, 1, 0, Mat(), histogram, 1, &histSize, &histRange);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
            Scalar(255, 255, 255), 2, 8, 0);
    }

    return histImage;
}

void histogram(Mat img1,const string& title1) {
    Mat histImage1 = GetHistogram(img1);
    imshow("Histogram " + title1, histImage1);
}

void ex1() {
    cout << "Reading images and exposure times ..." << endl;
    vector<Mat> images;
    vector<float> times;
    readImagesAndTimes(images, times);  // 이미지와 노출 시간 읽기
    cout << "Finished reading images" << endl;

    // 카메라 응답 함수(CRF) 계산
    cout << "Calculating Camera Response Function ..." << endl;
    Mat responseDebevec;
    Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
    calibrateDebevec->process(images, responseDebevec, times);  // CRF 계산

    // 이미지를 하나의 HDR 이미지로 병합
    cout << "Merging images into one HDR image ..." << endl;
    Mat hdrDebevec;
    Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
    mergeDebevec->process(images, hdrDebevec, times, responseDebevec);  // HDR 이미지 병합

    // Drago 방법을 사용한 톤매핑
    cout << "Tonemapping using Drago's method ..." << endl;
    Mat ldrDrago;
    Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0, 0.7, 0.85f);
    tonemapDrago->process(hdrDebevec, ldrDrago);  // 톤매핑
    ldrDrago = 3 * ldrDrago;
    ldrDrago.convertTo(ldrDrago, CV_8UC3, 255.0);

    // Reinhard 방법을 사용한 톤매핑
    cout << "Tonemapping using Reinhard's method ..." << endl;
    Mat ldrReinhard;
    Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
    tonemapReinhard->process(hdrDebevec, ldrReinhard);  // 톤매핑
    ldrReinhard.convertTo(ldrReinhard, CV_8UC3, 255.0);

    // Mantiuk 방법을 사용한 톤매핑
    cout << "Tonemapping using Mantiuk's method ..." << endl;
    Mat ldrMantiuk;
    Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
    tonemapMantiuk->process(hdrDebevec, ldrMantiuk);  // 톤매핑
    ldrMantiuk = 3 * ldrMantiuk;
    ldrMantiuk.convertTo(ldrMantiuk, CV_8UC3, 255.0);

    // 원본 사진과 톤매핑한 결과 출력
    cout << "Displaying images ..." << endl;
    showResized("Drago Tonemapped Image", ldrDrago);
    showResized("Reinhard Tonemapped Image", ldrReinhard);
    showResized("Mantiuk Tonemapped Image", ldrMantiuk);

    // 원본 사진과 톤매핑한 결과의 히스토그램 표시
    cout << "Displaying histograms ..." << endl;
    histogram(images[0] , "original 1");
    histogram(images[1], "original 2");
    histogram(images[2], "original 3");
    histogram(images[3], "original 4");
    waitKey(0); 
    histogram(ldrDrago, "Drago Tonemapped Image");
    histogram(ldrReinhard, "Reinhard Tonemapped Image");
    histogram(ldrMantiuk, "Mantiuk Tonemapped Image");

    waitKey(0);  // 키 입력 대기
    destroyAllWindows();  // 모든 윈도우 제거
}

int main() {
    cout << "EX 1" << endl;
    ex1();
}
