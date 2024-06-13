#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// �̹����� ���� �ð��� �о���� �Լ�
void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
    int numImages = 4;
    static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
    times.assign(timesArray, timesArray + numImages);  // ���� �ð� ����
    static const char* filenames[] = { "11/im.jpg", "11/im1.jpg", "11/im2.jpg", "11/im3.jpg" };
    for (int i = 0; i < numImages; i++) {
        Mat im = imread(filenames[i]);  // �̹����� �о��
        if (im.empty()) {
            cout << "Could not open or find the image: " << filenames[i] << endl;
            return;
        }
        images.push_back(im);  // �о�� �̹����� ���Ϳ� �߰�
    }
}

// �̹��� ũ�⸦ �����Ͽ� �����ִ� �Լ�
void showResized(const string& windowName, const Mat& img, int width = 500, int height = 700) {
    Mat resizedImg;
    resize(img, resizedImg, Size(width, height));  // �̹��� ũ�� ����
    namedWindow(windowName, WINDOW_AUTOSIZE);  // ������ ����
    imshow(windowName, resizedImg);  // �̹��� ������
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
    readImagesAndTimes(images, times);  // �̹����� ���� �ð� �б�
    cout << "Finished reading images" << endl;

    // ī�޶� ���� �Լ�(CRF) ���
    cout << "Calculating Camera Response Function ..." << endl;
    Mat responseDebevec;
    Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
    calibrateDebevec->process(images, responseDebevec, times);  // CRF ���

    // �̹����� �ϳ��� HDR �̹����� ����
    cout << "Merging images into one HDR image ..." << endl;
    Mat hdrDebevec;
    Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
    mergeDebevec->process(images, hdrDebevec, times, responseDebevec);  // HDR �̹��� ����

    // Drago ����� ����� �����
    cout << "Tonemapping using Drago's method ..." << endl;
    Mat ldrDrago;
    Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0, 0.7, 0.85f);
    tonemapDrago->process(hdrDebevec, ldrDrago);  // �����
    ldrDrago = 3 * ldrDrago;
    ldrDrago.convertTo(ldrDrago, CV_8UC3, 255.0);

    // Reinhard ����� ����� �����
    cout << "Tonemapping using Reinhard's method ..." << endl;
    Mat ldrReinhard;
    Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
    tonemapReinhard->process(hdrDebevec, ldrReinhard);  // �����
    ldrReinhard.convertTo(ldrReinhard, CV_8UC3, 255.0);

    // Mantiuk ����� ����� �����
    cout << "Tonemapping using Mantiuk's method ..." << endl;
    Mat ldrMantiuk;
    Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
    tonemapMantiuk->process(hdrDebevec, ldrMantiuk);  // �����
    ldrMantiuk = 3 * ldrMantiuk;
    ldrMantiuk.convertTo(ldrMantiuk, CV_8UC3, 255.0);

    // ���� ������ ������� ��� ���
    cout << "Displaying images ..." << endl;
    showResized("Drago Tonemapped Image", ldrDrago);
    showResized("Reinhard Tonemapped Image", ldrReinhard);
    showResized("Mantiuk Tonemapped Image", ldrMantiuk);

    // ���� ������ ������� ����� ������׷� ǥ��
    cout << "Displaying histograms ..." << endl;
    histogram(images[0] , "original 1");
    histogram(images[1], "original 2");
    histogram(images[2], "original 3");
    histogram(images[3], "original 4");
    waitKey(0); 
    histogram(ldrDrago, "Drago Tonemapped Image");
    histogram(ldrReinhard, "Reinhard Tonemapped Image");
    histogram(ldrMantiuk, "Mantiuk Tonemapped Image");

    waitKey(0);  // Ű �Է� ���
    destroyAllWindows();  // ��� ������ ����
}

int main() {
    cout << "EX 1" << endl;
    ex1();
}
