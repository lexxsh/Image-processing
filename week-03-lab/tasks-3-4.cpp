#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void Gradation1(Mat img) {
	for (int y = 0; y < img.rows; y++) {
		int k = y * 255 / img.rows;
		for (int x = 0; x < img.cols; x++) {
			Vec3b& pixel = img.at<Vec3b>(y, x);
			pixel -= Vec3b(k, k, k);
		}
	}
	imshow("12201928 이상혁 1", img);
}
void Gradation2(Mat img) {
	for (int y = (img.rows-1); y >=0 ; y--) {
		int k = (img.rows - 1-y) * 255 / img.rows;
		for (int x = 0; x < img.cols; x++) {
			Vec3b& pixel = img.at<Vec3b>(y, x);
			pixel -= Vec3b(k, k, k);
		}
	}
	imshow("12201928 이상혁 2", img);
}

Mat GetColorHistogram(Mat src) {
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;
	Mat b_hist, g_hist, r_hist;

	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	int hist_w = 512; // 히스토그램 이미지의 너비
	int hist_h = 400; // 히스토그램 이미지의 높이
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	return histImage;
}

void histogram(Mat img1,Mat img2) {
	Mat histImage1 = GetColorHistogram(img1);
	Mat histImage2 = GetColorHistogram(img2);
	imshow("12201928 이상혁 3", histImage1);
	imshow("12201928 이상혁 4", histImage2);
}

void MakeVideo(Mat img3, Mat img4,Mat img5) {
	resize(img4, img4, Size(img3.cols, img3.rows));
	Mat result1,result2,img5_gray,img5_mask;
	subtract(img3, img4, result1);
	resize(img5, img5, Size(img5.cols*0.9, img5.rows*0.9));
	cvtColor(img5, img5_gray, COLOR_BGR2GRAY);
	threshold(img5_gray, img5_mask, 180, 255, THRESH_BINARY_INV);
	Rect logoRoi = Rect(320, 340, img5.cols, img5.rows);
	Mat result1_roi(result1, logoRoi);
	img5.copyTo(result1_roi, img5_mask);
	imshow("12201928 이상혁 5", result1);
}
int main() {
	Mat img1 = imread("img2.jpg", 1);
	Mat img2 = imread("img2.jpg", 1);
	Mat img3 = imread("img3.jpg", 1);
	Mat img4 = imread("img4.jpg", 1);
	Mat img5 = imread("img5.jpg", 1);
	if (img1.empty() || img2.empty() || img3.empty() || img4.empty() || img5.empty()) {
		cerr << "image load failed!" << endl;
		return -1;
	}
	Gradation1(img1);
	Gradation2(img2);
	histogram(img1, img2);
	MakeVideo(img3, img4, img5);
	waitKey(0);
	destroyAllWindows();
} 