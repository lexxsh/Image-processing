#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

void SpreadSalts(Mat img, int R, int G, int B) {
	for (int n = 0; n < R; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			//Red Point 출력
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[1] = 0;
			img.at<Vec3b>(y, x)[2] = 255;
		}
	}
	for (int n = 0; n < G; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			//Green Point 출력
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[1] = 255;
			img.at<Vec3b>(y, x)[2] = 0; 
		}
	}
	for (int n = 0; n < B; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			//Blue Point 출력
			img.at<Vec3b>(y, x)[0] = 255;
			img.at<Vec3b>(y, x)[1] = 0;
			img.at<Vec3b>(y, x)[2] = 0;
		}
	}
}

void Count(Mat img) {
	int R_P = 0;
	int B_P = 0;
	int G_P = 0;
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			if (img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[1] == 0 && img.at<Vec3b>(y, x)[2] == 255) R_P++;
			if (img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[1] == 255 && img.at<Vec3b>(y, x)[2] == 0) G_P++;
			if (img.at<Vec3b>(y, x)[0] == 255 && img.at<Vec3b>(y, x)[1] == 0 && img.at<Vec3b>(y, x)[2] == 0) B_P++;
		}
	}
	cout << "Red : " << R_P << "\nGreen : " << G_P << "\nBlue : " << B_P << endl;
}
int main() {
	srand(time(NULL));
	Mat src_img = imread("img1.jpg", 1);
	if (src_img.empty()) {
		cerr << "image load failed!" << endl;
		return -1;
	}
	SpreadSalts(src_img, 30,50,70); // 소금효과 이미지에 추가, 앞부터 RGB 순서
	imshow("Task 1", src_img); // 이미지 출력
	Count(src_img);
	waitKey(0);
	destroyWindow("Task 1");
}
