#include<iostream>
#include<vector>

#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			//// max, min ����
			double minv = min({ b,g,r });
			double maxv = max({ b,g,r });
			//v �� �������ֱ�
			v = maxv;
			//s ����(������ �־��� ���Ĵ�� ����)
			if (v == 0) s = 0;
			else s = (maxv - minv) / maxv;
			//h ����
			if (maxv == r) h = 60 * (0 + ((g - b) / (maxv - minv)));
			if (maxv == g) h = 60 * (2 + ((b - r) / (maxv - minv)));
			if (maxv == b) h = 60 * (4 + ((r - g) / (maxv - minv)));
			else if (h < 0) h = h + 360;

			h /= 2.0; // H ���� 0~180 ������ ����
			h = max(min(h, 180.0), 0.0); // overflow ����
			s *= 255.0; // S ���� 0~255 ������ ����
			s = max(min(s, 255.0), 0.0); // overflow ����


			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}
	}
	return dst_img;
}
Mat MyinRange(const Mat& src_img, const Scalar& h1, const Scalar& h2) {
	Mat dst_img(src_img.size(), CV_8UC1); // ���� ä�� ����ũ �̹���

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			Vec3b pixel = src_img.at<Vec3b>(y, x);
			double h = pixel[0];

			// H ���� �־��� ���� ���� ������ ���(255), �ƴϸ� ������(0)���� ����
			if (h >= h1[0] && h <= h2[0]) {
				dst_img.at<uchar>(y, x) = 255;
			}
			else {
				dst_img.at<uchar>(y, x) = 0;
			}
		}
	}
	return dst_img;
}
string getColor(int hValue) {
	if (hValue >= 0 && hValue < 30)
		return "Red";
	else if (hValue >= 30 && hValue < 60)
		return "Orange";
	else if (hValue >= 60 && hValue < 90)
		return "Yellow";
	else if (hValue >= 90 && hValue < 150)
		return "Green";
	else if (hValue >= 150 && hValue < 210)
		return "Cyan";
	else if (hValue >= 210 && hValue < 270)
		return "Blue";
	else if (hValue >= 270 && hValue < 330)
		return "Magenta";
	else
		return "Red";
}
void PrintColor(Mat src_img) {
	Mat hsvimg = MyBgr2Hsv(src_img);
	// �� ���� �ȼ� ���� ������ ��
	map<string, int> colorCounts;

	// �� �ȼ��� H ���� Ȯ���Ͽ� ���� ��� �ۼ�
	for (int y = 0; y < hsvimg.rows; y++) {
		for (int x = 0; x < hsvimg.cols; x++) {
			Vec3b pixel = hsvimg.at<Vec3b>(y, x);
			int hValue = pixel[0]; // H ���� ù ��° ä�ο� �����
			string color = getColor(hValue);
			// �ش� ������ ������ ����
			colorCounts[color]++;
		}
	}

	// ���� ��� ���
	for (const auto& entry : colorCounts) {
		cout << "Color: " << entry.first << ", Count: " << entry.second << endl;
	}

	string bestColor;
	int maxCount = 0;
	for (const auto& entry : colorCounts) {
		if (entry.second > maxCount) {
			bestColor = entry.first;
			maxCount = entry.second;
		}
	}

	// ���� ���� ���� ���� ���
	cout << "Most frequent color: " << bestColor << ", Count: " << maxCount << endl;
}
Mat CvKmeans(Mat src_img, int k) {

	//2���� ���� -> 1���� ����
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x + src_img.rows) = (float)src_img.at<uchar>(y, x);
			}
		}
	}

	//opencv k-means ����
	Mat labels;
	Mat centers;
	int attemps = 5;

	kmeans(samples, k, labels,
		TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
		attemps, KMEANS_PP_CENTERS, centers);

	//1���� ���� => 2���� ����
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
					//�����Ǻ� ����� ���� �� ������ �߾Ӱ����� ��� ����
				}
			}
			else {
				dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	imshow("results", dst_img);
	waitKey(0);
	return dst_img;

}

int main() {
	// Homework
	//////////////////////////////////////////////1��
	Mat src_img = imread("7/tomato.jpg", 1);
	Mat src_img2 = imread("7/picture.jpg", 1);
	cout << "12201928 �̻���" << endl;

	Mat dst_img = MyBgr2Hsv(src_img);
	Mat dst_img2 = MyBgr2Hsv(src_img2);
	Mat dst_img1;
	//cvtColor(src_img, dst_img1, COLOR_BGR2HSV);

	//imshow("12201928_cvt", dst_img1);
	//imshow("12201928_hsv", dst_img);

	//Mat cvt[3];
	//split(dst_img1, cvt);
	//imshow("12201928_cvt_h", cvt[0]);
	//imshow("12201928_cvt_s", cvt[1]);
	//imshow("12201928_cvt_v", cvt[2]);

	//Mat hsv[3];
	//split(dst_img, hsv);
	//imshow("12201928_hsv_h", hsv[0]);
	//imshow("12201928_hsv_s", hsv[1]);
	//imshow("12201928_hsv_v", hsv[2]);

	//Mat mask1 = MyinRange(dst_img2, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255));
	//Mat mask2;
	//inRange(dst_img2, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255), mask2);
	//imshow("MyinRange", mask1);
	//imshow("inRange", mask2);
	//imshow("12201928_original", src_img2);
	//PrintColor(src_img2);
	//Mat result;
	//bitwise_and(src_img2, src_img2, result, mask1);
	//imshow("12201928_final", result);


	////////////////////////////////2��
	//Mat src_img3 = imread("7/beach.jpg", 1);
	//Mat dst_img3, dst_img4, dst_img5, dst_img6, dst_img7;
	//CvKmeans(src_img3, 1);
	//CvKmeans(src_img3, 2);
	//CvKmeans(src_img3, 3);
	//CvKmeans(src_img3, 4);
	//CvKmeans(src_img3, 5);

	////////////////////////////////3��
	CvKmeans(src_img2, 5);

	waitKey(0);
	destroyAllWindows();
}