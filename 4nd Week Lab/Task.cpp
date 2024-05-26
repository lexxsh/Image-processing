#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

double gaussian(float x, double sigma);
float distance(int x, int y, int i, int j);
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);

Mat medianfilter(Mat Img, int filterSize) {
	Mat srcImg;
	cvtColor(Img, srcImg, COLOR_BGR2GRAY);
	imshow("Input Image", srcImg);

	int width = srcImg.cols;
	int height = srcImg.rows;
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	int halfSize = filterSize / 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
				int k = 0;
				int m_tempImg[25];
				for (int b = -halfSize; b <= halfSize; b++) {
					for (int a = -halfSize; a <= halfSize; a++) {
						m_tempImg[k] = srcData[(y + b) * width + (x + a)];
						k++;
					}
				}
				k = 0;
				sort(m_tempImg, m_tempImg + filterSize * filterSize);
				dstData[y * width + x] = m_tempImg[filterSize * filterSize / 2];
		}
	}
	return dstImg;
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {
	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1);
}
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s) {
	int radius = diameter / 2;
	double gr, gs, wei;
	double tmp = 0;
	double sum = 0;

	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr)
				- (float)src_img.at<uchar>(c, r), sig_r);
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum;
}

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}
float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}
void canny() {
	cout << "--- doCanny edge detection() ---\n" << endl;
	Mat srcimg = imread("gear.jpg", 1);
	if (!srcimg.data) printf("No image data \n");
	Mat dst_0, dst_4, dst_1, dst_2, dst_3, dst_5;
	Canny(srcimg, dst_0, 0, 100, 3, false);
	Canny(srcimg, dst_1, 100, 100, 3, false);
	Canny(srcimg, dst_2, 100, 200, 3, false);
	Canny(srcimg, dst_3, 100, 127, 3, false);
	Canny(srcimg, dst_5, 100, 127, 3, true);
	Canny(srcimg, dst_4, 100, 127, 5, true);
	imshow("Original", srcimg);
	imshow("Ex: 0", dst_0);
	imshow("Ex: 1", dst_1);
	imshow("Ex: 2", dst_2);
	imshow("Ex: 3", dst_3);
	imshow("Ex: 4", dst_4);
	imshow("Ex: 5", dst_5);
	waitKey(0);
	destroyAllWindows();
}
void doMedian() {
	cout << "--- doMedianFillter() ---\n" << endl;
	Mat color_img = imread("salt_pepper2.png", 1);
	if (!color_img.data) printf("No image data \n");
	Mat resultImg1 = medianfilter(color_img, 3);
	Mat resultImg2 = medianfilter(color_img, 5);
	imshow("Median Test", resultImg1);
	imshow("Median Test2", resultImg2);
	waitKey(0);
	destroyAllWindows();
}
void dobilateral() {
	cout << "--- doBilateralEx() ---\n" << endl;
	Mat src_img = imread("rock.png", 0);
	Mat dst_img, dst_img2, dst_img3, dst_img4, dst_img5, dst_img6, dst_img7, dst_img8, dst_img9;
	if (!src_img.data) printf("No image data \n");
	myBilateral(src_img, dst_img, 5, 0.1, 0.1);
	myBilateral(src_img, dst_img2, 5, 0.1, 5);
	myBilateral(src_img, dst_img3, 5, 0.1, 200);
	myBilateral(src_img, dst_img4, 5, 0.25, 0.1);
	myBilateral(src_img, dst_img5, 5, 0.25, 5);
	myBilateral(src_img, dst_img6, 5, 0.25, 200);
	myBilateral(src_img, dst_img7, 5, 100, 0.1);
	myBilateral(src_img, dst_img8, 5, 100, 5);
	myBilateral(src_img, dst_img9, 5, 100, 200);

	imshow("Ex: 1", dst_img);
	imshow("Ex: 2", dst_img2);
	imshow("Ex: 3", dst_img3);
	imshow("Ex: 4", dst_img4);
	imshow("Ex: 5", dst_img5);
	imshow("Ex: 6", dst_img6);
	imshow("Ex: 7", dst_img7);
	imshow("Ex: 8", dst_img8);
	imshow("Ex: 9", dst_img9);
	waitKey(0);
	destroyAllWindows();
}

int main() {
	//doMedian();
	//Task #1 수행

	//dobilateral();
	//Task #2 수행
	
	//canny();
	//Task #3 수행
}