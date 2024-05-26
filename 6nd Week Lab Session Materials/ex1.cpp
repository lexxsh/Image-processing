#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

Mat doDft(Mat srcImg);
Mat getMagnitude(Mat complexImg);  
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);  
Mat centralize(Mat complex);  
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);
Mat padding(Mat img);
Mat doBPF(Mat srcImg);
int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);

Mat doDft(Mat srcImg) { 	
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg) {  	
	Mat planes[2];
	split(complexImg, planes);

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	 	magImg += Scalar::all(1);
	log(magImg, magImg);
	 	 
	return magImg;
}

Mat myNormalize(Mat src) { 	
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

Mat getPhase(Mat complexImg) { 	
	Mat planes[2];
	split(complexImg, planes);
	 
	Mat phaImg;
	phase(planes[0], planes[1], phaImg); 	
	return phaImg;
}

Mat centralize(Mat complex) { 	
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}
Mat setComplex(Mat magImg, Mat phaImg) {  	
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	 
	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);  
	Mat complexImg;
	merge(planes, 2, complexImg); 	
	return complexImg;
}
Mat doIdft(Mat complexImg) {  	
	Mat idftcvt;
	idft(complexImg, idftcvt);
	 
	Mat planes[2];
	split(idftcvt, planes);
	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);
	 	return dstImg;
}


Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);
	Mat padded;
	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}

Mat doBPF(Mat srcImg) {
	//<DFT>

	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); //min max 범위를 통일하고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); //이 min max 에 대해서 normalize
	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	Mat maskImg2 = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 200, Scalar::all(1), -1, -1, 0);
	circle(maskImg2, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat resultImg = maskImg - maskImg2;

	imshow("LowPassFilter", maskImg);
	imshow("HighPassFilter", maskImg2);
	imshow("BandPassFilter", resultImg);

	Mat magImg2;
	multiply(magImg, resultImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}
Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1,0,-1,-2,0,2,1,0,1 };
	int kernelY[3][3] = { -1,-2,-1,0,0,0,1,2,1 };

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) + abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		}
	}
	return dstImg;
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
	else return sum;
}
Mat freqeunSobel(Mat srcImg) {
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); //min max 범위를 통일하고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); //이 min max 에 대해서 normalizedd
	
	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	Mat maskImg2 = Mat::zeros(magImg.size(), CV_32F);
	int lineHeight = magImg.rows;
	int lineWidth = 200;

	maskImg(Rect(magImg.cols / 2 - lineWidth / 2, magImg.rows / 2 - lineHeight / 2, lineWidth, lineHeight)) = 1;
	
	int lineThickness = 200; 
	int lineWidth1 = magImg.cols; 


	maskImg(Rect(magImg.cols / 2 - lineWidth1 / 2, magImg.rows / 2 - lineThickness / 2, lineWidth1, lineThickness)) = 1;
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20 ,Scalar::all(0), -1, -1, 0);
	circle(maskImg2, Point(maskImg.cols / 2, maskImg.rows / 2), 10, Scalar::all(1), -1, -1, 0);
	maskImg -= maskImg2;
	imshow("Mask222", maskImg);



	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	// 결과 이미지 출력
	imshow("Result Image", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}
Mat removeHorizontalLines(Mat srcImg) {
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); 
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); 
	Mat maskImg = Mat::ones(magImg.size(), CV_32F);

	int lineHeight = magImg.rows;
	int lineWidth = 20; 


	maskImg(Rect(magImg.cols / 2 - lineWidth / 2, magImg.rows / 2 - lineHeight / 2, lineWidth, lineHeight)) = 0;

	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 1, Scalar::all(1), -1, -1, 0);
	imshow("Mask1", maskImg);
	
	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	// 결과 이미지 출력
	imshow("Result Image1", magImg2);
	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

int main() {
	Mat srcImg = imread("6/img1.jpg", 0);
	Mat srcImg1 = imread("6/img3.JPG", 0);
	Mat srcimg2 = imread("6/img2.jpg", 0);
	imshow("ORG", srcImg);
	imshow("BPF", doBPF(srcImg));
	imshow(" Sobel_Spatial", mySobelFilter(srcImg2));
	imshow(" sobel_freq", freqeunSobel(srcImg2));
	imshow("remove", removeHorizontalLines(srcImg1));
	waitKey(0);
	destroyAllWindows(); // 모든 창 종료
}