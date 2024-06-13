#include <iostream>
#include <math.h>
#include <time.h>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include<opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;


Mat getMyRotationMatrix(Point center, double angle) {
	double radians = angle * CV_PI / 180;
	double alpha = cos(radians);
	double beta = sin(radians);

	Mat matrix = (Mat_<double>(2, 3) <<
		alpha, beta, (1 - alpha) * center.x - beta * center.y,
		-beta, alpha, beta * center.x + (1 - alpha) * center.y);

	return matrix;
}
Mat cvHarrisCorner(Mat img) {
	Mat src = imread("10/card_per.png", 1);
	Mat harr;
	cornerHarris(img, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	int thresh = 125;
	Mat result = img.clone();
	vector<Point> corner_points;
	int minDist = 5; 
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh) {
				Point candidateCorner = Point(x, y);
				bool tooClose = false;
				for (auto storedCorner : corner_points) {
					double dist = norm(candidateCorner - storedCorner);
					if (dist < minDist) {
						tooClose = true;
						break;
					}
				}
				if (!tooClose) {
					corner_points.push_back(candidateCorner);
					circle(result, candidateCorner, 7, Scalar(255, 0, 255), 2, 8, 0);
				}
			}
		}
	}
	cout << "1¹øÂ° ÁÂÇ¥ : " << corner_points[0] << endl;
	cout << "2¹øÂ° ÁÂÇ¥ : " << corner_points[1] << endl;
	cout << "3¹øÂ° ÁÂÇ¥ : " << corner_points[2] << endl;
	cout << "4¹øÂ° ÁÂÇ¥ : " << corner_points[3] << endl;
	imshow("Source image", img);
	imshow("Harris image", harr_abs);
	imshow("Target image", result);
	waitKey(0);
	Mat dst, matrix;
	Point2f srcQuad[4];
	srcQuad[0] = corner_points[0];
	srcQuad[1] = corner_points[1];
	srcQuad[2] = corner_points[2];
	srcQuad[3] = corner_points[3];
	
	Point2f dstQuad[4];
	dstQuad[0] = Point2f(corner_points[3].x, corner_points[0].y);
	dstQuad[1] = Point2f(corner_points[2].x, corner_points[0].y);
	dstQuad[2] = Point2f(corner_points[2].x, corner_points[3].y);
	dstQuad[3] = corner_points[3];

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	imwrite("nonper.jpg", src);
	imwrite("per.jpg", dst);

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);
	destroyAllWindows();
	return result;
}
void Task1() {
	Mat getMyRotationMatrix(Point center, double angle);
	Mat src = imread("10/Lenna.png");
	if (src.empty()) {
		cerr << "Error loading the image!" << endl;
		return;
	}

	Point center = Point(src.cols / 2, src.rows / 2);
	Mat matrix = getRotationMatrix2D(center, 45.0, 1.0);
	Mat mymatrix = getMyRotationMatrix(center, 45.0);
	Mat dst, mydst;

	warpAffine(src, dst, matrix, src.size());
	warpAffine(src, mydst, mymatrix, src.size());

	imwrite("rot.jpg", dst);
	imwrite("myrot.jpg", mydst);


	imshow("rot.jpg", dst);
	imshow("myrot.jpg", mydst);
	waitKey(0);
	destroyAllWindows();
}
void Task2() {
	Mat img = imread("10/card_per.png", 1);
	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);
	Rect rect(10, 10, 480, 480);
	Mat result, bg_model, fg_model;
	grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);
	compare(result, GC_PR_FGD, result, CMP_EQ);
	imshow("ÀÌ»óÇõ",result);
	waitKey(0);
	cvHarrisCorner(result);
	waitKey(0);
}

int main() {
	Task1();

	//Task2();

}