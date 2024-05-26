#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2//features2d.hpp>
#include<opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;


void tesk1cvBlobDetection() {
    Mat img = imread("9/coin.png", IMREAD_COLOR);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 500;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 10000;
    params.filterByCircularity = true;
    params.minCircularity = 0.3;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    // Set blob detector
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    // Detect blobs
    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // Draw blobs
    Mat result;
    drawKeypoints(img, keypoints, result,
        Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    int Coin = keypoints.size();


    putText(result, "COIN: " + to_string(Coin), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
    imshow("coin", result);
    waitKey(0);
    destroyAllWindows();
}
void tesk2cvBlobDetection(Mat img) {
    // Set params
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 400;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 1000;
    params.filterByCircularity = true;
    params.minCircularity = 0.5;
    params.filterByConvexity = true;
    params.minConvexity = 0.8;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;

    // Set blob detector
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    // Detect blobs
    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // Draw blobs
    Mat result;
    drawKeypoints(img, keypoints, result,
        Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    int Coin = keypoints.size();
    string ans;
    if (Coin == 3) ans = "Triangle";
    if (Coin == 4) ans = "Rectangle";
    if (Coin == 5) ans = "Pentagon";
    if (Coin == 6) ans = "Hexagon";

    putText(result, ans , Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
    cout << "이미지 정답 : " << Coin << "각형" << endl;
    imshow("keypointzs", result);
    waitKey(0);
    destroyAllWindows();
}
Mat cvHarrisCorner(Mat img) {
    resize(img, img, Size(500, 300), 0, 0, INTER_CUBIC);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat harr;
    cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);

    int thresh = 125;
    Mat result = img.clone();
    for (int y = 0; y < harr.rows; y += 1) {
        for (int x = 0; x < harr.cols; x += 1) {
            if ((int)harr.at<float>(y, x) > thresh)
                circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
        }
    }

    imshow("Source image", img);
    imshow("Harris image", harr_abs);
    imshow("Target image", result);
    waitKey(0);
    destroyAllWindows();
    return result;
}
Mat warpPers(Mat img) {
    Mat dst;
    Point2f src_p[4], dst_p[4];

    src_p[0] = Point2f(0, 0);
    src_p[1] = Point2f(1200, 0);
    src_p[2] = Point2f(0, 800);
    src_p[3] = Point2f(1200, 800);

    dst_p[0] = Point2f(0, 0);
    dst_p[1] = Point2f(1200, 0);
    dst_p[2] = Point2f(0, 800);
    dst_p[3] = Point2f(1000, 600);

    Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
    warpPerspective(img, dst, pers_mat, Size(1200, 800));
    return dst;
}
void cvFeatureSIFT(Mat img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Ptr<SiftFeatureDetector>detector = SiftFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints);

    Mat result;
    drawKeypoints(img, keypoints, result, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    imwrite("sift_result.jpg", result);
    imshow("sift resutl", result);
    waitKey(0);
    destroyAllWindows();
}
int main() {
    //1

    /*tesk1cvBlobDetection();*/

    //2
    //Mat img = imread("9/1.jpg", IMREAD_COLOR);
    //Mat dst = cvHarrisCorner(img);
    //tesk2cvBlobDetection(dst);

    //Mat img1 = imread("9/2.jpg", IMREAD_COLOR);
    //Mat dst1 = cvHarrisCorner(img1);
    //tesk2cvBlobDetection(dst1);

    //Mat img2 = imread("9/3.jpg", IMREAD_COLOR);
    //Mat dst2 = cvHarrisCorner(img2);
    //tesk2cvBlobDetection(dst2);

    //Mat img3 = imread("9/4.jpg", IMREAD_COLOR);
    //Mat dst3 = cvHarrisCorner(img3);
    //tesk2cvBlobDetection(dst3);

    //3
    Mat img3 = imread("9/church.jpg", IMREAD_COLOR);
    Mat dst;
    img3.convertTo(dst, -1, 1, 50);

    imshow("briteness", dst);
    Mat result = warpPers(dst);
    imshow("warp", result);
    cvFeatureSIFT(result);
    cvFeatureSIFT(img3);
}