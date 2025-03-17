#include "preprocessing/pre.h"
#include "preprocessing/pclprocessing.h"
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;
Ptr<cv::SIFT> sift = cv::SIFT::create(
    1000,   // 增加特征点数量
    3,     
    0.03,  // 调低对比度阈值
    15,    // 调整边缘阈值
    1.6    
);
vector<DMatch> matchFeatures(const Mat& descriptors1, const Mat& descriptors2, float ratioThresh = 0.5);
vector<DMatch> matchFeatures(const Mat& descriptors1, const Mat& descriptors2, float ratioThresh)
{
    // 构建 FLANN 匹配器
    FlannBasedMatcher matcher(new flann::KDTreeIndexParams(5), 
                             new flann::SearchParams(50));
    
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
    
    // 应用 Lowe's ratio test
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }
    
    return goodMatches;
}
void detectFeatures(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors)
{
    sift->detectAndCompute(image, Mat(), keypoints, descriptors);
}
Mat drawFeatureMatches(const Mat& img1, const vector<KeyPoint>& keypoints1,
                      const Mat& img2, const vector<KeyPoint>& keypoints2,
                      const vector<DMatch>& matches)
{
    Mat matchImg;
    drawMatches(
        img1, keypoints1, 
        img2, keypoints2,
        matches, matchImg, 
        Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    return matchImg;
}
int main() 
{
    pre::CameraPreprocessor preprocessor;
    std::string CameraParamsPath_ = std::string(RESOURCE_DIR) + "/room/calibration_result.yml";
    preprocessor.loadCameraParams(CameraParamsPath_);
    cv::Mat img1 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_1.jpg");
    cv::Mat img2 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_2.jpg");
    img1 = preprocessor.preprocess(img1);
    img2 = preprocessor.preprocess(img2);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detectFeatures(img1, keypoints1, descriptors1);
    detectFeatures(img2, keypoints2, descriptors2);
    vector<DMatch> goodMatches = matchFeatures(descriptors1, descriptors2);

    vector<Point2f> points1, points2;
    for (int i = 0; i < goodMatches.size(); i++) {
        points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }
    Mat matchImg = drawFeatureMatches(img1, keypoints1, img2, keypoints2, goodMatches);
    imshow("Matched Features", matchImg);
    waitKey(0);
}