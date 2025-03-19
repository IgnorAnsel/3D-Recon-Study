#include "preprocessing/SFMFrontend.h"
#include "preprocessing/pre.h"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <vector>
int main() {
  pre::SFMFrontend sfmFrontend;
  pre::CameraPreprocessor preprocessor;
  preprocessor.loadCameraParams(std::string(RESOURCE_DIR) +
                                "/room/calibration_result.yml");
  cv::Mat img1 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_4.jpg");
  cv::Mat img2 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_5.jpg");
  img1 = preprocessor.preprocess(img1);
  img2 = preprocessor.preprocess(img2);
  std::vector<cv::Point2f> points1, points2;
  sfmFrontend.GetGoodMatches(img1, img2, points1, points2);
  cv::Mat F = sfmFrontend.ComputeFundamentalMatrix(points1, points2);
  sfmFrontend.TestFundamentalMatrix(points1, points2, F, img1, img2);
  cv::Mat K = preprocessor.getIntrinsicMatrix();
  cv::Mat E = sfmFrontend.ComputeEssentialMatrix(K, K, F);
  sfmFrontend.TestEssentialMatrix(points1, points2, E, K, K, img1, img2);
  cv::waitKey(0);
}