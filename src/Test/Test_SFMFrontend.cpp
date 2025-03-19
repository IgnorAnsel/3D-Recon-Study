#include "preprocessing/SFMFrontend.h"
#include "preprocessing/pclprocessing.h"
#include "preprocessing/pre.h"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <vector>
int main() {
  pre::SFMFrontend sfmFrontend;
  pre::CameraPreprocessor preprocessor;
  pre::PCLProcessing pclProcessor;
  preprocessor.loadCameraParams(std::string(RESOURCE_DIR) +
                                "/room/calibration_result.yml");
  std::string img1Path = std::string(RESOURCE_DIR) + "/room/room_4.jpg";
  std::string img2Path = std::string(RESOURCE_DIR) + "/room/room_5.jpg";
  cv::Mat img1 = cv::imread(img1Path);
  cv::Mat img2 = cv::imread(img2Path);
  img1 = preprocessor.preprocess(img1);
  img2 = preprocessor.preprocess(img2);
  std::vector<cv::Point2f> points1, points2;
  sfmFrontend.GetGoodMatches(img1, img2, points1, points2);
  cv::Mat F = sfmFrontend.ComputeFundamentalMatrix(points1, points2);
  // sfmFrontend.TestFundamentalMatrix(points1, points2, F, img1, img2);
  cv::Mat K = preprocessor.getIntrinsicMatrix();
  cv::Mat E = sfmFrontend.ComputeEssentialMatrix(K, K, F);
  // sfmFrontend.TestEssentialMatrix(points1, points2, E, K, K, img1, img2);
  cv::Mat R, t;
  sfmFrontend.ComputePoseFromEssentialMatrix(E, points1, points2, K, R, t);
  pclProcessor.initPCLWithNoCloud();
  cv::Mat R1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  cv::Mat t1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  pclProcessor.addCamera(R1, t1, 0, img1Path);
  pclProcessor.addCamera(R, t, 1, img2Path);
  std::vector<cv::Point3f> points3D =
      sfmFrontend.robustTriangulate(points1, points2, K, R1, t1, R, t);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      sfmFrontend.convertToPointCloud(points3D, points1, img1);
  pclProcessor.addPointCloud(*cloud);

  while (!pclProcessor.getViewer()->wasStopped()) {
    pclProcessor.getViewer()->spinOnce(100);
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}