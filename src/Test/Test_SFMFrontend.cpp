#include "preprocessing/SFMFrontend.h"
#include "preprocessing/pclprocessing.h"
#include "preprocessing/pre.h"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
int main() {
  pre::SFMFrontend sfmFrontend;
  pre::CameraPreprocessor preprocessor;
  pre::PCLProcessing pclProcessor;
  preprocessor.loadCameraParams(std::string(RESOURCE_DIR) +
                                "/room/calibration_result.yml");
  std::string img1Path = std::string(RESOURCE_DIR) + "/room/room_3.jpg";
  std::string img2Path = std::string(RESOURCE_DIR) + "/room/room_4.jpg";
  cv::Mat img1 = cv::imread(img1Path);
  cv::Mat img2 = cv::imread(img2Path);
  img1 = preprocessor.preprocess(img1);
  img2 = preprocessor.preprocess(img2);
  std::vector<cv::Point2f> points1, points2;
  sfmFrontend.GetGoodMatches(img1, img2, points1, points2);
  cv::Mat F = sfmFrontend.ComputeFundamentalMatrix(points1, points2);
  // sfmFrontend.TestFundamentalMatrix(points1, points2, F, img1, img2);
  cv::Mat K = preprocessor.getIntrinsicMatrix();
  std::cout << "K: " << K << std::endl;
  cv::Mat D = preprocessor.getDistortionCoefficients();
  cv::Mat E;
  cv::Mat R2, t2;
  cv::recoverPose(points1, points2, K, D, K, D, E, R2, t2);
  // sfmFrontend.TestEssentialMatrix(points1, points2, E, K, K, img1, img2);

  // sfmFrontend.ComputePoseFromEssentialMatrix(E, points1, points2, K, R2, t2);
  pclProcessor.initPCLWithNoCloud();
  cv::Mat R1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  cv::Mat t1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  pclProcessor.addCamera(R1, t1, 0, img1Path);
  pclProcessor.addCamera(R2, t2, 1, img2Path);
  std::vector<cv::Point3f> points3D =
      sfmFrontend.robustTriangulate(points1, points2, K, R1, t1, R2, t2, 10000);
  points3D = sfmFrontend.scaleToVisibleRange(points3D);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      sfmFrontend.convertToPointCloud(points3D, points1, img1);
  std::cout << "cloud size: " << cloud->size() << std::endl;
  pclProcessor.setGlobalCloud(cloud);
  pclProcessor.addPointCloud(*cloud, "test");

  while (!pclProcessor.getViewer()->wasStopped()) {
    pclProcessor.getViewer()->spinOnce(100);
    cv::waitKey(1);
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}