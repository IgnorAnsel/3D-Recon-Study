#ifndef SFMFrontend_H
#define SFMFrontend_H
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <vector>

namespace pre {
enum FeatureDetectorType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  FAST,
  GFTT,
  HARRIS,
  MSER,
  STAR,
  SIFT_CUDA
};
class SFMFrontend {
public:
  SFMFrontend();
  SFMFrontend(FeatureDetectorType detector_type);
  void createSIFT(int nfeatures = 0, int nOctaveLayers = 3,
                  double contrastThreshold = 0.040000000000000001,
                  double edgeThreshold = 10, double sigma = 1.6000000000000001,
                  bool enable_precise_upscale =
                      false); // 创建SIFT对象，默认参数与OpenCV一致
  void detectFeatures(const cv::Mat &image,
                      std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors,
                      const FeatureDetectorType &detectorType =
                          FeatureDetectorType::SIFT); // 检测特征点和描述符
  std::vector<cv::DMatch> matchFeatures(const cv::Mat &descriptors1,
                                        const cv::Mat &descriptors2,
                                        float ratioThresh = 0.5); // 特征匹配
  cv::Mat drawFeatureMatches(
      const cv::Mat &img1, const std::vector<cv::KeyPoint> &keypoints1,
      const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints2,
      const std::vector<cv::DMatch> &matches); // 绘制特征匹配
  cv::Mat
  Test_DrawFeatureMatches(const cv::Mat &img1, const cv::Mat &img2,
                          const FeatureDetectorType &detectorType =
                              FeatureDetectorType::SIFT); // 测试绘制特征匹配

  void GetGoodMatches(const std::vector<cv::DMatch> &matches,
                      const std::vector<cv::KeyPoint> &keypoints1,
                      const std::vector<cv::KeyPoint> &keypoints2,
                      std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2); // 获取好的匹配点
  void
  GetGoodMatches(const cv::Mat &img1, const cv::Mat &img2,
                 std::vector<cv::Point2f> &points1,
                 std::vector<cv::Point2f> &points2,
                 const FeatureDetectorType &detectorType =
                     FeatureDetectorType::SIFT); // 获取好的匹配点(快捷使用)

  cv::Mat ComputeFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                                   const std::vector<cv::Point2f> &points2,
                                   float threshold = 1.0); // 计算基础矩阵
  void TestFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                             const std::vector<cv::Point2f> &points2,
                             const cv::Mat &fundamentalMatrix,
                             const cv::Mat &img1, const cv::Mat &img2);
  cv::Mat
  ComputeEssentialMatrix(const cv::Mat &K1, const cv::Mat &K2,
                         const cv::Mat &fundamentalMatrix); // 计算本质矩阵
  void TestEssentialMatrix(const std::vector<cv::Point2f> &points1,
                           const std::vector<cv::Point2f> &points2,
                           const cv::Mat &essentialMatrix, const cv::Mat &K1,
                           const cv::Mat &K2, const cv::Mat &img1,
                           const cv::Mat &img2);
  ~SFMFrontend();

private:
  cv::Ptr<cv::SIFT> sift_;
  cv::Ptr<cv::ORB> orb_;
};
} // namespace pre

#endif // SFMFrontend_H