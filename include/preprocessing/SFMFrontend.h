#ifndef SFMFrontend_H
#define SFMFrontend_H
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

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
                  bool enable_precise_upscale = false);
  void detectFeatures(const cv::Mat &image,
                      std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors);
  std::vector<cv::DMatch> matchFeatures(const cv::Mat &descriptors1,
                                        const cv::Mat &descriptors2,
                                        float ratioThresh = 0.5);
  cv::Mat drawFeatureMatches(const cv::Mat &img1,
                             const std::vector<cv::KeyPoint> &keypoints1,
                             const cv::Mat &img2,
                             const std::vector<cv::KeyPoint> &keypoints2,
                             const std::vector<cv::DMatch> &matches);
  cv::Mat Test_DrawFeatureMatches(const cv::Mat &img1, const cv::Mat &img2,
                                  const FeatureDetectorType &detectorType);
  ~SFMFrontend();

private:
  cv::Ptr<cv::SIFT> sift_;
  cv::Ptr<cv::ORB> orb_;
  //  cv::Ptr<cv::xfeatures2d::SURF> surf_;
};
} // namespace pre

#endif // SFMFrontend_H