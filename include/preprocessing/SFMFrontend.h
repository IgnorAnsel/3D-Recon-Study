#ifndef SFMFrontend_H
#define SFMFrontend_H
#include "opencv2/opencv.hpp"
#include <iostream>
namespace pre {

class SFMFrontend {
public:
  SFMFrontend();
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
  ~SFMFrontend();

private:
  cv::Ptr<cv::SIFT> sift_;
};
} // namespace pre

#endif // SFMFrontend_H