#include "preprocessing/SFMFrontend.h"

namespace pre {
SFMFrontend::SFMFrontend() {}
SFMFrontend::SFMFrontend(FeatureDetectorType detector_type) {}
SFMFrontend::~SFMFrontend() {}
void SFMFrontend::createSIFT(int nfeatures, int nOctaveLayers,
                             double contrastThreshold, double edgeThreshold,
                             double sigma, bool enable_precise_upscale) {
  sift_ = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                           edgeThreshold, sigma, enable_precise_upscale);
}
std::vector<cv::DMatch> SFMFrontend::matchFeatures(const cv::Mat &descriptors1,
                                                   const cv::Mat &descriptors2,
                                                   float ratioThresh) {
  // 构建 FLANN 匹配器
  cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5),
                                new cv::flann::SearchParams(50));

  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

  // 应用 Lowe's ratio test
  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < knnMatches.size(); i++) {
    if (knnMatches[i][0].distance < ratioThresh * knnMatches[i][1].distance) {
      goodMatches.push_back(knnMatches[i][0]);
    }
  }

  return goodMatches;
}
void SFMFrontend::detectFeatures(const cv::Mat &image,
                                 std::vector<cv::KeyPoint> &keypoints,
                                 cv::Mat &descriptors) {
  sift_->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}
cv::Mat SFMFrontend::drawFeatureMatches(
    const cv::Mat &img1, const std::vector<cv::KeyPoint> &keypoints1,
    const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints2,
    const std::vector<cv::DMatch> &matches) {
  cv::Mat matchImg;
  drawMatches(img1, keypoints1, img2, keypoints2, matches, matchImg,
              cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  return matchImg;
}
cv::Mat
SFMFrontend::Test_DrawFeatureMatches(const cv::Mat &img1, const cv::Mat &img2,
                                     const FeatureDetectorType &detectorType) {
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  switch (detectorType) {
  case FeatureDetectorType::SIFT:
    createSIFT();
    detectFeatures(img1, keypoints1, descriptors1);
    detectFeatures(img2, keypoints2, descriptors2);
    break;
  default:
    break;
  }

  std::vector<cv::DMatch> matches = matchFeatures(descriptors1, descriptors2);
  return drawFeatureMatches(img1, keypoints1, img2, keypoints2, matches);
}

} // namespace pre