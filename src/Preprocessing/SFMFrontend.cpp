#include "preprocessing/SFMFrontend.h"
#include "preprocessing/console.h"
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
                                 cv::Mat &descriptors,
                                 const FeatureDetectorType &detectorType) {
  switch (detectorType) {
  case FeatureDetectorType::SIFT:
    sift_->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    break;
  case FeatureDetectorType::SURF:
    break;
  case FeatureDetectorType::ORB:
    break;
  default:
    break;
  }
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
void SFMFrontend::GetGoodMatches(const std::vector<cv::DMatch> &matches,
                                 const std::vector<cv::KeyPoint> &keypoints1,
                                 const std::vector<cv::KeyPoint> &keypoints2,
                                 std::vector<cv::Point2f> &points1,
                                 std::vector<cv::Point2f> &points2) {
  for (size_t i = 0; i < matches.size(); i++) {
    points1.push_back(keypoints1[matches[i].queryIdx].pt);
    points2.push_back(keypoints2[matches[i].trainIdx].pt);
  }
}
void SFMFrontend::GetGoodMatches(const cv::Mat &img1, const cv::Mat &img2,
                                 std::vector<cv::Point2f> &points1,
                                 std::vector<cv::Point2f> &points2,
                                 const FeatureDetectorType &detectorType) {
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> matches;
  switch (detectorType) {
  case FeatureDetectorType::SIFT:
    createSIFT();
  }
  detectFeatures(img1, keypoints1, descriptors1, detectorType);
  detectFeatures(img2, keypoints2, descriptors2, detectorType);
  matches = matchFeatures(descriptors1, descriptors2);
  GetGoodMatches(matches, keypoints1, keypoints2, points1, points2);
}

// -----------------------------------------求解矩阵----------------------------------------------------

cv::Mat
SFMFrontend::ComputeFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                                      const std::vector<cv::Point2f> &points2,
                                      float threshold) {
  cv::Mat fundamentalMatrix =
      cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, threshold, 0.99);
  return fundamentalMatrix;
}
void SFMFrontend::TestFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                                        const std::vector<cv::Point2f> &points2,
                                        const cv::Mat &fundamentalMatrix,
                                        const cv::Mat &img1,
                                        const cv::Mat &img2) {
  // 1. 创建用于显示的输出图像
  cv::Mat outImg;
  cv::hconcat(img1, img2, outImg);

  // 确保点对数量相等
  if (points1.size() != points2.size()) {
    std::cerr << Console::ERROR << "[SFMFrontend::TestFundamentalMatrix]"
              << "Point sets have different sizes" << std::endl;
    return;
  }

  // 2. 计算对极约束误差
  double totalError = 0.0;
  std::vector<double> errors;

  for (size_t i = 0; i < points1.size(); i++) {
    // 将点转换为齐次坐标
    cv::Mat p1 = (cv::Mat_<double>(3, 1) << points1[i].x, points1[i].y, 1.0);
    cv::Mat p2 = (cv::Mat_<double>(3, 1) << points2[i].x, points2[i].y, 1.0);

    // 计算对极约束误差: x2^T * F * x1 应接近0
    cv::Mat error = p2.t() * fundamentalMatrix * p1;
    double err = std::abs(error.at<double>(0, 0));
    errors.push_back(err);
    totalError += err;

    // 3. 在图像上绘制对应点
    cv::circle(outImg, points1[i], 3, cv::Scalar(0, 0, 255), -1);
    cv::circle(outImg, cv::Point2f(points2[i].x + img1.cols, points2[i].y), 3,
               cv::Scalar(0, 0, 255), -1);
    cv::line(outImg, points1[i],
             cv::Point2f(points2[i].x + img1.cols, points2[i].y),
             cv::Scalar(0, 255, 0), 1);
  }

  // 4. 为一些随机选取的点绘制对极线
  const int numLinesToDraw = std::min(10, static_cast<int>(points1.size()));
  std::vector<int> indices(points1.size());
  for (size_t i = 0; i < indices.size(); i++)
    indices[i] = i;
  std::random_shuffle(indices.begin(), indices.end());

  for (int i = 0; i < numLinesToDraw; i++) {
    int idx = indices[i];

    // 计算右图中对应左图点的对极线 (l' = F * x)
    cv::Mat p1 =
        (cv::Mat_<double>(3, 1) << points1[idx].x, points1[idx].y, 1.0);
    cv::Mat epiline1 = fundamentalMatrix * p1;

    // 计算左图中对应右图点的对极线 (l = F^T * x')
    cv::Mat p2 =
        (cv::Mat_<double>(3, 1) << points2[idx].x, points2[idx].y, 1.0);
    cv::Mat epiline2 = fundamentalMatrix.t() * p2;

    // 在右图中绘制对极线
    double a1 = epiline1.at<double>(0);
    double b1 = epiline1.at<double>(1);
    double c1 = epiline1.at<double>(2);

    // 求线与图像边界的交点
    double x0_2 = 0, x1_2 = img2.cols;
    double y0_2 = (-c1 - a1 * x0_2) / b1;
    double y1_2 = (-c1 - a1 * x1_2) / b1;

    // 绘制对极线
    cv::line(outImg, cv::Point(x0_2 + img1.cols, y0_2),
             cv::Point(x1_2 + img1.cols, y1_2), cv::Scalar(255, 0, 0), 1);

    // 在左图中绘制对极线
    double a2 = epiline2.at<double>(0);
    double b2 = epiline2.at<double>(1);
    double c2 = epiline2.at<double>(2);

    // 求线与图像边界的交点
    double x0_1 = 0, x1_1 = img1.cols;
    double y0_1 = (-c2 - a2 * x0_1) / b2;
    double y1_1 = (-c2 - a2 * x1_1) / b2;

    // 绘制对极线
    cv::line(outImg, cv::Point(x0_1, y0_1), cv::Point(x1_1, y1_1),
             cv::Scalar(255, 0, 0), 1);
  }

  // 5. 显示统计信息
  double avgError = totalError / points1.size();

  // 创建统一的格式化输出
  std::cout << Console::TEST
            << "========== 基础矩阵测试结果 ==========" << std::endl;
  std::cout << Console::TEST << "┌───────────────────┬───────────────┐"
            << std::endl;
  std::cout << Console::TEST << "│ 平均对极约束误差  │ " << std::fixed
            << std::setprecision(8) << std::setw(13) << avgError << " │"
            << std::endl;

  // 计算中位数误差
  std::sort(errors.begin(), errors.end());
  double medianError = errors[errors.size() / 2];
  std::cout << Console::TEST << "│ 中位数对极约束误差│ " << std::fixed
            << std::setprecision(8) << std::setw(13) << medianError << " │"
            << std::endl;
  std::cout << Console::TEST << "└───────────────────┴───────────────┘"
            << std::endl;

  // 添加评估结果
  std::string qualityAssessment;
  if (medianError < 0.005)
    qualityAssessment = "优秀";
  else if (medianError < 0.01)
    qualityAssessment = "良好";
  else if (medianError < 0.02)
    qualityAssessment = "可接受";
  else
    qualityAssessment = "较差";

  std::cout << Console::TEST << "基础矩阵质量评估: " << qualityAssessment
            << std::endl;
  std::cout << Console::TEST
            << "======================================" << std::endl;

  // 显示结果图像
  cv::imshow("Fundamental Matrix Test", outImg);
  cv::waitKey(0);
}
} // namespace pre