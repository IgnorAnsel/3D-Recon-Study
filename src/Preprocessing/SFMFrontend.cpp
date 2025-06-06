#include "preprocessing/SFMFrontend.h"
#include "preprocessing/console.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
namespace pre {
SFMFrontend::SFMFrontend() {}
SFMFrontend::SFMFrontend(const std::string &cameraParamsPath) {
  preprocessor_.loadCameraParams(cameraParamsPath);
  pclProcessor_.initPCLWithNoCloud();
  K = preprocessor_.getIntrinsicMatrix();
  D = preprocessor_.getDistortionCoefficients();
}
SFMFrontend::SFMFrontend(FeatureDetectorType detector_type) {}
SFMFrontend::~SFMFrontend() {}
bool SFMFrontend::haveImage(const std::string &imagePath, cv::Mat &image) {
  if (!cv::imread(imagePath).empty()) {
    image = cv::imread(imagePath);
    return true;
  }
  image = cv::Mat();
  return false;
}

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
  matcher.knnMatch(descriptors2, descriptors1, knnMatches, 2);

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
  matches = matchFeatures(descriptors1, descriptors2, 0.7);
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
}
cv::Mat SFMFrontend::ComputeEssentialMatrix(const cv::Mat &K1,
                                            const cv::Mat &K2,
                                            const cv::Mat &fundamentalMatrix) {
  if (K1.empty() || K2.empty() || fundamentalMatrix.empty()) {
    std::cerr << Console::ERROR
              << "[SFMFrontend::ComputeEssentialMatrix] Invalid input matrices!"
              << std::endl;
    return cv::Mat();
  }
  if (K1.cols != 3 || K1.rows != 3 || K2.cols != 3 || K2.rows != 3 ||
      fundamentalMatrix.cols != 3 || fundamentalMatrix.rows != 3) {
    std::cerr << Console::ERROR
              << "[SFMFrontend::ComputeEssentialMatrix] Invalid input matrix "
                 "dimensions!"
              << std::endl;
    return cv::Mat();
  }
  cv::Mat U, D, Vt;
  cv::SVDecomp(fundamentalMatrix, D, U, Vt,
               cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  cv::Mat Dn = cv::Mat::eye(3, 3, CV_64F);
  Dn.at<double>(2, 2) = 0;
  cv::Mat essentialMatrix = U * Dn * Vt;
  return essentialMatrix;
}
void SFMFrontend::TestEssentialMatrix(const std::vector<cv::Point2f> &points1,
                                      const std::vector<cv::Point2f> &points2,
                                      const cv::Mat &essentialMatrix,
                                      const cv::Mat &K1, const cv::Mat &K2,
                                      const cv::Mat &img1,
                                      const cv::Mat &img2) {
  // 1. 创建用于显示的输出图像
  cv::Mat outImg;
  cv::hconcat(img1, img2, outImg);

  // 确保点对数量相等
  if (points1.size() != points2.size()) {
    std::cerr << Console::ERROR << "点集数量不匹配" << std::endl;
    return;
  }

  // 2. 计算对极约束误差
  double totalError = 0.0;
  std::vector<double> errors;

  // // 转换为归一化坐标
  std::vector<cv::Point2f> normPoints1, normPoints2;
  // cv::undistortPoints(points1, normPoints1, K1, cv::Mat());
  // cv::undistortPoints(points2, normPoints2, K2, cv::Mat());
  normPoints1 = points1;
  normPoints2 = points2;
  for (size_t i = 0; i < normPoints1.size(); i++) {
    // 将归一化点转换为齐次坐标
    cv::Mat p1 =
        (cv::Mat_<double>(3, 1) << normPoints1[i].x, normPoints1[i].y, 1.0);
    cv::Mat p2 =
        (cv::Mat_<double>(3, 1) << normPoints2[i].x, normPoints2[i].y, 1.0);

    // 计算对极约束误差: x2'^T * E * x1' 应接近0
    cv::Mat error = p2.t() * essentialMatrix * p1;
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

  // 4. 验证本质矩阵的代数性质
  cv::SVD svd(essentialMatrix, cv::SVD::FULL_UV);
  std::vector<double> singularValues = {
      svd.w.at<double>(0), svd.w.at<double>(1), svd.w.at<double>(2)};

  // 从本质矩阵恢复相对姿态
  cv::Mat R1, R2, t;
  cv::decomposeEssentialMat(essentialMatrix, R1, R2, t);

  // 绘制一些对极线
  const int numLinesToDraw = std::min(10, static_cast<int>(points1.size()));
  std::vector<int> indices(points1.size());
  for (size_t i = 0; i < indices.size(); i++)
    indices[i] = i;
  std::random_shuffle(indices.begin(), indices.end());

  for (int i = 0; i < numLinesToDraw; i++) {
    int idx = indices[i];

    // 计算对极线
    cv::Mat p1 =
        (cv::Mat_<double>(3, 1) << normPoints1[idx].x, normPoints1[idx].y, 1.0);
    cv::Mat epiline1 = essentialMatrix * p1;

    // 将对极线从归一化坐标转回像素坐标
    double a = epiline1.at<double>(0);
    double b = epiline1.at<double>(1);
    double c = epiline1.at<double>(2);

    // 归一化线参数
    double norm = std::sqrt(a * a + b * b);
    a /= norm;
    b /= norm;
    c /= norm;

    // 将对极线转换到像素坐标系下
    cv::Mat lineInPixels = K2.t() * epiline1;
    a = lineInPixels.at<double>(0);
    b = lineInPixels.at<double>(1);
    c = lineInPixels.at<double>(2);

    // 归一化
    norm = std::sqrt(a * a + b * b);
    a /= norm;
    b /= norm;
    c /= norm;

    // 求线与图像边界的交点
    double x0 = 0, x1 = img2.cols;
    double y0 = (-c - a * x0) / b;
    double y1 = (-c - a * x1) / b;

    // 绘制对极线
    cv::line(outImg, cv::Point(x0 + img1.cols, y0),
             cv::Point(x1 + img1.cols, y1), cv::Scalar(255, 0, 0), 1);
  }

  // 5. 显示统计信息与评估
  double avgError = totalError / normPoints1.size();

  // 整理输出结果
  std::cout << Console::TEST
            << "========== 本质矩阵测试结果 ==========" << std::endl;
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

  // 显示奇异值
  std::cout << Console::TEST << "│ 奇异值比率 σ1/σ2  │ " << std::fixed
            << std::setprecision(8) << std::setw(13)
            << singularValues[0] / singularValues[1] << " │" << std::endl;
  std::cout << Console::TEST << "│ 最小奇异值 σ3     │ " << std::fixed
            << std::setprecision(8) << std::setw(13) << singularValues[2]
            << " │" << std::endl;
  std::cout << Console::TEST << "└───────────────────┴───────────────┘"
            << std::endl;

  // 恢复的旋转和平移
  std::cout << Console::TEST << "恢复的相对姿态:" << std::endl;
  std::cout << Console::TEST << "  旋转矩阵 R1:" << std::endl;
  for (int i = 0; i < 3; i++) {
    std::cout << Console::TEST << "    [";
    for (int j = 0; j < 3; j++) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(6)
                << R1.at<double>(i, j);
      if (j < 2)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }

  std::cout << Console::TEST << "  平移向量 t:" << std::endl;
  std::cout << Console::TEST << "    [" << std::setw(10) << t.at<double>(0)
            << ", " << std::setw(10) << t.at<double>(1) << ", " << std::setw(10)
            << t.at<double>(2) << "]" << std::endl;

  // 评估结果
  std::string qualityAssessment;
  if (medianError < 0.005)
    qualityAssessment = "优秀";
  else if (medianError < 0.01)
    qualityAssessment = "良好";
  else if (medianError < 0.02)
    qualityAssessment = "可接受";
  else
    qualityAssessment = "较差";

  std::cout << Console::TEST << "本质矩阵质量评估: " << qualityAssessment
            << std::endl;
  std::cout << Console::TEST
            << "======================================" << std::endl;

  // 显示结果图像
  cv::imshow("Essential Matrix Test", outImg);
}
void SFMFrontend::ComputePoseFromEssentialMatrix(
    const cv::Mat &E, const std::vector<cv::Point2f> &points1,
    const std::vector<cv::Point2f> &points2, const cv::Mat &K, cv::Mat &R,
    cv::Mat &t) {
  // 在这之前，已经对图像进行了畸变校正
  // std::vector<cv::Point2f> normalizedPts1, normalizedPts2;
  // cv::undistortPoints(points1, normalizedPts1, K, cv::Mat());
  // cv::undistortPoints(points2, normalizedPts2, K, cv::Mat());

  // 使用OpenCV的内置函数分解本质矩阵并恢复姿态
  cv::Mat mask;
  // int inliers =
  //     cv::recoverPose(E, normalizedPts1, normalizedPts2, K, R, t, mask);
  int inliers = cv::recoverPose(E, points1, points2, K, R, t, mask);
  std::cout << Console::INFO << "从本质矩阵恢复相机姿态:" << std::endl;
  std::cout << Console::INFO << "  • 内点数量: " << inliers << "/"
            << points1.size() << std::endl;
  std::cout << Console::INFO << "  • 旋转矩阵 R:" << std::endl;
  std::cout << Console::INFO << R << std::endl;

  std::cout << Console::INFO << "  • 平移向量 t:" << std::endl;
  std::cout << Console::INFO << t << std::endl;

  return;
}
std::vector<cv::Point3f>
SFMFrontend::robustTriangulate(const std::vector<cv::Point2f> &points1,
                               const std::vector<cv::Point2f> &points2,
                               const cv::Mat &K, const cv::Mat &R1,
                               const cv::Mat &t1, const cv::Mat &R2,
                               const cv::Mat &t2, float reprojectionThreshold) {

  // 计算投影矩阵
  cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
  cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);

  R1.copyTo(P1.colRange(0, 3));
  t1.copyTo(P1.col(3));
  P1 = K * P1;

  R2.copyTo(P2.colRange(0, 3));
  t2.copyTo(P2.col(3));
  P2 = K * P2;

  // 三角化所有点
  cv::Mat points4D;
  cv::triangulatePoints(P1, P2, points1, points2, points4D);

  // 将点转换为3D点并进行重投影测试
  std::vector<cv::Point3f> points3D;
  // std::vector<bool> inliers(points1.size(), false);
  cv::convertPointsFromHomogeneous(points4D.t(), points3D);

  // for (int i = 0; i < points4D.cols; ++i) {
  //   // 转换为三维点
  //   cv::Mat X = points4D.col(i);
  //   X /= X.at<double>(3, 0);

  //   cv::Point3f point3D(static_cast<float>(X.at<double>(0, 0)),
  //                       static_cast<float>(X.at<double>(1, 0)),
  //                       static_cast<float>(X.at<double>(2, 0)));
  //   // std::cout << "point3D: " << point3D << std::endl;
  //   // 重投影测试
  //   cv::Mat X3D =
  //       (cv::Mat_<double>(4, 1) << point3D.x, point3D.y, point3D.z, 1);

  //   // 重投影到相机1
  //   // 确保两个矩阵具有相同的数据类型
  //   cv::Mat X3D_converted;
  //   X3D.convertTo(X3D_converted, P1.type());

  //   cv::Mat x1 = P1 * X3D_converted;
  //   cv::Point2f reprojPoint1(x1.at<double>(0) / x1.at<double>(2),
  //                            x1.at<double>(1) / x1.at<double>(2));
  //   float error1 = cv::norm(reprojPoint1 - points1[i]);

  //   // 重投影到相机2
  //   cv::Mat x2 = P2 * X3D_converted;
  //   cv::Point2f reprojPoint2(x2.at<double>(0) / x2.at<double>(2),
  //                            x2.at<double>(1) / x2.at<double>(2));
  //   float error2 = cv::norm(reprojPoint2 - points2[i]);
  //   // std::cout << "error1: " << error1 << ", error2: " << error2 <<
  //   std::endl;
  //   //  检查重投影误差
  //   if (error1 < reprojectionThreshold && error2 < reprojectionThreshold) {

  //     inliers[i] = true;
  //     points3D.push_back(point3D);
  //   }
  // }

  // std::cout << "Triangulation inliers: "
  //           << std::count(inliers.begin(), inliers.end(), true) << "/"
  //           << points1.size() << std::endl;

  return points3D;
}
bool SFMFrontend::find_transform(cv::Mat &K, std::vector<cv::KeyPoint> &p1,
                                 std::vector<cv::KeyPoint> &p2, cv::Mat &R,
                                 cv::Mat &T, cv::Mat &mask) {
  //根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
  double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
  cv::Point2d principle_point(K.at<double>(2), K.at<double>(5));

  std::vector<cv::Point2f> _p1, _p2;
  for (int i = 0; i < p1.size(); i++) {
    _p1.push_back(p1[i].pt);
    _p2.push_back(p2[i].pt);
  }

  //根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
  cv::Mat E = cv::findEssentialMat(_p1, _p2, focal_length, principle_point,
                                   cv::RANSAC, 0.999, 1.0, mask);
  if (E.empty())
    return false;

  double feasible_count = cv::countNonZero(mask);
  std::cout << (int)feasible_count << " -in- " << p1.size() << std::endl;
  //对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
  if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
    return false;

  //分解本征矩阵，获取相对变换
  int pass_count =
      cv::recoverPose(E, _p1, _p2, R, T, focal_length, principle_point, mask);

  //同时位于两个相机前方的点的数量要足够大
  if (((double)pass_count) / feasible_count < 0.7)
    return false;
  return true;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
SFMFrontend::convertToPointCloud(const std::vector<cv::Point3f> &points3D,
                                 const std::vector<cv::Point2f> &imagePoints,
                                 const cv::Mat &image) {

  // 创建PCL点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  // std::cout << Console::INFO << "[convertToPointCloud]"
  //           << "3D points: " << points3D.size() << std::endl;
  // 设置点云基本属性
  cloud->width = points3D.size();
  cloud->height = 1;       // 无序点云
  cloud->is_dense = false; // 可能包含无效点
  cloud->points.resize(points3D.size());

  bool hasColor = !image.empty() && imagePoints.size() == points3D.size();

  // 转换每个点
  for (size_t i = 0; i < points3D.size(); ++i) {
    // 复制坐标
    cloud->points[i].x = points3D[i].x;
    cloud->points[i].y = points3D[i].y;
    cloud->points[i].z = points3D[i].z;
    // 添加颜色信息
    if (hasColor) {
      int x = static_cast<int>(std::round(imagePoints[i].x));
      int y = static_cast<int>(std::round(imagePoints[i].y));

      // 检查点是否在图像范围内
      if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
        if (image.channels() == 3) {
          // BGR图像
          cv::Vec3b color = image.at<cv::Vec3b>(y, x);
          cloud->points[i].b = 255; // color[0];
          cloud->points[i].g = 255; // color[1];
          cloud->points[i].r = 255; // color[2];
        } else if (image.channels() == 1) {
          // 灰度图像
          uchar gray = image.at<uchar>(y, x);
          cloud->points[i].r = 255; // gray;
          cloud->points[i].g = 255; // gray;
          cloud->points[i].b = 255; // gray;
        }
      } else {
        // 默认颜色：白色
        cloud->points[i].r = 255;
        cloud->points[i].g = 255;
        cloud->points[i].b = 255;
      }
    } else {
      // 无颜色信息时使用默认颜色：白色
      cloud->points[i].r = 255;
      cloud->points[i].g = 255;
      cloud->points[i].b = 255;
    }
  }
  std::cout << Console::INFO << "[convertToPointCloud]"
            << " Point cloud generated with " << cloud->points.size()
            << " points." << std::endl;
  return cloud;
}
std::vector<cv::Point3f>
SFMFrontend::scaleToVisibleRange(std::vector<cv::Point3f> &points3D) {
  if (points3D.empty()) {
    std::cerr << "Error: No 3D points available for scaling!" << std::endl;
    return points3D;
  }

  // 计算每个坐标轴的最小值和最大值
  float minX = points3D[0].x, maxX = points3D[0].x;
  float minY = points3D[0].y, maxY = points3D[0].y;
  float minZ = points3D[0].z, maxZ = points3D[0].z;

  for (const auto &point : points3D) {
    minX = std::min(minX, point.x);
    maxX = std::max(maxX, point.x);
    minY = std::min(minY, point.y);
    maxY = std::max(maxY, point.y);
    minZ = std::min(minZ, point.z);
    maxZ = std::max(maxZ, point.z);
  }

  // 计算每个坐标轴的缩放因子，目标范围是1e7到1e9之间
  float scaleFactorX = 10000000.0f / (maxX - minX);
  float scaleFactorY = 1000000000.0f / (maxY - minY);
  float scaleFactorZ = 10000000.0f / (maxZ - minZ);

  // 对所有点进行缩放
  for (auto &point : points3D) {
    point.x = point.x * scaleFactorX;
    point.y = point.y * scaleFactorY;
    point.z = point.z * scaleFactorZ;
  }

  return points3D;
}
std::vector<cv::Point3f>
SFMFrontend::homogeneous2euclidean(const cv::Mat &points4D) {
  // CV_Assert(points4D.rows == 4);  // 确保是4行N列的齐次坐标
  // CV_Assert(points4D.type() == CV_32F || points4D.type() == CV_64F); //
  // 支持浮点或双精度

  const int num_points = points4D.cols; // 点的数量
  std::vector<cv::Point3f> points3D;
  points3D.reserve(num_points);

  for (int i = 0; i < num_points; ++i) {
    // 获取第i个点的齐次坐标 (x, y, z, w)
    const double *p4 = points4D.ptr<double>(0) + 4 * i; // 假设双精度输入
    const double w = p4[3];                             // 第4个分量是w

    if (std::abs(w) < 1e-9) {
      points3D.emplace_back(0, 0, 0); // 无效点设为原点（或抛出异常）
      continue;
    }

    const double inv_w = 1.0 / w;
    const double x = p4[0] * inv_w;
    const double y = p4[1] * inv_w;
    const double z = p4[2] * inv_w;

    points3D.emplace_back(static_cast<float>(x), static_cast<float>(y),
                          static_cast<float>(z));
  }

  return points3D;
}
std::vector<cv::Point3f> SFMFrontend::twoViewEuclideanReconstruction(
    cv::Mat &img1, cv::Mat &img2, FeatureDetectorType detector_type,
    bool isProcessed, int best_cam, int next_cam) {
  if (!isProcessed) {
    img1 = preprocessor_.preprocess(img1);
    img2 = preprocessor_.preprocess(img2);
  }
  std::vector<cv::Point2f> points1, points2;
  GetGoodMatches(img1, img2, points1, points2, detector_type);
  cv::Mat F = ComputeFundamentalMatrix(points1, points2);
  cv::Mat E;
  cv::Mat R2, t2;
  cv::recoverPose(points1, points2, K, D, K, D, E, R2, t2);
  cv::Mat R1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  cv::Mat t1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  pclProcessor_.addCamera(R1, t1, best_cam, img1);
  pclProcessor_.addCamera(R2, t2, next_cam, img2);
  std::vector<cv::Point3f> points3D =
      robustTriangulate(points1, points2, K, R1, t1, R2, t2, 0.1);
  // points3D = sfmFrontend.scaleToVisibleRange(points3D);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      convertToPointCloud(points3D, points1, img1);
  // std::cout << "cloud size: " << cloud->size() << std::endl;
  pclProcessor_.addPointCloud(*cloud, std::to_string(time(nullptr)));
  return points3D;
}
std::vector<cv::Point3f> SFMFrontend::twoViewEuclideanReconstruction(
    cv::Mat &img1, cv::Mat &img2, const cv::Mat &InputR, const cv::Mat &Inputt,
    cv::Mat &OutputR, cv::Mat &Outputt, FeatureDetectorType detector_type,
    bool isProcessed, int best_cam, int next_cam, bool addCamera1,
    bool addCamera2) {
  if (!isProcessed) {
    img1 = preprocessor_.preprocess(img1);
    img2 = preprocessor_.preprocess(img2);
  }
  std::vector<cv::Point2f> points1, points2;
  GetGoodMatches(img1, img2, points1, points2, detector_type);
  cv::Mat F = ComputeFundamentalMatrix(points1, points2);
  cv::Mat E;
  cv::recoverPose(points1, points2, K, D, K, D, E, OutputR, Outputt);
  if (addCamera1)
    pclProcessor_.addCamera(InputR, Inputt, best_cam, img1);
  if (addCamera2)
    pclProcessor_.addCamera(OutputR, Outputt, next_cam, img2);
  std::vector<cv::Point3f> points3D = robustTriangulateWithFilter(
      points1, points2, K, InputR, Inputt, OutputR, Outputt, 10000);
  // points3D = sfmFrontend.scaleToVisibleRange(points3D);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      convertToPointCloud(points3D, points1, img1);
  pclProcessor_.addPointCloud(*cloud, "cloud" + std::to_string(best_cam) + "_" +
                                          std::to_string(next_cam));
  return points3D;
}
void SFMFrontend::show() {
  std::cout << Console::INFO
            << "Cloud size :" << pclProcessor_.getGlobalCloud()->size()
            << std::endl;
  while (!pclProcessor_.getViewer()->wasStopped()) {
    pclProcessor_.getViewer()->spinOnce(100);
    cv::waitKey(1);
  }
}
void SFMFrontend::processShow() {
  bool isok = false;
  while (!pclProcessor_.getViewer()->wasStopped()) {
    pclProcessor_.getViewer()->spinOnce(100);
    if (!isok) {
      incrementalSFM();
      isok = true;
    }
    cv::waitKey(1);
  }
}
void SFMFrontend::processImageNodes(std::vector<ImageNode> &all_nodes,
                                    float ratio_threshold,
                                    int min_match_count) {
  for (size_t base_idx = 0; base_idx < all_nodes.size(); ++base_idx) {
    ImageNode &base_node = all_nodes[base_idx];
    if (base_node.descriptors.empty())
      continue;

    std::vector<int> match_counts(base_node.keypoints.size(), 0);

    for (size_t target_idx = 0; target_idx < all_nodes.size(); ++target_idx) {
      if (target_idx == base_idx)
        continue;
      ImageNode &target_node = all_nodes[target_idx];
      if (target_node.descriptors.empty())
        continue;

      // 使用FLANN匹配器（适合SIFT/SURF）
      cv::Ptr<cv::DescriptorMatcher> matcher =
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(base_node.descriptors, target_node.descriptors,
                        knn_matches, 2);

      // 应用比率测试
      std::vector<cv::DMatch> good_matches;
      for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() < 2)
          continue;
        if (knn_matches[i][0].distance <
            ratio_threshold * knn_matches[i][1].distance) {
          good_matches.push_back(knn_matches[i][0]);
        }
      }

      // 更新匹配次数
      for (const auto &match : good_matches) {
        int query_idx = match.queryIdx;
        if (query_idx >= 0 &&
            query_idx < static_cast<int>(match_counts.size())) {
          match_counts[query_idx]++;
        }
      }
    }

    // 保留匹配次数足够的特征点
    base_node.points.clear();
    for (size_t i = 0; i < match_counts.size(); ++i) {
      if (match_counts[i] >= min_match_count) {
        base_node.points.push_back(base_node.keypoints[i]);
        base_node.points_descriptors.push_back(base_node.descriptors.row(i));
      }
    }
  }
}
void SFMFrontend::processImageGraph(std::map<int, ImageNode> &image_graph,
                                    float ratio_threshold,
                                    int min_match_count) {
  // 遍历所有基底节点 (base_node)
  for (auto &base_pair : image_graph) {
    ImageNode &base_node = base_pair.second;
    if (base_node.descriptors.empty())
      continue;

    // 初始化匹配次数计数器：索引对应 base_node.keypoints 的索引
    std::vector<int> match_counts(base_node.keypoints.size(), 0);

    // 遍历所有目标节点 (target_node)
    for (auto &target_pair : image_graph) {
      ImageNode &target_node = target_pair.second;

      // 跳过自身匹配
      if (target_node.image_id == base_node.image_id)
        continue;
      if (target_node.descriptors.empty())
        continue;

      // 使用 FLANN 匹配器进行特征匹配
      cv::Ptr<cv::DescriptorMatcher> matcher =
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(base_node.descriptors, target_node.descriptors,
                        knn_matches, 2);

      // 应用比率测试筛选优质匹配
      std::vector<cv::DMatch> good_matches;
      for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() < 2)
          continue;
        if (knn_matches[i][0].distance <
            ratio_threshold * knn_matches[i][1].distance) {
          good_matches.push_back(knn_matches[i][0]);
        }
      }

      // 统计基底节点特征点的匹配次数
      for (const auto &match : good_matches) {
        int query_idx = match.queryIdx;
        if (query_idx >= 0 &&
            query_idx < static_cast<int>(match_counts.size())) {
          match_counts[query_idx]++;
        }
      }
    }

    // 保留匹配次数达到阈值的特征点到 points 中
    base_node.points.clear();
    for (size_t i = 0; i < match_counts.size(); ++i) {
      if (match_counts[i] >= min_match_count) {
        base_node.points.push_back(base_node.keypoints[i]);
        base_node.points_descriptors.push_back(base_node.descriptors.row(i));
      }
    }
  }
}
void SFMFrontend::processImageGraph(float ratio_threshold,
                                    int min_match_count) {
  // 遍历所有基底节点 (base_node)
  for (auto &base_pair : image_graph_) {
    // std::cout << Console::INFO
    //           << "Processing base node id: " << base_pair.second.image_id
    //           << " || before: " << base_pair.second.keypoints.size()
    //           << std::endl;
    ImageNode &base_node = base_pair.second;
    if (base_node.descriptors.empty())
      continue;

    // 初始化匹配次数计数器：索引对应 base_node.keypoints 的索引
    std::vector<int> match_counts(base_node.keypoints.size(), 0);

    // 遍历所有目标节点 (target_node)
    for (auto &target_pair : image_graph_) {
      ImageNode &target_node = target_pair.second;

      // 跳过自身匹配
      if (target_node.image_id == base_node.image_id)
        continue;
      if (target_node.descriptors.empty())
        continue;

      // 使用 FLANN 匹配器进行特征匹配
      cv::Ptr<cv::DescriptorMatcher> matcher =
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(base_node.descriptors, target_node.descriptors,
                        knn_matches, 2);
      matcher->knnMatch(target_node.descriptors, base_node.descriptors,
                        knn_matches, 2);
      // 应用比率测试筛选优质匹配
      std::vector<cv::DMatch> good_matches;
      for (size_t i = 0; i < knn_matches.size(); ++i) {
        if (knn_matches[i].size() < 2)
          continue;
        if (knn_matches[i][0].distance <
            ratio_threshold * knn_matches[i][1].distance) {
          good_matches.push_back(knn_matches[i][0]);
        }
      }

      // 统计基底节点特征点的匹配次数
      for (const auto &match : good_matches) {
        int query_idx = match.queryIdx;
        if (query_idx >= 0 &&
            query_idx < static_cast<int>(match_counts.size())) {
          match_counts[query_idx]++;
        }
      }
    }

    // 保留匹配次数达到阈值的特征点到 points 中
    base_node.points.clear();
    for (size_t i = 0; i < match_counts.size(); ++i) {
      if (match_counts[i] >= min_match_count) {
        base_node.points.push_back(base_node.keypoints[i]);
        base_node.points_descriptors.push_back(base_node.descriptors.row(i));
      }
    }
    // std::cout << Console::INFO << "Later: " << base_node.points.size()
    //           << std::endl;
  }
}
std::map<int, ImageNode> SFMFrontend::getImageGraph() { return image_graph_; }
void SFMFrontend::populateImageGraph(std::map<int, ImageNode> &imageGraph,
                                     const std::string &filePathBegin,
                                     int startImageId) {
  cv::Mat image;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  for (int image_id = startImageId;; image_id++) {
    std::string imagePath = filePathBegin + std::to_string(image_id) + ".jpg";
    if (!haveImage(imagePath, image)) {
      break;
    }
    image = preprocessor_.preprocess(image);
    detectFeatures(image, keypoints, descriptors, FeatureDetectorType::SIFT);
    ImageNode node;
    node.image = image;
    node.image_id = image_id;
    node.keypoints = keypoints;
    node.descriptors = descriptors;
    imageGraph.insert({image_id, node});
  }
}
void SFMFrontend::populateImageGraph(const std::string &filePathBegin,
                                     int startImageId) {
  cv::Mat image;
  createSIFT();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  for (int image_id = startImageId;; image_id++) {
    std::string imagePath = filePathBegin + std::to_string(image_id) + ".jpg";
    if (!haveImage(imagePath, image)) {
      break;
    }
    image = preprocessor_.preprocess(image);
    detectFeatures(image, keypoints, descriptors, FeatureDetectorType::SIFT);
    ImageNode node;
    node.image = image;
    node.image_id = image_id;
    node.keypoints = keypoints;
    node.descriptors = descriptors;
    image_graph_.insert({image_id, node});
  }
}
void SFMFrontend::populateEdges(int min_matches_threshold) {
  edges_.clear();
  std::vector<int> image_ids;

  // 收集所有图像ID
  for (const auto &pair : image_graph_) {
    image_ids.push_back(pair.first);
  }

  // 遍历所有图像对 (i < j 避免重复)
  for (size_t i = 0; i < image_ids.size(); ++i) {
    int id_i = image_ids[i];
    ImageNode &node_i = image_graph_[id_i];
    if (node_i.points_descriptors.empty())
      continue; // 跳过无优化描述子的节点

    for (size_t j = i + 1; j < image_ids.size(); ++j) {
      int id_j = image_ids[j];
      ImageNode &node_j = image_graph_[id_j];
      if (node_j.points_descriptors.empty())
        continue;

      // 使用knnMatch + 比率测试
      cv::Ptr<cv::DescriptorMatcher> matcher =
          cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher->knnMatch(node_i.points_descriptors, node_j.points_descriptors,
                        knn_matches, 2);

      // 筛选优质匹配
      std::vector<cv::DMatch> good_matches;
      for (const auto &match_pair : knn_matches) {
        if (match_pair.size() < 2)
          continue;
        if (match_pair[0].distance < 0.8 * match_pair[1].distance) {
          good_matches.push_back(match_pair[0]);
        }
      }

      // 若匹配数超过阈值，建立边
      if (good_matches.size() >= min_matches_threshold) {
        edges_.emplace_back(id_i, id_j);
      }
    }
  }
}
void SFMFrontend::printGraphAsMatrix() {
  if (image_graph_.empty()) {
    std::cout << "图像图为空." << std::endl;
    return;
  }
  // 1. 收集所有图像ID并排序
  std::vector<int> image_ids;
  for (const auto &pair : image_graph_) {
    image_ids.push_back(pair.first);
  }
  std::sort(image_ids.begin(), image_ids.end());
  // 2. 创建ID到矩阵索引的映射
  std::map<int, int> id_to_index;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    id_to_index[image_ids[i]] = i;
  }
  // 3. 初始化邻接矩阵（全0）
  int n = image_ids.size();
  std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
  // 4. 填充邻接矩阵
  for (const auto &edge : edges_) {
    int i = id_to_index[edge.first];
    int j = id_to_index[edge.second];
    matrix[i][j] = 1;
    matrix[j][i] = 1; // 无向图对称
  }
  // 5. 打印矩阵
  std::cout << "邻接矩阵 (1表示存在边):\n";
  // 打印列标题（图像ID）
  std::cout << "     "; // 对齐空白
  for (int id : image_ids) {
    std::cout << std::setw(4) << id << " ";
  }
  std::cout << "\n";
  // 打印矩阵行
  for (int i = 0; i < n; ++i) {
    std::cout << std::setw(4) << image_ids[i] << " "; // 行标题
    for (int j = 0; j < n; ++j) {
      std::cout << std::setw(4) << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
void SFMFrontend::deleteEdges(int i, int j) {
  edges_.erase(std::remove_if(edges_.begin(), edges_.end(),
                              [i, j](const auto &edge) {
                                return (edge.first == i && edge.second == j) ||
                                       (edge.first == j && edge.second == i);
                              }),
               edges_.end());
}
bool SFMFrontend::getEdges(int &i, int &j, const bool &isDeleteEdge) {
  if (edges_.empty()) {
    std::cout << Console::WARNING << "No edges available.\n";
    return false;
  }
  std::pair<int, int> edge = edges_.back();
  i = edge.first;
  j = edge.second;
  std::cout << Console::INFO << "Selected edge: (" << i << ", " << j << ")\n";
  if (isDeleteEdge)
    deleteEdges(i, j); // 删除边
  return true;
}
void SFMFrontend::registerImage(int ID) {
  if (ID >= image_graph_.size() || ID < 0) {
    std::cout << Console::WARNING << "Invalid image ID.\n";
    return;
  }
  image_graph_[ID].isRegistered = true;
}
void SFMFrontend::getEdgesWithBestPoints(int &i, int &j) {
  if (edges_.empty()) {
    std::cout << Console::WARNING << "No edges available.\n";
    return;
  }

  double best_score = -1.0;
  int best_i = -1, best_j = -1;

  for (const auto &edge : edges_) {
    int current_i = edge.first;
    int current_j = edge.second;

    // Get the descriptors for both images
    const cv::Mat &descriptors1 = image_graph_[current_i].points_descriptors;
    const cv::Mat &descriptors2 = image_graph_[current_j].points_descriptors;

    if (descriptors1.empty() || descriptors2.empty()) {
      continue;
    }

    // Use FLANN matcher
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Apply Lowe's ratio test
    std::vector<cv::DMatch> good_matches;
    for (size_t m = 0; m < knn_matches.size(); m++) {
      if (knn_matches[m].size() < 2)
        continue;
      if (knn_matches[m][0].distance < 0.8 * knn_matches[m][1].distance) {
        good_matches.push_back(knn_matches[m][0]);
      }
    }

    double avg_ratio = 0.0;
    for (const auto &match : good_matches) {
      if (knn_matches[match.queryIdx].size() >= 2) {
        avg_ratio += knn_matches[match.queryIdx][0].distance /
                     knn_matches[match.queryIdx][1].distance;
      }
    }
    if (!good_matches.empty()) {
      avg_ratio /= good_matches.size();
    }

    double current_score = good_matches.size() * (1.0 - avg_ratio);

    if (current_score > best_score) {
      best_score = current_score;
      best_i = current_i;
      best_j = current_j;
    }
  }

  if (best_i == -1 || best_j == -1) {
    std::cout << Console::WARNING
              << "No valid edges with good matches found.\n";
    return;
  }

  i = best_i;
  j = best_j;
  std::cout << Console::INFO << "Selected best edge: (" << i << ", " << j
            << ") with score: " << best_score << "\n";
}
void SFMFrontend::getEdgesWithMaxPoints(int &i, int &j) {
  if (edges_.empty()) {
    std::cout << Console::WARNING << "No edges available.\n";
    return;
  }
  int maxPoints = 0;
  int Points = 0;
  for (auto &edge : edges_) {
    Points = image_graph_[edge.first].points.size() +
             image_graph_[edge.second].points.size();
    std::cout << Console::INFO << "Edge: (" << edge.first << ", " << edge.second
              << ") Points: " << Points << "\n";
    if (maxPoints < Points) {
      maxPoints = Points;
      i = edge.first;
      j = edge.second;
    }
  }
  std::cout << Console::INFO << "Selected edge: (" << i << ", " << j << ")\n";
}
void SFMFrontend::showEdgesMatchs(int i, int j) {
  cv::Mat img1 = image_graph_[i].image.clone();
  cv::Mat img2 = image_graph_[j].image.clone();
  cv::Mat img3;
  std::vector<cv::DMatch> matches = matchFeatures(
      image_graph_[i].points_descriptors, image_graph_[j].points_descriptors);
  img3 = drawFeatureMatches(img1, image_graph_[i].points, img2,
                            image_graph_[j].points, matches);
  cv::imshow("Matches" + std::to_string(i) + std::to_string(j), img3);
}
void SFMFrontend::showAllEdgesMatchs() {
  for (auto &edge : edges_) {
    showEdgesMatchs(edge.first, edge.second);
  }
  cv::waitKey(0);
}
// 增量式SFM
// 先取出两个做欧式结构恢复，得出基本的点云
// 循环取边，PnP，BA，更新点云
void SFMFrontend::incrementalSFM() {
  int base_cam, next_cam;
  getEdgesWithBestPoints(base_cam, next_cam);
  deleteEdges(base_cam, next_cam);
  std::cout << Console::INFO << "Incremental SFM started.\n";
  cv::Mat img1 = image_graph_[base_cam].image;
  cv::Mat img2 = image_graph_[next_cam].image;
  image_graph_[base_cam].R = cv::Mat::eye(3, 3, CV_64F);
  image_graph_[base_cam].t = cv::Mat::zeros(3, 1, CV_64F);
  points3D_ = twoViewEuclideanReconstruction(
      img1, img2, image_graph_[base_cam].R, image_graph_[base_cam].t,
      image_graph_[next_cam].R, image_graph_[next_cam].t,
      pre::FeatureDetectorType::SIFT, true, base_cam, next_cam); // 初始的点集
  image_graph_[base_cam].points3d = points3D_;
  image_graph_[next_cam].points3d = points3D_;
  std::set<int> registered_cams = {base_cam, next_cam};
  registerImage(base_cam);
  registerImage(next_cam);
  while (!edges_.empty()) {
    // 选择与已注册相机有最多匹配的未注册相机
    int best_cam = -1, max_matches = 0, ref_cam = -1;
    for (const auto &edge : edges_) {
      int cam1 = edge.first, cam2 = edge.second;
      bool cam1_reg = registered_cams.count(cam1);
      bool cam2_reg = registered_cams.count(cam2);
      if (cam1_reg != cam2_reg) { // 找到连接已注册和未注册的边
        int candidate = cam1_reg ? cam2 : cam1;
        int ref = cam1_reg ? cam1 : cam2;

        std::cout << "找到已经注册的参考相机" << ref << "和未注册的候选相机"
                  << candidate << std::endl;
        // 获取两视图间的匹配数
        std::vector<cv::Point2f> pts1, pts2;
        GetGoodMatches(image_graph_[ref].image, image_graph_[candidate].image,
                       pts1, pts2, FeatureDetectorType::SIFT);
        if (pts1.size() > max_matches) {
          max_matches = pts1.size();
          best_cam = candidate;
          ref_cam = ref;
        }
      }
    }
    if (best_cam == -1) {
      std::cerr << Console::WARNING << "No suitable next camera." << std::endl;
      break;
    }
    // 步骤3：使用PnP估计新相机的位姿
    ImageNode &new_node = image_graph_[best_cam];
    ImageNode &ref_node = image_graph_[ref_cam];
    std::cout << "处理：" << best_cam << "和" << ref_cam << std::endl;

    // 获取匹配点对应的3D-2D对应
    // 求解PnP
    cv::Mat rvec_new, tvec_new, R_new;
    cv::Mat rvec_ref, tvec_ref, R_ref;
    std::vector<cv::DMatch> matches =
        matchFeatures(new_node.points_descriptors, ref_node.points_descriptors);
    std::vector<cv::Point3f> matched_points3D_new;
    std::vector<cv::Point2f> matched_points2D_new;
    for (int i = 0; i < ref_node.points3d.size() && i < new_node.points.size();
         i++) {
      matched_points3D_new.push_back(ref_node.points3d[i]);
      matched_points2D_new.push_back(new_node.points[i].pt);
    }
    // for (const auto &match : matches) {
    //   matched_points3D_new.push_back(ref_node.points3d[match.queryIdx]);
    //   matched_points2D_new.push_back(new_node.points[match.trainIdx].pt);
    // }

    // 调用 solvePnPRansac
    cv::solvePnPRansac(matched_points3D_new, matched_points2D_new, K,
                       cv::noArray(), rvec_new, tvec_new);

    cv::Rodrigues(rvec_new, R_new);
    new_node.R = R_new.clone();
    new_node.t = tvec_new.clone();

    if (!new_node.isRegistered) {
      std::cout << "registering new node" << best_cam << std::endl;
      pclProcessor_.addCamera(new_node.R, new_node.t, best_cam, new_node.image);
      // 步骤4：添加新相机到已注册列表
      registered_cams.insert(best_cam);
      registerImage(best_cam);
    }
    // 步骤5：三角化新相机与参考相机的新匹配点
    std::vector<cv::Point2f> new_pts1, new_pts2;
    GetGoodMatches(ref_node.image, new_node.image, new_pts1, new_pts2,
                   FeatureDetectorType::SIFT);
    // 计算投影矩阵
    cv::Mat P1, P2;
    cv::hconcat(ref_node.R, ref_node.t, P1); // 使用ref_node的当前位姿
    P1 = K * P1;

    if (ref_node.R.empty() || ref_node.t.empty()) {
      std::cerr << Console::ERROR << "ref_node.R or ref_node.t is empty."
                << std::endl;
      continue;
    }
    cv::hconcat(new_node.R, new_node.t, P2); // 使用新相机的世界位姿
    P2 = K * P2;
    // 三角化新点
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, new_pts1, new_pts2, points4D);
    std::vector<cv::Point3f> new_points = homogeneous2euclidean(points4D);
    new_node.points3d = new_points;
    // 输出new_points
    new_points.erase(
        std::remove_if(new_points.begin(), new_points.end(),
                       [](const cv::Point3f &p) { return p.z < 0; }),
        new_points.end()); // 去除z小于0的点

    // 合并到全局点云
    points3D_.insert(points3D_.end(), new_points.begin(), new_points.end());
    // 更新点云显示
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
        convertToPointCloud(new_points, new_pts1, new_node.image);
    pclProcessor_.addPointCloud(*cloud,
                                "global_cloud" + std::to_string(best_cam));
    // 删除处理过的边
    deleteEdges(ref_cam, best_cam);
  }
  // 最终显示
  show();
}
void SFMFrontend::incrementalSFM2() {
  int base_cam, next_cam;
  getEdgesWithBestPoints(base_cam, next_cam);
  deleteEdges(base_cam, next_cam);
  std::cout << Console::INFO << "Incremental SFM started.\n";

  // 初始化基础相机
  image_graph_[base_cam].R = cv::Mat::eye(3, 3, CV_64F);
  image_graph_[base_cam].t = cv::Mat::zeros(3, 1, CV_64F);

  // 两视图重建
  points3D_ = twoViewEuclideanReconstruction(
      image_graph_[base_cam].image, image_graph_[next_cam].image,
      image_graph_[base_cam].R, image_graph_[base_cam].t,
      image_graph_[next_cam].R, image_graph_[next_cam].t,
      pre::FeatureDetectorType::SIFT, true, base_cam, next_cam);

  // 初始两个相机的注册
  image_graph_[base_cam].points3d = points3D_;
  image_graph_[next_cam].points3d = points3D_;
  image_graph_[base_cam].isRegistered = true;
  image_graph_[next_cam].isRegistered = true;
  std::set<int> registered_cams = {base_cam, next_cam};
  registerImage(base_cam);
  registerImage(next_cam);

  // 跟踪特征点和3D点之间的对应关系
  std::map<int, std::map<int, int>>
      point2d_to_point3d; // camera_id -> {2d_point_idx -> 3d_point_idx}
  std::map<int, std::set<int>>
      point3d_to_cameras; // 3d_point_idx -> set of camera_ids

  // 为初始重建建立这种对应关系
  // 这需要根据twoViewEuclideanReconstruction实现补充，这里只是概念演示

  while (!edges_.empty()) {
    // 选择下一个最佳相机进行注册
    int best_cam = -1, max_inliers = 0, ref_cam = -1;
    std::vector<cv::Point3f> best_matched_points3D;
    std::vector<cv::Point2f> best_matched_points2D;

    for (const auto &edge : edges_) {
      int cam1 = edge.first, cam2 = edge.second;
      bool cam1_reg = registered_cams.count(cam1);
      bool cam2_reg = registered_cams.count(cam2);

      if (cam1_reg != cam2_reg) { // 仅考虑连接已注册和未注册相机的边
        int candidate = cam1_reg ? cam2 : cam1;
        int ref = cam1_reg ? cam1 : cam2;

        if (image_graph_[candidate].isRegistered)
          continue; // 跳过已注册的相机

        std::cout << "找到已注册参考相机" << ref << "和未注册候选相机"
                  << candidate << std::endl;

        // 为PnP准备3D-2D对应关系
        std::vector<cv::Point3f> matched_points3D;
        std::vector<cv::Point2f> matched_points2D;

        // 获取特征匹配
        std::vector<cv::DMatch> matches =
            matchFeatures(image_graph_[ref].points_descriptors,
                          image_graph_[candidate].points_descriptors);

        // 筛选有对应3D点的匹配
        for (const auto &match : matches) {
          int ref_feat_idx = match.queryIdx;
          if (ref_feat_idx < image_graph_[ref].points3d.size()) {
            matched_points3D.push_back(
                image_graph_[ref].points3d[ref_feat_idx]);
            matched_points2D.push_back(
                image_graph_[candidate].points[match.trainIdx].pt);
          }
        }

        if (matched_points3D.size() >= 8) { // 至少需要8个点进行RANSAC PnP
          // 尝试PnP求解位姿，计算内点数量
          cv::Mat rvec, tvec;
          std::vector<int> inliers;
          if (cv::solvePnPRansac(matched_points3D, matched_points2D, K,
                                 cv::noArray(), rvec, tvec, false, 100, 8.0,
                                 0.99, inliers, cv::SOLVEPNP_EPNP)) {
            if (inliers.size() > max_inliers) {
              max_inliers = inliers.size();
              best_cam = candidate;
              ref_cam = ref;

              // 只保留内点
              std::vector<cv::Point3f> inlier_points3D;
              std::vector<cv::Point2f> inlier_points2D;
              for (int idx : inliers) {
                inlier_points3D.push_back(matched_points3D[idx]);
                inlier_points2D.push_back(matched_points2D[idx]);
              }
              best_matched_points3D = inlier_points3D;
              best_matched_points2D = inlier_points2D;
            }
          }
        }
      }
    }

    if (best_cam == -1) {
      std::cerr << Console::WARNING << "No suitable next camera found."
                << std::endl;
      break;
    }

    // 最终PnP求解最佳下一个相机的位姿
    ImageNode &new_node = image_graph_[best_cam];
    ImageNode &ref_node = image_graph_[ref_cam];
    std::cout << "处理：" << best_cam << "和" << ref_cam << std::endl;

    cv::Mat rvec, tvec, R;
    if (!cv::solvePnPRansac(best_matched_points3D, best_matched_points2D, K,
                            cv::noArray(), rvec, tvec, cv::SOLVEPNP_ITERATIVE,
                            500, 2.0, 0.999, cv::noArray(),
                            cv::SOLVEPNP_EPNP)) {
      std::cerr << Console::ERROR << "Final PnP failed for camera " << best_cam
                << std::endl;
      deleteEdges(ref_cam, best_cam);
      continue;
    }

    cv::Rodrigues(rvec, R);
    new_node.R = R.clone();
    new_node.t = tvec.clone();
    new_node.isRegistered = true;

    // 注册新相机
    registered_cams.insert(best_cam);
    registerImage(best_cam);
    pclProcessor_.addCamera(new_node.R, new_node.t, best_cam, new_node.image);

    // 对已注册相机进行三角化，获取更多的3D点
    for (int reg_cam : registered_cams) {
      if (reg_cam == best_cam)
        continue;

      // 获取两相机间的特征匹配
      std::vector<cv::Point2f> pts1, pts2;
      GetGoodMatches(image_graph_[reg_cam].image, new_node.image, pts1, pts2,
                     FeatureDetectorType::SIFT);

      // 构建投影矩阵
      cv::Mat P1, P2;
      cv::hconcat(image_graph_[reg_cam].R, image_graph_[reg_cam].t, P1);
      P1 = K * P1;

      cv::hconcat(new_node.R, new_node.t, P2);
      P2 = K * P2;

      // 三角化获取3D点
      cv::Mat points4D;
      cv::triangulatePoints(P1, P2, pts1, pts2, points4D);
      std::vector<cv::Point3f> triangulated_points =
          homogeneous2euclidean(points4D);

      // 过滤3D点：检查深度和重投影误差
      std::vector<cv::Point3f> filtered_points;
      std::vector<cv::Point2f> filtered_pts2;

      for (int i = 0; i < triangulated_points.size(); i++) {
        // 检查点是否在相机前方
        if (triangulated_points[i].z <= 0)
          continue;

        // 计算重投影误差
        cv::Mat pt3d_hom =
            (cv::Mat_<double>(4, 1) << triangulated_points[i].x,
             triangulated_points[i].y, triangulated_points[i].z, 1);

        cv::Mat proj1 = P1 * pt3d_hom;
        cv::Point2f proj_pt1(proj1.at<double>(0) / proj1.at<double>(2),
                             proj1.at<double>(1) / proj1.at<double>(2));

        cv::Mat proj2 = P2 * pt3d_hom;
        cv::Point2f proj_pt2(proj2.at<double>(0) / proj2.at<double>(2),
                             proj2.at<double>(1) / proj2.at<double>(2));

        double err1 = cv::norm(pts1[i] - proj_pt1);
        double err2 = cv::norm(pts2[i] - proj_pt2);

        const double MAX_REPROJECTION_ERROR = 2.0; // 像素

        if (err1 < MAX_REPROJECTION_ERROR && err2 < MAX_REPROJECTION_ERROR) {
          filtered_points.push_back(triangulated_points[i]);
          filtered_pts2.push_back(pts2[i]);
        }
      }
      new_node.points3d = filtered_points;
      // 更新全局点云
      for (const auto &pt : filtered_points) {
        points3D_.push_back(pt);
      }

      // 更新新相机的3D点
      new_node.points3d.insert(new_node.points3d.end(), filtered_points.begin(),
                               filtered_points.end());

      // 更新点云可视化
      if (!filtered_points.empty()) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
            convertToPointCloud(filtered_points, filtered_pts2, new_node.image);
        pclProcessor_.addPointCloud(*cloud, "cloud_" + std::to_string(reg_cam) +
                                                "_" + std::to_string(best_cam));
      }
    }

    // 删除已处理的边
    deleteEdges(ref_cam, best_cam);

    // 每隔几个相机执行一次局部光束法平差优化
    if (registered_cams.size() % 3 == 0) {
      // localBundleAdjustment(registered_cams, point2d_to_point3d);
    }
  }

  // 全局光束法平差
  // globalBundleAdjustment();

  // 显示最终结果
  show();
}
std::vector<cv::Point3f> SFMFrontend::robustTriangulateWithFilter(
    const std::vector<cv::Point2f> &points1,
    const std::vector<cv::Point2f> &points2, const cv::Mat &K,
    const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2, const cv::Mat &t2,
    float maxDepth) {
  cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
  cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);

  R1.copyTo(P1.colRange(0, 3));
  t1.copyTo(P1.col(3));
  P1 = K * P1;

  R2.copyTo(P2.colRange(0, 3));
  t2.copyTo(P2.col(3));
  P2 = K * P2;

  cv::Mat points4D;
  cv::triangulatePoints(P1, P2, points1, points2, points4D);

  std::vector<cv::Point3f> points3D;
  for (int i = 0; i < points4D.cols; ++i) {
    cv::Mat x = points4D.col(i);
    x /= x.at<float>(3); // 齐次坐标归一化

    // 关键修改点：显式指定矩阵类型和转换
    cv::Mat p_mat =
        (cv::Mat_<double>(3, 1) << static_cast<double>(x.at<float>(0)),
         static_cast<double>(x.at<float>(1)),
         static_cast<double>(x.at<float>(2)));

    // 转换为相机坐标系（注意所有矩阵都使用CV_64F）
    cv::Mat p_cam1 = R1 * p_mat + t1;
    cv::Mat p_cam2 = R2 * p_mat + t2;

    // 检查深度（Z值）是否有效
    if (p_cam1.at<double>(2) > 0 && p_cam2.at<double>(2) > 0 &&
        p_cam1.at<double>(2) < maxDepth && p_cam2.at<double>(2) < maxDepth) {
      points3D.emplace_back(static_cast<float>(p_mat.at<double>(0)),
                            static_cast<float>(p_mat.at<double>(1)),
                            static_cast<float>(p_mat.at<double>(2)));
    }
  }
  return points3D;
}
void SFMFrontend::shunxuSFM(const std::string &filePathBegin) {
  int currentCam = 0;
  cv::Mat lastImage;
  cv::Mat currneImage;
  cv::Mat lastR, lastT;
  cv::Mat currR, currT;
  std::cout << filePathBegin + std::to_string(currentCam) + ".jpg" << std::endl;
  while (haveImage((filePathBegin + std::to_string(currentCam) + ".jpg"),
                   currneImage)) {
    if (currentCam == 0) {
      lastImage = currneImage.clone();
      lastR = cv::Mat::eye(3, 3, CV_64F);
      lastT = cv::Mat::zeros(3, 1, CV_64F);
      currentCam++;
      continue;
    }
    twoViewEuclideanReconstruction(lastImage, currneImage, lastR, lastT, currR,
                                   currT, FeatureDetectorType::SIFT, false,
                                   currentCam - 1, currentCam, true, false);

    lastImage = currneImage.clone();
    lastR = currR.clone();
    lastT = currT.clone();
    std::cout << "currentCam: " << currentCam << std::endl;
    currentCam++;
  }
  show();
}
} // namespace pre