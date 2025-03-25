#include "preprocessing/SFMFrontend.h"
#include "preprocessing/console.h"
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
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
  std::vector<bool> inliers(points1.size(), false);
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

  std::cout << "Triangulation inliers: "
            << std::count(inliers.begin(), inliers.end(), true) << "/"
            << points1.size() << std::endl;

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
  std::cout << Console::INFO << points3D.size() << std::endl;
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
    std::cout << Console::INFO << cloud->points[i].x << ","
              << cloud->points[i].y << "," << cloud->points[i].z << std::endl;
    // 添加颜色信息
    if (hasColor) {
      int x = static_cast<int>(std::round(imagePoints[i].x));
      int y = static_cast<int>(std::round(imagePoints[i].y));

      // 检查点是否在图像范围内
      if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
        if (image.channels() == 3) {
          // BGR图像
          cv::Vec3b color = image.at<cv::Vec3b>(y, x);
          cloud->points[i].b = color[0];
          cloud->points[i].g = color[1];
          cloud->points[i].r = color[2];
        } else if (image.channels() == 1) {
          // 灰度图像
          uchar gray = image.at<uchar>(y, x);
          cloud->points[i].r = gray;
          cloud->points[i].g = gray;
          cloud->points[i].b = gray;
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
void SFMFrontend::twoViewEuclideanReconstruction(
    cv::Mat &img1, cv::Mat &img2, FeatureDetectorType detector_type) {
  img1 = preprocessor_.preprocess(img1);
  img2 = preprocessor_.preprocess(img2);
  std::vector<cv::Point2f> points1, points2;
  GetGoodMatches(img1, img2, points1, points2, detector_type);
  cv::Mat F = ComputeFundamentalMatrix(points1, points2);
  cv::Mat E;
  cv::Mat R2, t2;
  cv::recoverPose(points1, points2, K, D, K, D, E, R2, t2);
  cv::Mat R1 = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  cv::Mat t1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  pclProcessor_.addCamera(R1, t1, 0, img1);
  pclProcessor_.addCamera(R2, t2, 1, img2);
  std::vector<cv::Point3f> points3D =
      robustTriangulate(points1, points2, K, R1, t1, R2, t2, 0.1);
  // points3D = sfmFrontend.scaleToVisibleRange(points3D);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      convertToPointCloud(points3D, points1, img1);
  std::cout << "cloud size: " << cloud->size() << std::endl;
  pclProcessor_.addPointCloud(*cloud, std::to_string(time(nullptr)));
}
void SFMFrontend::twoViewEuclideanReconstruction(
    cv::Mat &img1, cv::Mat &img2, const cv::Mat &InputR, const cv::Mat &Inputt,
    cv::Mat &OutputR, cv::Mat &Outputt, FeatureDetectorType detector_type) {
  img1 = preprocessor_.preprocess(img1);
  img2 = preprocessor_.preprocess(img2);
  std::vector<cv::Point2f> points1, points2;
  GetGoodMatches(img1, img2, points1, points2, detector_type);
  cv::Mat F = ComputeFundamentalMatrix(points1, points2);
  cv::Mat E;
  cv::recoverPose(points1, points2, K, D, K, D, E, OutputR, Outputt);
  pclProcessor_.addCamera(InputR, Inputt, 0, img1);
  pclProcessor_.addCamera(OutputR, Outputt, 1, img2);
  std::vector<cv::Point3f> points3D = robustTriangulate(
      points1, points2, K, InputR, Inputt, OutputR, Outputt, 10000);
  // points3D = sfmFrontend.scaleToVisibleRange(points3D);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud =
      convertToPointCloud(points3D, points1, img1);
  pclProcessor_.addPointCloud(*cloud, std::to_string(time(nullptr)));
}
void SFMFrontend::show() {
  while (!pclProcessor_.getViewer()->wasStopped()) {
    pclProcessor_.getViewer()->spinOnce(100);
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
    std::cout << Console::INFO
              << "Processing base node id: " << base_pair.second.image_id
              << " || before: " << base_pair.second.keypoints.size()
              << std::endl;
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
    std::cout << Console::INFO << "Later: " << base_node.points.size()
              << std::endl;
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
} // namespace pre