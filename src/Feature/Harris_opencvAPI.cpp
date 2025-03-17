#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main() {
// 使用之前在CMakeLists.txt中定义的资源目录
#ifdef RESOURCE_DIR
  std::string imagePath = std::string(RESOURCE_DIR) + "/room/room_1.jpg";
#else
  std::string imagePath = "chessboard.jpg"; // 默认路径
#endif

  // 读取输入图像
  cv::Mat src = cv::imread(imagePath);
  if (src.empty()) {
    std::cerr << "Error: Could not open or find the image: " << imagePath
              << std::endl;
    return -1;
  }

  // 显示原始图像
  cv::imshow("Original Image", src);

  // 转换为灰度图
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

  // 角点检测参数
  int blockSize = 2; // 邻域大小
  int ksize = 3;     // Sobel算子的孔径参数
  double k = 0.04;   // Harris检测器自由参数

  // Harris角点检测
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(src.size(), CV_32FC1);

  // 执行Harris角点检测
  cv::cornerHarris(gray, dst, blockSize, ksize, k);

  // 归一化角点强度
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // 标记角点
  cv::Mat result = src.clone();
  for (int i = 0; i < dst_norm.rows; i++) {
    for (int j = 0; j < dst_norm.cols; j++) {
      // 设置阈值，只保留强角点
      if ((int)dst_norm.at<float>(i, j) > 100) {
        // 绘制角点，参数：图像、点、半径、颜色、粗细
        cv::circle(result, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 1,
                   cv::LINE_AA);
      }
    }
  }

  // 显示结果
  cv::imshow("Harris Corners", result);

  // 等待用户按键退出
  cv::waitKey(0);

  return 0;
}