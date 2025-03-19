#ifndef PRE_H
#define PRE_H

#include "config.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
namespace pre {
class CameraPreprocessor {
public:
  CameraPreprocessor();
  CameraPreprocessor(const std::string &calibFilePath);
  // 加载相机参数
  bool loadCameraParams(const std::string &calibFilePath);
  // 获取相机内参矩阵
  cv::Mat getIntrinsicMatrix() const;
  // 获取畸变系数
  cv::Mat getDistortionCoefficients() const;
  // 对图像进行去畸变处理
  cv::Mat undistort(const cv::Mat &image) const;
  // 预处理函数，包括去畸变...(待扩展)
  cv::Mat preprocess(const cv::Mat &image) const;

private:
  cv::Mat intrinsicMatrix_;        // 相机内参矩阵
  cv::Mat distortionCoefficients_; // 畸变系数
};
} // namespace pre

#endif