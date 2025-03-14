#ifndef PRE_H
#define PRE_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>

namespace pre {
    class CameraPreprocessor {
        public:
            CameraPreprocessor();
            CameraPreprocessor(const std::string& calibFilePath);
            bool loadCameraParams(const std::string& calibFilePath);
            cv::Mat getIntrinsicMatrix() const; // 获取相机内参矩阵
            cv::Mat getDistortionCoefficients() const; // 获取畸变系数
            cv::Mat undistort(const cv::Mat& image) const; // 对图像进行去畸变处理
            cv::Mat preprocess(const cv::Mat& image) const; // 预处理函数，包括去畸变...(待扩展)
        private:
            cv::Mat intrinsicMatrix_; // 相机内参矩阵
            cv::Mat distortionCoefficients_; // 畸变系数
    };
}

#endif