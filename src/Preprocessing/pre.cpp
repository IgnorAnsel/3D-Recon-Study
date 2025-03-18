#include "preprocessing/pre.h"
#include "preprocessing/console.h"
namespace pre {
CameraPreprocessor::CameraPreprocessor() {}
CameraPreprocessor::CameraPreprocessor(const std::string &calibFilePath) {
  loadCameraParams(calibFilePath);
}
bool CameraPreprocessor::loadCameraParams(const std::string &calibFilePath) {
  cv::FileStorage fs(calibFilePath, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open camera calibration file: " << calibFilePath
              << std::endl;
    return false;
  }
  fs["cameraMatrix"] >> intrinsicMatrix_;
  fs["distCoeffs"] >> distortionCoefficients_;
  fs.release();
  return true;
}
cv::Mat CameraPreprocessor::getIntrinsicMatrix() const {
  return intrinsicMatrix_;
}
cv::Mat CameraPreprocessor::getDistortionCoefficients() const {
  return distortionCoefficients_;
}
cv::Mat CameraPreprocessor::undistort(const cv::Mat &image) const {
  cv::Mat undistortedImage;
  if (intrinsicMatrix_.empty() || distortionCoefficients_.empty()) {
    std::cerr
        << Console::WARNING
        << "CameraPreprocessor: Camera parameters are not set: Please use "
           "loadCameraParams() first"
        << std::endl;

    return image;
  }
  cv::undistort(image, undistortedImage, intrinsicMatrix_,
                distortionCoefficients_);
  return undistortedImage;
}
cv::Mat CameraPreprocessor::preprocess(const cv::Mat &image) const {
  cv::Mat preprocessedImage;
  preprocessedImage = undistort(image);
  // 后续扩展处理位置

  // 返回预处理后的图像
  return preprocessedImage;
}
} // namespace pre