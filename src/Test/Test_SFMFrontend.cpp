#include "preprocessing/SFMFrontend.h"
#include "preprocessing/pre.h"
int main() {
  pre::SFMFrontend sfmFrontend;
  pre::CameraPreprocessor preprocessor;
  cv::Mat img1 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_1.jpg");
  cv::Mat img2 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_2.jpg");
  img1 = preprocessor.preprocess(img1);
  img2 = preprocessor.preprocess(img2);
  cv::Mat resualtImage_ =
      sfmFrontend.Test_DrawFeatureMatches(img1, img2, pre::SIFT);
  cv::imshow("result", resualtImage_);
  cv::waitKey(0);
}