#include "preprocessing/SFMFrontend.h"

int main(int argc, char **argv) {
  pre::SFMFrontend sfmFrontend(std::string(RESOURCE_DIR) +
                               "/room/calibration_result.yml");
  cv::Mat img1 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_8.jpg");
  cv::Mat img2 = cv::imread(std::string(RESOURCE_DIR) + "/room/room_9.jpg");
  sfmFrontend.twoViewEuclideanReconstruction(img1, img2,
                                             pre::FeatureDetectorType::SIFT);
  sfmFrontend.show();
}