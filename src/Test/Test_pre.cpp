#include "preprocessing/pre.h"
#include <opencv2/highgui.hpp>

int main()
{
    pre::CameraPreprocessor preprocessor;
    std::string CameraParamsPath_ = std::string(RESOURCE_DIR) + "/room/calibration_result.yml";
    preprocessor.loadCameraParams(CameraParamsPath_);
    cv::Mat image = cv::imread(std::string(RESOURCE_DIR) + "/room/room_1.jpg");
    cv::imshow("original image",image);
    image = preprocessor.preprocess(image);
    cv::imshow("preprocessed image", image);
    cv::waitKey();
    return 0;
}