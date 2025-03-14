#include "pre.h"

namespace pre {
    CameraPreprocessor::CameraPreprocessor() {}
    CameraPreprocessor::CameraPreprocessor(const std::string& calibFilePath) {
        loadCameraParams(calibFilePath);
    }
}