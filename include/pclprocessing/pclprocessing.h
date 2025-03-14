#ifndef PCLPROCESSING_H
#define PCLPROCESSING_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>

namespace pc {
    class PCLProcessing {
    public:
        PCLProcessing();
        bool initPCL();
        bool initPCL(const std::string &cloudPath);
        bool loadCloud(const std::string &cloudPath);
        bool saveCloud(const std::string &cloudPath);
        template <typename PointT>
        bool addCloud(const pcl::PointCloud<PointT> &cloud);
        bool addCamera(const cv::Mat &R, const cv::Mat &t);
        ~PCLProcessing();
    };
}

#endif // PCLPROCESSING_H