#ifndef PCLPROCESSING_H
#define PCLPROCESSING_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
namespace pc {
    class PCLProcessing {
    public:
        PCLProcessing();
        bool initPCL();
        bool initPCL(const std::string &cloudPath);
        template <typename PointT>
        bool initPCL(const pcl::PointCloud<PointT> &cloud, const std::string viewerName = "PCL Viewer");
        bool loadPointCloud(const std::string &cloudPath);
        bool savePointCloud(const std::string &cloudPath);
        template <typename PointT>
        bool addPointCloud(const pcl::PointCloud<PointT> &cloud);
        template <typename PointT>
        bool addPointCloud(const pcl::PointCloud<PointT> &cloud, const std::string &frameName);
        bool addCamera(const cv::Mat &R, const cv::Mat &t);
        ~PCLProcessing();
    private:
        pcl::PointCloud<pcl::PointXYZ>::Ptr globalCloud_;
        pcl::visualization::PCLVisualizer viewer_;
        bool isInit_;
    };
}

#endif // PCLPROCESSING_H