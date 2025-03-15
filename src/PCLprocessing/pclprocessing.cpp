#include "pclprocessing/pclprocessing.h"

// pc => pclprocessing
namespace pc {
    PCLProcessing::PCLProcessing() {}
    PCLProcessing::~PCLProcessing() {}
    bool PCLProcessing::initPCL() {
        if(isInit_)
            return false;
        viewer_ = pcl::visualization::PCLVisualizer("PCL Viewer");
        viewer_.setBackgroundColor(0, 0, 0);
        
        isInit_ = true;
        return true;
    }
    bool PCLProcessing::initPCL(const std::string &cloudPath) {
        if(isInit_)
            return false;
        viewer_ = pcl::visualization::PCLVisualizer("PCL Viewer");
        viewer_.setBackgroundColor(0, 0, 0);
        isInit_ = true;
        return true;
    }
    template <typename PointT>
    bool PCLProcessing::addPointCloud(const pcl::PointCloud<PointT> &cloud) {
        if(!isInit_)
            return false;
        std::string cloudID =  std::to_string(time(nullptr)) + "_" + "cloud";
        viewer_.addPointCloud<PointT>(cloud.makeShared(), cloudID);
        return true;
    }
}