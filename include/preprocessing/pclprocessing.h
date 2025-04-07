#ifndef PCLPROCESSING_H
#define PCLPROCESSING_H

#include "config.h"
#include "preprocessing/console.h"
#include <opencv2/opencv.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
struct CameraInfo {
  cv::Mat R;             // 旋转矩阵
  cv::Mat t;             // 平移向量
  cv::Mat K;             // 相机内参(可选)
  int frameId;           // 对应的帧ID
  cv::Mat img;           // 对应的图像(可选)
  std::string imagePath; // 对应的图像路径(可选)
};

namespace pre {
class PCLProcessing {
public:
  PCLProcessing();
  bool initPCL();                             // 初始化PCL(默认)
  bool initPCL(const std::string &cloudPath); // 初始化PCL(从文件中读取点云)
  bool initPCLWithNoCloud();
  template <typename PointT>
  bool initPCL(const pcl::PointCloud<PointT> &cloud,
               const std::string viewerName =
                   "PCL Viewer"); // 初始化PCL(从点云中读取,并能修改视图名字)
  bool loadPointCloud(const std::string &cloudPath); // 从文件中读取点云
  bool savePointCloud(const std::string &cloudPath); // 保存点云到文件
  template <typename PointT>
  bool addPointCloud(const pcl::PointCloud<PointT> &cloud); // 添加点云
  template <typename PointT>
  bool addPointCloud(const pcl::PointCloud<PointT> &cloud,
                     const std::string &frameName); // 添加点云,并设置帧名字
  bool addCamera(
      const cv::Mat &R, const cv::Mat &t, int frameId,
      const std::string &imagePath = ""); // 添加相机(主要用于SFM运动结构恢复)
  bool addCamera(const cv::Mat &R, const cv::Mat &t, int frameId,
                 const cv::Mat &image);
  void
  visualizeCameraInPointCloud(const CameraInfo &camera); // 在点云中可视化相机
  void setGlobalCloud(
      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud); // 设置全局点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr getGlobalCloud(); // 获取全局点云
  pcl::visualization::PCLVisualizer::Ptr getViewer(); // 获取PCL可视化器
  ~PCLProcessing();

private:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalCloud_;
  std::vector<CameraInfo> cameras_; // 存储所有相机信息
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  bool isInit_ = false;
};
template <typename PointT>
bool PCLProcessing::addPointCloud(const pcl::PointCloud<PointT> &cloud,
                                  const std::string &frameName) {
  if (!isInit_)
    return false;
  std::string cloudID = frameName;

  // 创建过滤后的点云
  typename pcl::PointCloud<PointT>::Ptr filtered_cloud(
      new pcl::PointCloud<PointT>);

  // 统计离群点移除滤波器
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud.makeShared());
  sor.setMeanK(50);            // 每个点分析的邻近点数
  sor.setStddevMulThresh(1.0); // 标准偏差乘数阈值
  sor.filter(*filtered_cloud);

  // 添加过滤后的点云到视图
  viewer_->addPointCloud<PointT>(filtered_cloud, cloudID);
  // viewer_->addPointCloud<PointT>(cloud.makeShared(), cloudID);
  std::cout << Console::INFO
            << "[addPointCloud] Adding point cloud: " << cloudID
            << " before size:" << cloud.size()
            << " filtered size:" << filtered_cloud->size() << "\n"
            << std::endl;
  return true;
}
} // namespace pre

#endif // PCLPROCESSING_H