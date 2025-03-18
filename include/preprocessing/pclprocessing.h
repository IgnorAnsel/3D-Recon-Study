#ifndef PCLPROCESSING_H
#define PCLPROCESSING_H

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
struct CameraInfo {
  cv::Mat R;             // 旋转矩阵
  cv::Mat t;             // 平移向量
  cv::Mat K;             // 相机内参(可选)
  int frameId;           // 对应的帧ID
  std::string imagePath; // 对应的图像路径(可选)
};

namespace pre {
class PCLProcessing {
public:
  PCLProcessing();
  bool initPCL();                             // 初始化PCL(默认)
  bool initPCL(const std::string &cloudPath); // 初始化PCL(从文件中读取点云)
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
  viewer_->addPointCloud<PointT>(cloud.makeShared(), cloudID);
  return true;
}
} // namespace pre

#endif // PCLPROCESSING_H