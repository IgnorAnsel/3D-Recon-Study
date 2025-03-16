#include "pclprocessing/pclprocessing.h"

// pc => pclprocessing
namespace pc {
PCLProcessing::PCLProcessing() {}
PCLProcessing::~PCLProcessing() {}
bool PCLProcessing::initPCL() {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("PCL Viewer"));
  viewer_->setBackgroundColor(0, 0, 0);

  isInit_ = true;
  return true;
}
bool PCLProcessing::initPCL(const std::string &cloudPath) {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("PCL Viewer"));
  viewer_->setBackgroundColor(0, 0, 0);
  isInit_ = true;
  return true;
}
template <typename PointT>
bool PCLProcessing::initPCL(const pcl::PointCloud<PointT> &cloud,
                            const std::string viewerName) {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer(viewerName));
  viewer_->setBackgroundColor(0, 0, 0);
  isInit_ = true;
  return true;
}
bool PCLProcessing::loadPointCloud(const std::string &cloudPath) {
  // if(!isInit_)
  //     return false;
  // addPointCloud(cloudPath);
  // return true;
  return false;
}
template <typename PointT>
bool PCLProcessing::addPointCloud(const pcl::PointCloud<PointT> &cloud) {
  if (!isInit_)
    return false;
  std::string cloudID = std::to_string(time(nullptr)) + "_" + "cloud";
  viewer_->addPointCloud<PointT>(cloud.makeShared(), cloudID);
  return true;
}
// template <typename PointT>
// bool PCLProcessing::addPointCloud(const pcl::PointCloud<PointT> &cloud,
//                                   const std::string &frameName) {
//   if (!isInit_)
//     return false;
//   std::string cloudID = frameName;
//   viewer_->addPointCloud<PointT>(cloud.makeShared(), cloudID);
//   return true;
// }
bool PCLProcessing::addCamera(const cv::Mat &R, const cv::Mat &t, int frameId,
                              const std::string &imagePath) {
  if (!isInit_)
    return false;

  if (R.empty() || t.empty() || R.type() != CV_64F || t.type() != CV_64F)
    return false;

  if (R.rows != 3 || R.cols != 3 || t.rows != 3 || t.cols != 1)
    return false;

  CameraInfo camera;
  camera.R = R.clone();
  camera.t = t.clone();
  camera.frameId = frameId;
  camera.imagePath = imagePath;

  cameras_.push_back(camera);

  // 可视化相机位置(在点云中添加相机模型)
  visualizeCameraInPointCloud(camera);

  return true;
}
void PCLProcessing::visualizeCameraInPointCloud(const CameraInfo &camera) {
  if (!globalCloud_ || !viewer_)
    return;

  // 计算相机中心
  cv::Mat C = -camera.R.t() * camera.t;
  double px = C.at<double>(0, 0);
  double py = C.at<double>(1, 0);
  double pz = C.at<double>(2, 0);

  // 添加相机位置点
  pcl::PointXYZRGB camPoint;
  camPoint.x = px;
  camPoint.y = py;
  camPoint.z = pz;
  camPoint.r = 255; // 红色表示相机位置
  camPoint.g = 0;
  camPoint.b = 0;
  globalCloud_->points.push_back(camPoint);

  // 创建指示相机朝向的箭头
  std::string arrowID = "camera_dir_" + std::to_string(camera.frameId);
  
  // 计算相机朝向 (z轴方向)
  cv::Mat zAxis = camera.R.col(2);
  Eigen::Vector3f direction(zAxis.at<double>(0, 0), 
                           zAxis.at<double>(1, 0), 
                           zAxis.at<double>(2, 0));
  
  // 相机朝向箭头的终点
  pcl::PointXYZ arrowStart(px, py, pz);
  pcl::PointXYZ arrowEnd(px + direction[0], 
                        py + direction[1], 
                        pz + direction[2]);
  
  // 添加箭头
  viewer_->addArrow(arrowEnd, arrowStart, 0, 0, 255, false, arrowID);
  
  // 添加相机坐标系
  Eigen::Matrix3f rotation;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation(i, j) = camera.R.at<double>(i, j);
    }
  }
  
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
  transform.translation() << px, py, pz;
  transform.rotate(rotation);
  
  std::string coordID = "camera_coord_" + std::to_string(camera.frameId);
  viewer_->addCoordinateSystem(0.5, transform, coordID);
}
void PCLProcessing::setGlobalCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  globalCloud_ = cloud;
}
pcl::visualization::PCLVisualizer::Ptr PCLProcessing::getViewer() {
  return viewer_;
}
} // namespace pc