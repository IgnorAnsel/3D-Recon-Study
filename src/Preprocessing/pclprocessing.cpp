#include "preprocessing/pclprocessing.h"
#include "preprocessing/console.h"
#include <string>

// pc => pclprocessing
namespace pre {
PCLProcessing::PCLProcessing() {}
PCLProcessing::~PCLProcessing() {}
bool PCLProcessing::initPCL() {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(
      new pcl::visualization::PCLVisualizer("PCL Viewer"));
  viewer_->setBackgroundColor(0, 0, 0);

  isInit_ = true;
  return true;
}
bool PCLProcessing::initPCLWithNoCloud() {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(
      new pcl::visualization::PCLVisualizer("PCL Viewer"));
  viewer_->setBackgroundColor(0, 0, 0);
  isInit_ = true;
  globalCloud_ = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  return true;
}
bool PCLProcessing::initPCL(const std::string &cloudPath) {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(
      new pcl::visualization::PCLVisualizer("PCL Viewer"));
  viewer_->setBackgroundColor(0, 0, 0);
  isInit_ = true;
  return true;
}
template <typename PointT>
bool PCLProcessing::initPCL(const pcl::PointCloud<PointT> &cloud,
                            const std::string viewerName) {
  if (isInit_)
    return false;
  viewer_ = pcl::visualization::PCLVisualizer::Ptr(
      new pcl::visualization::PCLVisualizer(viewerName));
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
bool PCLProcessing::addCamera(const cv::Mat &R, const cv::Mat &t, int frameId,
                              const cv::Mat &image) {
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
  camera.img = image.clone();

  cameras_.push_back(camera);

  // 可视化相机位置(在点云中添加相机模型)
  visualizeCameraInPointCloud(camera);

  return true;
}
void PCLProcessing::visualizeCameraInPointCloud(const CameraInfo &camera) {
  if (!globalCloud_ || !viewer_) {
    std::cerr << "Global cloud or viewer is not initialized!" << std::endl;
    return;
  }

  // 计算相机中心（世界坐标系）
  cv::Mat C = -camera.R.t() * camera.t;
  double px = C.at<double>(0, 0);
  double py = C.at<double>(1, 0);
  double pz = C.at<double>(2, 0);

  // 添加相机位置点（红色）
  pcl::PointXYZRGB camPoint;
  camPoint.x = px;
  camPoint.y = py;
  camPoint.z = pz;
  camPoint.r = 255;
  camPoint.g = 0;
  camPoint.b = 0;
  globalCloud_->points.push_back(camPoint);

  // --- 修正1：调整箭头方向 ---
  // 相机的观察方向是相机坐标系的 +Z 轴方向
  cv::Mat zAxis = camera.R.col(2); // R 是世界到相机坐标系的旋转矩阵，col(2)
                                   // 是世界坐标系中的相机 Z 轴方向
  Eigen::Vector3f direction(zAxis.at<double>(0), zAxis.at<double>(1),
                            zAxis.at<double>(2));
  direction.normalize();

  // 箭头应从相机位置指向方向
  pcl::PointXYZ arrowStart(px, py, pz);
  pcl::PointXYZ arrowEnd(px + direction[0], py + direction[1],
                         pz + direction[2]);

  // 修正箭头参数顺序：起点 -> 终点
  std::string arrowID = "camera_dir_" + std::to_string(camera.frameId);
  viewer_->addArrow(arrowEnd, arrowStart, 0, 0, 255, false,
                    arrowID); // 蓝色箭头

  // --- 修正2：正确转换视锥体角点 ---
  double far_dist = 2.0;             // 远平面距离
  double fovX = 60.0 * M_PI / 180.0; // 水平视场角
  double fovY = 45.0 * M_PI / 180.0; // 垂直视场角

  // 计算远平面半宽半高
  double halfWidth = far_dist * tan(fovX / 2);
  double halfHeight = far_dist * tan(fovY / 2);

  // 相机坐标系中的远平面角点（Z 轴正方向）
  std::vector<cv::Mat> cornersCam(4);
  cornersCam[0] =
      (cv::Mat_<double>(3, 1) << halfWidth, halfHeight, far_dist); // 右上
  cornersCam[1] =
      (cv::Mat_<double>(3, 1) << -halfWidth, halfHeight, far_dist); // 左上
  cornersCam[2] =
      (cv::Mat_<double>(3, 1) << -halfWidth, -halfHeight, far_dist); // 左下
  cornersCam[3] =
      (cv::Mat_<double>(3, 1) << halfWidth, -halfHeight, far_dist); // 右下

  // 将角点转换到世界坐标系
  std::vector<pcl::PointXYZ> cornersWorld(4);
  for (int i = 0; i < 4; ++i) {
    cv::Mat cornerWorldMat = camera.R.t() * cornersCam[i] + C; // 正确转换公式
    cornersWorld[i].x = cornerWorldMat.at<double>(0);
    cornersWorld[i].y = cornerWorldMat.at<double>(1);
    cornersWorld[i].z = cornerWorldMat.at<double>(2);
  }

  // 绘制视锥体连线（黄色）
  pcl::PointXYZ camPos(px, py, pz);
  for (int i = 0; i < 4; ++i) {
    std::string lineId =
        "cone_line_" + std::to_string(i) + "_" + std::to_string(camera.frameId);
    viewer_->addLine(camPos, cornersWorld[i], 255, 255, 0,
                     lineId); // 相机到远平面角点
  }

  // 绘制远平面四边形
  viewer_->addLine(cornersWorld[0], cornersWorld[1], 255, 255, 0,
                   "far_line_0_1" + std::to_string(camera.frameId));
  viewer_->addLine(cornersWorld[1], cornersWorld[2], 255, 255, 0,
                   "far_line_1_2" + std::to_string(camera.frameId));
  viewer_->addLine(cornersWorld[2], cornersWorld[3], 255, 255, 0,
                   "far_line_2_3" + std::to_string(camera.frameId));
  viewer_->addLine(cornersWorld[3], cornersWorld[0], 255, 255, 0,
                   "far_line_3_0" + std::to_string(camera.frameId));
}
void PCLProcessing::setGlobalCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
  globalCloud_ = cloud;
}
pcl::visualization::PCLVisualizer::Ptr PCLProcessing::getViewer() {
  return viewer_;
}
template bool PCLProcessing::addPointCloud<pcl::PointXYZRGB>(
    const pcl::PointCloud<pcl::PointXYZRGB> &cloud);
template bool PCLProcessing::addPointCloud<pcl::PointXYZ>(
    const pcl::PointCloud<pcl::PointXYZ> &cloud);
} // namespace pre