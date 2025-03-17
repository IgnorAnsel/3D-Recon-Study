#include "preprocessing/pclprocessing.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
void createCameraPath(pre::PCLProcessing &pclProcessor, int numCameras = 20);
int main() {
  // 创建PCLProcessing实例
  pre::PCLProcessing pclProcessor;

  // 初始化PCL
  if (!pclProcessor.initPCL()) {
    std::cerr << "初始化PCL失败" << std::endl;
    return -1;
  }

  // 创建一个简单的点云
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());

  // 向点云中添加一些点
  for (float x = -5.0; x <= 5.0; x += 0.1) {
    for (float y = -5.0; y <= 5.0; y += 0.1) {
      pcl::PointXYZRGB point;
      point.x = x;
      point.y = y;
      // 创建一个平面，z=sin(x)*cos(y)
      point.z = sin(x) * cos(y);

      // 根据位置设置颜色
      point.r = static_cast<uint8_t>((x + 5) / 10.0 * 255);
      point.g = static_cast<uint8_t>((y + 5) / 10.0 * 255);
      point.b = static_cast<uint8_t>((point.z + 1) / 2.0 * 255);

      cloud->points.push_back(point);
    }
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;

  // 设置全局点云指针
  pclProcessor.setGlobalCloud(cloud);

  // 添加点云到可视化器
  pclProcessor.addPointCloud<pcl::PointXYZRGB>(*cloud, "main_cloud");

  // 创建几个虚拟的相机位置
  for (int i = 0; i < 5; i++) {
    double angle = i * (2 * M_PI / 5);
    double radius = 7.0;

    // 创建相机的外参矩阵 [R|t]
    cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);

    // 1. 设置相机位置（平移向量t）
    double tx = radius * cos(angle);
    double ty = radius * sin(angle);
    double tz = 2.0;

    extrinsic.at<double>(0, 3) = tx;
    extrinsic.at<double>(1, 3) = ty;
    extrinsic.at<double>(2, 3) = tz;

    // 2. 创建从世界坐标系到相机坐标系的旋转
    // 我们希望相机看向原点，所以相机的z轴应指向世界坐标系的原点

    // 相机位置向量
    cv::Mat cameraPos = (cv::Mat_<double>(3, 1) << tx, ty, tz);

    // 相机z轴方向 = 归一化的(相机位置 - 目标点)
    cv::Mat zAxis = cameraPos.clone();
    cv::normalize(zAxis, zAxis);

    // 相机y轴（假设世界up向量是Z轴正方向）
    cv::Mat worldUp = (cv::Mat_<double>(3, 1) << 0, 0, 1);

    // 相机x轴 = y轴 × z轴 (右手坐标系)
    cv::Mat xAxis = worldUp.cross(zAxis);
    cv::normalize(xAxis, xAxis);

    // 相机y轴 = z轴 × x轴
    cv::Mat yAxis = zAxis.cross(xAxis);

    // 构建旋转矩阵R
    cv::Mat R(3, 3, CV_64F);
    xAxis.copyTo(R.col(0));
    yAxis.copyTo(R.col(1));
    zAxis.copyTo(R.col(2));

    // 将R复制到外参矩阵的左上3x3部分
    R.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));

    // 3. 从外参矩阵提取R和t
    cv::Mat rotation = extrinsic(cv::Rect(0, 0, 3, 3));
    cv::Mat translation = extrinsic(cv::Rect(3, 0, 1, 3));

    // 添加相机
    std::string imagePath = "camera_" + std::to_string(i) + ".jpg";
    pclProcessor.addCamera(rotation, translation, i, imagePath);

    std::cout << "添加相机 " << i << " 在位置: ("
              << translation.at<double>(0, 0) << ", "
              << translation.at<double>(1, 0) << ", "
              << translation.at<double>(2, 0) << ")" << std::endl;
  }
  createCameraPath(pclProcessor, 20);
  // 设置点云可视化参数
  // pclProcessor.getViewer().setPointCloudRenderingProperties(
  //     pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "main_cloud");

  // 主循环，保持窗口打开直到用户手动关闭
  while (!pclProcessor.getViewer()->wasStopped()) {
    pclProcessor.getViewer()->spinOnce(100);
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return 0;
}
// 创建沿路径移动的相机序列
void createCameraPath(pre::PCLProcessing &pclProcessor, int numCameras) {
  // 初始外参矩阵 [R|t]
  cv::Mat extrinsic = cv::Mat::eye(4, 4, CV_64F);

  // 初始相机位置 - 起点
  double radius = 7.0;
  double height = 0.0;
  double angle = 0.0;

  // 设置初始位置
  extrinsic.at<double>(0, 3) = radius * cos(angle);
  extrinsic.at<double>(1, 3) = radius * sin(angle);
  extrinsic.at<double>(2, 3) = height;

  // 每步变化量
  double angleStep = 2 * M_PI / 40; // 每步旋转角度
  double heightStep = 1.6;          // 每步升高高度
  double radiusStep = -0.1; // 每步半径变化（负值表示向内螺旋）

  for (int i = 0; i < numCameras; i++) {
    // 当前相机位置
    double tx = extrinsic.at<double>(0, 3);
    double ty = extrinsic.at<double>(1, 3);
    double tz = extrinsic.at<double>(2, 3);

    // 相机位置向量
    cv::Mat cameraPos = (cv::Mat_<double>(3, 1) << tx, ty, tz);

    // 相机Z轴方向 = 从相机指向原点（待观察物体中心）
    cv::Mat targetPos = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat zAxis = targetPos - cameraPos;
    cv::normalize(zAxis, zAxis);

    // 相机Y轴（假设世界up向量是Z轴正方向）
    cv::Mat worldUp = (cv::Mat_<double>(3, 1) << 0, 0, 1);

    // 相机X轴 = Y轴 × Z轴
    cv::Mat xAxis = worldUp.cross(zAxis);
    cv::normalize(xAxis, xAxis);

    // 相机Y轴 = Z轴 × X轴
    cv::Mat yAxis = zAxis.cross(xAxis);

    // 构建旋转矩阵R
    cv::Mat R(3, 3, CV_64F);
    xAxis.copyTo(R.col(0));
    yAxis.copyTo(R.col(1));
    zAxis.copyTo(R.col(2));

    // 更新外参矩阵
    R.copyTo(extrinsic(cv::Rect(0, 0, 3, 3)));

    // 提取当前R和t
    cv::Mat rotation = extrinsic(cv::Rect(0, 0, 3, 3));
    cv::Mat translation = extrinsic(cv::Rect(3, 0, 1, 3));

    // 添加相机
    std::string imagePath = "camera_path_" + std::to_string(i) + ".jpg";
    pclProcessor.addCamera(rotation, translation, i + 100, imagePath);

    std::cout << "添加路径相机 " << i << " 在位置: ("
              << translation.at<double>(0, 0) << ", "
              << translation.at<double>(1, 0) << ", "
              << translation.at<double>(2, 0) << ")" << std::endl;

    // 计算下一个相机位置（更新外参矩阵）
    angle += angleStep;
    radius += radiusStep;
    height += heightStep;

    // 计算新的位置
    double new_tx = radius * cos(angle);
    double new_ty = radius * sin(angle);
    double new_tz = height;

    // 更新外参矩阵中的平移部分
    extrinsic.at<double>(0, 3) = new_tx;
    extrinsic.at<double>(1, 3) = new_ty;
    extrinsic.at<double>(2, 3) = new_tz;
  }
}