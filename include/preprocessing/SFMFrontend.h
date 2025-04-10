#ifndef SFMFrontend_H
#define SFMFrontend_H
#include "config.h"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <utility>
#include <vector>

#include "preprocessing/pclprocessing.h"
#include "preprocessing/pre.h"
namespace pre {
enum FeatureDetectorType {
  SIFT,
  SURF,
  ORB,
  BRISK,
  AKAZE,
  FAST,
  GFTT,
  HARRIS,
  MSER,
  STAR,
  SIFT_CUDA
};
struct Image {
  cv::Mat image;                       // 图像
  std::vector<cv::KeyPoint> keypoints; // 关键点(未优化)
  cv::Mat descriptors;                 // 关键点描述子
  std::vector<cv::Point2f> points;     // 关键点(优化)
};
struct ImageNode {
  cv::Mat image;                       // 图像
  int image_id;                        // 图像唯一标识
  std::vector<cv::KeyPoint> keypoints; // 关键点
  std::vector<cv::KeyPoint> points;    // 关键点(优化)
  cv::Mat points_descriptors;          // 关键点描述子
  cv::Mat descriptors;                 // 描述子
  cv::Mat R = cv::Mat();               // 相机旋转矩阵
  cv::Mat t = cv::Mat();               // 相机平移向量
  bool isRegistered;                   // 是否已注册
  std::vector<cv::Point3f> points3d;   // 图像三角化的3D坐标
};
struct Track {
  int track_id;                                  // 轨迹唯一ID
  cv::Point3f point3d;                           // 3D坐标
  std::vector<std::pair<int, int>> observations; // (image_id, keypoint_idx)
};
class SFMFrontend {
public:
  SFMFrontend();
  SFMFrontend(const std::string &cameraParamsPath);
  SFMFrontend(FeatureDetectorType detector_type);
  bool haveImage(const std::string &imagePath, cv::Mat &image);

  void createSIFT(int nfeatures = 0, int nOctaveLayers = 3,
                  double contrastThreshold = 0.040000000000000001,
                  double edgeThreshold = 10, double sigma = 1.6000000000000001,
                  bool enable_precise_upscale =
                      false); // 创建SIFT对象，默认参数与OpenCV一致
  void detectFeatures(const cv::Mat &image,
                      std::vector<cv::KeyPoint> &keypoints,
                      cv::Mat &descriptors,
                      const FeatureDetectorType &detectorType =
                          FeatureDetectorType::SIFT); // 检测特征点和描述符
  std::vector<cv::DMatch> matchFeatures(const cv::Mat &descriptors1,
                                        const cv::Mat &descriptors2,
                                        float ratioThresh = 0.5); // 特征匹配
  cv::Mat drawFeatureMatches(
      const cv::Mat &img1, const std::vector<cv::KeyPoint> &keypoints1,
      const cv::Mat &img2, const std::vector<cv::KeyPoint> &keypoints2,
      const std::vector<cv::DMatch> &matches); // 绘制特征匹配
  cv::Mat
  Test_DrawFeatureMatches(const cv::Mat &img1, const cv::Mat &img2,
                          const FeatureDetectorType &detectorType =
                              FeatureDetectorType::SIFT); // 测试绘制特征匹配

  void GetGoodMatches(const std::vector<cv::DMatch> &matches,
                      const std::vector<cv::KeyPoint> &keypoints1,
                      const std::vector<cv::KeyPoint> &keypoints2,
                      std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2); // 获取好的匹配点
  void
  GetGoodMatches(const cv::Mat &img1, const cv::Mat &img2,
                 std::vector<cv::Point2f> &points1,
                 std::vector<cv::Point2f> &points2,
                 const FeatureDetectorType &detectorType =
                     FeatureDetectorType::SIFT); // 获取好的匹配点(快捷使用)

  cv::Mat ComputeFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                                   const std::vector<cv::Point2f> &points2,
                                   float threshold = 1.0); // 计算基础矩阵
  void TestFundamentalMatrix(const std::vector<cv::Point2f> &points1,
                             const std::vector<cv::Point2f> &points2,
                             const cv::Mat &fundamentalMatrix,
                             const cv::Mat &img1, const cv::Mat &img2);
  cv::Mat
  ComputeEssentialMatrix(const cv::Mat &K1, const cv::Mat &K2,
                         const cv::Mat &fundamentalMatrix); // 计算本质矩阵
  void TestEssentialMatrix(const std::vector<cv::Point2f> &points1,
                           const std::vector<cv::Point2f> &points2,
                           const cv::Mat &essentialMatrix, const cv::Mat &K1,
                           const cv::Mat &K2, const cv::Mat &img1,
                           const cv::Mat &img2);
  void ComputePoseFromEssentialMatrix(const cv::Mat &E,
                                      const std::vector<cv::Point2f> &points1,
                                      const std::vector<cv::Point2f> &points2,
                                      const cv::Mat &K, cv::Mat &R,
                                      cv::Mat &t); // 从本质矩阵计算姿态
  std::vector<cv::Point3f>
  robustTriangulate(const std::vector<cv::Point2f> &points1,
                    const std::vector<cv::Point2f> &points2, const cv::Mat &K,
                    const cv::Mat &R1, const cv::Mat &t1, const cv::Mat &R2,
                    const cv::Mat &t2,
                    float reprojectionThreshold = 5.0); // 稳健三角化
  bool find_transform(cv::Mat &K, std::vector<cv::KeyPoint> &p1,
                      std::vector<cv::KeyPoint> &p2, cv::Mat &R, cv::Mat &T,
                      cv::Mat &mask);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertToPointCloud(
      const std::vector<cv::Point3f> &points3D,
      const std::vector<cv::Point2f> &imagePoints = std::vector<cv::Point2f>(),
      const cv::Mat &image = cv::Mat());
  std::vector<cv::Point3f>
  scaleToVisibleRange(std::vector<cv::Point3f> &points3D);
  std::vector<cv::Point3f> homogeneous2euclidean(const cv::Mat &points4D);
  std::vector<cv::Point3f> twoViewEuclideanReconstruction(
      cv::Mat &img1, cv::Mat &img2,
      FeatureDetectorType detector_type = FeatureDetectorType::SIFT,
      bool isProcessed = false, int best_cam = 0,
      int next_cam = 1); // 双视图欧式结构恢复和构建稀疏点云
  std::vector<cv::Point3f> twoViewEuclideanReconstruction(
      cv::Mat &img1, cv::Mat &img2, const cv::Mat &InputR,
      const cv::Mat &Inputt, cv::Mat &OutputR, cv::Mat &Outputt,
      FeatureDetectorType detector_type = FeatureDetectorType::SIFT,
      bool isProcessed = false, int best_cam = 0, int next_cam = 1,
      bool addCamera1 = true,
      bool addCamera2 = true); // 双视图欧式结构恢复和构建稀疏点云
  void show();
  void processShow();
  void processImageNodes(std::vector<ImageNode> &all_nodes,
                         float ratio_threshold = 0.7f, int min_match_count = 2);
  void processImageGraph(std::map<int, ImageNode> &image_graph,
                         float ratio_threshold = 0.7f, int min_match_count = 2);
  std::map<int, ImageNode> getImageGraph(); // 获取图像图
  void processImageGraph(float ratio_threshold = 0.7f, int min_match_count = 2);
  void populateImageGraph(std::map<int, ImageNode> &imageGraph,
                          const std::string &filePathBegin,
                          int startImageId = 0);
  void populateImageGraph(const std::string &filePathBegin,
                          int startImageId = 0);
  void populateEdges(int min_matches_threshold = 30); // 建立图像图中的边
  void printGraphAsMatrix();                          // 打印图像图
  void deleteEdges(int i, int j); // 删除图像图中的边
  bool getEdges(int &i, int &j,
                const bool &isDeleteEdge = true); // 获取图像图中的边
  void incrementalSFM();                          // 增量式SFM
  void incrementalSFM2();                         // 增量式SFM
  void getEdgesWithMaxPoints(int &i, int &j); // 获取图像图中点数最多的边
  void getEdgesWithBestPoints(int &i,
                              int &j); // 获取图像图中能作为做好初始点的边
  void showEdgesMatchs(int i, int j); // 显示图像图中的匹配点
  void showAllEdgesMatchs();          // 显示每一条边的匹配点
  void registerImage(int ID);         // 注册图像
  void shunxuSFM(const std::string &filePathBegin); // 顺序式SFM
  std::vector<cv::Point3f>
  robustTriangulateWithFilter(const std::vector<cv::Point2f> &points1,
                              const std::vector<cv::Point2f> &points2,
                              const cv::Mat &K, const cv::Mat &R1,
                              const cv::Mat &t1, const cv::Mat &R2,
                              const cv::Mat &t2, float maxDepth);
  ~SFMFrontend();

private:
  pre::CameraPreprocessor preprocessor_; // 相机预处理
  pre::PCLProcessing pclProcessor_;      // 点云处理
  cv::Mat K;                             // 相机内参
  cv::Mat D;                             // 相机畸变参数
  cv::Ptr<cv::SIFT> sift_;
  cv::Ptr<cv::ORB> orb_;
  std::map<int, ImageNode> image_graph_;   // 图像图（key为image_id）
  std::vector<std::pair<int, int>> edges_; // 图像间的边（连接关系）
  int next_track_id_ = 0;                  // 轨迹ID自增计数器
  std::vector<cv::Point3f> points3D_;      // 三维点集
};
} // namespace pre

#endif // SFMFrontend_H