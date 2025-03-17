#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

Mat SRC; // 存储灰度输入图像
Mat DST; // 存储绘制角点后的结果图
void my_cornerHarris(Mat src, Mat &dst, int ksize, double k) {
  Mat Ix, Iy;
  Mat M(src.size(), CV_32FC3); // M = [Ix^2, Ix*Iy; Ix*Iy, Iy^2] * w(x, y)
  Mat R(src.size(), CV_32FC1); // R = det(M) - k * trace(M)^2
  Mat Ixx, Iyy, Ixy;

  Sobel(src, Ix, CV_32FC1, 1, 0, ksize);
  Sobel(src, Iy, CV_32FC1, 0, 1, ksize);

  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      M.at<Vec3f>(i, j) = Vec3f(Ix.at<float>(i, j) * Ix.at<float>(i, j),
                                Ix.at<float>(i, j) * Iy.at<float>(i, j),
                                Iy.at<float>(i, j) * Iy.at<float>(i, j));
    }
  }
  //高斯滤波对M矩阵进行加权求和
  GaussianBlur(M, M, Size(ksize, ksize), 2, 2);

  //求得Harris响应值
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      float A = M.at<Vec3f>(i, j)[0]; // Ix*Ix
      float C = M.at<Vec3f>(i, j)[1]; // Ix*Iy
      float B = M.at<Vec3f>(i, j)[2]; // Iy*Iy

      //响应值计算公式
      R.at<float>(i, j) = (A * B - C * C) - (k * (A + B) * (A + B));
    }
  }
  dst = R.clone();
}
void NMS(Mat &src) //手写非极大值抑制
{
  int i, j, k, l, cnt = 0;
  //遍历图像
  for (i = 2; i < src.rows; i++)
    for (j = 2; j < src.cols; j++)

      //采用5×5窗口，小于中心像素置零
      for (k = i - 2; k <= i + 2; k++)
        for (l = j - 2; l <= j + 2; l++)
          if (src.at<float>(k, l) <= src.at<float>(i, j) && k != i && l != j &&
              src.at<float>(k, l) > 0)
            src.at<float>(i, j) = 0;
}

void draw_Point(Mat src, Mat SRC, int T) //在原图标注角点
{
  cvtColor(SRC, DST, COLOR_GRAY2BGR);
  int Num_of_corner = 0;
  for (int row = 0; row < src.rows; row++) {
    uchar *Currunt_row = src.ptr(row); //行指针
    for (int col = 0; col < src.cols; col++) {
      int R_value = Currunt_row[col]; //提取处理后的角点响应
      if (R_value >= T) {
        //颜色渐变
        Num_of_corner++; //计算角点数
        int R, G, B;

        if (R_value <= 63.75) {
          B = 255;
          G = int(255 * R_value / 63.75);
          R = 0;
          circle(DST, Point(col, row), 3, Scalar(B, G, R), 1,
                 LINE_MAX); //圈出大于阈值的角点
        } else if (R_value <= 127.5) {
          B = 255 - int(255 * (R_value - 63.75) / 63.75);
          G = 255;
          R = 0;
          circle(DST, Point(col, row), 3, Scalar(B, G, R), 1,
                 LINE_MAX); //圈出大于阈值的角点
        } else if (R_value <= 191.25) {
          B = 0;
          G = 255;
          R = int(255 * (R_value - 127.5) / 63.75);
          circle(DST, Point(col, row), 3, Scalar(B, G, R), 1,
                 LINE_MAX); //圈出大于阈值的角点
        } else if (R_value <= 255) {
          B = 0;
          G = 255 - saturate_cast<uchar>(255 * (R_value - 191.25) / 63.75);
          R = 255;
          circle(DST, Point(col, row), 3, Scalar(B, G, R), 1,
                 LINE_MAX); //圈出大于阈值的角点
        }
      }
    }
    Currunt_row++;
  }
  cout << "total nums of corner is:" << Num_of_corner << endl;
}
void Gradient_change(Mat src) //通道阈值渐变（channel：B0,G1,R2)
{
  Mat dst = Mat::zeros(src.size(), CV_8UC3);
  for (int row = 0; row < src.rows; row++) {
    for (int col = 0; col < src.cols; col++) {
      int BGR_value = src.at<uchar>(row, col); //三通道像素指针(RGB)
      if (BGR_value <= 63.75) {
        dst.at<Vec3b>(row, col)[0] = 255;
        dst.at<Vec3b>(row, col)[1] = int(255 * BGR_value / 63.75);
        dst.at<Vec3b>(row, col)[2] = 0;
      } else if (BGR_value <= 127.5) {
        dst.at<Vec3b>(row, col)[0] =
            255 - int(255 * (BGR_value - 63.75) / 63.75);
        dst.at<Vec3b>(row, col)[1] = 255;
        dst.at<Vec3b>(row, col)[2] = 0;
      } else if (BGR_value <= 191.25) {
        dst.at<Vec3b>(row, col)[0] = 0;
        dst.at<Vec3b>(row, col)[1] = 255;
        dst.at<Vec3b>(row, col)[2] = int(255 * (BGR_value - 127.5) / 63.75);
      } else if (BGR_value <= 255) {
        dst.at<Vec3b>(row, col)[0] = 0;
        dst.at<Vec3b>(row, col)[1] =
            255 - saturate_cast<uchar>(255 * (BGR_value - 191.25) / 63.75);
        dst.at<Vec3b>(row, col)[2] = 255;
      }
    }
  }
  imshow("My_Gradient_change", dst);
}
int Threshold = 30;
int K = 400;
void Harris_detaction(int, void *) // Harris角点检测回调函数
{
  Threshold = getTrackbarPos("T_value", "Harris_detaction");
  K = getTrackbarPos("k_value", "Harris_detaction");
  Mat dst = Mat::zeros(SRC.size(), CV_32FC1);
  int blockSize = 2;     //矩阵M的维数(二维以上的原理不太清楚)
  int ksize = 3;         //窗口大小
  double k = K * 0.0001; //阈值k

  //求出每一个像素点的Harris响应值(使用OpenCV API)
  cornerHarris(SRC, dst, blockSize, ksize, k);
  // my_cornerHarris(SRC, dst,ksize, k);

  NMS(dst); //手写非极大值抑制

  normalize(dst, dst, 0, 255, NORM_MINMAX, CV_32FC1,
            Mat());                //将Harris响应值归一化
  convertScaleAbs(dst, dst, 1, 0); //将Harris响应值转为整型（uchar）
  Gradient_change(dst);            //绘制角点响应分布图(渐变)

  imshow("Harris_callback", dst);  //角点响应分布图
  draw_Point(dst, SRC, Threshold); //在原图标注角点
  imshow("Harris_detaction", DST); // result
}
void Harris_threshold_arrange() //滑动窗口
{
  namedWindow("Harris_detaction", WINDOW_NORMAL);
  // createTrackbar("T_value","Harris_detaction",&Threshold,255,Harris_detaction);
  // //阈值T
  // createTrackbar("k_value","Harris_detaction",&K,700,Harris_detaction);
  // //阈值k
  createTrackbar("T_value", "Harris_detaction", nullptr, 255, Harris_detaction);
  createTrackbar("k_value", "Harris_detaction", nullptr, 700, Harris_detaction);
  Harris_detaction(0, 0); // 确保滑动条已创建后再调用

  waitKey(0);
}

bool Pic_show(Mat src, const char *param) //展示图片
{
  if (src.empty()) {
    cout << "图片打开失败！\n";
    return false;
  } else
    imshow(param, src);

  waitKey(0);
  return true;
}

int main(int argv, char **argc) {
  std::cout << RESOURCE_DIR << std::endl;
  Mat img = imread(string(RESOURCE_DIR) + "/room/room_1.jpg", IMREAD_COLOR);
  if (img.empty()) {
    cout << "Failed to load image!" << endl;
    return -1;
  }
  Pic_show(img, "input");

  // 转换为灰度图并存入全局变量 SRC
  cvtColor(img, SRC, COLOR_BGR2GRAY); // 注意：原图应为彩色，这里需要先转灰度

  Harris_threshold_arrange();
  return 0;
}
