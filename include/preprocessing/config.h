#ifndef CONFIG_H
#define CONFIG_H

// 取消注释下面这行即可启用DEBUG模式
// #define DEBUG_MODE

// 添加到config.h或相关头文件
#ifdef DEBUG_MODE
#define DEBUG(x) std::cout << Console::DEBUG << x << std::endl

// 美化矩阵输出
#define DEBUG_MATRIX(mat)                                                      \
  do {                                                                         \
    std::cout << Console::DEBUG << "  ┌";                                      \
    for (int i = 0; i < mat.cols; ++i)                                         \
      std::cout << "──────────";                                               \
    std::cout << "┐" << std::endl;                                             \
    for (int i = 0; i < mat.rows; ++i) {                                       \
      std::cout << Console::DEBUG << "  │";                                    \
      for (int j = 0; j < mat.cols; ++j) {                                     \
        std::cout << std::fixed << std::setprecision(6) << std::setw(10)       \
                  << mat.at<double>(i, j) << " ";                              \
      }                                                                        \
      std::cout << "│" << std::endl;                                           \
    }                                                                          \
    std::cout << Console::DEBUG << "  └";                                      \
    for (int i = 0; i < mat.cols; ++i)                                         \
      std::cout << "──────────";                                               \
    std::cout << "┘" << std::endl;                                             \
  } while (0)

// 美化向量输出
#define DEBUG_VECTOR(vec)                                                      \
  do {                                                                         \
    std::cout << Console::DEBUG << "  [";                                      \
    for (int i = 0; i < vec.cols * vec.rows; ++i) {                            \
      if (vec.rows > 1)                                                        \
        std::cout << std::fixed << std::setprecision(6) << std::setw(10)       \
                  << vec.at<double>(i, 0);                                     \
      else                                                                     \
        std::cout << std::fixed << std::setprecision(6) << std::setw(10)       \
                  << vec.at<double>(0, i);                                     \
      if (i < vec.cols * vec.rows - 1)                                         \
        std::cout << ", ";                                                     \
    }                                                                          \
    std::cout << "]" << std::endl;                                             \
  } while (0)
#else
#define DEBUG(x)                                                               \
  do {                                                                         \
  } while (0)
#define DEBUG_MATRIX(mat)                                                      \
  do {                                                                         \
  } while (0)
#define DEBUG_VECTOR(vec)                                                      \
  do {                                                                         \
  } while (0)
#endif

#endif // CONFIG_H