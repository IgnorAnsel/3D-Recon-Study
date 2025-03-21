#ifndef CONSOLE_H
#define CONSOLE_H

#include "config.h"
#include <iostream>
#include <string>
/**
 * @brief 控制台输出工具类
 *
 * 提供预定义的彩色控制台输出格式
 * 使用示例:
 *     std::cerr << Console::WARNING << " 这是一条警告信息" << std::endl;
 *     std::cout << Console::INFO << " 这是一条信息" << std::endl;
 */
class Console {
private:
  // ANSI颜色代码
  static const std::string RED;
  static const std::string GREEN;
  static const std::string YELLOW;
  static const std::string BLUE;
  static const std::string MAGENTA;
  static const std::string CYAN;
  static const std::string RESET;

public:
  // 预定义的消息类型
  static const std::string ERROR;   // 红色 [ERROR]
  static const std::string WARNING; // 黄色 [Warning]
  static const std::string INFO;    // 青色 [Info]
  static const std::string SUCCESS; // 绿色 [Success]
  static const std::string DEBUG;   // 蓝色 [Debug]
  static const std::string TEST;    // 紫色 [TEST]

  // 清除颜色
  std::ostream &reset(std::ostream &os);
};

#endif // CONSOLE_H