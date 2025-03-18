#ifndef CONSOLE_H
#define CONSOLE_H

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

  // 清除颜色
  static std::ostream &reset(std::ostream &os) { return os << RESET; }
};

// 颜色代码定义
const std::string Console::RED = "\033[31m";
const std::string Console::GREEN = "\033[32m";
const std::string Console::YELLOW = "\033[33m";
const std::string Console::BLUE = "\033[34m";
const std::string Console::MAGENTA = "\033[35m";
const std::string Console::CYAN = "\033[36m";
const std::string Console::RESET = "\033[0m";

// 预定义消息类型
const std::string Console::ERROR = RED + "[ERROR] " + RESET;
const std::string Console::WARNING = YELLOW + "[Warning] " + RESET;
const std::string Console::INFO = CYAN + "[Info] " + RESET;
const std::string Console::SUCCESS = GREEN + "[Success] " + RESET;
const std::string Console::DEBUG = BLUE + "[Debug] " + RESET;

#endif // CONSOLE_H