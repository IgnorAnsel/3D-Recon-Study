#include "preprocessing/console.h"

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
const std::string Console::TEST = MAGENTA + "[TEST] " + RESET;

std::ostream &Console::reset(std::ostream &os) { return os << RESET; }