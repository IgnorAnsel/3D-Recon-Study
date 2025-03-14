project(3D-RECON-STUDY)
cmake_minimum_required(VERSION 3.10)

# 建议添加C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(PCL 1.14 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)
# 资源目录设置正确
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/dataset" CACHE PATH "dataset directory")
add_definitions(-DRESOURCE_DIR="${RESOURCE_DIR}")

# 安装命令正确
install(DIRECTORY ${RESOURCE_DIR} DESTINATION .)

# 子目录添加正确
add_subdirectory(src/Feature)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)