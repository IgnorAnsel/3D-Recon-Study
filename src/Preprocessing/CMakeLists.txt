# 预处理模块
add_library(Preprocessing
    pre.cpp  
    pclprocessing.cpp
    SFMFrontend.cpp
    console.cpp
)

target_include_directories(Preprocessing PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(Preprocessing
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

set_target_properties(Preprocessing PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"  # 确保值被引号包裹
    LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"   # 可选添加动态库路径
)