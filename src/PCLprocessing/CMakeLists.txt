# 预处理模块
add_library(PCLprocessing
    pclprocessing.cpp  
)

target_include_directories(PCLprocessing PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(PCLprocessing
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

set_target_properties(PCLprocessing PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"  # 确保值被引号包裹
    LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}"   # 可选添加动态库路径
)