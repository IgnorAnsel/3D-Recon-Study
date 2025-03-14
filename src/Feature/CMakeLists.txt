project(Feature)

add_executable(Harris Harris.cpp)
add_executable(Harris_opencvAPI Harris_opencvAPI.cpp)
target_link_libraries(Harris PRIVATE
        PRIVATE
        ${OpenCV_LIBS}
        )
target_link_libraries(Harris_opencvAPI PRIVATE
        PRIVATE
        ${OpenCV_LIBS}
        )
add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PCL_CXX_FLAGS}")