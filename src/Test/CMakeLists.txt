project(Test)

add_executable(Test_pre Test_pre.cpp)

target_link_libraries(Test_pre PRIVATE
        PRIVATE
        Preprocessing
        ${OpenCV_LIBS}
        )
add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PCL_CXX_FLAGS}")