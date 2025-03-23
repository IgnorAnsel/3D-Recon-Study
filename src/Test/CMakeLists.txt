project(Test)

add_executable(Test_pre Test_pre.cpp)
add_executable(Test_pclprocessing Test_pclprocessing.cpp)
add_executable(Test_2OSSFM Test_2OSSFM.cpp)
add_executable(Test_SFMFrontend Test_SFMFrontend.cpp)
add_executable(Test_SFMFrontend2 Test_SFMFrontend2.cpp)
target_link_libraries(Test_pre PRIVATE
        PRIVATE
        Preprocessing
        ${OpenCV_LIBS}
        )
target_link_libraries(Test_pclprocessing PRIVATE
        Preprocessing
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
)
target_link_libraries(Test_2OSSFM PRIVATE
        Preprocessing
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)
target_link_libraries(Test_SFMFrontend PRIVATE
        Preprocessing
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)
target_link_libraries(Test_SFMFrontend2 PRIVATE
        Preprocessing
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
)
add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PCL_CXX_FLAGS}")