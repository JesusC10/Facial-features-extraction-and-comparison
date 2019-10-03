cmake_minimum_required(VERSION 3.14)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
message(${OpenCV_LIBS})

include(/Users/yannlelorier/Downloads/dlib-19.18/dlib/cmake)


set(FeatureExtraction_Include /usr/local/Cellar/opencv/4.1.1_2/include/opencv4)
set(Dlib_feat /usr/local/Cellar/dlib/19.17/include/)

include_directories(${FeatureExtraction_Include})
include_directories(${Dlib_feat})



set(OP ${OpenCV_LIBS} "opencv_face")
message(${OP})

add_executable(FeatureExtraction Executable.cpp NeuralNetworkTemplate.h)
target_include_directories(FeatureExtraction PRIVATE /usr/local/Cellar/opencv/4.1.1_2/lib)

target_link_libraries(FeatureExtraction ${OP} dlib::dlib)