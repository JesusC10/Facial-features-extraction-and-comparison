cmake_minimum_required(VERSION 3.14)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 11)

#set( SOURCE )

#set( HEADERS )

find_package(OpenCV REQUIRED)
message(${OpenCV_LIBS})



set(FeatureExtraction_include /usr/local/Cellar/opencv/4.1.1_2/include/opencv4/) #Yann

include_directories(${FeatureExtraction_include})



set(OP ${OpenCV_LIBS} "opencv_face")
message(${OP})

add_executable(FeatureExtraction main.cpp)
target_include_directories(FeatureExtraction PRIVATE /usr/local/Cellar/opencv/4.1.1_2/lib/)

target_link_libraries(FeatureExtraction ${OP})

