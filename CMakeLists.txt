cmake_minimum_required(VERSION 3.14)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
message(${OpenCV_LIBS})


#DLIB intalado, cambiar el include. (homebrew)
set(FeatureExtraction_Include /usr/local/Cellar/opencv/4.1.1_2/include/opencv4)

include_directories(${FeatureExtraction_Include})



set(OP ${OpenCV_LIBS} "opencv_face")
message(${OP})

add_executable(FeatureExtraction Executable.cpp)
target_include_directories(FeatureExtraction PRIVATE /usr/local/Cellar/opencv/4.1.1_2/lib)

target_link_libraries(FeatureExtraction ${OP})