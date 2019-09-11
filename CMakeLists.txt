cmake_minimum_required(VERSION 3.14)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 11)

#set( SOURCE )

#set( HEADERS )

find_package(OpenCV REQUIRED)

set(FeatureExtraction_include /usr/local/Cellar/opencv/4.1.1_2/include/opencv4/) #Yann

include_directories(${FeatureExtraction_include})

add_executable(FeatureExtraction main.cpp)
link_directories(/usr/local/Cellar/opencv/4.1.1_2/lib)

target_link_libraries(FeatureExtraction ${OpenCV_LIBS})

