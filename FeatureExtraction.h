//
// Created by Yann Le Lorier on 2019-10-19.
//

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp" //comparehist is here
#include "NeuralNetworkTemplate.h"
#include <dlib/opencv.h> //to convert objects from opencv to dlib and viceversa
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <sstream>


class FeatureExtraction {
private:
    anet_type net;

public:
    FeatureExtraction(std::string url) {
        dlib::deserialize(url) >> net;
    }
    cv::Mat ComputeDescriptorForFace(cv::Mat &face); //comms w module 2
    double compareFeaturesCV(cv::Mat h1, cv::Mat h2, int method);
    double compareFeaturesDlib(dlib::matrix<float,128,1> &descriptor1, dlib::matrix<float,128,1> &descriptor2);
    cv::Mat convertToMat(dlib::matrix<float,128,1> descriptor); //comms w module 4
    dlib::matrix<float,128,1> toDlib(cv::Mat descriptor);
};



#ifndef FEATUREEXTRACTION_FEATUREEXTRACTION_H
#define FEATUREEXTRACTION_FEATUREEXTRACTION_H

#endif //FEATUREEXTRACTION_FEATUREEXTRACTION_H
