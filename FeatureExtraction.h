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

using namespace std;



class FeatureExtraction {
private:
    anet_type net;

public:
    FeatureExtraction(string url) {
        dlib::deserialize(url) >> net;
    }
    cv::Mat ComputeDescriptorForFaceCV(cv::Mat &face);
    dlib::matrix<float,0,1> ComputeDescriptorForFaceDlib(cv::Mat &face); //comms w module 2
    double compareFeaturesCV(cv::Mat h1, cv::Mat h2, int method);
    double compareFeaturesDlib(dlib::matrix<float,0,1> &descriptor1, dlib::matrix<float,0,1> &descriptor2);
    cv::Mat convertToMat(dlib::matrix<float,0,1> descriptor); //comms w module 4
};


cv::Mat FeatureExtraction::ComputeDescriptorForFaceCV(cv::Mat &face){
    dlib::cv_image<dlib::bgr_pixel> cvImage(face);
    dlib::matrix<dlib::rgb_pixel> dlibImage;
    dlib::assign_image(dlibImage, cvImage);
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, dlibImage);
    dlib::matrix<float,0,1> faceDescriptor = mean(mat(net(faces))); //computing face descriptor
//    std::vector<dlib::matrix<float,0,1>> faceDescriptor = net(faces);

    cv::Mat descriptor = dlib::toMat(faceDescriptor);
    return descriptor;
}

dlib::matrix<float,0,1> FeatureExtraction::ComputeDescriptorForFaceDlib(cv::Mat &face){
    dlib::cv_image<dlib::bgr_pixel> cvImage(face);
    dlib::matrix<dlib::rgb_pixel> dlibImage;
    dlib::assign_image(dlibImage, cvImage);
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, dlibImage);
    dlib::matrix<float,0,1> faceDescriptor = mean(mat(net(faces))); //computing face descriptor
//    std::vector<dlib::matrix<float,0,1>> faceDescriptor = net(faces);

    return faceDescriptor;
}



double FeatureExtraction::compareFeaturesCV(cv::Mat h1, cv::Mat h2, int method){
    /*
     * Methods: From 0 to 5
     * 0: Correlation
     * 1: Chi-squared
     * 2: Intersection
     * 3: Bhattacharyya
     * 4: Synonym
     * 5: Alternative Chi-Squared
     */
    double dist = 0.0;
    dist = compareHist(h1,h2,method);
    return dist;


}

double FeatureExtraction::compareFeaturesDlib(dlib::matrix<float,0,1> &descriptor1, dlib::matrix<float,0,1> &descriptor2){
//    dlib::matrix<float,0,1> desc1;
//    dlib::matrix<float,0,1> desc2;

//    for (int i = 0; i <128; ++i) { //as in 128D vector
//        desc1(i,0) = descriptor1.at<float>(i,0);
//        desc2(i,0) = descriptor2.at<float>(i,0);
//    }
    std::vector<dlib::matrix<float,0,1>> faces;
    faces.push_back(descriptor1);
    faces.push_back(descriptor2);


    return length(faces[0]-faces[1]);
}

cv::Mat FeatureExtraction::convertToMat(dlib::matrix<float,0,1> descriptor){
    cv::Mat cvDescriptor = dlib::toMat(descriptor);
    return cvDescriptor;
}
