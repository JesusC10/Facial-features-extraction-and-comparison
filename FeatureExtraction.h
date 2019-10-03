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
    FeatureExtraction() {
        dlib::deserialize("./ResNetModel/dlib_face_recognition_resnet_model_v1.dat") >> net;
    }
    cv::Mat ComputeDescriptorForFace(cv::Mat &face);
    double compareFeatures(cv::Mat h1, cv::Mat h2, int method);
};


cv::Mat FeatureExtraction::ComputeDescriptorForFace(cv::Mat &face){
    if (face.channels()>1){
        cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);
    }
    //como decir que use el struct de grayscale_pixel_traits?
    dlib::cv_image<unsigned char> cvImage(face);
    dlib::matrix<unsigned char> dlibImage;
    dlib::assign_image(dlibImage, cvImage);
    std::vector<dlib::matrix<unsigned char>> faces(1, dlibImage);
    dlib::matrix<float,0,1> faceDescriptor = mean(mat(net(faces))); //computing face descriptor

    cv::Mat descriptor = dlib::toMat(faceDescriptor);
    return descriptor;
}



double FeatureExtraction::compareFeatures(cv::Mat h1, cv::Mat h2, int method){
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