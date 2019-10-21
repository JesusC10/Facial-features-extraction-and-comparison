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

#include "FeatureExtraction.h" //including our class


cv::Mat FeatureExtraction::ComputeDescriptorForFace(cv::Mat &face){
    dlib::cv_image<dlib::bgr_pixel> cvImage(face);
    dlib::matrix<dlib::rgb_pixel> dlibImage;
    dlib::assign_image(dlibImage, cvImage);
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, dlibImage);
    dlib::matrix<float,128,1> faceDescriptor = mean(mat(net(faces))); //computing face descriptor
//    std::vector<dlib::matrix<float,0,1>> faceDescriptor = net(faces);


    return dlib::toMat(faceDescriptor);
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
     * 6: Euclidean distance or write EUCL_DIST
     */
    double dist = 0.0;

    if (method == 6)
        dist = cv::norm(h1,h2);
    else
        dist = compareHist(h1,h2,method);

    return dist;


}

double FeatureExtraction::compareFeaturesDlib(dlib::matrix<float,128,1> &descriptor1, dlib::matrix<float,128,1> &descriptor2){
    std::vector<dlib::matrix<float,128,1>> faces;
    faces.push_back(descriptor1);
    faces.push_back(descriptor2);

    return length(faces[0]-faces[1]);
}

cv::Mat FeatureExtraction::convertToMat(dlib::matrix<float,128,1> descriptor){
    cv::Mat cvDescriptor = dlib::toMat(descriptor);
    return cvDescriptor;
}

dlib::matrix<float,128,1> FeatureExtraction::toDlib(cv::Mat descriptor){
    //how to convert without losing info
    dlib::matrix<float,128,1> res; //losing info (rounding error)
    for (int i = 0; i < 128; ++i) {
        res(i,0) = descriptor.at<float>(i,0);
        std::cout << res(i,0)<< std::endl;
        std::cout << descriptor.at<float>(i,0) << std::endl;
    }
    return res;
}
