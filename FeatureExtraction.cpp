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

#include "FeatureExtraction.h" 

cv::Mat FeatureExtraction::ComputeDescriptorForFace(cv::Mat &face){
    dlib::cv_image<dlib::bgr_pixel> cvImage(face);  // Wraps the cv::Mat image into a dlib::cv_image
    dlib::matrix<dlib::rgb_pixel> dlibImage; // Instantiates a dlibImage
    dlib::assign_image(dlibImage, cvImage); // Sets the cvImage to a dlibImage
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, dlibImage); 
    dlib::matrix<float,128,1> faceDescriptor = mean(mat(net(faces))); // Computes face descriptor
//    std::vector<dlib::matrix<float,0,1>> faceDescriptor = net(faces);
    return dlib::toMat(faceDescriptor); // Transforms it into a cv::Mat
}


// Method that computes the Eucledian Distance between two cv::Mat descriptors
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
    std::vector<dlib::matrix<float,128,1>> faces; //Create a vector to insert the descriptors
    faces.push_back(descriptor1);
    faces.push_back(descriptor2);
    return length(faces[0]-faces[1]); // Calculates the distance between the two faces
}

// Method to convert a dlib::matrix into a cv::Mat
cv::Mat FeatureExtraction::convertToMat(dlib::matrix<float,128,1> descriptor){
    cv::Mat cvDescriptor = dlib::toMat(descriptor); 
    return cvDescriptor;
}
