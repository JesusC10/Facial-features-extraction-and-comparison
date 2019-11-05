
#include "FeatureExtraction.h"


// Receives an image in the cv::Mat format, computes the descriptor as a dlib object,
// and returns the descriptor in the cv::Mat format.
// Easy-peasy.
cv::Mat FeatureExtraction::ComputeDescriptorForFace(cv::Mat &face){
    dlib::cv_image<dlib::bgr_pixel> cvImage(face);  // Wraps the cv::Mat image into a dlib::cv_image
    dlib::matrix<dlib::rgb_pixel> dlibImage; // Instantiates a dlibImage
    dlib::assign_image(dlibImage, cvImage); // Sets the cvImage to a dlibImage
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, dlibImage);
    dlib::matrix<float,128,1> faceDescriptor = mean(mat(net(faces))); // Computes face descriptor
    cv::Mat descr = dlib::toMat(faceDescriptor);
    return descr.clone(); // Transforms it into a cv::Mat
}


// This method allows you to compare histograms if that is your objective,
// or to compare descriptors, which is what we are doing in this case.
double FeatureExtraction::compareFeaturesCV(cv::Mat h1, cv::Mat h2, int method){
    /*
     * Methods: From 0 to 6
     * 0: Correlation
     * 1: Chi-squared
     * 2: Intersection
     * 3: Bhattacharyya
     * 4: Synonym
     * 5: Alternative Chi-Squared
     * 6: Euclidean distance or write EUCL_DIST
     */
    double dist = 0.0;

    if (method == 6) {
        dist = cv::norm(h1, h2, cv::NORM_L2);
    }
    else
        dist = compareHist(h1,h2,method);

    return dist;
}
