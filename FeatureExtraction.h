#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp" //comparehist is here
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;



class FeatureExtraction {
private:
    Ptr<LBPHFaceRecognizer> model;
    vector<Mat> histograms; //to access the histograms,
    //they must be calculated all at once, so an attribute was declared

    Ptr<LBPHFaceRecognizer> LBPH();

public:
    FeatureExtraction() {
        model = LBPH();
    }
    Mat getHistogram(int index);
    double compareFeatures(Mat h1, Mat h2, int method);
    void trainDataSet(vector<Mat> samples, vector<int> labels);
    void updateDataSet(vector<Mat> samples, vector<int> labels);
};


Ptr<LBPHFaceRecognizer> FeatureExtraction::LBPH(){
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); //Instantiates the LBPHFaceRecognizer class.
    return model;
}

void FeatureExtraction::trainDataSet(vector<Mat> samples, vector<int> labels){
    model->train(samples, labels); // Calls the train method in LBPHFaceRecognizer.
    // It associates labels with their corresponding faces (from the images vector).
    histograms = model->getHistograms();
}

void FeatureExtraction::updateDataSet(vector<Mat> samples, vector<int> labels){
    model->update(samples, labels); // Calls the train method in LBPHFaceRecognizer.
    // It associates labels with their corresponding faces (from the images vector).
    histograms = model->getHistograms();
}

Mat FeatureExtraction::getHistogram(int index){
    return histograms[index];
}

double FeatureExtraction::compareFeatures(Mat h1, Mat h2, int method){
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