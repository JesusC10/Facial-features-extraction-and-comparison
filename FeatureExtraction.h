#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;



class FeatureExtraction {
private:
    vector<Mat> samples;
    vector<int> labels;
    Ptr<LBPHFaceRecognizer> model;

    Ptr<LBPHFaceRecognizer> LBPH(vector<Mat> samples, vector<int> labels);

public:
    FeatureExtraction(vector<Mat> s, vector<int> l) {
        samples = s;
        labels = l;
        model = LBPH(samples, labels);
    }
    vector<int> HitOrMiss(vector<int> labelsToCompare, vector<Mat> imagesToCompare); //known labels is just for testing purposes
    void printHist();
};


Ptr<LBPHFaceRecognizer> FeatureExtraction::LBPH(vector<Mat> samples, vector<int> labels){

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); // Instantiates the LBPHFaceRecognizer class.
    model->train(samples, labels); // Calls the train method in LBPHFaceRecognizer. It associates labels with their corresponding faces (from the images vector).

    return model;

}

vector<int> FeatureExtraction::HitOrMiss(vector<int> labelsToCompare, vector<Mat> imagesToCompare){ //known labels is just for testing purposes
    vector<int> predictedLabels;
    vector<int> hm;
    int hit = 0, miss = 0;

    for (int i = 0; i < imagesToCompare.size(); i++) {
        int item = model->predict(imagesToCompare[i]);
        predictedLabels.push_back(item);


        cout << "Predicted: " << predictedLabels[i] << "/ Actual: " << labelsToCompare[i] << endl;

        if(labelsToCompare[i] == predictedLabels[i])
            hit++;
        else
            miss++;


    }

    hm.push_back(hit);
    hm.push_back(miss);
    return hm;
}


void FeatureExtraction::printHist(){
    cout << "Model Information:" << endl;
    string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
                               model->getRadius(),
                               model->getNeighbors(),
                               model->getGridX(),
                               model->getGridY(),
                               model->getThreshold());
    cout << model_info << endl;
    // We could get the histograms for example:
    vector<Mat> histograms = model->getHistograms();

    cout << "Size of the histograms: " << histograms[0].total() << endl;
    cout << "Histograms [0]: " << histograms[0] << endl;
}