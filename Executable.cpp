#include "FeatureExtraction.h"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;



static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


string pathCreator(string basePath, string fileName){
    return basePath+fileName;
}

int main(int argc, const char *argv[]) {

    //Declaration of Variables
    // Get the filename to your CSV.
    string fn_csv = string(argv[1]);

    vector<Mat> samples; // Basic container of images for OpenCV. Each entry corresponds to the intensity of each pixel in a certain image.
    vector<int> labels; // Control variable for each test subject.

    int pos = fn_csv.rfind("/");
    string basePath = fn_csv.substr(0,pos+1);

    //Checks if the file is available.
    //Reading a part of the DB.
    try {
        read_csv(fn_csv, samples, labels);
    }catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(samples.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }

    // How to get to input_images_pgm directory
    //read_csv(pathCreator(basePath, "input_data.csv"), imagesToCompare, labelsToCompare);

    FeatureExtraction *fe = new FeatureExtraction();
    fe->trainDataSet(samples, labels);

    //

    double comparisonRes;
    Mat incomingImage = fe->getHistogram(80);

    for (int i = 0; i <= 5; ++i) {
        for (int j = 0; j < 90; ++j) {
            comparisonRes = fe->compareFeatures(incomingImage,fe->getHistogram(j),i);
            cout << "Comparison result w/ method " << i << ": "<< comparisonRes << " " << j << endl;

        }
    }



    return 0;

}