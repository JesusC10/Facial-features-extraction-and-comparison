#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;

vector<int> precisionRate(Ptr<LBPHFaceRecognizer>, vector<int>, vector<Mat>);
void printHist(Ptr<LBPHFaceRecognizer>);
string pathCreator(string, string);
Ptr<LBPHFaceRecognizer> LBPH(vector<Mat>, vector<int>);


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

int main(int argc, const char *argv[]) {

    //Declaration of Variables
    // Get the filename to your CSV.
    string fn_csv = string(argv[1]);

    vector<Mat> samples; // Basic container of images for OpenCV. Each entry corresponds to the intensity of each pixel in a certain image.
    vector<int> labels; // Control variable for each test subject.

    vector<Mat> imagesToCompare; //This are the images that we want to know if they exist in the DB.
    vector<int> labelsToCompare; //This are the labels that we are going to assign to the test images.


    int pos = fn_csv.rfind("/");
    string basePath = fn_csv.substr(0,pos+1);
    string pathToTest = pathCreator(basePath, "test_data.csv");

    Ptr<LBPHFaceRecognizer> model;

    //Checks if the file is available.
    //Reading a part of the DB.
    try {
        read_csv(pathToTest, samples, labels);
    }catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(samples.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }

    model = LBPH(samples, labels);



    // How to get to input_images_pgm directory
    read_csv(pathCreator(basePath, "input_data.csv"), imagesToCompare, labelsToCompare);

    labelsToCompare = {-1,-1,-1,-1,-1,-1,-1,3,2,-1};

    vector<int> hm = precisionRate(model, labelsToCompare, imagesToCompare);

    cout << "HITS: " << hm[0] << endl;
    cout << "MISSES: " << hm[1] << endl;
    

    //Print Historgram
    printHist(model);

    return 0;

}

Ptr<LBPHFaceRecognizer> LBPH(vector<Mat> samples, vector<int> labels){

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); // Instantiates the LBPHFaceRecognizer class.
    model->train(samples, labels); // Calls the train method in LBPHFaceRecognizer. It associates labels with their corresponding faces (from the images vector).

    return model;

}

vector<int> precisionRate(Ptr<LBPHFaceRecognizer> model, vector<int> labelsToCompare, vector<Mat> imagesToCompare){ //known labels is just for testing purposes
    vector<int> predictedLabels;
    vector<int> hm;
    int hit = 0, miss = 0;

    for (int i = 0; i < imagesToCompare.size(); i++) {
        int item = model->predict(imagesToCompare[i]);
        predictedLabels.push_back(item);


        cout << "Predicted: " << predictedLabels[i] << "/Actual: " << labelsToCompare[i] << endl;

        if(labelsToCompare[i] == predictedLabels[i])
            hit++;
        else
            miss++;


    }

        hm.push_back(hit);
        hm.push_back(miss);
    return hm;
}

string pathCreator(string basePath, string fileName){
    return basePath+fileName;
}

void printHist(Ptr<LBPHFaceRecognizer> model){
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