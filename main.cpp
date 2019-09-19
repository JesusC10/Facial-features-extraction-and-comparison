#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;

vector<int> PrecisionRate(Ptr<LBPHFaceRecognizer>, vector<int>, vector<Mat>);
string pathCreator(string, string);

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
    // Get the filename to your CSV.
    string fn_csv = string(argv[1]);

    vector<Mat> images; // Basic container of images for OpenCV. Each entry corresponds to the intensity of each pixel in a certain image.
    vector<int> labels; // Control variable for each test subject.

    //Checks if the file is available.
    try {
        read_csv(fn_csv, images, labels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }



    vector<Mat> testSamples;
    vector<int> knownLabels;

    int pos = fn_csv.rfind("/");

    string basePath = fn_csv.substr(0,pos+1);

    string pathToTest = pathCreator(basePath, "test_data.csv");


    try {
        read_csv(pathToTest, testSamples, knownLabels);
    }catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    // TEST WITH MULTIPLE PHOTOS
    // BEGIN
    Ptr<LBPHFaceRecognizer> testModel = LBPHFaceRecognizer::create();
    testModel->train(testSamples, knownLabels);


    // How to get to input_images_pgm directory

    vector<Mat> imagesToCompare;
    vector<int> labelsToCompare;

    read_csv(pathCreator(basePath, "input_data.csv"), imagesToCompare, labelsToCompare);

    labelsToCompare = {-1,-1,-1,-1,-1,-1,-1,-1,3,2};

    vector<int> hm = PrecisionRate(testModel, labelsToCompare, imagesToCompare);

    cout << "HITS: " << hm[0] << endl;
    cout << "MISSES: " << hm[1] << endl;



    // FOR ONE TEST ONLY: UNCOMMENT THE NEXT 6 LINES
    //Mat testSample;
    //int testlabel;
    //testSample = images[images.size() - 1]; // Store the last image of the last test subject.
    //testLabel = labels[labels.size() - 1]; // Store the last label of the last test subject.
    //images.pop_back(); // Pop the last image of the last test subject (to avoid overlapping between the test image and the vector holding the images.
    //labels.pop_back(); // Pop the last label of the last test subject (to avoid overlapping between the test label and the vector holding the labels


    // The LBPHFaceRecognizer uses Extended Local Binary Patterns
    // (it's probably configurable with other operators at a later
    // point), and has the following default values
    //
    //      radius = 1
    //      neighbors = 8
    //      grid_x = 8
    //      grid_y = 8
    //
    // So if you want a LBPH FaceRecognizer using a radius of
    // 2 and 16 neighbors, call the factory method with:
    //
    //      cv::face::LBPHFaceRecognizer::create(2, 16);
    //
    // And if you want a threshold (e.g. 123.0) call it with its default values:
    //
    //      cv::face::LBPHFaceRecognizer::create(1,8,8,8,123.0)

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(); // Instantiates the LBPHFaceRecognizer class.
    model->train(images, labels); // Calls the train method in LBPHFaceRecognizer. It associates labels with their corresponding faces (from the images vector).

    //int predictedLabel = model->predict(testSample); // Predicts the label of a given face after the train method has been called (comparison).

    // To get the confidence of a prediction call the model with:
    //
//    int predictedLabel = -1;
//    double confidence = 1.0;
//    model->predict(testSample, predictedLabel, confidence);

    //string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    //cout << result_message << endl;
    // First we'll use it to set the threshold of the LBPHFaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    // model->setThreshold(0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    // predictedLabel = model->predict(testSample);
    // cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
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
    return 0;

}

vector<int> PrecisionRate(Ptr<LBPHFaceRecognizer> model, vector<int> labelsToCompare, vector<Mat> imagesToCompare){ //known labels is just for testing purposes
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