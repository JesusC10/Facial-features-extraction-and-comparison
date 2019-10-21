#include "FeatureExtraction.cpp"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
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
            images.push_back(imread(path, 1));
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

    FeatureExtraction *fe = new FeatureExtraction("/Yann Le Lorier/TEC/Semestre 5/Software engineering/git_repo/git_repo2/Facial-features-extraction-and-comparison/ResNetModel/dlib_face_recognition_resnet_model_v1.dat");


    bool flag = false;
    double hits = 0;
    int misses = 0;


//    for (int i = 0; i < 10; ++i) {
//        cv::Mat descriptor1 = fe->ComputeDescriptorForFace(samples[i]);
//        for (int j = 0; j < samples.size(); ++j) {
//            if(i != j){
//                cv::Mat descriptor2 = fe->ComputeDescriptorForFace(samples[j]);
//                dlib::matrix<float,128,1> dlib1 = fe->toDlib(descriptor1); //conversion problem (Wrapper?)
//                dlib::matrix<float,128,1> dlib2 = fe->toDlib(descriptor2);
//                double compareRes = fe->compareFeaturesDlib(dlib1, dlib2);
//                cout << "Comparison between picture " << i << " and picture " << j << " => " << compareRes << " | Prediction: ";
//                if(compareRes<0.60){
//                    flag = true;
//                    cout << "They are the same person. ";
//                }
//                else{
//                    flag = false;
//                    cout << "They aren't the same person. ";
//                }
//                if ((labels[i]==labels[j] && flag) || (labels[i]!=labels[j] && !flag)){
//                    cout << "Correct!" << endl;
//                    hits++;
//
//                }
//                else{
//                    cout << "Wrong." << endl;
//                    misses++;
//                }
//            }
//        }
//        cout << descriptor1 << endl;
//    }
//    cout << "Accuracy after rotation: " << hits*100/720 << endl;

    cv::Mat descriptor1 = fe->ComputeDescriptorForFace(samples[0]);
    cv::Mat descriptor2 = fe->ComputeDescriptorForFace(samples[1]);

    double compare = fe->compareFeaturesCV(descriptor1,descriptor2, EUCL_DIST);

    cout << compare << endl;

    return 0;
}
