#include "FeatureExtraction.cpp"

using namespace cv;
using namespace cv::face;
using namespace std;


// This function is not necessary for the project, it is just a way to feed images to the program, useful for tests.
// If you have no need for this method, simply comment it out and feed the main function with images manually (we recommend using cv::imread)
// How to use: receives the path to the file, a vector<cv::Mat> called images, a vector<int> called labels, and the separator for the file (';').
// This is because the CSV should have a format that looks like this (for one line):
// /path/to/your/project/directory/celeb_db/s1/1.jpg;0
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


// In this example images are loaded from the directory ./Python_CreateCSV/celeb_images.celeb_images.csv

// Run the project with the path leading to the csv called celeb_images.csv as the only argument
// if you commented read_csv, delete line 45, and lines 50 through 66,
// as you will no longer feed the program with images loaded from the disk.

// To make your life easier, we've added a simple python program in Python_CreateCSV, where you only have to give
// /The/Path/To/The/Project/ as the argument for the main
// (in unix-based systems, for now.)
int main(int argc, const char *argv[]) {

    string fn_csv = string(argv[1]);

    vector<Mat> samples; // Basic container of images for OpenCV.
    vector<int> labels; // Control variable for each test subject.

    int pos = fn_csv.rfind("/");
    string basePath = fn_csv.substr(0,pos+1);

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

    // The instantiation of our class, receives as only argument the ResNet model
    // taken from the dlib website, available at: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    // Put this file wherever you please, but for yout convenience we have added it to the project in the following directory:
    // ./ResNetModel/dlib_face_recognition_resnet_model_v1.dat
    FeatureExtraction *fe = new FeatureExtraction("/Yann Le Lorier/TEC/Semestre 5/Software engineering/git_repo/git_repo2/Facial-features-extraction-and-comparison/ResNetModel/dlib_face_recognition_resnet_model_v1.dat");


    bool flag = false;
    double hits = 0;
    int misses = 0;

    // A big for loop comparing the first 720 images among themselves.
    for (int i = 0; i < 10; ++i) {
        cv::Mat descriptor1 = fe->ComputeDescriptorForFace(samples[i]);
        for (int j = 0; j < samples.size(); ++j) {
            if(i != j){
                cv::Mat descriptor2 = fe->ComputeDescriptorForFace(samples[j]);
                // compareFeaturesCV(cv::Mat desc1, cv::Mat desc2, int method)
                // NOTE: the method goes from 0 to 6, 0 to 5 are methods for comparing histograms (used in LBPH),
                // while the method 6 is the euclidean distance comparator, useful for descriptor comparison
                double compareRes = fe->compareFeaturesCV(descriptor1, descriptor2, EUCL_DIST);
                cout << "Comparison between picture " << i << " and picture " << j << " => " << compareRes << " | Prediction: ";
                // The threshold was chosen to minimize the type II error rate.
                // For a 0.54 threshold, the accuracy is kept to 96%,
                // while getting a type II error rate of 0.003%
                // To improve accuracy, consider a threshold between 0.55 and 0.60, but
                // doing so shoots the Type II error rate to 1%
                if(compareRes<0.54){
                    flag = true;
                    cout << "They are the same person. ";
                }
                else if (compareRes==0){

                }
                else{
                    flag = false;
                    cout << "They aren't the same person. ";
                }
                if ((labels[i]==labels[j] && flag) || (labels[i]!=labels[j] && !flag)){
                    cout << "Correct!" << endl;
                    hits++;

                }
                else{
                    cout << "Wrong." << endl;
                    misses++;
                }
            }
        }
    }
    cout << "Accuracy after rotation: " << hits*100/1190 << endl;

    return 0;
}
