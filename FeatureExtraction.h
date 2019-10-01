#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp" //comparehist is here
#include <dlib/dnn.h>
#include <dlib/opencv.h> //to convert objects from opencv to dlib and viceversa
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <sstream>
//using namespace cv;
//using namespace cv::face;
using namespace std;

//IMPLEMENT DLIBBBBBBBB

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                                                        dlib::input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;

class FeatureExtraction {
private:
    anet_type net;
    //dlib::matrix<dlib::rgb_pixel> image; //dlib structure
    cv::Ptr<cv::face::LBPHFaceRecognizer> model;
    std::vector<cv::Mat> histograms; //to access the histograms,
    //they must be calculated all at once, so an attribute was declared

    cv::Ptr<cv::face::LBPHFaceRecognizer> LBPH();

public:
    FeatureExtraction() {
        model = LBPH();
        dlib::deserialize("/Users/yannlelorier/Downloads/dlib_face_recognition_resnet_model_v1.dat") >> net;
    }
    cv::Mat ComputeDescriptorForFace(cv::Mat face);
    cv::Mat getHistogram(int index);
    double compareFeatures(cv::Mat h1, cv::Mat h2, int method);
};


cv::Ptr<cv::face::LBPHFaceRecognizer> FeatureExtraction::LBPH(){
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create(); //Instantiates the LBPHFaceRecognizer class.
    return model;
}

cv::Mat FeatureExtraction::ComputeDescriptorForFace(cv::Mat face){
    dlib::cv_image<dlib::bgr_pixel> image(face);
    dlib::matrix<dlib::rgb_pixel> matrix;
    dlib::assign_image(matrix, image);
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces(1, matrix);
    dlib::matrix<float,0,1> faceDescriptor = mean(mat(net(faces))); //computing face descriptor
    face = dlib::toMat(matrix); //converting data structure of dlib to opencv structure
    return face;
}

cv::Mat FeatureExtraction::getHistogram(int index){
    return histograms[index];
}

double FeatureExtraction::compareFeatures(cv::Mat h1, cv::Mat h2, int method){
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