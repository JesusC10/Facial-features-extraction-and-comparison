HOW DOES IT WORK:


This module works using a neural network model that is used to generate vector descriptors given a set of normalized images. 



HOW TO USE IT:


On the main executable:
1) Include the "FeatureExtracion.cpp" file.

2) Create an instance of the FeatureExtraction class and send to the constructor the path for the following file as a parameter: 
    "dlib_face_recognition_resnet_model_v1.dat". 
    
    
     Example: 
     
     
         FeatureExtraction * fe = new FeatureExtraction("dlib_face_recognition_resnet_model_v1.dat");
         
         
     The file can be found in ResNetModel directory.
     

3) Create two instances of the cv::Mat class and set each one of them to the FeatureExtraction method ComputeDescriptorForFace(image), where image is an indivdual output image (cv::Mat) obtained from the previous or next module, depending on the use case. These instances will create two descriptors that will be compared between one another.


     Example: 
     
     
        cv::Mat descriptor1 = fe->ComputeDescriptorForFace(image_1);       
        cv::Mat descriptor2 = fe->ComputeDescriptorForFace(image_2);

4) Create a variable (double) that will be used to store the Eucledian Distance between one descriptor and the other. For this, a call to the FeatureExtraction method compareFeaturesCV(descriptor1, descriptor2, EUCL_DIST) is needed. EUCL_DIST is a global variable that is set to 6.


    Example:
    
    
        double distance = fe->compareFeaturesCV(descriptor1,descriptor2, EUCL_DIST);
        
The variable distance is the value that will determine whether or not two image descriptors correspond to the same person.
