# Facial-features-extraction-and-comparison
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
