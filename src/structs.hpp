#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>



/// <summary>
/// An option that is used withen the engine. It decides what feature matcher we going to use. Asift requires a file input. Orb is in Opencv, window matcher is implemented manually.
/// </summary>
enum DetectorOption {
    ORB,			// We use the ORB algorithm that is implemented in the OPENCV
    ASIFT,			// We use the ASIFT Algorithm that is being obtained through ASIFT online
    WINDOW_MATCHER	// We use the naive window matching algorithm to find matches in all the window. Specially since the baseline is small.
};



/// <summary>
/// Loss function that will be used to compare in template matching.
/// </summary>
enum TemplateMatcherLoss {
    SUM_OF_ABSOLUTE_DIFF,
    SUM_OF_SQUARED_DIFF,
    NORMALIZED_CROSS_CORRELATION
};


/// <summary>
///     An abstraction over the Feature mappings. It takes 2 images and by using the ORB model provided for 
///     free by OpenCV, it generates a set of Feature maps and holds other very useful information. However, for 
///     this project, the most useful variable is the matchingPoints since it defines the set of points in image 'a'
///     and their corresponding points in image 'b'
/// 
/// USEFULL REFERENCES:
///     https://github.com/santosderek/Brute-Force-Matching-using-ORB-descriptors/blob/master/src/main.cpp
/// </summary>
struct ImageFeatureMatcher {
    // Feature Points.
    std::vector<cv::KeyPoint> keypointsBaseImage, keypointsTargetImage;

    // Find descriptors.
    cv::Mat descriptorsBaseImage, descriptorsTargetImage;

    // Vector where matches will be stored.
    std::vector<cv::DMatch> matches;

    // vector of points matched...
    std::vector <std::pair <cv::Point2f, cv::Point2f> > matchingPoints;

    /* How things go: Keypoints -> descriptors -> DMatches -> matchingPoints*/
    ImageFeatureMatcher(cv::Mat& baseImage, cv::Mat& targetImage) {
        
        /*Computing the keypoints*/
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();      // Using ORB detection for keypoints.
        detector->detect(baseImage, this->keypointsBaseImage);          // Detect keypoints for first image
        detector->detect(targetImage, this->keypointsTargetImage);      // Detect for second image.
        detector.release();                                             // Release the detector.


        /*Computing the descriptors*/
        cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();                                 // Using the ORB engine again.
        extractor->compute(baseImage, this->keypointsBaseImage, this->descriptorsBaseImage);            // Get descriptors from first feature points
        extractor->compute(targetImage, this->keypointsTargetImage, this->descriptorsTargetImage);      // Get descriptors for second image.
        extractor.release();

        // Create Brute-Force Matcher. Other Algorithms are 'non-free'.
        cv::BFMatcher brue_force_matcher = cv::BFMatcher(cv::NORM_HAMMING, true);

        // Find matches and store in matches vector.
        brue_force_matcher.match(
            (const cv::OutputArray)this->descriptorsBaseImage, 
            (const cv::OutputArray)this->descriptorsTargetImage, 
            this->matches);

        // Sort them in order of their distance. The less distance, the better.
        std::sort(
            this->matches.begin(), 
            this->matches.end(), 
            [](cv::DMatch& a, cv::DMatch& b) { return a.distance < b.distance; }); 
        this->matchingPoints.reserve(this->matches.size());

        /* Filling the pixels matches */
        for (int i = 0; i < this->matches.size(); i++) {
            int baseInd = this->matches[i].queryIdx;
            cv::Point2f u_1 = this->keypointsBaseImage[baseInd].pt;

            auto targetInd = this->matches[i].trainIdx;
            cv::Point2f u_2 = this->keypointsTargetImage[targetInd].pt;

            /*For this task, we only grab the points with the same horizontal coordinates.*/
            //std::cout << "length: " << u_1.y  << " - " <<  u_2.y << ":" << std::fabs(u_1.y - u_2.y) << std::endl;
            if (std::fabs(u_1.y - u_2.y) <= 0.05f) {
                //
                this->matchingPoints.push_back({ u_1 , u_2});
            }
        }
        printf("[DEBUG]: ImageFeatureMatcher: size of the feature points: %d\n", this->matchingPoints.size());
    }
};