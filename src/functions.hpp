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

#include "structs.hpp"


/// <summary>
///     Reads the data from the attached file. 
/// 
/// NOTE:
///     File inputs would have the following pattern
///        left1.png
///        right1.png
///         ..
///         ..
/// </summary>
/// <returns>pairs of the read images.</returns>
std::vector<std::pair<cv::Mat, cv::Mat>> readTaskImages() {
    std::string directoryName = "res/";
    std::vector<std::pair<cv::Mat, cv::Mat>> ret;
    for (int i = 1; i <= 5; i++) {
        std::string left = directoryName + "left" + std::to_string(i) + ".png";
        std::string right = directoryName + "right" + std::to_string(i) + ".png";

        cv::Mat leftImg = cv::imread(left);
        cv::Mat rightImg = cv::imread(right);
        if (leftImg.empty()) {
            std::cout << "can't find image at: " << left << std::endl;
            continue;
        }
        if (rightImg.empty()) {
            std::cout << "can't find image at: " << right << std::endl;
            continue;
        }

        ret.push_back({ rightImg,leftImg });
    }
    return ret;
}



/// <summary>
/// 
/// </summary>
/// <param name="filename"></param>
/// <returns></returns>
std::vector<std::pair<cv::Point2f, cv::Point2f>> loadASIFT(const std::string& filename) {
    std::vector<std::pair<cv::Point2f, cv::Point2f>> ret;
    std::ifstream file(filename);
    if (!file) {
        printf("[ERROR]: [loadASIFT]: Can't open the file: %s\n", filename.c_str());
        return ret;
    }
    uint32_t n;
    file >> n;
    ret.reserve(n);

    for (uint32_t i = 0; i < n; i++) {
        float x1, y1, x2, y2;
        file >> x1;
        file >> y1;
        file >> x2;
        file >> y2;

        //std::cout << "rading point: " << x1 << " " << y1 << " " << x2 << " " << x2 << std::endl;
        cv::Point2f p1(x1,y1);
        cv::Point2f p2(x2, y2);

        if (std::fabs(y1 - y2) <= 0.05f) {
        //if (y1 == y2) {
            ret.push_back({ p1, p2 });
        }
    }
    printf("[DEBUG]: [loadASIFT]: Loaded %d matches\n", ret.size());
    return ret;
}



/// <summary>
/// Crop the subwindow at the given position from the given image. This is a legacy function that was implemented statically.
///  Now it is just an overlap of the opencv cropping funciton. Note that this function returns a copy of the matrix!
/// </summary>
/// <param name="w_size"></param>
/// <param name="w1pos"></param>
/// <param name="image"></param>
/// <returns></returns>
cv::Mat cropWindow(const int& w_size, const std::pair<int, int>& w1pos, const cv::Mat& image) {
    return image(cv::Rect(w1pos.second, w1pos.first, w_size, w_size)).clone();
}


