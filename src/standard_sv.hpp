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

#include "functions.hpp"

/// <summary>
/// Caculates the 3D spetial point from the given parameters. 
/// </summary>
/// <param name="u_1">relative pixel coordinate in image plane 1</param>
/// <param name="u_2">relative pixel coordinate in image plane 2</param>
/// <param name="baseline">camera baseline</param>
/// <param name="M_int"></param>
/// <param name="c_1"></param>
/// <param name="c_2"></param>
/// <returns></returns>
cv::Point3f standard_stereo(
	const cv::Point2f& u_1, const cv::Point2f& u_2,
	//const cv::Point3f& c_1, const cv::Point3f& c_2,
	const cv::Mat& M_int,
	const float& b, const float ppmm, const float& fl
	) {

	/*Extracting the intrinsic important parameters*/
	float fx = M_int.at<float>({ 0, 0 });		float fy = M_int.at<float>({ 1, 1 });
	float ox = M_int.at<float>({ 0, 2 });		float oy = M_int.at<float>({ 1, 2 });

	/*Fully using the intrinsic parameters of the camera.*/
	float d = std::fabs(u_1.x - u_2.x);
    printf("[DEBUG]: [standard_stereo]: disparity is: (%.2f)\n", d);
    // d = (d == 0) ? 0.000001f : d;               // To avoid infinity
	cv::Point3f point3d;
	point3d.x = (b * (u_1.x - ox)) / d;
	point3d.y = (b * fx * (u_1.y - oy)) / (fy * d);
	point3d.z = (b * fx) / d;
	
    printf("[DEBUG]: [standard_stereo]: Constructed 3D point: (%.2f, %.2f, %.2f)\n", point3d.x, point3d.y, point3d.z);

	return point3d;
}






/// <summary>
/// Calculate the difference sum between 2 matrices of the same size.
/// </summary>
/// <param name="M1"></param>
/// <param name="M2"></param>
/// <returns></returns>
static int sumOfAbsDiff(const cv::Mat& M1, const cv::Mat& M2) {
    int rdsum = 0, bdsum = 0, gdsum = 0;

    for (int r = 0; r < M1.size().height; r++) {
        for (int c = 0; c < M1.size().width; c++) {
            cv::Vec3b mc1 = M1.at<cv::Vec3b>({ r,c });
            cv::Vec3b mc2 = M2.at<cv::Vec3b>({ r,c });

            rdsum += std::abs(int(mc1[0]) - int(mc2[0]));
            bdsum += std::abs(int(mc1[1]) - int(mc2[1]));
            gdsum += std::abs(int(mc1[2]) - int(mc2[2]));
        }
    }

    return rdsum + bdsum + gdsum;
}




/// <summary>
/// 
/// </summary>
/// <param name="windowsize"></param>
/// <param name="w1pos">to know which pixel we want to compare (from first image)</param>
/// <param name="image1"></param>
/// <param name="image2"></param>
/// <returns></returns>
std::pair<int, int> templateMatching(
    const int& w_size, const std::pair<int, int>& w1pos,
    const cv::Mat& image1, const cv::Mat& image2,
    const TemplateMatcherLoss& option = TemplateMatcherLoss::SUM_OF_ABSOLUTE_DIFF)
{
    if (image1.size().height != image2.size().height) {
        throw std::runtime_error("Image heights are not the same. Template matching require both images to have the same sizes");
    }

    const int im1_w = image1.size().width; const int im1_h = image1.size().height;
    const int im2_w = image2.size().width; const int im2_h = image2.size().height;
    cv::Mat M_w = cropWindow(w_size, w1pos, image1);

    int bestSum = 0;                // the column of the lowest sum
    int smallestSum = INT_MAX;      // the smallest absolute difference sum of the current subwindow.

    /*Use the template matching algorithm*/
    for (int c = 0; c < im2_w - w_size; c++) { // per column in the second image x-axis
        // right image subwindow
        cv::Mat M_w_2 = cropWindow(w_size, { w1pos.first, c }, image2);

        /*Match the 2 cropped windows and use the loss option.*/
        int temp = sumOfAbsDiff(M_w, M_w_2);

        if (temp < smallestSum) {
            bestSum = c;
            smallestSum = temp;
        }
    }

    //printf("[DEBUG]: [templateMatching]: Window at (%d, %d) have been matched to (%d, %d)\n\n", w1pos.first, w1pos.second, w1pos.first, bestSum);
    return { w1pos.first, bestSum };
}







/// <summary>
/// Uses the simple stereo to solve the task.
/// </summary>
/// <param name="filename"></param>
/// <param name="matchings"></param>
/// <param name="depthMap"></param>
/// <param name="M_int"></param>
void useSimpleStereo(
    const std::string& filename, 
    const std::vector<std::pair<cv::Point2f, cv::Point2f>>& matchings, 
    cv::Mat& depthMap, 
    const cv::Mat& M_int, 
    const float& baseline = BASELINE, 
    const float& pixel_space_ratio = PIXEL_SPACE_RATIO, 
    const float& focal_length = FOCAL_LENGTH
) {

#ifdef CREATE_CLOUD
    std::ofstream file(filename);
#endif

    for (const std::pair<cv::Point2f, cv::Point2f>& p_pair : matchings) {
        cv::Point3f point3d = standard_stereo(p_pair.first, p_pair.second, M_int, baseline, pixel_space_ratio, focal_length);
        if (point3d.z >= FAR_PLANE || point3d.z <= NEAR_PLANE) {
            printf("Point out of range\n");
            continue;
        }

        double& color = depthMap.at<double>({ (int)p_pair.first.x, (int)p_pair.first.y });
        double nz = (double)point3d.z / (double)FAR_PLANE;
        color = 1.0f - nz;

#ifdef CREATE_CLOUD
        file << point3d.x << " " << point3d.y << " " << point3d.z << std::endl;
#endif

    }
#ifdef CREATE_CLOUD
    file.close();
#endif
#ifdef SHOW_DEPTH_MAP
    cv::imshow(WINDOW_NAME, depthMap);
#endif
    cv::waitKey(0);
}