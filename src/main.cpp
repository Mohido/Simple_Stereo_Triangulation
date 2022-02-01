#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>

/*Window name*/
#define WINDOW_NAME "Display Image"

/*Options to set or unset.*/
//#define SHOW_SIMILAR_FEATURES							// Show the similar features. It will show something only if the matcher used is "ORB"
#define SHOW_DEPTH_MAP									// Shows the depth map of the processed points. Closer to camera = brighter
#define CREATE_CLOUD									// Creates the Point cloud. 
#define WINDOW_MATCHER_CREATE_DISPARITY_MAP				// Shows and creates the disparity map of the Window matcher.

/*Camera settings*/
#define BASELINE				120						// mm
#define PIXEL_SPACE_RATIO		166.67					// pixel/mm
#define FOCAL_LENGTH			3.8						// mm
#define FAR_PLANE				100000.0f				// far plane in mm = 100M
#define NEAR_PLANE				FOCAL_LENGTH			// near plane in mm

/*Window matcher sittengs*/
#define WINDOW_MATCHER_SIZE		10						// window size = 10px width and 10px height

/*Definitions were define above because they are used in the whole project structure.*/
#include "structs.hpp"
#include "functions.hpp"
#include "standard_sv.hpp"


/*constant variables area*/
const static DetectorOption DETECTOR = ASIFT;


/*Main program area*/
int main(int argc, char** argv)
{
	/*Camera Intrinsic Matrix. Gotten from the file malag.mat in the cg elte website.*/
	static cv::Mat M_int(3, 3, CV_32F);
	M_int.at<float>(0,0) = 621.18f;	M_int.at<float>(0, 1) = 0.0f;		M_int.at<float>(0, 2) = 404.0f;
	M_int.at<float>(1,0) = 0.0f;	M_int.at<float>(1, 1) = 621.18f;	M_int.at<float>(1, 2) = 309.0f;
	M_int.at<float>(2,0) = 0.0f;	M_int.at<float>(2, 1) = 0.0f;		M_int.at<float>(2, 2) = 1.0f;

	// Loading the images from the res file.
	std::vector<std::pair<cv::Mat, cv::Mat>>		imagePairs	= readTaskImages();	

	// Creating a window handler.
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

// Show the similar points computed via ORB functionality. NOTE: This only keeps the features on the horizontal line.
#ifdef SHOW_SIMILAR_FEATURES 
	for (int i = 0; i < imagePairs.size(); i++) {
		std::pair<cv::Mat, cv::Mat>		pr = imagePairs[i];
		cv::Mat		output_image;
		
		if (DETECTOR == DetectorOption::ORB) {
			ImageFeatureMatcher		temp(pr.first, pr.second);
			//printf("Size of the matches in orb: %d\n", temp.matchingPoints.size());
			cv::drawMatches(
				pr.first, temp.keypointsBaseImage,
				pr.second, temp.keypointsTargetImage,
				temp.matches,
				output_image);
			cv::imshow(WINDOW_NAME, output_image);
		}
		else if (DETECTOR == DetectorOption::ASIFT) {
			std::string filename = "res/asift_matchings/image" + std::to_string(i + 1) + ".txt";
			loadASIFT(filename);

			// TODO: Implement a viewer for feature points.
		}
		cv::waitKey(0);
	}
#endif

	switch (DETECTOR) {
	case ORB:
	{
		for (int i = 0; i < imagePairs.size(); i++) {
			// Load the data
			std::pair<cv::Mat, cv::Mat> pr = imagePairs[i];
			cv::Mat depthMap = cv::Mat::zeros(pr.first.size(), CV_64FC1);
			ImageFeatureMatcher temp(pr.first, pr.second);

			// Use simple stereo algorithm
			std::string filename = "orb_cloud" + std::to_string(i) + ".xyz";
			useSimpleStereo(filename, temp.matchingPoints, depthMap, M_int);
		}
	}
	break;
	case ASIFT:
	{
		for (int i = 0; i < imagePairs.size(); i++) {
			// Load the data
			std::pair<cv::Mat, cv::Mat> pr = imagePairs[i];
			std::string asif_filename = "res/asift_matchings/image" + std::to_string(i + 1) + ".txt";
			std::vector<std::pair<cv::Point2f, cv::Point2f>> matchings = loadASIFT(asif_filename);
			cv::Mat depthMap = cv::Mat::zeros(pr.first.size(), CV_64FC1);
			if (matchings.size() == 0) {
				printf("[ERROR]: [main]: No matching for the given file.\n");
				continue;
			}

			// Use simple stereo algorithm
			std::string filename = "asif_cloud" + std::to_string(i) + ".xyz";
			useSimpleStereo(filename, matchings, depthMap, M_int);
		}
	}
	break;
	case WINDOW_MATCHER:
	{	
		/*Matching window algorthm*/
		/* First we create the window sizes and say that our pixels are that huge => pixels / mm  = windowsize*pixels/mm. 
			And the output depth image should be of size: Image size/ window size. 
			The caliberation matrix should be edited to handle the new results.
				fx = f*(pixels/mm)*windowsize
				fy = f*(pixels/mm)*windowsize
				ox = ox / windowsize
				oy = oy / windowsize
				*/
		
		/*Camera caliberation for the new imageplane*/
		//cv::Mat wm_M_int = M_int.clone();
		//// We edit the focal length to the new scaled world. Pixels of the focal length ratio are scaled.
		//wm_M_int.at<float>({0,0}) *= WINDOW_MATCHER_SIZE;	wm_M_int.at<float>({1,1}) *= WINDOW_MATCHER_SIZE;
		//// We edit the centers of the world to the new pixels.
		//wm_M_int.at<float>(0, 2) /= WINDOW_MATCHER_SIZE;	wm_M_int.at<float>(1, 2) /= WINDOW_MATCHER_SIZE;


		for (int i = 0; i < imagePairs.size(); i++) {
			// Time start
			std::chrono::steady_clock::time_point startTime = std::chrono::high_resolution_clock::now();
			std::cout << "Starting template matching for image " << i << std::endl;

			// Matching points for the window. Rach of the images should have a matching point on the other image
			std::vector<std::pair<cv::Point2f, cv::Point2f>> matchings;
			matchings.reserve(imagePairs[i].first.size().area());		// reserving for the points. All points will have a map to the pixel from next block.

			// disparityMap to show the image.
			cv::Mat disparityMap = cv::Mat::zeros(imagePairs[i].first.size(), CV_32FC1);

			// Depth map that will be used to fill the data
			cv::Mat depthMap = cv::Mat::zeros(imagePairs[i].first.size(), CV_64FC1);

			// Do template matching and get the u1,u2 similar feature points. Window starts at top,left corner. (We convolute it.)
			for (int r = 0; r < imagePairs[i].first.size().height / WINDOW_MATCHER_SIZE; r++) {
				for (int c = 0; c < imagePairs[i].first.size().width / WINDOW_MATCHER_SIZE; c++) {
					std::pair<int, int> w1pos = { r * WINDOW_MATCHER_SIZE , c * WINDOW_MATCHER_SIZE };
					std::pair<int,int> matchedPosition = templateMatching(
						WINDOW_MATCHER_SIZE, 
						w1pos,
						imagePairs[i].first, 
						imagePairs[i].second);


					// Push back all the points and their coorespondies.
					for (int dr = 0; dr < WINDOW_MATCHER_SIZE; dr++) {
						float p1cr = float(r * WINDOW_MATCHER_SIZE + dr);			// Current point coordinate of hte left image pixel
						float p2cr = float(matchedPosition.first + dr);				// Pixel from second image. Note that we don't need to conver it to r-space

						for (int dc = 0 ; dc < WINDOW_MATCHER_SIZE; dc++) {
							float p1cc = float(c * WINDOW_MATCHER_SIZE + dc);		// Current point coordinate of hte left image pixel
							float p2cc = float(matchedPosition.second + dc);		// Pixel from second image. Note that we don't need to conver it to r-space
							
							// Points on the same pixel refe that they have been taking by the same camera. Which is wrong. 
							//		Therefore, we can either introduce a bias, or elimenate these points from accuring.
							//		Elememnating them is not very good since we don't have a lot of textures, the scene we are working on is almost textureless.
							if (std::fabs(p2cc - p1cc) < 1.0) {
								continue;
							}

							matchings.push_back({ cv::Point2f( p1cc, p1cr), cv::Point2f(p2cc, p2cr)});
							// printf("[WINDOW_MATCHER]: Pixel from (%.2f, %.2f) is matched to (%.2f, %.2f)\n", p1cr, p1cc, p2cr, p2cc);
						}
					}


#ifdef WINDOW_MATCHER_CREATE_DISPARITY_MAP
					// Calculating the disparity ratio of the horizontal displacement. difference / width. => difference higher == brighter image.
					float disparity_ratio = std::fabs(w1pos.second - matchedPosition.second) / (float)(imagePairs[i].first.size().width );
					// Fill the disparity map region
					for (int dr = r * WINDOW_MATCHER_SIZE; dr < WINDOW_MATCHER_SIZE * (r + 1); dr++) {
						for (int dc = c * WINDOW_MATCHER_SIZE; dc < WINDOW_MATCHER_SIZE * (c + 1); dc++) {
							float& color = disparityMap.at<float>({ dc, dr }); // the opencv is a column based matrix. 
							color = disparity_ratio;
						}
					}
#endif
				}

				printf("[DEBUG]: [WINDOW_MATCHER]: Done horizontal row at: %d\n", r);
			}

			std::chrono::steady_clock::time_point endtime = std::chrono::high_resolution_clock::now();
			float duration = std::chrono::duration_cast<std::chrono::duration<float>>(endtime - startTime).count();
			printf("[DEBUG]: [WINDOW_MATCHER]: Duration spent in finding Window template matching: %.5f\n", duration);

			// Forming the sterio vision
			std::string filename_wm = "window_matcher_" + std::to_string(i) + ".xyz";
			useSimpleStereo(filename_wm, matchings, depthMap, M_int);


#ifdef WINDOW_MATCHER_CREATE_DISPARITY_MAP
			std::string disparityFilename = "WM_disparitymap_" + std::to_string(i) + "_winsize_" + std::to_string(WINDOW_MATCHER_SIZE) + ".png";
			disparityMap.convertTo(disparityMap, CV_8UC3, 255.0);
			cv::imwrite(disparityFilename, disparityMap);

			cv::imshow(WINDOW_NAME, disparityMap);
			cv::waitKey(0);
#endif
		}
	}
	break;
	default:
		printf("The matcher is not implemented yet.\n");
		break;
	}

    cv::waitKey(0);
    return 0;
}


