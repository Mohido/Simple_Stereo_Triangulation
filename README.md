# Simple_Stereo_Triangulation

Simple stereo triangulation stands for a 3D scene reconstruction from 2 image pairs. Simple stereo requires the 2 images to have the same rotation on z-axis, y-axis, and x-axis. The camera calibration should be given. 
---

This project is actually an assignment in the Computer Vision course I have taken. We were asked to create a "points cloud" using 2 images. Given the following parameters: The baseline is 12cm. The pixel size is 166.67 pixel/millimeter, the focal length is 3.8mm. The camera calibration matrix (Intrinsic matrix) is given as well. It is being hardcoded in the first lines in the src/main.cpp file.

---

## Features:
* Standard(simple) stereo triangulation.
* Disparity map viewer.
* Depth map viewer.
* 2 different Feature detection techniques were used: ASIFT, and ORB.
* Brute-force Horizental template matching (window matching). 
* Minimum absolute difference is being used in Template matching.
---

## Project deployment on Windows:
I am using CMAKE files to make your life easier with the deployment. The Cmake was implemented to work on windows though (Others OS were not considered). If you have a windows system, then running the CMakeList file will just build and deploy everything. You can setup different settings/options in the main.cpp. 
OpenCV 4.5.3 files and debug library is being attached ini the project files as well. So you don't need to download it. If you want to use a newer version then feel free to edit the setup. 

---

# Credits and Resources:
* OpenCV 4.5.3
* Course website: http://cg.elte.hu/index.php/computer-vision/
* Useful resources for learners: https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw
