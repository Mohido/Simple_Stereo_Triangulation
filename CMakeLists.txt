# CMakeList.txt : CMake project for Stereo_Vision_3D_Reconstruction, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Stereo_Vision_3D_Reconstruction")


# Copying Files to the out folder.
file(COPY res DESTINATION "${CMAKE_BINARY_DIR}/")
file (GLOB SV_SRC 
			"src/*.cpp" 
			"src/*.hpp"
			)

# Add source to this project's executable.
add_executable (Stereo_Vision_3D_Reconstruction ${SV_SRC})

IF(WIN32)
	IF (NOT OpenCV_FOUND)
		find_library(OpenCV_LIBS NAMES opencv_world453d PATHS ${CMAKE_SOURCE_DIR}/libs/opencv)
		IF (OpenCV_LIBS)
			set(OpenCV_FOUND ON)
			set(OpenCV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/utils/opencv")
			include_directories( ${OpenCV_INCLUDE_DIRS} )
			target_link_libraries( Stereo_Vision_3D_Reconstruction ${OpenCV_LIBS} )
			MESSAGE( "OpenCV Includes are: ${OpenCV_INCLUDE_DIRS}")
			MESSAGE( "OpenCV libraries are: ${OpenCV_LIBS}")
			MESSAGE("Found OpenCV In the project directory")
		ENDIF()
	ENDIF()
ELSEIF(LINUX)
	# TODO: Linux implementation should be here. However, I have no Linux machine to try it on it.
ENDIF(WIN32)