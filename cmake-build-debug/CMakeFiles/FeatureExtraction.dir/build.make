# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/FeatureExtraction.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FeatureExtraction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FeatureExtraction.dir/flags.make

CMakeFiles/FeatureExtraction.dir/Executable.cpp.o: CMakeFiles/FeatureExtraction.dir/flags.make
CMakeFiles/FeatureExtraction.dir/Executable.cpp.o: ../Executable.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FeatureExtraction.dir/Executable.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FeatureExtraction.dir/Executable.cpp.o -c /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/Executable.cpp

CMakeFiles/FeatureExtraction.dir/Executable.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FeatureExtraction.dir/Executable.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/Executable.cpp > CMakeFiles/FeatureExtraction.dir/Executable.cpp.i

CMakeFiles/FeatureExtraction.dir/Executable.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FeatureExtraction.dir/Executable.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/Executable.cpp -o CMakeFiles/FeatureExtraction.dir/Executable.cpp.s

# Object files for target FeatureExtraction
FeatureExtraction_OBJECTS = \
"CMakeFiles/FeatureExtraction.dir/Executable.cpp.o"

# External object files for target FeatureExtraction
FeatureExtraction_EXTERNAL_OBJECTS =

FeatureExtraction: CMakeFiles/FeatureExtraction.dir/Executable.cpp.o
FeatureExtraction: CMakeFiles/FeatureExtraction.dir/build.make
FeatureExtraction: /usr/local/lib/libopencv_gapi.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_stitching.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_aruco.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_bgsegm.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_bioinspired.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_ccalib.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_dnn_objdetect.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_dpm.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_freetype.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_fuzzy.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_hfs.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_img_hash.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_line_descriptor.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_quality.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_reg.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_rgbd.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_saliency.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_sfm.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_stereo.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_structured_light.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_superres.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_surface_matching.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_tracking.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_videostab.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_xfeatures2d.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_xobjdetect.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_xphoto.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_face.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_shape.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_datasets.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_plot.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_text.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_dnn.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_highgui.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_ml.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_phase_unwrapping.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_optflow.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_ximgproc.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_video.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_videoio.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_imgcodecs.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_objdetect.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_calib3d.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_features2d.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_flann.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_photo.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_imgproc.4.1.1.dylib
FeatureExtraction: /usr/local/lib/libopencv_core.4.1.1.dylib
FeatureExtraction: CMakeFiles/FeatureExtraction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FeatureExtraction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FeatureExtraction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FeatureExtraction.dir/build: FeatureExtraction

.PHONY : CMakeFiles/FeatureExtraction.dir/build

CMakeFiles/FeatureExtraction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FeatureExtraction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FeatureExtraction.dir/clean

CMakeFiles/FeatureExtraction.dir/depend:
	cd /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug /Users/hectorrmanrique/CLionProjects/Facial-features-extraction-and-comparison/cmake-build-debug/CMakeFiles/FeatureExtraction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FeatureExtraction.dir/depend

