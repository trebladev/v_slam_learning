# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xuan/slam14

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xuan/slam14/build

# Include any dependencies generated for this target.
include CMakeFiles/stereoVision.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereoVision.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereoVision.dir/flags.make

CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o: CMakeFiles/stereoVision.dir/flags.make
CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o: ../ch5/stereoVision.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xuan/slam14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o -c /home/xuan/slam14/ch5/stereoVision.cpp

CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xuan/slam14/ch5/stereoVision.cpp > CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.i

CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xuan/slam14/ch5/stereoVision.cpp -o CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.s

# Object files for target stereoVision
stereoVision_OBJECTS = \
"CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o"

# External object files for target stereoVision
stereoVision_EXTERNAL_OBJECTS =

stereoVision: CMakeFiles/stereoVision.dir/ch5/stereoVision.cpp.o
stereoVision: CMakeFiles/stereoVision.dir/build.make
stereoVision: /usr/local/lib/libopencv_dnn.so.4.2.0
stereoVision: /usr/local/lib/libopencv_gapi.so.4.2.0
stereoVision: /usr/local/lib/libopencv_highgui.so.4.2.0
stereoVision: /usr/local/lib/libopencv_ml.so.4.2.0
stereoVision: /usr/local/lib/libopencv_objdetect.so.4.2.0
stereoVision: /usr/local/lib/libopencv_photo.so.4.2.0
stereoVision: /usr/local/lib/libopencv_stitching.so.4.2.0
stereoVision: /usr/local/lib/libopencv_video.so.4.2.0
stereoVision: /usr/local/lib/libopencv_videoio.so.4.2.0
stereoVision: /usr/local/lib/libpango_glgeometry.so
stereoVision: /usr/local/lib/libpango_plot.so
stereoVision: /usr/local/lib/libpango_python.so
stereoVision: /usr/local/lib/libpango_scene.so
stereoVision: /usr/local/lib/libpango_tools.so
stereoVision: /usr/local/lib/libpango_video.so
stereoVision: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
stereoVision: /usr/local/lib/libopencv_calib3d.so.4.2.0
stereoVision: /usr/local/lib/libopencv_features2d.so.4.2.0
stereoVision: /usr/local/lib/libopencv_flann.so.4.2.0
stereoVision: /usr/local/lib/libopencv_imgproc.so.4.2.0
stereoVision: /usr/local/lib/libopencv_core.so.4.2.0
stereoVision: /usr/local/lib/libpango_geometry.so
stereoVision: /usr/local/lib/libtinyobj.so
stereoVision: /usr/local/lib/libpango_display.so
stereoVision: /usr/local/lib/libpango_vars.so
stereoVision: /usr/local/lib/libpango_windowing.so
stereoVision: /usr/local/lib/libpango_opengl.so
stereoVision: /usr/lib/x86_64-linux-gnu/libGLEW.so
stereoVision: /usr/lib/x86_64-linux-gnu/libOpenGL.so
stereoVision: /usr/lib/x86_64-linux-gnu/libGLX.so
stereoVision: /usr/lib/x86_64-linux-gnu/libGLU.so
stereoVision: /usr/local/lib/libpango_image.so
stereoVision: /usr/local/lib/libpango_packetstream.so
stereoVision: /usr/local/lib/libpango_core.so
stereoVision: CMakeFiles/stereoVision.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xuan/slam14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stereoVision"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereoVision.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereoVision.dir/build: stereoVision

.PHONY : CMakeFiles/stereoVision.dir/build

CMakeFiles/stereoVision.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereoVision.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereoVision.dir/clean

CMakeFiles/stereoVision.dir/depend:
	cd /home/xuan/slam14/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xuan/slam14 /home/xuan/slam14 /home/xuan/slam14/build /home/xuan/slam14/build /home/xuan/slam14/build/CMakeFiles/stereoVision.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereoVision.dir/depend

