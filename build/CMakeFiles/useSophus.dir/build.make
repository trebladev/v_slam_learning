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
include CMakeFiles/useSophus.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/useSophus.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/useSophus.dir/flags.make

CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o: CMakeFiles/useSophus.dir/flags.make
CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o: ../ch4/useSophus.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xuan/slam14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o -c /home/xuan/slam14/ch4/useSophus.cpp

CMakeFiles/useSophus.dir/ch4/useSophus.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/useSophus.dir/ch4/useSophus.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xuan/slam14/ch4/useSophus.cpp > CMakeFiles/useSophus.dir/ch4/useSophus.cpp.i

CMakeFiles/useSophus.dir/ch4/useSophus.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/useSophus.dir/ch4/useSophus.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xuan/slam14/ch4/useSophus.cpp -o CMakeFiles/useSophus.dir/ch4/useSophus.cpp.s

# Object files for target useSophus
useSophus_OBJECTS = \
"CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o"

# External object files for target useSophus
useSophus_EXTERNAL_OBJECTS =

useSophus: CMakeFiles/useSophus.dir/ch4/useSophus.cpp.o
useSophus: CMakeFiles/useSophus.dir/build.make
useSophus: CMakeFiles/useSophus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xuan/slam14/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable useSophus"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/useSophus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/useSophus.dir/build: useSophus

.PHONY : CMakeFiles/useSophus.dir/build

CMakeFiles/useSophus.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/useSophus.dir/cmake_clean.cmake
.PHONY : CMakeFiles/useSophus.dir/clean

CMakeFiles/useSophus.dir/depend:
	cd /home/xuan/slam14/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xuan/slam14 /home/xuan/slam14 /home/xuan/slam14/build /home/xuan/slam14/build /home/xuan/slam14/build/CMakeFiles/useSophus.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/useSophus.dir/depend
