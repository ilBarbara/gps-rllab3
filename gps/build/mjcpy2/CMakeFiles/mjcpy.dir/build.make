# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build

# Include any dependencies generated for this target.
include mjcpy2/CMakeFiles/mjcpy.dir/depend.make

# Include the progress variables for this target.
include mjcpy2/CMakeFiles/mjcpy.dir/progress.make

# Include the compile flags for this target's objects.
include mjcpy2/CMakeFiles/mjcpy.dir/flags.make

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o: mjcpy2/CMakeFiles/mjcpy.dir/flags.make
mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o: /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mjcpy2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mjcpy.dir/mjcpy2.cpp.o -c /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mjcpy2.cpp

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mjcpy.dir/mjcpy2.cpp.i"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mjcpy2.cpp > CMakeFiles/mjcpy.dir/mjcpy2.cpp.i

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mjcpy.dir/mjcpy2.cpp.s"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mjcpy2.cpp -o CMakeFiles/mjcpy.dir/mjcpy2.cpp.s

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.requires:

.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.requires

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.provides: mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.requires
	$(MAKE) -f mjcpy2/CMakeFiles/mjcpy.dir/build.make mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.provides.build
.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.provides

mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.provides.build: mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o


mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o: mjcpy2/CMakeFiles/mjcpy.dir/flags.make
mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o: /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mujoco_osg_viewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o -c /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mujoco_osg_viewer.cpp

mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.i"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mujoco_osg_viewer.cpp > CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.i

mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.s"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2/mujoco_osg_viewer.cpp -o CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.s

mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.requires:

.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.requires

mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.provides: mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.requires
	$(MAKE) -f mjcpy2/CMakeFiles/mjcpy.dir/build.make mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.provides.build
.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.provides

mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.provides.build: mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o


# Object files for target mjcpy
mjcpy_OBJECTS = \
"CMakeFiles/mjcpy.dir/mjcpy2.cpp.o" \
"CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o"

# External object files for target mjcpy
mjcpy_EXTERNAL_OBJECTS =

lib/mjcpy.so: mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o
lib/mjcpy.so: mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o
lib/mjcpy.so: mjcpy2/CMakeFiles/mjcpy.dir/build.make
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libboost_python.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
lib/mjcpy.so: /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjpro/bin/libmujoco131.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libosg.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libosgViewer.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libOpenThreads.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libosgGA.so
lib/mjcpy.so: lib/libboost_numpy.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libboost_python.so
lib/mjcpy.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
lib/mjcpy.so: mjcpy2/CMakeFiles/mjcpy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../lib/mjcpy.so"
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mjcpy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mjcpy2/CMakeFiles/mjcpy.dir/build: lib/mjcpy.so

.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/build

mjcpy2/CMakeFiles/mjcpy.dir/requires: mjcpy2/CMakeFiles/mjcpy.dir/mjcpy2.cpp.o.requires
mjcpy2/CMakeFiles/mjcpy.dir/requires: mjcpy2/CMakeFiles/mjcpy.dir/mujoco_osg_viewer.cpp.o.requires

.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/requires

mjcpy2/CMakeFiles/mjcpy.dir/clean:
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 && $(CMAKE_COMMAND) -P CMakeFiles/mjcpy.dir/cmake_clean.cmake
.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/clean

mjcpy2/CMakeFiles/mjcpy.dir/depend:
	cd /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/src/3rdparty/mjcpy2 /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2 /home2/wsdm/gyy/sjh_project/gps-tf1.3.0/build/mjcpy2/CMakeFiles/mjcpy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mjcpy2/CMakeFiles/mjcpy.dir/depend
