cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE Release)
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_genmsg()

#eigen stuff
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})


set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

# I COMMENTED THESE TWO OUT NOW
#####set (CMAKE_C_FLAGS   "-Ofast -fopenmp" )
#####set (CMAKE_CXX_FLAGS "-Ofast -fopenmp" )

#set (CMAKE_CXX_FLAGS   "-march=atom -mtune=atom -Ofast " )
#set (CMAKE_CXX_FLAGS "-msse3 -ip  -no-prec-div         -parallel -O3 -fPIC" )

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
## Declare a cpp executable
include_directories(include/AprilTags)
file(GLOB libsrc
    "src/*.cc"
)

rosbuild_add_executable(ros_april
                        src/ros_april.cpp
                        src/kf_generic.cpp
                        ${libsrc})

#rosbuild_add_executable(ros_april_kf_test
#                        src/april_kf_main.cpp
#                        src/kf_generic.cpp)

target_link_libraries(ros_april)
