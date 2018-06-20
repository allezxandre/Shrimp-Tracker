CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(ShrimpTracker)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES(/usr/local/include)

# eigen 2 or 3
find_path(EIGEN_INCLUDE_DIR Eigen/Core
	/usr/local/include/eigen3
	/usr/local/include/eigen2
	/usr/local/include/eigen
	/usr/include/eigen3
	/usr/include/eigen2
	/usr/include/eigen
	/opt/ros/diamondback/stacks/geometry/eigen/include
)

# include all libs so far
include_directories(${EIGEN_INCLUDE_DIR})


find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv)
FIND_PACKAGE(Ceres REQUIRED)
INCLUDE_DIRECTORIES(${CERES_INCLUDES} ${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${CERES_LIBS})

add_definitions(-Wall -g -ggdb -O3)

ADD_EXECUTABLE(ceres_shrimp src/ceres_shrimp.cpp)
TARGET_LINK_LIBRARIES(ceres_shrimp ${CERES_LIBRARIES} gflags ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(elliptic_shrimp src/elliptic_shrimp.cpp)
TARGET_LINK_LIBRARIES(elliptic_shrimp ${CERES_LIBRARIES} gflags ${OpenCV_LIBRARIES})

