cmake_minimum_required( VERSION 3.17 )

set(CMAKE_CXX_FLAGS "-std=c++14")

project(learn_eigen)

include_directories("/usr/include/eigen3")

set(Sophus_DIR "/home/xuan/Sophus/build")
find_package( Sophus REQUIRED)

find_package(Pangolin REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${Ceres_INCLUDE_DIRS})

add_executable(learn_eigen ch3/helloeigen.cpp)

add_executable(useGeometry ch3/useGeometry.cpp)

add_executable(TransformExanple ch3/coordinateTransform.cpp)

add_executable(useSophus ch4/useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)

add_executable(imageBasics ch5/imageBasics.cpp)
add_executable(undistortImage ch5/undistortImage.cpp)
target_link_libraries(undistortImage ${OpenCV_LIBS})
target_link_libraries(imageBasics ${OpenCV_LIBS})

add_executable(stereoVision ch5/stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(gaussNewton ch6/gaussNewton.cpp)
target_link_libraries(gaussNewton ${OpenCV_LIBS})
