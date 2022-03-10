cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_FLAGS "-std=c++11")

project(learn_eigen)

include_directories("/usr/include/eigen3")

set(Sophus_DIR "/home/xuan/Sophus/build")
find_package( Sophus REQUIRED)

include_directories(${Sophus_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(learn_eigen ch3/helloeigen.cpp)

add_executable(useGeometry ch3/useGeometry.cpp)

add_executable(TransformExanple ch3/coordinateTransform.cpp)

add_executable(useSophus ch4/useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)

add_executable(imageBasics ch5/imageBasics.cpp)
target_link_libraries(imageBasics ${OpenCV_LIBS})