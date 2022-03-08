cmake_minimum_required( VERSION 2.8 )

project(learn_eigen)

include_directories("/usr/include/eigen3")

set(Sophus_DIR "/home/xuan/Sophus/build")
find_package( Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(learn_eigen ch3/helloeigen.cpp)

add_executable(useGeometry ch3/useGeometry.cpp)

add_executable(TransformExanple ch3/coordinateTransform.cpp)

add_executable(useSophus ch4/useSophus.cpp)
target_link_libraries(useSophus Sophus::Sophus)