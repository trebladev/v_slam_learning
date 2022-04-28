cmake_minimum_required( VERSION 3.17 )


add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")
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

include_directories(common)

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

add_executable(ceresCurveFitting ch6/ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

add_executable(orb_cv ch7/orb_cv.cpp common/Timing.h)
target_link_libraries(orb_cv ${OpenCV_LIBS})

#add_executable(orb_self ch7/orb_self.cpp common/Timing.h)
#target_link_libraries(orb_self ${OpenCV_LIBS})