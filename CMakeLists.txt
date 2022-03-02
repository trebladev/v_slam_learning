cmake_minimum_required( VERSION 2.8 )

project(learn_eigen)

include_directories("/usr/include/eigen3")

add_executable(learn_eigen ch3/helloeigen.cpp)
