cmake_minimum_required(VERSION 3.24)
project(open_mp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

add_executable(open_mp main.cpp)
