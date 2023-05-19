cmake_minimum_required(VERSION 3.24)
project(MPI_simple)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(MPI_simple main.cpp)

target_link_libraries(MPI_simple ${MPI_CXX_LIBRARIES})