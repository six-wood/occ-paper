cmake_minimum_required(VERSION 3.3)
project(mapping)

set (CMAKE_CXX_STANDARD 14)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(pybind11)
include_directories(${pybind11_INCLUDE_DIRS})
pybind11_add_module(mapping mapping.cpp)
