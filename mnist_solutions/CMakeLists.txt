cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(mnist_solutions CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include(${PROJECT_SOURCE_DIR}/../cmake/conan.cmake)
conan_cmake_run(
  CONANFILE conanfile.txt
  BASIC_SETUP
  CMAKE_TARGETS # individual targets to link to
  BUILD
    missing 
)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    include/*.h
    demo/*.h
)
file(GLOB SRCS
    src/*.cpp
    demo/helpers.cpp
)

add_subdirectory(logistic_regression_solution)
add_subdirectory(catboost_solution)
add_subdirectory(mlp_solution)
add_subdirectory(tensorflow_solution)

