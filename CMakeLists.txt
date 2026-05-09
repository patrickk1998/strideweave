cmake_minimum_required(VERSION 3.15)
project(neotorch LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_index src/neotorch/_index.cpp)
target_compile_features(_index PRIVATE cxx_std_17)

install(TARGETS _index DESTINATION neotorch)
