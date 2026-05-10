cmake_minimum_required(VERSION 3.15)
project(neotorch LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_index src/neotorch/_index.cpp)
target_compile_features(_index PRIVATE cxx_std_17)

pybind11_add_module(_data src/neotorch/_data.cpp)
target_compile_features(_data PRIVATE cxx_std_17)

pybind11_add_module(_tensor src/neotorch/_tensor.cpp)
target_compile_features(_tensor PRIVATE cxx_std_17)

install(TARGETS _index _data _tensor DESTINATION neotorch)
