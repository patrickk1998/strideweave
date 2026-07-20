cmake_minimum_required(VERSION 3.15)
project(strideweave LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_index src/strideweave/core/native/_index.cpp)
target_compile_features(_index PRIVATE cxx_std_17)

pybind11_add_module(_einops src/strideweave/einops/native/_einops.cpp)
target_compile_features(_einops PRIVATE cxx_std_17)

pybind11_add_module(
    _carrier
    src/strideweave/core/native/_carrier.cpp
    src/strideweave/carriers/cpu/native/_cpu.cpp
)
target_compile_features(_carrier PRIVATE cxx_std_17)
target_include_directories(
    _carrier
    PRIVATE
    src/strideweave/core/native
    src/strideweave/carriers/cpu/native
)

pybind11_add_module(_operation src/strideweave/core/native/_operation.cpp)
target_compile_features(_operation PRIVATE cxx_std_17)

pybind11_add_module(_tensor src/strideweave/core/native/_tensor.cpp)
target_compile_features(_tensor PRIVATE cxx_std_17)

pybind11_add_module(_move src/strideweave/carriers/move/native/_move.cpp)
target_compile_features(_move PRIVATE cxx_std_17)

install(TARGETS _index _einops _carrier _operation _tensor _move DESTINATION strideweave)
