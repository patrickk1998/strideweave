cmake_minimum_required(VERSION 3.15)
project(strideweave LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

option(
    STRIDEWEAVE_STRICT_WARNINGS
    "Treat warnings in StrideWeave native targets as errors"
    OFF
)
option(
    STRIDEWEAVE_SANITIZERS
    "Build StrideWeave native targets with address and undefined sanitizers"
    OFF
)

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

set(
    STRIDEWEAVE_NATIVE_TARGETS
    _index
    _einops
    _carrier
    _operation
    _tensor
    _move
)

if(STRIDEWEAVE_STRICT_WARNINGS)
    if(MSVC)
        foreach(target IN LISTS STRIDEWEAVE_NATIVE_TARGETS)
            target_compile_options(${target} PRIVATE /W4 /WX)
        endforeach()
    else()
        foreach(target IN LISTS STRIDEWEAVE_NATIVE_TARGETS)
            target_compile_options(
                ${target}
                PRIVATE
                -Wall
                -Wextra
                -Wshadow
                -Wconversion
                -Wsign-conversion
                -Wnon-virtual-dtor
                -Werror
            )
        endforeach()
    endif()
endif()

if(STRIDEWEAVE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(FATAL_ERROR "StrideWeave sanitizers require Clang or GCC")
    endif()
    foreach(target IN LISTS STRIDEWEAVE_NATIVE_TARGETS)
        target_compile_options(
            ${target} PRIVATE -O1 -g -fsanitize=address,undefined
                              -fno-omit-frame-pointer
        )
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

install(TARGETS _index _einops _carrier _operation _tensor _move DESTINATION strideweave)
