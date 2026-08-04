cmake_minimum_required(VERSION 3.15)
project(strideweave LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

file(
    GLOB STRIDEWEAVE_CPU_OPERATION_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/strideweave/carriers/cpu/native/ops/*.cpp"
)

foreach(source IN LISTS STRIDEWEAVE_CPU_OPERATION_SOURCES)
    file(RELATIVE_PATH relative_source "${CMAKE_CURRENT_SOURCE_DIR}" "${source}")
    set_property(
        SOURCE "${source}"
        APPEND
        PROPERTY COMPILE_DEFINITIONS
                 "STRIDEWEAVE_KERNEL_SOURCE=\"${relative_source}\""
    )
endforeach()

pybind11_add_module(
    _carrier
    src/strideweave/core/native/_carrier.cpp
    src/strideweave/carriers/cpu/native/_cpu.cpp
    src/strideweave/carriers/cpu/native/_cpu_registry.cpp
    ${STRIDEWEAVE_CPU_OPERATION_SOURCES}
)
target_compile_features(_carrier PRIVATE cxx_std_17)
target_include_directories(
    _carrier
    PRIVATE
    src/strideweave/core/native
    src/strideweave/carriers/cpu/native
)

include(TestBigEndian)
test_big_endian(STRIDEWEAVE_BIG_ENDIAN)
if(STRIDEWEAVE_BIG_ENDIAN)
    set(STRIDEWEAVE_ENDIANNESS big)
else()
    set(STRIDEWEAVE_ENDIANNESS little)
endif()
math(EXPR STRIDEWEAVE_POINTER_BITS "${CMAKE_SIZEOF_VOID_P} * 8")

set(
    STRIDEWEAVE_PROVENANCE_MANIFEST
    "${CMAKE_CURRENT_BINARY_DIR}/generated/strideweave/verification/_native_provenance.json"
)
set(STRIDEWEAVE_PROVENANCE_SOURCE_ARGS ${STRIDEWEAVE_CPU_OPERATION_SOURCES})
list(TRANSFORM STRIDEWEAVE_PROVENANCE_SOURCE_ARGS PREPEND "--source=")

add_custom_command(
    OUTPUT "${STRIDEWEAVE_PROVENANCE_MANIFEST}"
    COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_provenance.py"
        "--project-root=${CMAKE_CURRENT_SOURCE_DIR}"
        "--compile-commands=${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json"
        "--artifact=$<TARGET_FILE:_carrier>"
        "--output=${STRIDEWEAVE_PROVENANCE_MANIFEST}"
        "--cache-dir=${CMAKE_CURRENT_BINARY_DIR}/generated/provenance-cache"
        "--compiler=${CMAKE_CXX_COMPILER}"
        "--compiler-id=${CMAKE_CXX_COMPILER_ID}"
        "--cmake-version=${CMAKE_VERSION}"
        "--architecture=${CMAKE_SYSTEM_PROCESSOR}"
        "--operating-system=${CMAKE_SYSTEM_NAME}"
        "--endianness=${STRIDEWEAVE_ENDIANNESS}"
        "--pointer-bits=${STRIDEWEAVE_POINTER_BITS}"
        ${STRIDEWEAVE_PROVENANCE_SOURCE_ARGS}
        "--build-input=${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "--build-input=${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
    DEPENDS
        _carrier
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_provenance.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
        ${STRIDEWEAVE_CPU_OPERATION_SOURCES}
    VERBATIM
    COMMAND_EXPAND_LISTS
)
add_custom_target(
    strideweave_kernel_provenance ALL
    DEPENDS "${STRIDEWEAVE_PROVENANCE_MANIFEST}"
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
install(
    FILES "${STRIDEWEAVE_PROVENANCE_MANIFEST}"
    DESTINATION strideweave/verification
)
