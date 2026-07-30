cmake_minimum_required(VERSION 3.16)
project(flash_kda_ascend LANGUAGES CXX ASC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN SDK path
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

# Architecture: 2201 for Atlas A2/A3, 3510 for Ascend 950
set(CATLASS_ARCH "2201" CACHE STRING "Target architecture (2201 or 3510)")

# catlass submodule
add_subdirectory(catlass)

# AscendC kernel sources
set(KERNEL_SOURCES
    src/flash_kda.cpp
)

# Kernel operator library (compiled as ASC language)
add_library(flash_kda_kernel STATIC
    src/fwd_kernel1.cpp
    src/fwd_kernel2.cpp
)
set_target_properties(flash_kda_kernel PROPERTIES LANGUAGE ASC)
target_include_directories(flash_kda_kernel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/catlass/include
)
target_compile_definitions(flash_kda_kernel PRIVATE
    CATLASS_ARCH=${CATLASS_ARCH}
)

# Host-side library with pybind11
if(BUILD_PYBIND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND Python3::Interpreter -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(_C src/flash_kda.cpp)
    target_include_directories(_C PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/catlass/include
    )
    target_link_libraries(_C PRIVATE flash_kda_kernel)
    target_compile_definitions(_C PRIVATE CATLASS_ARCH=${CATLASS_ARCH})
endif()
