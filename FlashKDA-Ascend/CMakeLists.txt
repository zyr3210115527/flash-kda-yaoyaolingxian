# FlashKDA-Ascend build.
#
# Modeled on catlass's own torch-extension template, which is the only
# reference in-tree that is known to compile:
#   catlass/.agents/skills/catlass-example-to-torch-intf/templates/CMakeLists.txt
#
# Three things the inherited CMakeLists got wrong and that are fixed here:
#   1. `find_package(ASC)` was missing. It must run *before* project(), or the
#      ASC language is unknown and configuration fails immediately.
#   2. catlass was pulled in with add_subdirectory(), which builds all ~70 of
#      its examples and its test tree. catlass is header-only for our purposes;
#      it belongs on the include path and nowhere else.
#   3. Nothing linked torch, torch_npu, or the Ascend runtime, so even a
#      successful compile could not have linked.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Run: source <cann>/set_env.sh")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

find_package(ASC REQUIRED)
project(flash_kda_ascend LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2201 = Atlas A2/A3 (Ascend 910B/910C), 3510 = Ascend 950
if(NOT DEFINED CATLASS_ARCH OR NOT CATLASS_ARCH)
  set(CATLASS_ARCH "2201")
endif()
message(STATUS "FlashKDA-Ascend: CATLASS_ARCH=${CATLASS_ARCH}")

add_compile_definitions(CATLASS_ARCH=${CATLASS_ARCH})
add_compile_options($<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-${CATLASS_ARCH}>)
add_compile_options(
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-macro-redefined>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-ignored-attributes>
  $<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-value>)
add_link_options($<$<COMPILE_LANGUAGE:ASC>:-Wno-unused-command-line-argument>)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}/torch/share/cmake")
find_package(Torch REQUIRED)

set(CATLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catlass/include)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CATLASS_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${Python3_SITELIB}/torch_npu/include
  ${ASCEND_HOME_PATH}/include)

link_directories(${ASCEND_HOME_PATH}/lib64 ${Python3_SITELIB}/torch_npu/lib)

# Device code. The .asc extension is what selects the Ascend compiler; the
# language property must be set per source file, not per target (the inherited
# version used set_target_properties(... LANGUAGE ASC), which does nothing).
set(FLASH_KDA_KERNEL_SOURCES
  src/fwd_kernel1.asc
  src/fwd_kernel2.asc)
set_source_files_properties(${FLASH_KDA_KERNEL_SOURCES} PROPERTIES LANGUAGE ASC)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(_C src/flash_kda.cpp ${FLASH_KDA_KERNEL_SOURCES})

target_link_libraries(_C PRIVATE
  ${TORCH_LIBRARIES}
  torch_npu
  ascendc_runtime
  runtime
  ascendcl
  tiling_api
  platform)

install(TARGETS _C
  LIBRARY DESTINATION .
  COMPONENT _python_extension_lib)
