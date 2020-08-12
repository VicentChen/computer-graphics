cmake_minimum_required(VERSION 3.9)

project(PBR)

set(PBR_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PBR_BINARY_DIR ${CMAKE_BINARY_DIR}/Products)
set(DEPENDENCY_DIR ${CMAKE_BINARY_DIR}/Dependencies)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

add_subdirectory(External)
add_subdirectory(PBR)
