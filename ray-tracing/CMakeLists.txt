cmake_minimum_required(VERSION 3.9)

project(ray-tracing)

set(RT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEPENDENCY_DIR ${CMAKE_BINARY_DIR}/dependencies)

file(MAKE_DIRECTORY ${DEPENDENCY_DIR})
file(MAKE_DIRECTORY ${DEPENDENCY_DIR}/include)
file(MAKE_DIRECTORY ${DEPENDENCY_DIR}/lib)

add_subdirectory(external)
add_subdirectory(in-a-week)
add_subdirectory(the-next-week)
add_subdirectory(rest-of-life)