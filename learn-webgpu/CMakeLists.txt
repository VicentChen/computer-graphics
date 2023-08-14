cmake_minimum_required(VERSION 3.25)

project(
	LearnWebGPU
	LANGUAGES CXX C
)

set(
	CMAKE_CXX_STANDARD 17
	COMPILE_WARNING_AS_ERROR ON
)

add_subdirectory(glfw)
add_subdirectory(webgpu)
add_subdirectory(glfw3webgpu)
include_directories(webgpu/include)
include_directories(glfw3webgpu)

file(GLOB WebGPU_SRC *.cpp *.h)
add_executable(LearnWebGPU ${WebGPU_SRC})
target_link_libraries(LearnWebGPU PRIVATE glfw webgpu glfw3webgpu)

target_copy_webgpu_binaries(LearnWebGPU)