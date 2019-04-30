include_directories(${DEPENDENCY_DIR}/include)
include_directories("include/")

file(GLOB TNW_SRC "src/*.cpp")
file(GLOB TNW_INC "include/*.h")

add_executable(the-next-week ${TNW_INC} ${TNW_SRC})
