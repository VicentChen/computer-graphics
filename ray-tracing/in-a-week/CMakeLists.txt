include_directories(${DEPENDENCY_DIR}/include)
include_directories("include/")

file(GLOB IAW_SRC "src/*.cpp")
file(GLOB IAW_INC "include/*.h")

# add_executable(in-a-week ${IAW_INC} ${IAW_SRC})
