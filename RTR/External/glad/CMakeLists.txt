project(glad)

set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")
set(SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c)
set(INC_DICT ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INSTALL_INC_DICT ${CMAKE_INSTALL_RPATH}/include)
set(INSTALL_LIB_DICT ${CMAKE_INSTALL_RPATH}/lib)
file(MAKE_DIRECTORY ${INSTALL_LIB_DICT} ${INSTALL_INC_DICT})

include_directories(${INC_DICT})
add_library(glad STATIC ${SRC_LIST})
install(TARGETS glad ARCHIVE DESTINATION ${INSTALL_LIB_DICT})
install(DIRECTORY ${INC_DICT}/ DESTINATION ${INSTALL_INC_DICT})