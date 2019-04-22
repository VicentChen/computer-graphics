set(EXTERNAL_PROJECT_DIR "${PROJECT_BINARY_DIR}/External")
set(EXTERNAL_DOWNLOAD_DIR "${RTR_ROOT_DIR}/External/Downloads")
set(EXTERNAL_BINARY_DIR "${PROJECT_BINARY_DIR}/Dependencies")

function(DownloadExternal name url target md5)
    set(file_download_path "${EXTERNAL_DOWNLOAD_DIR}/${target}")
    if(NOT EXISTS ${file_download_path})
        message(STATUS "Downloading ${name}...")
        file(DOWNLOAD ${url} ${file_download_path} SHOW_PROGRESS EXPECTED_MD5 ${md5})
    endif()
    message(STATUS "${name} downloaded.")
    file(COPY ${file_download_path} DESTINATION ${EXTERNAL_PROJECT_DIR}/)
endfunction(DownloadExternal)

# TODO: change to fit different build type
set(BUILD_COMMAND_OPTS --target install --config ${CMAKE_BUILD_TYPE})
file(MAKE_DIRECTORY ${EXTERNAL_DOWNLOAD_DIR})

###############
# Google Test #
###############
set(GTEST_PROJECT_DIR "${EXTERNAL_PROJECT_DIR}/googletest-release-1.8.1")
DownloadExternal("Google Test" "https://github.com/google/googletest/archive/release-1.8.1.tar.gz" "googletest-release-1.8.1.tar.gz" "2E6FBEB6A91310A16EFE181886C59596")
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf googletest-release-1.8.1.tar.gz  WORKING_DIRECTORY ${EXTERNAL_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} 
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BINARY_DIR}
    -DBUILD_GMOCK=OFF
    -Dgtest_force_shared_crt=ON
    -G ${CMAKE_GENERATOR}
    -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    ${GTEST_PROJECT_DIR}
    WORKING_DIRECTORY ${GTEST_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} --build ${GTEST_PROJECT_DIR} --target install --config RelWithDebInfo)
execute_process(COMMAND ${CMAKE_COMMAND} --build ${GTEST_PROJECT_DIR} --target install --config debug)

########
# GLFW #
########
set(GLFW_PROJECT_DIR "${EXTERNAL_PROJECT_DIR}/glfw-3.2.1")
DownloadExternal("GLFW" "https://github.com/glfw/glfw/archive/3.2.1.tar.gz" "glfw-3.2.1.tar.gz" "91B8250B6EDCC26C9F5205555070A504")
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf glfw-3.2.1.tar.gz  WORKING_DIRECTORY ${EXTERNAL_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} 
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BINARY_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DGLFW_BUILD_EXAMPLES=OFF
    -DGLFW_BUILD_TESTS=OFF
    -DGLFW_BUILD_DOCS=OFF
    -G ${CMAKE_GENERATOR}
    -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    ${GLFW_PROJECT_DIR}
    WORKING_DIRECTORY ${GLFW_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} --build ${GLFW_PROJECT_DIR} ${BUILD_COMMAND_OPTS})

########
# GLAD #
########
set(GLAD_PROJECT_DIR "${EXTERNAL_PROJECT_DIR}/glad")
file(COPY ${RTR_ROOT_DIR}/External/glad/ DESTINATION ${GLAD_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} 
    -DCMAKE_INSTALL_PREFIX=${EXTERNAL_BINARY_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -G ${CMAKE_GENERATOR}
    -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
    ${GLAD_PROJECT_DIR}
    WORKING_DIRECTORY ${GLAD_PROJECT_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} --build ${GLAD_PROJECT_DIR} ${BUILD_COMMAND_OPTS})