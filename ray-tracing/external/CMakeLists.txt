set(EXTERNAL_DOWNLOAD_DIR "${RT_ROOT_DIR}/external/downloads")

file(MAKE_DIRECTORY ${EXTERNAL_DOWNLOAD_DIR})

function(CloneExternal name url)
    if(NOT EXISTS "${EXTERNAL_DOWNLOAD_DIR}/${name}")
        message(STATUS "Cloning ${name}...")
        execute_process(COMMAND "git" "clone" ${url} WORKING_DIRECTORY ${EXTERNAL_DOWNLOAD_DIR})
    endif()
    message(STATUS "${name} cloned")
endfunction()

#########
# svpng #
#########
set(SVPNG_SRC_DIR "${EXTERNAL_DOWNLOAD_DIR}/svpng")
CloneExternal("svpng" "https://github.com/miloyip/svpng")
file(COPY "${SVPNG_SRC_DIR}/svpng.inc" DESTINATION ${DEPENDENCY_DIR}/include/)