set(EXTERNAL_DOWNLOAD_DIR "${RT_ROOT_DIR}/external/downloads")

file(MAKE_DIRECTORY ${EXTERNAL_DOWNLOAD_DIR})

function(CloneExternal name url)
    if(NOT EXISTS "${EXTERNAL_DOWNLOAD_DIR}/${name}")
        message(STATUS "Cloning ${name}...")
        execute_process(COMMAND "git" "clone" ${url} WORKING_DIRECTORY ${EXTERNAL_DOWNLOAD_DIR})
    endif()
    message(STATUS "${name} cloned")
endfunction()

function(DownloadExternal name url target md5)
    set(file_download_path "${EXTERNAL_DOWNLOAD_DIR}/${target}")
    if(NOT EXISTS ${file_download_path})
        message(STATUS "Downloading ${name}...")
        file(DOWNLOAD ${url} ${file_download_path} SHOW_PROGRESS EXPECTED_MD5 ${md5})
    endif()
    message(STATUS "${name} downloaded.")
    file(COPY ${file_download_path} DESTINATION ${EXTERNAL_PROJECT_DIR}/)
endfunction(DownloadExternal)

#########
# svpng #
#########
set(SVPNG_SRC_DIR "${EXTERNAL_DOWNLOAD_DIR}/svpng")
CloneExternal("svpng" "https://github.com/miloyip/svpng")
file(COPY "${SVPNG_SRC_DIR}/svpng.inc" DESTINATION ${DEPENDENCY_DIR}/include/)

#######
# stb #
#######
set(STB_IMG_SRC_DIR "${EXTERNAL_DOWNLOAD_DIR}/stb")
CloneExternal("stb" "https://github.com/nothings/stb")
file(COPY "${STB_IMG_SRC_DIR}/stb_image.h" DESTINATION ${DEPENDENCY_DIR}/include/)

##########
# images #
##########
set(IMG_DIR "${EXTERNAL_DOWNLOAD_DIR}/img")
file(MAKE_DIRECTORY "IMG_DIR")
DownloadExternal("earth" "https://eoimages.gsfc.nasa.gov/images/imagerecords/78000/78349/arctic_vir_2012147_lrg.jpg" "img/earth.jpg" "7D4D5C98A924C4411CCE09519FE61B80")
file(COPY "${IMG_DIR}/" DESTINATION ${DEPENDENCY_DIR}/img/)