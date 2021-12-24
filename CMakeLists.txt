cmake_minimum_required(VERSION 3.16)

# for armhf
set(CROSS_COMPILE_TOOL /media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/rk1109_rk1126_linux/prebuilts/gcc/linux-x86/arm/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf)
set(CMAKE_C_COMPILER ${CROSS_COMPILE_TOOL}-gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE_TOOL}-g++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl")
project(LightTrack)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv4)
include_directories(${OpenCV_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(LightTrack 
    src/main.cpp 
    src/LightTrack.cpp 
    )

target_link_libraries(LightTrack -lopencv_world -lrknn_api -lade -llibjasper -llibjpeg-turbo -llibpng -llibprotobuf -llibtiff -llibwebp -lquirc -ltegra_hal -lzlib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswresample -lswscale)

## install target
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/lighttrack_demo)
install(TARGETS LightTrack RUNTIME DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(FILES lib/librknn_api.so DESTINATION ./)
install(FILES 01.mp4 DESTINATION ./)
