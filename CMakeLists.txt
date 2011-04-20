cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

# project name
project(object_attention)
list(APPEND CMAKE_MODULE_PATH ${object_attention_SOURCE_DIR})

# LIBRARIES

find_package(OpenNI)
include_directories(${OPENNI_INCLUDE_DIRS})

SET(ENV{OpenCV_DIR} "/home/dedan/Downloads/opencv/build/")
find_package(OpenCV REQUIRED)

include_directories(/usr/local/include/siftfast/)


# make and link executable
add_executable(obj_att main.cpp)
target_link_libraries(obj_att ${OpenCV_LIBS} /usr/local/lib/libsiftfast.so)
