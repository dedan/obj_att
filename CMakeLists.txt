cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

# project name
project(object_attention)
list(APPEND CMAKE_MODULE_PATH ${object_attention_SOURCE_DIR})

# LIBRARIES
find_package(PCL 1.0 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenNI)
include_directories(${OPENNI_INCLUDE_DIRS})

find_package(wxWidgets)
include(${wxWidgets_USE_FILE})
include_directories(${wxWidgets_INCLUDE_DIRS})

find_package(VTK)
include(${VTK_USE_FILE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

FIND_PACKAGE( OpenCV REQUIRED )


# make and link executable
add_executable(obj_att main.cpp)
target_link_libraries(obj_att ${PCL_LIBRARIES})
MESSAGE(${PCL_LIBRARIES})


add_executable(cv_test cv_test.cpp)
target_link_libraries(cv_test ${OpenCV_LIBS} ${PCL_LIBRARIES})
