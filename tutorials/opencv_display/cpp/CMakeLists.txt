# Standalone build for the opencv_display tutorial. Needs OpenCV in addition to
# the SDK. Resolve the SDK against the in-repo build tree (source <repo>/env.sh
# first, or pass -DCMAKE_PREFIX_PATH=<repo>/sdk/linux/build). See tutorials/README.md.
#
# Every .cpp here builds to its own binary named after the file.
cmake_minimum_required(VERSION 3.16)
project(ef_tutorial_opencv_display CXX)
find_package(ef REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
file(GLOB TUT_SRCS CONFIGURE_DEPENDS "*.cpp")
foreach(src ${TUT_SRCS})
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE ef::ef ${OpenCV_LIBS})
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()
