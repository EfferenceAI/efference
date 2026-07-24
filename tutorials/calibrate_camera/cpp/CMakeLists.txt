# Standalone build for the calibrate_camera tutorial. Needs OpenCV and the
# prebuilt camera-calibration solver (ef::camera_cal), available only where the
# per-arch libcamera_cal.so ships (sdk/linux/lib/<arch>/). Resolve the SDK against
# the in-repo build tree (source <repo>/env.sh first). See tutorials/README.md.
#
# Every .cpp here builds to its own binary named after the file.
cmake_minimum_required(VERSION 3.16)
project(ef_tutorial_calibrate_camera CXX)
find_package(ef REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
if(NOT TARGET ef::camera_cal)
    message(FATAL_ERROR "calibrate_camera needs libcamera_cal.so, absent for this architecture.")
endif()
file(GLOB TUT_SRCS CONFIGURE_DEPENDS "*.cpp")
foreach(src ${TUT_SRCS})
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE ef::ef ef::camera_cal ${OpenCV_LIBS})
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
endforeach()
