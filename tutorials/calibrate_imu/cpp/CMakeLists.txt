# Standalone build for the calibrate_imu tutorial. Needs the prebuilt IMU
# calibration solver (ef::imu_cal), available only where the per-arch
# libimu_cal.so ships (sdk/linux/lib/<arch>/). Resolve the SDK against the in-repo
# build tree (source <repo>/env.sh first). See tutorials/README.md.
#
# Every .cpp here builds to its own binary named after the file.
cmake_minimum_required(VERSION 3.16)
project(ef_tutorial_calibrate_imu CXX)
find_package(ef REQUIRED)
if(NOT TARGET ef::imu_cal)
    message(FATAL_ERROR "calibrate_imu needs libimu_cal.so, absent for this architecture.")
endif()
file(GLOB TUT_SRCS CONFIGURE_DEPENDS "*.cpp")
foreach(src ${TUT_SRCS})
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE ef::ef ef::imu_cal)
endforeach()
