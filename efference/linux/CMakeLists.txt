# Efference SDK — Linux (C++). Self-contained: this directory is its own CMake
# project. Build it directly:
#
#   ./build.sh                       # or:
#   cmake -S . -B build && cmake --build build
#
# Produces the `ef` library, the ef-info diagnostic, and (by default) the C++
# examples (whose sources are shared across languages under the repo's
# top-level examples/ directory). libusb-1.0 is the only external dependency.
cmake_minimum_required(VERSION 3.16)
project(efference_sdk_linux VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EF_BUILD_EXAMPLES "Build the example programs" ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED IMPORTED_TARGET libusb-1.0)

# ---- the SDK library -------------------------------------------------------
add_library(ef
    src/device.cpp
    src/usb_transport.cpp
)
add_library(ef::ef ALIAS ef)

target_include_directories(ef
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(ef PRIVATE PkgConfig::LIBUSB)
target_compile_features(ef PUBLIC cxx_std_17)
target_compile_options(ef PRIVATE -Wall -Wextra)

# ---- ef-info diagnostic ----------------------------------------------------
add_executable(ef-info tools/ef_info.cpp)
target_link_libraries(ef-info PRIVATE ef::ef)

# ---- ef-config: list caps / set the session configuration -------------------
add_executable(ef-config tools/ef_config.cpp)
target_link_libraries(ef-config PRIVATE ef::ef)

# ---- ef-stream: capture the MCAP data plane to a .mcap (M3) -----------------
add_executable(ef-stream tools/ef_stream.cpp)
target_link_libraries(ef-stream PRIVATE ef::ef)

# ---- ef-view: live viewer (video + IMU). OPTIONAL — only built when SDL2 +
#      ffmpeg (libavcodec/libavutil) are present, so the core SDK stays
#      libusb-only. Install: apt install libsdl2-dev libavcodec-dev libavutil-dev
pkg_check_modules(SDL2 IMPORTED_TARGET sdl2)
pkg_check_modules(AVCODEC IMPORTED_TARGET libavcodec libavutil libswscale)
if(SDL2_FOUND AND AVCODEC_FOUND)
    add_executable(ef-view tools/ef_view.cpp)
    target_link_libraries(ef-view PRIVATE ef::ef PkgConfig::SDL2 PkgConfig::AVCODEC)
    message(STATUS "ef-view: enabled (SDL2 + libavcodec found)")
else()
    message(STATUS "ef-view: skipped (need libsdl2-dev + libavcodec-dev/libavutil-dev)")
endif()

# ---- examples (shared sources live at the repo-level examples/ dir) ---------
if(EF_BUILD_EXAMPLES)
    set(EX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../examples)
    add_executable(example1_serial_number
        "${EX_DIR}/wired/example 1 - serial number/cpp/main.cpp")
    target_link_libraries(example1_serial_number PRIVATE ef::ef)
endif()

# ---- install ---------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS ef
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
