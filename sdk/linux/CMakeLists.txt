# Efference SDK — Linux (C++). Self-contained CMake project:
#
#   ./build.sh                       # or:
#   cmake -S . -B build && cmake --build build
#
# Produces the `ef` library, the command-line tools, and the examples.
# libusb-1.0 is the only required dependency.
cmake_minimum_required(VERSION 3.16)
project(efference_sdk_linux VERSION 1.0.0 LANGUAGES C CXX)

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
# Protobuf control plane: the wire is nanopb-encoded ef.proto (shared with the
# device). Compile the generated ef.pb.c + the vendored nanopb runtime in.
set(EF_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../proto)
set(EF_NANOPB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanopb)
add_library(ef
    src/device.cpp
    src/device_stream.cpp
    src/device_recording.cpp
    src/device_ota.cpp
    src/device_system.cpp
    src/enums.cpp
    src/mat.cpp
    src/usb_connection.cpp
    src/stream_assembler.cpp
    src/stream_reader.cpp
    src/udp_stream_reader.cpp
    src/mcap.cpp
    src/mcap_replay_reader.cpp
    ${EF_PROTO_DIR}/ef.pb.c
    ${EF_NANOPB_DIR}/pb_common.c
    ${EF_NANOPB_DIR}/pb_encode.c
    ${EF_NANOPB_DIR}/pb_decode.c
)
add_library(ef::ef ALIAS ef)

target_include_directories(ef
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/internal
            ${EF_PROTO_DIR} ${EF_NANOPB_DIR}
)
target_link_libraries(ef PRIVATE PkgConfig::LIBUSB)
target_compile_features(ef PUBLIC cxx_std_17)
target_compile_options(ef PRIVATE -Wall -Wextra)
# Static lib is linked into PIE executables; nanopb's message descriptors need
# position-independent code to avoid text-relocation errors.
set_target_properties(ef PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The isoc data-plane reader runs a libusb event thread.
find_package(Threads REQUIRED)
target_link_libraries(ef PRIVATE Threads::Threads)

# ---- optional H264/H265 decode + colour convert for retrieve_image ---------
# Compiled into `ef` only when libavcodec is present, so the RAW/NV12 path stays
# dependency-free. Decode users: apt install libavcodec-dev libavutil-dev libswscale-dev.
option(EF_WITH_FFMPEG "Decode H264/H265 + colour-convert in retrieve_image (needs libavcodec)" ON)
if(EF_WITH_FFMPEG)
    pkg_check_modules(EF_AV IMPORTED_TARGET libavcodec libavutil libswscale)
    if(EF_AV_FOUND)
        target_link_libraries(ef PRIVATE PkgConfig::EF_AV)
        target_compile_definitions(ef PRIVATE EF_WITH_FFMPEG=1)
        message(STATUS "retrieve_image decode: enabled (libavcodec found)")
    else()
        message(STATUS "retrieve_image decode: RAW/NV12 only (need libavcodec-dev)")
    endif()
endif()

# ---- optional BLE connection (BlueZ via GLib/GIO) ---------------------------
# Compiled into `ef` only when GLib/GIO + OpenSSL are present, so the core SDK
# stays libusb-only for USB-only users. BLE users: apt install libglib2.0-dev
# libssl-dev.
option(EF_ENABLE_BLE "Build the optional BLE connection (needs GLib/GIO + OpenSSL)" ON)
if(EF_ENABLE_BLE)
    pkg_check_modules(GIO IMPORTED_TARGET gio-2.0 gio-unix-2.0)
    find_package(OpenSSL)   # libcrypto: BLE password challenge-response (PBKDF2/HMAC)
    if(GIO_FOUND AND OPENSSL_FOUND)
        target_sources(ef PRIVATE src/ble_connection.cpp)
        target_link_libraries(ef PRIVATE PkgConfig::GIO OpenSSL::Crypto)
        target_compile_definitions(ef PRIVATE EF_HAVE_BLE=1)
        message(STATUS "BLE connection: enabled (GLib/GIO + OpenSSL found)")
    else()
        message(STATUS "BLE connection: skipped (need libglib2.0-dev + libssl-dev)")
    endif()
endif()

# ---- tools ------------------------------------------------------------------
# ef: the control tool — one subcommand per SDK verb (info/health/record/...).
add_executable(ef-cli tools/ef.cpp)
set_target_properties(ef-cli PROPERTIES OUTPUT_NAME ef)
target_link_libraries(ef-cli PRIVATE ef::ef)

# ef-grab: the open/grab/retrieve loop demo.
add_executable(ef-grab tools/ef_grab.cpp)
target_link_libraries(ef-grab PRIVATE ef::ef)

# OpenCV: used by the live viewer and the OpenCV example. OPTIONAL — the core
# SDK (ef) never links it; only display/interop tooling does. The ef::toCvMat
# bridge (include/ef/opencv.hpp) is header-only. Install: apt install libopencv-dev.
find_package(OpenCV QUIET COMPONENTS core imgproc highgui)

# efference-viewer: live video + IMU viewer. OPTIONAL — needs OpenCV + the ffmpeg
# decode path (retrieve_image with VIEW::BGR). Install: apt install libopencv-dev
# libavcodec-dev libavutil-dev libswscale-dev.
if(OpenCV_FOUND AND EF_AV_FOUND)
    add_executable(efference-viewer tools/efference_viewer.cpp)
    target_link_libraries(efference-viewer PRIVATE ef::ef ${OpenCV_LIBS})
    target_include_directories(efference-viewer PRIVATE ${OpenCV_INCLUDE_DIRS})
    message(STATUS "efference-viewer: enabled (OpenCV ${OpenCV_VERSION} + libavcodec found)")
else()
    message(STATUS "efference-viewer: skipped (need libopencv-dev + libavcodec-dev)")
endif()

# ---- examples (<repo>/examples/, one executable per .cpp) -------------------
# Examples live at the repository root (../../examples relative to this file),
# not under sdk/linux. Every examples/<group>/<name>.cpp builds to a binary
# named <group>_<name> (e.g. wired_01_serial_number). Drop in a new .cpp and
# re-run cmake — it is picked up automatically.
set(EF_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../examples")
if(EF_BUILD_EXAMPLES)
    file(GLOB EF_EXAMPLE_SRCS CONFIGURE_DEPENDS
         "${EF_EXAMPLES_DIR}/wired/*.cpp"
         "${EF_EXAMPLES_DIR}/wireless/*.cpp")
    foreach(src ${EF_EXAMPLE_SRCS})
        get_filename_component(name ${src} NAME_WE)         # 01_serial_number
        get_filename_component(dir  ${src} DIRECTORY)
        get_filename_component(grp  ${dir} NAME)            # wired | wireless
        set(out "${grp}_${name}")                           # wired_01_serial_number
        add_executable(example_${out} ${src})
        set_target_properties(example_${out} PROPERTIES OUTPUT_NAME ${out})
        target_link_libraries(example_${out} PRIVATE ef::ef)
    endforeach()

    # examples/opencv/*.cpp additionally link OpenCV (they use ef::toCvMat +
    # cv::imshow) and decode over USB via retrieve_image(VIEW::BGR), so they need
    # the ffmpeg path too. Built only when BOTH are present; otherwise skipped.
    if(OpenCV_FOUND AND EF_AV_FOUND)
        file(GLOB EF_OPENCV_EXAMPLE_SRCS CONFIGURE_DEPENDS
             "${EF_EXAMPLES_DIR}/opencv/*.cpp")
        foreach(src ${EF_OPENCV_EXAMPLE_SRCS})
            get_filename_component(name ${src} NAME_WE)
            set(out "opencv_${name}")
            add_executable(example_${out} ${src})
            set_target_properties(example_${out} PROPERTIES OUTPUT_NAME ${out})
            target_link_libraries(example_${out} PRIVATE ef::ef ${OpenCV_LIBS})
            target_include_directories(example_${out} PRIVATE ${OpenCV_INCLUDE_DIRS})
        endforeach()
    else()
        message(STATUS "examples/opencv: skipped (need libopencv-dev + libavcodec-dev)")
    endif()
endif()

# ---- install ---------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS ef
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
