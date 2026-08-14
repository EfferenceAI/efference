# Efference SDK, Linux (C++). Self-contained CMake project:
#
#   ./build.sh                       # or:
#   cmake -S . -B build && cmake --build build
#
# Produces the `ef` library and the `ef-cli` control tool. Tutorials under
# ../../tutorials build separately against this build tree (find_package(ef)).
# libusb-1.0 is the only required dependency.
cmake_minimum_required(VERSION 3.16)
project(efference_sdk_linux VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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
    src/update_check.cpp
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

# The update-check client POSTs to the update service (src/update_check.cpp), so
# libcurl is required: apt install libcurl4-openssl-dev.
find_package(CURL REQUIRED)
target_link_libraries(ef PRIVATE CURL::libcurl)
# Overridable at configure time for a self-hosted check service:
#   cmake -DEF_UPDATE_CHECK_URL=https://my-host/ota/check
if(EF_UPDATE_CHECK_URL)
    target_compile_definitions(ef PRIVATE
        EF_UPDATE_CHECK_URL_DEFAULT="${EF_UPDATE_CHECK_URL}")
endif()
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
# ef-cli: the control tool, one subcommand per SDK verb (info/health/record/...).
# The binary is ef-cli; the library keeps the ef/ef::ef name.
add_executable(ef-cli tools/ef.cpp)
target_link_libraries(ef-cli PRIVATE ef::ef)

# ef-decrypt: turn an encrypted recording back into the plain container. Built
# from the vendored device sources under tools/vendor/ (see its README), so the
# host reads exactly the format the device writes. OPTIONAL: needs libcrypto,
# which BLE also uses. Install: apt install libssl-dev.
find_package(OpenSSL QUIET)   # also probed under EF_ENABLE_BLE below
if(OPENSSL_FOUND)
    add_executable(ef-decrypt
        tools/ef_decrypt.c
        tools/vendor/enc_sink.c
        tools/vendor/efr_keyid.c)
    target_include_directories(ef-decrypt PRIVATE tools/vendor)
    target_link_libraries(ef-decrypt PRIVATE OpenSSL::Crypto)
    # O_CLOEXEC: upstream builds these with _GNU_SOURCE, so ask for it rather
    # than inheriting it from whatever C dialect happens to be the default.
    target_compile_definitions(ef-decrypt PRIVATE _GNU_SOURCE)
    set_target_properties(ef-decrypt PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
    # Vendored as-is; -Werror here would make an upstream refresh a build break.
    target_compile_options(ef-decrypt PRIVATE -Wall -Wextra)
    message(STATUS "ef-decrypt: enabled (OpenSSL found)")
else()
    message(STATUS "ef-decrypt: skipped (need libssl-dev)")
endif()

# OpenCV: used by the live viewer and the OpenCV example. OPTIONAL, the core
# SDK (ef) never links it; only display/interop tooling does. The ef::toCvMat
# bridge (include/ef/OpenCV.hpp) is header-only. Install: apt install libopencv-dev.
find_package(OpenCV QUIET COMPONENTS core imgproc highgui)

# ---- calibration solvers (prebuilt, closed source) --------------------------------
# The camera and IMU field-calibration solvers ship as per-architecture BINARIES
# alongside the public headers ef/Checkerboard.h and ef/ImuCalib.h. Two independent
# libs, each bundling the shared linalg. When a .so is absent, the things that need
# it are skipped and the rest builds as usual.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(EF_CAL_ARCH linux-aarch64)      # arm64 Linux
else()
    set(EF_CAL_ARCH linux-x86_64)
endif()
set(EF_CAMERA_CAL_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/${EF_CAL_ARCH}/libcamera_cal.so)
set(EF_IMU_CAL_LIB    ${CMAKE_CURRENT_SOURCE_DIR}/lib/${EF_CAL_ARCH}/libimu_cal.so)
if(EXISTS ${EF_CAMERA_CAL_LIB})
    add_library(camera_cal SHARED IMPORTED)
    set_target_properties(camera_cal PROPERTIES IMPORTED_LOCATION ${EF_CAMERA_CAL_LIB})
    add_library(ef::camera_cal ALIAS camera_cal)
    message(STATUS "camera_cal: enabled (${EF_CAL_ARCH})")
else()
    message(STATUS "camera_cal: skipped (no ${EF_CAMERA_CAL_LIB})")
endif()
if(EXISTS ${EF_IMU_CAL_LIB})
    add_library(imu_cal SHARED IMPORTED)
    set_target_properties(imu_cal PROPERTIES IMPORTED_LOCATION ${EF_IMU_CAL_LIB})
    add_library(ef::imu_cal ALIAS imu_cal)
    message(STATUS "imu_cal: enabled (${EF_CAL_ARCH})")
else()
    message(STATUS "imu_cal: skipped (no ${EF_IMU_CAL_LIB})")
endif()

# efference-viewer: live video + IMU viewer. OPTIONAL, needs OpenCV + the ffmpeg
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

# ---- tutorials build-tree export -------------------------------------------
# Tutorials (../../tutorials/<topic>/cpp) are standalone mini-projects that link
# the SDK via find_package(ef). Export the targets from the build tree so a
# tutorial resolves against this in-repo build with no install step: point its
# CMAKE_PREFIX_PATH at build/ (env.sh does this), then find_package(ef) works.
# The generated efConfig.cmake also re-declares the imported calibration targets
# when their per-arch .so is present, so OpenCV/IMU tutorials can link them.
export(TARGETS ef NAMESPACE ef:: FILE "${CMAKE_BINARY_DIR}/efTargets.cmake")
set(EF_EXPORT_CAL_LIB "${EF_CAMERA_CAL_LIB}")
set(EF_EXPORT_IMU_LIB "${EF_IMU_CAL_LIB}")
configure_file(cmake/efConfig.cmake.in "${CMAKE_BINARY_DIR}/efConfig.cmake" @ONLY)

# ---- install ---------------------------------------------------------------
# Optional (build.sh --install): places ef-cli on PATH system-wide. The library,
# headers, and calibration .so ship too so find_package(ef) works off-tree.
include(GNUInstallDirs)
install(TARGETS ef-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET ef-decrypt)
    install(TARGETS ef-decrypt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(TARGETS ef
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(TARGET camera_cal)
    install(FILES ${EF_CAMERA_CAL_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(TARGET imu_cal)
    install(FILES ${EF_IMU_CAL_LIB} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
