# Standalone build for the serial_number tutorial. Links the SDK via find_package(ef),
# resolved against the in-repo build tree (source <repo>/env.sh first, or pass
# -DCMAKE_PREFIX_PATH=<repo>/sdk/linux/build). See tutorials/README.md.
#
# Every .cpp here builds to its own binary named after the file, so a topic can
# hold more than one program: just drop in another .cpp and rebuild.
cmake_minimum_required(VERSION 3.16)
project(ef_tutorial_serial_number CXX)
find_package(ef REQUIRED)
file(GLOB TUT_SRCS CONFIGURE_DEPENDS "*.cpp")
foreach(src ${TUT_SRCS})
    get_filename_component(name "${src}" NAME_WE)
    add_executable(${name} "${src}")
    target_link_libraries(${name} PRIVATE ef::ef)
endforeach()
