cmake_minimum_required(VERSION 3.10)
project(particles_in_a_box)

set(CMAKE_CXX_STANDARD 17)

# ----------------------------------------------------
# Build Type Options
# ----------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)   # default to Debug
endif()

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fno-inline")

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ----------------------------------------------------
# Find packages
# ----------------------------------------------------
find_package(SFML 2.5 COMPONENTS graphics window system  REQUIRED)
find_package(X11 REQUIRED)

# ----------------------------------------------------
# Create executable for particles_in_a_box
# ----------------------------------------------------
add_executable(particles_in_a_box "${CMAKE_SOURCE_DIR}/particles_in_a_box.cpp")

# ----------------------------------------------------
# Include directories
# ----------------------------------------------------
target_include_directories(particles_in_a_box PRIVATE "../../.../inc")

target_link_libraries(particles_in_a_box
    sfml-graphics
    sfml-window
    sfml-system
    ${X11_LIBRARIES}
    X11
    Xext
    Xrandr
)
