cmake_minimum_required(VERSION 3.10)
project(n_body_gravity_inelastic)

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
# Create executable for n_body_gravity_inelastic
# ----------------------------------------------------
add_executable(n_body_gravity_inelastic "${CMAKE_SOURCE_DIR}/n_body_gravity_inelastic.cpp")

# ----------------------------------------------------
# Include directories
# ----------------------------------------------------
target_include_directories(n_body_gravity_inelastic PRIVATE "../../inc")

target_link_libraries(n_body_gravity_inelastic
    sfml-graphics
    sfml-window
    sfml-system
    ${X11_LIBRARIES}
    X11
    Xext
    Xrandr
)
