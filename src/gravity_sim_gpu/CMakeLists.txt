cmake_minimum_required(VERSION 3.10) 

# 1. PROJECT DECLARATION (Still required for .cu files)
project(gpu_n_body_gravity_inelastic LANGUAGES CXX CUDA) 

# 2. ALL VARIABLE SETTINGS (No major changes here)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
# ... Build flags/options follow ...

# ----------------------------------------------------
# 3. Find packages (CRITICAL CHANGE HERE)
# ----------------------------------------------------
find_package(SFML 2.5 COMPONENTS graphics window system  REQUIRED)
find_package(X11 REQUIRED)
# ⚠️ RE-INTRODUCE: Need this to populate the ${CUDA_LIBRARIES} variable
find_package(CUDA REQUIRED) 

# ----------------------------------------------------
# 4. Create executable 
# ----------------------------------------------------
add_executable(gpu_n_body_gravity_inelastic "${CMAKE_SOURCE_DIR}/gpu_n_body_gravity_inelastic.cu")

# ----------------------------------------------------
# Include directories
# ----------------------------------------------------
target_include_directories(gpu_n_body_gravity_inelastic PRIVATE "../../inc")

# ----------------------------------------------------
# 5. Link libraries (CRITICAL CHANGE HERE)
# ----------------------------------------------------
target_link_libraries(gpu_n_body_gravity_inelastic PRIVATE 
    sfml-graphics
    sfml-window
    sfml-system
    ${X11_LIBRARIES}
    X11
    Xext
    Xrandr
    # ⚠️ REPLACE THE BROKEN TARGET WITH THE ROBUST VARIABLE:
    ${CUDA_LIBRARIES} 
)