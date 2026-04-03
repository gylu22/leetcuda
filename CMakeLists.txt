cmake_minimum_required(VERSION 3.19)
project(CuteProject LANGUAGES CUDA CXX C)

# 1. 显式设置 C++ 和 CUDA 标准 (CuTe 3.0+ 需要 C++17 或更高)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. 针对 Ada 架构 (SM 89) 进行优化
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUTLASS_NVCC_ARCHS 89 CACHE STRING "SM architectures requested.")

# 3. 开启“Headers Only”模式，极大地加快配置速度和编译时间
# 这是因为对 CuTe 练习代码来说，只需要 CUTLASS 的头文件，不需要编译其庞大的静态库和测试
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

# 4. 引入子目录
include_directories(${PROJECT_SOURCE_DIR}/third-party/cutlass/include)
add_subdirectory(practice/cute)
add_subdirectory(practice/visual_layout)
