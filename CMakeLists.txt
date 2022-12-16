cmake_minimum_required(VERSION 2.8)

project(efanna2e)
include_directories(${PROJECT_SOURCE_DIR}/include)
#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO -g3)

add_definitions (-DEVAL_RECALL)
#add_definitions (-DGET_DIST_COMP)
#add_definitions (-DPROFILE) # [ARC-SJ] Enable to measure the performance bottleneck
#add_definitions(-DMMAP_HUGETLB) # [ARC-SJ] Enable to allocate huge page instead of malloc
#add_definitions (-DTHREAD_LATENCY)

add_subdirectory(src)
add_subdirectory(tests)
