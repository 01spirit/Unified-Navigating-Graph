cmake_minimum_required(VERSION 3.15)
project(ANNS)

# cmake setting
set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Boost
# set(Boost_INCLUDEDIRS "specify if needed")
# set(Boost_ROOT "specify if needed")
# set(Boost_DIR "specify if needed")
find_package(Boost COMPONENTS REQUIRED program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message(FATAL_ERROR "no Boost supprot")
endif()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

find_library(STATGRAB_LIBRARY statgrab)

# BLAS
find_package(BLAS REQUIRED)
if (BLAS_FOUND)
    include_directories(${BLAS_INCLUDE_DIRS})
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
else()
    message(FATAL_ERROR "no BLAS support")
endif()

# build options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DNDEBUG -Ofast -march=native -mtune=native")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})

# add subdirectories
add_subdirectory(vamana)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(apps)
add_subdirectory(third_party)

add_executable(RptreeTest src/rptree_test.cpp)
target_link_libraries(RptreeTest ${PROJECT_NAME} Vamana ${GTEST_LIBRARIES} pthread statgrab)

add_executable(KmeansTest src/kmeans_test.cpp)
target_link_libraries(KmeansTest ${PROJECT_NAME} Vamana ${GTEST_LIBRARIES} pthread statgrab ${BLAS_LIBRARIES})