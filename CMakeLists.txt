cmake_minimum_required(VERSION 2.8.3)
project(refusion)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    src
    include
    helpers
    cpptoml/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

if(DEFINED ENV{CKDOCKER})
    find_package(CUDA REQUIRED)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--std=c++11;-arch=sm_50;-O3)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
    include_directories(${CUDA_INCLUDE_DIRS})
    cuda_add_library(refusion STATIC
            src/utils/rgbd_image.cu
            src/marching_cubes/mesh.cu
            src/marching_cubes/mesh_extractor.cu
            src/tsdfvh/heap.cu
            src/tsdfvh/hash_table.cu
            src/tsdfvh/tsdf_volume.cu
            src/tracker/tracker.cu)
    target_link_libraries(refusion eigen_wrapper ${CUDA_LIBRARIES})
    add_library(eigen_wrapper src/tracker/eigen_wrapper.cpp)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

file(GLOB src_files *.cpp */*.cpp)
add_executable(main ${src_files} src/utils/FrameStream.cpp)
#add_executable(main src/example/example.cpp src/example/fr_parser.cpp)

if(DEFINED ENV{CKDOCKER})
    target_link_libraries(main refusion eigen_wrapper ${OpenCV_LIBS} ${CUDA_LIBRARIES})
endif()