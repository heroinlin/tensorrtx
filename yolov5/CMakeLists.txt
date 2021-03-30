cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(TensorRT_version 7.2.2.3)
set(cuda_version 10.2)
set(cudnn_version 8.0.5.39)
set(version ${TensorRT_version}_cu${cuda_version})
set(workroot /home/heroin)
set(tensorRT_root ${workroot}/softwares/TensorRT/TensorRT-${version})
set(CUDA_BIN_PATH ${workroot}/softwares/cuda-${cuda_version})
set(CUDA_TOOLKIT_ROOT_DIR ${workroot}/softwares/cuda-${cuda_version})
set(CUDNN_ROOT_DIR ${workroot}/softwares/cudnn/cudnn-${cuda_version}-linux-x64-v${cudnn_version})
include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# cudnn
include_directories(${CUDNN_ROOT_DIR}/include)
link_directories(${CUDNN_ROOT_DIR}/lib64)

# tensorrt
include_directories(${tensorRT_root}/include)
link_directories(${tensorRT_root}/lib)

# link cuda
find_package(CUDA ${cuda_version} REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)


# opencv
set(OPENCV_PATH /home/heroin/thirdpartys/opencv/opencv-4.5.1)
set(OPENCV_3RDPARTY_LIB_PATH ${OPENCV_PATH}/lib/opencv4/3rdparty)
include_directories(${OPENCV_PATH}/include/opencv4)
set(OpenCV_LIBS ${OPENCV_PATH}/lib/libopencv_dnn.so ${OPENCV_PATH}/lib/libopencv_highgui.so ${OPENCV_PATH}/lib/libopencv_imgproc.so ${OPENCV_PATH}/lib/libopencv_core.so ${OPENCV_PATH}/lib/libopencv_imgcodecs.so)


add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5s.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

