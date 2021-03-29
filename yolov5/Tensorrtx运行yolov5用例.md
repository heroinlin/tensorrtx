# Tensorrtx运行yolov5用例
## 环境搭建
环境： TensorRT7.2.2.3， cuda10.2, cudnn8.0.5.39, pythorch1.7.1_cuda10.1， opencv-4.5.1+ contrib

## 编译源码
打开yolov5目录

根据环境路径更改CMakeLists.txt文件

```cmake
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


add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)


```

完成修改后进行cmake编译

```shell
mkdir build
cd build
cmake ..
make -j8
```

如此，完成包含yolov5自定义层的cuda库编译，生成libmyplugins.so库和可执行文件yolov5

## 设置执行环境

```shell
TensorRT_version=7.2.2.3
cuda_version=10.2
cudnn_version=8.0.5.39
pytorch_version=1.7.1
version=${TensorRT_version}_cu${cuda_version}
workroot=/home/heroin
cuda_root=${workroot}/softwares/cuda-${cuda_version}
cudnn_root=${workroot}/softwares/cudnn/cudnn-${cuda_version}-linux-x64-v${cudnn_version}
TensorRT_folder=${workroot}/softwares/TensorRT/TensorRT-${version}

export CUDA_TOOLKIT_ROOT_DIR=${cuda_root}
export CUDA_BIN_PATH=${cuda_root}

export PATH=${cuda_root}/bin:$PATH
export C_INCLUDE_PATH=${cudnn_root}/include:${cuda_root}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${cudnn_root}/include:${cuda_root}/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=${cudnn_root}/lib64:${cuda_root}/lib64:$LD_LIBRARY_PATH:${TensorRT_folder}/lib

export pycuda_env=${workroot}/thirdpartys/pycuda/pycuda
export pytorch_env=${workroot}/thirdpartys/pytorch/pytorch${pytorch_version}_cu10.1/lib/python3.7/site-packages
export tensorRT_env=${workroot}/thirdpartys/TensorRT/TensorRT${version}/lib/python3.7/site-packages
export onnx_env=${workroot}/thirdpartys/onnx_env/lib/python3.7/site-packages
export extra_env=${workroot}/thirdpartys/extra/lib/python3.7/site-packages
export PYTHONPATH=${pytorch_env}:${pycuda_env}:${tensorRT_env}:${onnx_env}:${extra_env}
```

## 实例测试

首先使用`gen_wts.py`文件将pytorch版本的模型文件转换为`.wts`格式的文件

为实现参数化设置模型，修改`gen_wts.py`如下

```python
import os
import torch
import struct
import argparse
working_root = os.path.split(os.path.realpath(__file__))[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--model_pt',
                        type=str,
                        help='pytorch model file path',
                        default=os.path.join(working_root,
                                             "weights/yolov5s.pt"))
    parser.add_argument('-w',
                        '--model_wts',
                        type=str,
                        help='wts type model file path',
                        default=os.path.join(working_root,
                                             "weights/yolov5s.wts"))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    pt_model_path = args.model_pt
    wts_model_path = args.model_wts
    if not os.path.exists(pt_model_path):
        print("pytorch modle file is not exists! please check path!")
        exit()

    # Initialize
    device = torch.device('cpu')
    # Load model
    model = torch.load(pt_model_path, map_location=device)['model']  
    model.float().eval() # load to FP32
    model = model.to(device)

    f = open(wts_model_path, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
    

if __name__ == '__main__':
    main()

```

转换模型输入如下命令生成`.wts`格式文件

```shell
python gen_wts.py -p weights/yolov5s.pt -w weights/yolov5s.wts
```

再使用以下命令生成`.engine`文件

```shell
./build/yolov5 -s weights/yolov5s.wts weights/yolov5s.engine s
# ./build/yolov5 -s weights/yolov5l.wts weights/yolov5l.engine l
```

> 注： 如果使用Float16或INT8运行，需要对yolov5.cpp内代码进行修改
>
> ```C++
> #define USE_FP16   *// set USE_INT8 or USE_FP16 or USE_FP32*
> ```
>
> 使用USE_INT8时还需准备好coco_calib图像文件夹进行校正准备，具体代码也在yolov5.cpp文件内
>
> ```c++
> Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./data/coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
> ```
> 修改后需要重新编译生效。


* 使用python运行Tensorrt模型

  ```shell
  python yolov5_trt.py
  ```

* 使用C++方式运行Tensorrt模型

  ```shell
  ./build/yolov5 -d weights/yolov5s.engine data/images/
  # ./build/yolov5 -d weights/yolov5l.engine data/images/
  # ./build/yolov5 -d weights/yolov5s_int8.engine data/images/
  ```

  
> tensorrt可以通过set workspace来设置使用的显存空间

