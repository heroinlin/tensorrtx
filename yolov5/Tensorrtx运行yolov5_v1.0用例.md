# Tensorrtx运行yolov5_v1.0用例
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


add_executable(yolov5 ${PROJECT_SOURCE_DIR}/yolov5s.cpp)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

```

对yolos.cpp 文件中createEngine函数做如下修改适应其他规模网络

```c++
static int get_width(int x, float gw, int divisor = 8)
{
    //return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0)
    {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}


static int get_depth(int x, float gd)
{
    if (x == 1)
    {
        return 1;
    }
    else
    {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    std::string wts_name = "";
    float gd = 0.0f, gw = 0.0f;
    wts_name = "../yolov5s.wts";
    gd = 0.33;
    gw = 0.5;


    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBnLeaky(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = bottleneckCSP(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBnLeaky(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = bottleneckCSP(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBnLeaky(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = bottleneckCSP(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBnLeaky(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = bottleneckCSP(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBnLeaky(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * get_width(512, gw) * 2 * 2));
    for (int i = 0; i < get_width(512, gw) * 2 * 2; i++)
    {
        deval[i] = 1.0;
    }
    Weights deconvwts11{DataType::kFLOAT, deval, get_width(512, gw) * 2 * 2};
    IDeconvolutionLayer *deconv11 = network->addDeconvolutionNd(*conv10->getOutput(0), get_width(512, gw), DimsHW{2, 2}, deconvwts11, emptywts);
    deconv11->setStrideNd(DimsHW{2, 2});
    deconv11->setNbGroups(get_width(512, gw));
    weightMap["deconv11"] = deconvwts11;

    ITensor *inputTensors12[] = {deconv11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = bottleneckCSP(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBnLeaky(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    Weights deconvwts15{DataType::kFLOAT, deval, get_width(256, gw) * 2 * 2};
    IDeconvolutionLayer *deconv15 = network->addDeconvolutionNd(*conv14->getOutput(0), get_width(256, gw), DimsHW{2, 2}, deconvwts15, emptywts);
    deconv15->setStrideNd(DimsHW{2, 2});
    deconv15->setNbGroups(get_width(256, gw));

    ITensor *inputTensors16[] = {deconv15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);
    auto bottleneck_csp17 = bottleneckCSP(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer *conv18 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.18.weight"], weightMap["model.18.bias"]);
    auto conv19 = convBnLeaky(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.19");
    ITensor *inputTensors20[] = {conv19->getOutput(0), conv14->getOutput(0)};
    auto cat20 = network->addConcatenation(inputTensors20, 2);
    auto bottleneck_csp21 = bottleneckCSP(network, weightMap, *cat20->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.21");
    //yolo layer 1
    IConvolutionLayer *conv22 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.22.weight"], weightMap["model.22.bias"]);
    auto conv23 = convBnLeaky(network, weightMap, *bottleneck_csp21->getOutput(0), get_width(512, gw), 3, 2, 1, "model.23");
    ITensor *inputTensors24[] = {conv23->getOutput(0), conv10->getOutput(0)};
    auto cat24 = network->addConcatenation(inputTensors24, 2);
    auto bottleneck_csp25 = bottleneckCSP(network, weightMap, *cat24->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.25");
    IConvolutionLayer *conv26 = network->addConvolutionNd(*bottleneck_csp25->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.26.weight"], weightMap["model.26.bias"]);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection *pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor *inputTensors_yolo[] = {conv26->getOutput(0), conv22->getOutput(0), conv18->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

```

通过设置 wts_name，gd，gw变量控制网络规模，目前测试s, l网络正常运行 

在yololayer.h文件中添加如下代码

```C++
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
```

> 参考链接: https://github.com/wang-xinyu/tensorrtx/issues/251

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
cd build
./yolov5 -s 
```


* 使用python运行Tensorrt模型

  ```shell
  python yolov5_trt.py
  ```

* 使用C++方式运行Tensorrt模型

  ```shell
  cd build
  ./yolov5 -d data/images/
  ```
  



