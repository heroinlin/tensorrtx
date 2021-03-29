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


./build/yolov5 -s weights/yolov5s.wts weights/yolov5s.engine s
# ./build/yolov5 -s weights/yolov5l.wts weights/yolov5l1.engine l
# ./build/yolov5 -d weights/yolov5s.engine data/images/
# ./build/yolov5 -d weights/yolov5l.engine data/images/

# ./release/yolov5 -d weights/yolov5s.engine data/images/
# ./release/yolov5 -d weights/yolov5l.engine data/images/
# ./release/yolov5 -s weights/yolov5l.wts weights/yolov5l_int8.engine l
# ./release/yolov5 -d weights/yolov5l_int8.engine data/images/

# ./release_32/yolov5 -s weights/yolov5l.wts weights/yolov5l_float32.engine l
# ./release_32/yolov5 -d weights/yolov5l_float32.engine data/images/
