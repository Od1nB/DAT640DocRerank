# Install Guide
- [Source](https://www.tensorflow.org/install/gpu)

## Hardware Reqs
- NVIDIA® GPU card with CUDA® architectures 3.5, 5.0, 6.0, 7.0, 7.5, 8.0 and higher than 8.0. See the list of [ CUDA®-enabled GPU cards](https://developer.nvidia.com/cuda-gpus)

## Software Reqs
- tensorflow version >= 2.5.0
- [NVIDIA® GPU drivers](https://www.nvidia.com/drivers)—CUDA® 11.2 requires 450.80.02 or higher.
- [CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)—TensorFlow supports CUDA® 11.2 (TensorFlow >= 2.5.0)
- [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA® Toolkit
- [cuDNN SDK >= 8.1.0](https://developer.nvidia.com/cudnn)
- (Optional)[TensorRT 6.0](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_6) to improve latency and throughput for inference on some models.

## Windows Install
- [source](https://www.tensorflow.org/install/gpu#windows_setup)
- Install Software Reqs
- Add the CUDA®, CUPTI, and cuDNN installation directories to the %PATH% environmental variable'
- Do [this](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) after downloading both CUDA and cuDNN SDK
- Make sure the installed NVIDIA software packages match the versions listed in Software Reqs above. In particular, TensorFlow will not load without the `cuDNN64_8.dll` file.
