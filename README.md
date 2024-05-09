# NvDsInferYoloEfficientNMS for Gst-nvinferserver

This repository provides a custom implementation of parsing function to the [Gst-nvinferserver](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html) plugin when use YOLOv7/YOLOv9 model served by Triton Server using the [Efficient NMS](https://github.com/NVIDIA/TensorRT/tree/master/plugin/efficientNMSPlugin) plugin exported by ONNX.


By using the parsing function provided by `NvDsInferYoloEfficientNMS`, handling the number of classes dynamically becomes easier. This eliminates the need to hardcode the number of classes, allowing the same plugin to be used for different YOLOv9 models with varying numbers of classes.

- Use parse function `NvDsInferYoloEfficientNMS` for YOLO Exported ONNX model with End2End


# Deployment Guide for NvDsInferYoloEfficientNMS


## Cloning Repository and Installation

To clone the repository and build\install the `libnvds_infer_yolo_efficient_nms.so` library, follow these steps:


```bash
# Clone the repository
git clone https://github.com/levipereira/nvdsinfer_yolo_efficient_nms.git

# Copy the repository to the desired location
cp -R nvdsinfer_yolo_efficient_nms/ /opt/nvidia/deepstream/deepstream/sources/libs/

# Set the CUDA_VER environment variable (check your deepstream cuda version.  The DS 6.4 use cuda 12.2)
export CUDA_VER=12.2

# Navigate to the directory containing the nvdsinfer_yolov9_efficient_nms library
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_yolo_efficient_nms

# Build the project using the provided MakeFile
make -f MakeFile all
make -f MakeFile install

```
Install Location:

`/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolo_efficient_nms.so`

Usage on Deepstream

Snippet [Gst-nvinferserver](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinferserver.html)  Configuration File for YOLO Models
```
  postprocess {
    labelfile_path: "labels.txt"
    detection {
      num_detected_classes: 80
      custom_parse_bbox_func: "NvDsInferYoloEfficientNMS"
    }
  }
  custom_lib {
    path : "/opt/nvidia/deepstream/deepstream/lib/libnvds_infer_yolo_efficient_nms.so"
  }
```

