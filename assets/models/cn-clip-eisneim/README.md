---
license: apache-2.0
language:
- zh
pipeline_tag: zero-shot-classification
tags:
- clip
- multi modal
---
Chinese-CLIP Model Deployment: ONNX

those Onnx file is converted using this [script](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/deployment_En.md)
you will likely to encounter this Error while converting:
```
Exporting the operator 'aten::unflatten' to ONNX opset version 13 is not supported.
```
so I uploaded those converted file for your convenience.
中文CLIP模型 [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)