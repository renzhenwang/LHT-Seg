# LHT-Seg

This repository contains the PyTorch implementation for a preliminary exploration of applying our Label Hierarchy Transition (LHT) method to semantic segmentation. It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation) and [HieraSeg](https://github.com/lingorX/HieraSeg/tree/main/Pytorch).

## Results and Models

| Dataset | Backbone | Crop Size | mIoU (single scale) | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cityscapes | Resnet101 | 512x1024 | 81.41 | [config](configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024_LHT.py) | [log_file](https://github.com/renzhenwang/LHT-Seg/releases/download/v1.0/20250831_112054.log) | [weights](https://github.com/renzhenwang/LHT-Seg/releases/download/v1.0/iter_80000.pth) |


## Usage

### Installation 

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation) for installation and dataset preparation.

### Requirement 

We use four RTX 4090 GPUs for training. These are the packages and versions used in our environment. 
Other configurations have not been tested.

```
pytorch-cuda = 11.8
Pytorch = 2.0.0 
torchvision = 0.15.0
mmcv = 2.0.0
mmengine = 0.10.7
mmsegmentation == 1.2.2

```

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU
```

For example, with configuration configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024_LHT.py and pretrained weights iter_80000.pth:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024_LHT.py iter_80000.pth 4
```

### Training

To train on Cityscapes with a `ResNet-101` backbone and 4 gpus, run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024_LHT.py 4
```
