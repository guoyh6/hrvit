# HRViT
Code for MICCAI2025 paper.
To be updated

## Package Versions
```
python=3.8.12
pytorch=1.7.1=py3.8_cuda10.1.243_cudnn7.6.3_0
torchvision=0.8.2=py38_cu101
monai-weekly==0.9.dev2152
nibabel==3.2.1
omegaconf==2.1.1
timm==0.4.12
torchprofile==0.0.4
```

## Prepare Dataset
1. Download BTCV from https://www.synapse.org/#!Synapse:syn3376386 and MSD BraTS from http://medicaldecathlon.com/
2. Additionally for BTCV, copy the `lib/data/transunet.json` into the data folder.
3. Set the `data_path` in the config file as the above data folder.

## Acknowledgment
Our code is built on SCD.
```
@inproceedings{zhou2023token,
  title={Token Sparsification for Faster Medical Image Segmentation},
  author={Zhou, Lei and Liu, Huidong and Bae, Joseph and He, Junjun and Samaras, Dimitris and Prasanna, Prateek},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={743--754},
  year={2023},
  organization={Springer}
}
```
