<h1 align="center"> PDF-HR: Pose Distance Fields for Humanoid Robots
</h1>

<div align="center">

[[Website]](https://gaoyukang33.github.io/PDF-HR/)
[[Arxiv]](https://arxiv.org/abs/2602.04851)



[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div align='center'>

## Overview

PDF-HR (Pose Distance Fields for Humanoid Robots) is a lightweight, plug-and-play pose prior framework that provides a continuous and differentiable manifold of robot pose distributions for robust optimization and control.
This repo covers the core PDF-HR model and its integration pipelines across diverse humanoid tasks. You should be able to drop this prior directly into your own reinforcement learning baselines as a reward shaping term or regularizer to substantially improve motion plausibility, without requiring complex tuning.


## TODO

- [x] Release PDF-HR checkpoint
- [x] Release PDF-HR quick demo for pose denoising
- [ ] Release PDF-HR training pipeline
- [ ] Release Motion tracking with PDF-HR training pipeline
- [ ] Release Motion retargeting with PDF-HR pipeline



# Installation

PDF-HR codebase is built on [Mimickit](https://github.com/xbpeng/MimicKit), which provides a suite of motion imitation methods for training motion controllers. 

## Setup

First create your conda environment:

```bash
conda create -n pdfhr python=3.8.20 -y
conda activate pdfhr
```

After that, install the requirements:
```
pip install -r requirements.txt
```


## Data Preparation

Download the assets and precomputed data from [here](https://drive.google.com/drive/folders/1o1CqwYTmlNwtccKglNEZ39PpUnyfrntn?usp=sharing), and then extract the files to the following structure:
```bash
PDF_HR/
├── prior_ckpts/
│   └── PDFHR_epoch50.pt
└── data/
    └── assets/
        └── g1/
```

# Quick Demo for Pose Denoising

To run the demo, simply execute the following command:
```
python PDF_HR/scripts/pose_denoising.py --urdf_path PDF_HR/data/assets/g1 --model_path PDF_HR/prior_ckpts/
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Citation
If you find this codebase helpful, please cite:
```
@article{
      gu2026pdfhr,
      title={PDF-HR: Pose Distance Fields for Humanoid Robots}, 
      author={Yi Gu and Yukang Gao and Yangchen Zhou and Xingyu Chen and Yixiao Feng and Mingle Zhao and Yunyang Mo and Zhaorui Wang and Lixin Xu and Renjing Xu},
      year={2026},
      journal={arXiv preprint arXiv:2602.04851}
}
      