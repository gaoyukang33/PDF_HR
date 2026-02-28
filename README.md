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
- [x] Release PDF-HR training pipeline
- [x] Release Motion tracking with PDF-HR training pipeline
- [ ] Release Motion retargeting with PDF-HR pipeline



# Installation

PDF-HR codebase is built on [Mimickit](https://github.com/xbpeng/MimicKit), which provides a suite of motion imitation methods for training motion controllers. 

## Setup

First create your conda environment:

```bash
conda create -n pdfhr python=3.8.20 -y
conda activate pdfhr
```

Download Isaac Gym by following the [installation guide](https://developer.nvidia.com/isaac-gym).

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```


After that, install the requirements:
```
pip install -r requirements.txt
```


## Data Preparation

Download the assets and precomputed data from [here](https://drive.google.com/drive/folders/1o1CqwYTmlNwtccKglNEZ39PpUnyfrntn?usp=sharing), and then extract the files to the following structure:
```bash
precomputed_data/
└── sampling_pose_L1.pt

prior_ckpts/
└──PDFHR_epoch50.pt

data/
├── assets/
│   └── g1/
└── motions/
    └── g1/
```

# PDF-HR Training 

An efficient PyTorch training script for the MLP (`PDFHR_Adapter`). By leveraging memory mapping (`mmap=True`),  it trains directly from massive `.pt` files while keeping RAM usage low.

## Data Format Requirements

To ensure memory mapping works correctly, the input `.pt` file must contain raw `torch.Tensor` objects. 

Supported structures:
* **Dictionary (Recommended):** `{"db": X, "dis": Y}` (Keys can be customized via `--x-key` and `--y-key`).
* **Tuple/List:** `(X, Y)`

*Note: The first dimension (number of samples, N) of both X and Y must match exactly.*

## Training

Adjust the data path and run the following command to start training:

```bash
python scripts/train_PDF.py \
    --data ./precomputed_data/sampling_pose_L1.pt \
    --batch-size 65536 \
    --epochs 50 
```

## Outputs

* Checkpoints are automatically saved to `../prior_ckpts` (customizable via `--ckpt-dir`). 


# Motion Tracking Training

To train a motion tracker with PDF-HR, run the following command:
```
./run.sh 1 full_args/args_deepmimic_PDFHR/deepmimic_g1_args_backflip_PDFHR.txt
```
Parameters description:

- `use_PDFHR_reward`: True  # trigger for PDFHR reward 
- `reward_PDFHR_dist_w`: 0.2 # trigger for PDFHR reward 
- `reward_PDFHR_dist_scale`: 1.0
- `reward_PDFHR_mode`: "static"  # static or dynamic
- `reward_PDFHR_tolerance`: 0.05
- `PDFHR_model_path`: "prior_ckpts/PDFHR_epoch50.pt"
- `use_PDFHR_early_termination`: False  # default as False
- `PDFHR_et_threshold`: 0.4

## Distributed Training
To use distributed training with multi-GPU (CUDA:2,3,4,5):
```
./run_dist.sh ./full_args/args_add_PDFHR/add_g1_args_backflip_PDFHR.txt 2 3 4 5
```



## Testing

We use viser to test the model, avoiding desktop requirement. Run the following command:
```
./test.sh 1 $MODEL_FOLDER $MODEL_FILE
```
or
```
./test.sh 1 $MODEL_FOLDER
```
- `MODEL_FILE` specifies the `.pt` file that contains the parameters of the trained model.
- If no `MODEL_FILE`, directly use `MODEL_FOLDER/model.pt` to infer.


<details>
<summary>To visualize the vault_box, modify the vault_box as visual. Click to expand code for reference.</summary>

```xml
<?xml version="1.0"?>
<robot name="vault_box">
  <link name="object">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 2.0 0.6"/>
      </geometry>
    </collision>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.4 2.0 0.6"/>
      </geometry>
      <material name="grey">
        <color rgba="0.3 0.3 0.3 1.0"/>
      </material>
    </visual>
  </link>
</robot>
```
</details>

# Quick Demo for Pose Denoising

To run the demo, simply execute the following command:
```
python PDF-HR/scripts/pose_denoising.py --urdf_path path/to/your_robot.urdf --model_path path/to/PDFHR_model.pt
```
## Citation
If you find this codebase helpful, please cite:
```
@article{
      gu2026pdfhr,
      title={PDF-HR: Pose Distance Fields for Humanoid Robots}, 
      author={Yi Gu and Yukang Gao and Yangchen Zhou and Xingyu Chen and Yixiao Feng and Mingle Zhao and Yunyang Mo and Zhaorui Wang and Lixin Xu and Renjing Xu},
      year={2026},
      journal={arXiv preprint arXiv:2602.04851}
}
      