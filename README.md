<h1 align="center"> PDF-HR: Pose Distance Fields for Humanoid Robots
</h1>

<div align="center">

[[Website]](https://gaoyukang33.github.io/PDF-HR/)
[[arXiv]](https://arxiv.org/abs/2602.04851)

![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg) ![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg) ![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

</div align='center'>

## Overview

This repository includes the official implementation of PDF-HR model training and downstream humanoid control experiments.

- `ManifoldTraining/`: dataset preprocessing, sample generation, and PDF-HR training.
- `MimicKit_PDFHR/`: the pinned MimicKit source plus the PDF-HR reward integration, experiment configurations, and visualization tools.
- `PDFHR/`: implementation of PDF-HR models and checkpoint loading. Training and downstream code both import this package.



## Installation

Create the shared `PDFHR` Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate PDFHR
```

### PDF-HR training

Install the shared `PDFHR` package in editable mode:

```bash
python -m pip install -e .
```

This makes the model definitions and checkpoint-loading utilities available to the training code. See [ManifoldTraining/README.md](ManifoldTraining/README.md) for dataset preprocessing, sample generation, and training instructions.

### Downstream Humanoid Control

Download and install [Isaac Gym](https://developer.nvidia.com/isaac-gym), then install the downstream dependencies:

```bash
python -m pip install -r MimicKit_PDFHR/requirements.txt
```

The downstream experiments use the same editable `PDFHR` package installed above. Follow [MimicKit_PDFHR/README.md](MimicKit_PDFHR/README.md) for training, testing and evaluation.

## Citation

If you find this work useful, please cite:

```bibtex
@article{gu2026pdfhr,
  title   = {Pose Distance Fields for Humanoid Robots},
  author  = {Gu, Yi and Gao, Yukang and Zhou, Yangchen and Chen, Xingyu and Feng, Yixiao and Zhao, Mingle and Mo, Yunyang and Wang, Zhaorui and Xu, Lixin and Xu, Renjing},
  journal = {arXiv preprint arXiv:2602.04851},
  year    = {2026}
}
```

## References

### MimicKit

```bibtex
@article{MimicKitPeng2025,
  title         = {MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control},
  author        = {Peng, Xue Bin},
  year          = {2025},
  eprint        = {2510.13794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GR},
  url           = {https://arxiv.org/abs/2510.13794}
}
```

### NRDF

```bibtex
@inproceedings{He_2024_CVPR,
  author    = {He, Yannan and Tiwari, Garvita and Birdal, Tolga and Lenssen, Jan Eric and Pons-Moll, Gerard},
  title     = {NRDF: Neural Riemannian Distance Fields for Learning Articulated Pose Priors},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {1661--1671}
}
```

## License

Except where otherwise noted, this project is licensed under the [Apache License 2.0](LICENSE). Third-party components remain subject to their respective licenses.
