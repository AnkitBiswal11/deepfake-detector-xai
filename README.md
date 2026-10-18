# DeepFake Detector XAI

<p align="center">
  <img width="85%" src="./demo/logo.png" />
</p>

<p align="center">
  <i>An explainable deepfake detection and facial artifact forensics pipeline with PyTorch and Grad-CAM.</i>
</p>

## Introduction

This repository provides an integrated deep learning framework for facial manipulation detection, feature visualization, and model interpretability. While standard classifiers operate as black boxes, this project pairs high-performing CNN and Transformer backbones with **layer-wise Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize internal convolutional feature activations—highlighting blending seams, boundary warping, and frequency artifacts in real time.

The core architectures and dataset configurations are adapted and extended from research by [Fudan Vision and Learning Lab](https://fvl.fudan.edu.cn/):

- **CNN Models:**
  - [ResNet](https://arxiv.org/abs/1512.03385)
  - [Xception](https://arxiv.org/abs/1610.02357)
  - [EfficientNet](https://arxiv.org/abs/1905.11946)
  - [MesoNet](https://arxiv.org/abs/1809.00888)
  - [GramNet](https://arxiv.org/abs/2002.00133)
  - [Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues](https://arxiv.org/abs/2007.09355)
  - [Multi-attentional Deepfake Detection](https://arxiv.org/abs/2103.02406)

- **Transformer Models:**
  - [ViT](https://arxiv.org/abs/2010.11929)
  - [M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection](https://arxiv.org/abs/2104.09770)

---

## Explainable AI (Grad-CAM) Visual Forensics

By hooking into the final convolutional residual block (`net.0.layer4`), the framework computes activation gradients to pinpoint where the model detects synthesis anomalies:

<p align="center">
  <img width="70%" src="./demo/gradcam_output.jpg" alt="Grad-CAM Output" />
</p>

- **Red / Yellow Regions:** High gradient activation (regions driving the "Fake / Manipulated" decision, such as facial blending borders and mouth boundaries).
- **Blue / Cyan Regions:** Low-importance background context features suppressed during inference.

---

## Model Zoo and Baselines

### FaceForensics++ (FF-DF) Dataset AUC

| Method          | RAW   | C23   | C40   | Model Weights Checkpoint                                                                                                                                                                                                                                                          |
| --------------- | ----- | ----- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ResNet50        | 97.61 | 94.87 | 84.95 | [RAW](https://drive.google.com/file/d/1OPb5wQditLd2dd9O4Lx3uLEfei1e6WUt/view?usp=sharing) / [C23](https://drive.google.com/file/d/186i-BiOfl_-JPkTESROJ-035Tsl75P__/view?usp=sharing) / [C40](https://drive.google.com/file/d/1HmBu1UDVP-YK8mMXdcVUFWmmXaQ9k3Ts/view?usp=sharing) |
| Xception        | 97.84 | 95.24 | 86.27 | [RAW](https://drive.google.com/file/d/15FJreogGs70SIyJSIiNrAICJyHOHGOcp/view?usp=sharing) / [C23](https://drive.google.com/file/d/1tNvZ4DqLiDjL_C2YV89ZDPZhjvOSunnR/view?usp=sharing) / [C40](https://drive.google.com/file/d/1f9STu_JWCh9HIVJ_6iCnjE3h5W0V2Lbh/view?usp=sharing) |
| EfficientNet-b4 | 97.89 | 95.61 | 87.12 | [RAW](https://drive.google.com/file/d/10skQENw6di98vgCW8Sd8Pf4iKJUdxEeH/view?usp=sharing) / [C23](https://drive.google.com/file/d/1Y6wCqh2MwDRgEHyCnL0f_JCCdbnlquz2/view?usp=sharing) / [C40](https://drive.google.com/file/d/1c_bPzgGWwkOdT-4CoZTBse63TxcjgEVN/view?usp=sharing) |
| Meso4           | 85.14 | 77.14 | 60.13 | [RAW](https://drive.google.com/file/d/1LFCGiO6BVZlFLE1CMV_MDqFsOLMxperg/view?usp=sharing) / [C23](https://drive.google.com/file/d/1jJsUY8aA5eQf-ol4eqA1ICAav6Io_bRE/view?usp=sharing) / [C40](https://drive.google.com/file/d/1fCcUxninIfDB1TV5cDNW9Ou_7sAyTJRV/view?usp=sharing) |
| MesoInception4  | 95.45 | 84.13 | 71.31 | [RAW](https://drive.google.com/file/d/16NQSsEKMqFiFb94ZsD3w2LT5mBuBdU_Q/view?usp=sharing) / [C23](https://drive.google.com/file/d/1i4D3Eo1Yb3Tej1g_JnRjRLsJHATISpNf/view?usp=sharing) / [C40](https://drive.google.com/file/d/1R6gwhH-KE123GRLq7Svcg1KxHj9gjCk7/view?usp=sharing) |
| GramNet         | 97.65 | 95.16 | 86.21 | [RAW](https://drive.google.com/file/d/1UdDmcEEAEWEm9rhd6KRPdNar70yjht1d/view?usp=sharing) / [C23](https://drive.google.com/file/d/1oGmLgMcm-XnsVT6abNApGDhDS7ML6-IT/view?usp=sharing) / [C40](https://drive.google.com/file/d/1W9aamR5CLqROwDuQ_1EGfJ3zq1k1HuT7/view?usp=sharing) |
| F3Net           | 99.95 | 97.52 | 90.43 | [RAW](https://drive.google.com/file/d/1mUNeR-r5vi-dNtxw4wIBxGaLZculxfQR/view?usp=sharing) / [C23](https://drive.google.com/file/d/1epecuHtJTLH-T9rTpm92djzVG5RHBnXg/view?usp=sharing) / [C40](https://drive.google.com/file/d/1RFuIk-0pzBS_jaZxEsu3TpupxWV4s4Ne/view?usp=sharing) |
| MAT             | 97.90 | 95.59 | 87.06 | [RAW](https://drive.google.com/file/d/1FHGy1A4veezdiuJCrUGJDdp79pahM4hA/view?usp=sharing) / [C23](https://drive.google.com/file/d/1gwavqYZFljl3nvebzthm_JekmT1clWkb/view?usp=sharing) / [C40](https://drive.google.com/file/d/1e_Byw7zvTgYD1tsrVbVuBCOVNGzwvhFg/view?usp=sharing) |
| ViT             | 96.72 | 93.45 | 82.97 | [RAW](https://drive.google.com/file/d/1fA6yYfC2fggpGujLNIUZoooFGW8f8Ima/view?usp=sharing) / [C23](https://drive.google.com/file/d/1kcKeSoTD1s0EWs0yqvkojNBR4lRR7Jdh/view?usp=sharing) / [C40](https://drive.google.com/file/d/1X_esyW89arAdXmNVQBtUn215LYXjsho0/view?usp=sharing) |
| M2TR            | 99.50 | 97.93 | 92.89 | [RAW](https://drive.google.com/file/d/1_HaPE6r7Zzof2mmLmmc4fbIbqyWs17S0/view?usp=sharing) / [C23](https://drive.google.com/file/d/1XRIllA6p5YnITztl1burwcr5l7LAcpqv/view?usp=sharing) / [C40](https://drive.google.com/file/d/1xhclIjoh8GkVvoVefjDY-itdaV0VaMxY/view?usp=sharing) |

### Celeb-DF Dataset AUC

| Method          | Celeb-DF | Model Checkpoint                                                                              |
| --------------- | -------- | --------------------------------------------------------------------------------------------- |
| ResNet50        | 98.51    | [CelebDF](https://drive.google.com/file/d/1dcek81dAiSx6iJMYazncdrX-6cYBVYww/view?usp=sharing) |
| Xception        | 99.05    | [CelebDF](https://drive.google.com/file/d/19OL8lNeyFE3-23WX9JN9FPBbGK1ML_Z2/view?usp=sharing) |
| EfficientNet-b4 | 99.44    | [CelebDF](https://drive.google.com/file/d/1y4DrlZ5-B8Cl9sRYTqFYcs5pfcERQurj/view?usp=sharing) |
| Meso4           | 73.04    | [CelebDF](https://drive.google.com/file/d/1toHsZzMbo0ul9utmeF8QHvHdPkZPHBZh/view?usp=sharing) |
| MesoInception4  | 75.87    | [CelebDF](https://drive.google.com/file/d/1bQQtyKpG-ZN0RChZkmk3I76HnAiZs0Bj/view?usp=sharing) |
| GramNet         | 98.67    | [CelebDF](https://drive.google.com/file/d/1Zt9hsoU7cNh5mmMbKiAnRFeF63ycY8O1/view?usp=sharing) |
| F3Net           | 96.47    | [CelebDF](https://drive.google.com/file/d/1iHYJnvCmakjUs0QJH53ZgwwaWvudOp34/view?usp=sharing) |
| MAT             | 99.02    | [CelebDF](https://drive.google.com/file/d/1WApYy9ekgEq-Kt8641f9fAdThy1-ndqB/view?usp=sharing) |
| ViT             | 96.73    | [CelebDF](https://drive.google.com/file/d/1gzh9WlUE50sQ3meVAiagujADi6VR4t5F/view?usp=sharing) |
| M2TR            | 99.76    | [CelebDF](https://drive.google.com/file/d/19mPqJ1DzkPr89VHVjHD2b0dqiZjufZnG/view?usp=sharing) |

---

## Installation & Setup

Environment specifications: Python 3.9 on Windows/Linux with NVIDIA CUDA acceleration.

```powershell
# 1. Install PyTorch with CUDA 11.3 support
pip install --no-deps torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f [https://download.pytorch.org/whl/cu113/torch_stable.html](https://download.pytorch.org/whl/cu113/torch_stable.html)

# 2. Install pinned dependencies (avoids NumPy 2.x and OpenCV conflicts)
pip install "numpy==1.23.5" "opencv-python<4.9.0" "opencv-python-headless<4.9.0" matplotlib "kornia==0.6.4" timm==0.5.4 fvcore yacs gdown
Quick Start: Single-Image Inference & Grad-CAM\

1. Download Checkpoint Weights
Download the pre-trained ResNet-50 weights fine-tuned on the FaceForensics++ C23 benchmark:
gdown "[https://drive.google.com/uc?id=186i-BiOfl_-JPkTESROJ-035Tsl75P](https://drive.google.com/uc?id=186i-BiOfl_-JPkTESROJ-035Tsl75P)__" -O resnet50_c23.pyth

2. Run Inference and Generate Heatmap
Target the final residual convolutional block (net.0.layer4) to compute class activations:
$env:PYTHONPATH = "."
python visualization/gradcam.py --model ResNet50 --layer net.0.layer4 --pth resnet50_c23.pyth --img demo/manipulated_face.jpg --save_path demo/gradcam_output.jpg

Output:
python run.py --cfg resnet50.yaml

Citations
@inproceedings{wang2021m2tr,
  title={M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection},
  author={Wang, Junke and Wu, Zuxuan and Ouyang, Wenhao and Han, Xintong and Chen, Jingjing and Lim, Ser-Nam and Jiang, Yu-Gang},
  booktitle={ICMR},
  year={2022}
}

@article{selvaraju2017gradcam,
  title={Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization},
  author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedaldi, Andrea and Parikh, Devi and Batra, Dhruv},
  journal={ICCV},
  year={2017}
}

---

### What to Do Next

1. In VS Code, open `README.md`.
2. Press `Ctrl + A`, paste the markdown block above, and save with `Ctrl + S`.
3. Check `git status` in your terminal to verify that only `README.md` and your updated scripts are modified.

<FollowUp label="Would you like to stage and commit this updated README to Git?" query="Show me the Git commands to commit this updated README.md and push it to GitHub."/>
