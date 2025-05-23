# **ONDA-Pose**  
**Occlusion-Aware Neural Domain Adaptation for Self-Supervised6D Object Pose Estimation**  

ONDA-Pose is a **self-supervised** framework for **6D object pose estimation**, leveraging **Neural Radiance Fields (NeRFs)** to bridge the **real-synthetic domain gap** via a **render-and-compare self-training strategy**.

## **✨ Key Features**
- 🎨 **CAD-like Radiance Field:** Translates real training images into the synthetic domain with NeRFs.
- 🔧 **Global Pose Refinement:** Refines all pseudo object poses jointly for improved consistency.
- 🌀 **Render-and-Compare:** Performs self-supervised training with minimal domain gap.

## **🚀 Quick Start**
### **1. Clone Repository**
```bash
git clone https://github.com/tan-tao11/ONDA-Pose.git
cd ONDA-Pose
```

### **2. Install Dependencies**
```bash
conda create -n onda-pose python=3.9 -y
conda activate onda-pose
pip install -r requirements.txt
```

### **3. Prepare Datasets**
Dataset Structure:
```bash
datasets/
  ├── lm/
  │   ├── carf/         # Training data for Carf
  │   └── estimator/    # Data for pose estimator
  └── lmo/
      ├── carf/
      └── estimator/
```

🔗 Download demo data:
[OneDrive Link](https://1drv.ms/u/c/054882095addfd6a/ET8Nx87wIgVLvX_yTxkH3ZMBaa6EoDRCksNUSJG3VDhp7Q?e=p756Ga)

### **4. Download Pre-trained Weights**
```bash
mkdir -p weights/gdrn/lm
```
🔗 Download weights from [Self6dpp](https://github.com/THU-DA-6D-Pose-Group/Self6dpp):
[OneDrive Link](https://1drv.ms/u/c/054882095addfd6a/EXkfGthAF2hFsEgJhNHIa5cBN7XR-ELVALWfefOjmv4V1Q?e=4rNqoX)

### **5. Run Training**
**Train Carf:**
```bash
python train_carf.py --config configs/carf/lm/carf_pretrain.yaml  # Pre-training of geometry block
python train_carf.py --config configs/carf/lm/carf_texture_train.yaml # Training of radiance block
```
**Render images and masks:**
```bash
python render.py --config configs/Carf/lm/carf_texture_train.yaml data.image_size=[480,640] resume=True render.save_path=datasets/lm/estimator/test/000001/rgb_syn batch_size=1 data.preload=false
```
**Train pose estimator:**
```bash
python pose_pipeline.py --config configs/estimator/lm/pred_init.yaml  # Predict initial poses for rendered images
python pose_pipeline.py --config configs/estimator/lm/pose_refine.yaml  # Globally refine all initial poses
python pose_pipeline.py --config configs/estimator/lm/self_train.yaml  # Self-train pose estimator
```

## 📖 Citation
If you use ONDA-Pose in your research, please cite:
```bash
@inproceedings{tan2025onda,
  title     = {ONDA-Pose: Occlusion-Aware Neural Domain Adaptation for Self-Supervised 6D Object Pose Estimation},
  author    = {Tao Tan and Qiulei Dong},
  booktitle = {CVPR},
  year      = {2025}
}
```

## 🎯 Acknowledgements
Built upon the excellent work of:

- [Self6dpp](https://github.com/THU-DA-6D-Pose-Group/Self6dpp)
- [GDR-Net](https://github.com/THU-DA-6D-Pose-Group/GDR-Net)
- [Tex-Pose](https://github.com/HanzhiC/TexPose)

We sincerely thank the authors for their valuable contributions.
