[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

# 3D Flow Prediction via Latent Diffusion

## 📖 Overview

This repository predicts **3D resin flow fields** (velocity vx, vy, vz) in fibrous composite microstructures using a **latent diffusion model**. Given a 2D microstructure image and 2D flow field (where vz=0), the model predicts the full 3D flow field including the out-of-plane vz component.

### Architecture Pipeline

```
2D Microstructure + 2D Flow (vz=0)
         ↓
    E2D Encoder (Stage 2)
         ↓
    Latent z (8 channels)
         ↓
    Diffusion U-Net (denoising)
         ↓
    D3D Decoder (Stage 1)
         ↓
    3D Flow Field (vx, vy, vz)
```

### Training Pipeline

The model is trained in 3 stages:

1. **Stage 1 (VAE 3D)**: Train E3D + D3D on 3D velocity fields for reconstruction
2. **Stage 2 (VAE 2D + Alignment)**: Train E2D aligned to E3D latent space with cross-reconstruction (E2D→D3D)
3. **Diffusion**: Train U-Net to denoise latents conditioned on 2D microstructure + 2D velocity

---

## ⬇️ Getting Started

```bash
git clone https://github.com/your-repo/Diffusion_model_project
cd Diffusion_model_project
```

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate
# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The 3D dataset should be placed at `C:\Users\alexd\Downloads\dataset_3d` (or specify via `--dataset-dir`).

Required files in `dataset_3d/x/`:
- `domain.pt` - 2D microstructure (N, 1, H, W)
- `U_2d.pt` - 2D velocity input (N, num_slices, 3, H, W) where vz=0
- `U.pt` - 3D velocity target (N, num_slices, 3, H, W)
- `statistics.json` - Normalization statistics

---

## 🚀 How to Run

### Stage 1: Train 3D VAE (E3D + D3D)

```bash
cd VAE_model
python train_3d_vae_only.py \
    --dataset-dir "C:/Users/alexd/Downloads/dataset_3d" \
    --save-dir "trained/dual_vae_stage1_3d" \
    --latent-channels 8 \
    --batch-size 2 \
    --num-epochs 50 \
    --per-component-norm
```

### Stage 2: Train 2D Encoder with Alignment (E2D)

```bash
cd VAE_model
python train_2d_with_cross.py \
    --dataset-dir "C:/Users/alexd/Downloads/dataset_3d" \
    --save-dir "trained/dual_vae_stage2_2d" \
    --stage1-checkpoint "trained/dual_vae_stage1_3d" \
    --latent-channels 8 \
    --batch-size 2 \
    --num-epochs 50 \
    --lambda-align 0.1 \
    --lambda-cross 1.0 \
    --per-component-norm
```

### Stage 3: Train Latent Diffusion Model

```bash
cd Diffusion_model
python train.py \
    --root-dir "C:/Users/alexd/Downloads/dataset_3d" \
    --vae-encoder-path "../VAE_model/trained/dual_vae_stage2_2d" \
    --vae-decoder-path "../VAE_model/trained/dual_vae_stage1_3d" \
    --predictor-type latent-diffusion \
    --in-channels 17 --out-channels 8 \
    --features 64 128 256 512 1024 \
    --attention "3..2" \
    --batch-size 3 \
    --num-epochs 100 \
    --learning-rate 1e-4 \
    --use-3d True
```

### Inference

```bash
cd Inference
python inference.py \
    "../Diffusion_model/trained/YOUR_MODEL_DIR" \
    --vae-encoder-path "../VAE_model/trained/dual_vae_stage2_2d" \
    --vae-decoder-path "../VAE_model/trained/dual_vae_stage1_3d" \
    --dataset-dir "C:/Users/alexd/Downloads/dataset_3d" \
    --index 0
```

### Grid Search (Hyperparameter Tuning)

```bash
cd Diffusion_model
python gridsearch_diffusion.py
```

---

## 📂 Project Structure

```
├── Diffusion_model/           # Latent diffusion model
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   ├── gridsearch_diffusion.py
│   ├── config.py              # CLI arguments
│   ├── src/
│   │   ├── predictor.py       # LatentDiffusionPredictor
│   │   ├── diffusion.py       # DDPM scheduler
│   │   ├── helper.py          # Training loop
│   │   ├── physics.py         # Physics-informed losses
│   │   └── unet/              # U-Net architecture
│   └── utils/
│       └── dataset.py         # MicroFlowDataset
│
├── VAE_model/                 # VAE components
│   ├── train_3d_vae_only.py   # Stage 1 training
│   ├── train_2d_with_cross.py # Stage 2 training
│   ├── inference_vae.py       # VAE visualization
│   ├── src/
│   │   ├── vae/               # Standard VAE (E3D/D3D)
│   │   └── dual_vae/          # Dual-branch VAE
│   └── utils/
│       └── dataset.py         # MicroFlowDatasetVAE
│
├── Inference/
│   └── inference.py           # End-to-end inference with Napari visualization
│
└── requirements.txt
```

---

## 🔧 Key Configuration Options

### Diffusion Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--in-channels` | 17 | 1 (microstructure) + 8 (latent) + 8 (time embed) |
| `--out-channels` | 8 | Must match VAE latent_channels |
| `--features` | [64,128,256,512,1024] | U-Net depth (5 levels) |
| `--attention` | "3..2" | Attention at levels 3+ with 2 heads |
| `--num-slices` | 11 | Number of z-slices in 3D volume |

### Physics Losses (Optional)

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda-div` | 0.0 | Divergence loss (mass conservation) |
| `--lambda-flow` | 0.0 | Flow-rate consistency |
| `--lambda-smooth` | 0.0 | Gradient smoothness |
| `--lambda-laplacian` | 0.0 | Laplacian smoothness |

---

## 📝 Removed Legacy Code

The following legacy code from the original 2D→2D flow prediction project has been removed:

- **Legacy predictors**: `VelocityPredictor`, `PressurePredictor` (replaced by `LatentDiffusionPredictor`)
- **Legacy losses**: `mass_conservation_loss`, `mass_consv_loss`, `normalized_exp_loss` (replaced by `src/physics.py`)
- **Legacy config options**: `predictor-type velocity/pressure` choices

The physics-informed losses (`lambda_div`, `lambda_flow`, `lambda_smooth`, `lambda_laplacian`) are retained as they are still used for optional training regularization.

---

## Questions

For questions, please open an issue or contact the maintainers.