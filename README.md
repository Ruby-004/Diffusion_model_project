# Latent Diffusion Model for Microstructure-Based Flow Prediction

This project implements a machine learning pipeline for predicting **3D resin flow velocity fields** in fibrous composite microstructures. The approach combines a **Dual-Branch Variational Autoencoder (VAE)** with a **latent diffusion process** to predict 3D velocity fields (with non-zero vertical component) from 2D velocity inputs.

**Key idea**: Given a 2D velocity field (where the vertical component $v_z = 0$), the model predicts the full 3D velocity field with realistic vertical flow components.

**Disclaimer**: This model is trained on CFD simulations that relies in several physical assumptions. With the added model errors, this model doesn't guarantee 100% fidelity in representation of real life flow phenomena. When applied in industry, the model should be used only as a supplementary tool and validated through additional testing.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Steps to Reproduce Results](#steps-to-reproduce-results)
3. [Program Structure and Design](#program-structure-and-design)
4. [Key Features](#key-features)

---

## Quick Start

### Prerequisites
- **Python 3.11** (tested)
- **CUDA 12.6+** for GPU acceleration (required for training)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/Ruby-004/Diffusion_model_project.git
cd Diffusion_model_project
```
```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
```
```bash
# Activate (Windows PowerShell)
venv\Scripts\activate
# Activate (Linux/macOS)
# source venv/bin/activate
```
```bash
# Upgrade core packaging tools (recommended)
python -m pip install --upgrade pip setuptools wheel
```


```bash
# Install PyTorch with CUDA 12.6 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For newer GPUs (RTX 40xx/50xx with CUDA 12.8):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
# Install remaining dependencies
pip install -r requirements.txt
```
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Steps to Reproduce Results

### 1. Download Pre-trained Models and Dataset

Pre-trained models and the dataset are hosted on Zenodo: [**https://doi.org/10.5281/zenodo.18341260**](https://doi.org/10.5281/zenodo.18341260)

**Downloads:**

All files are available from the same Zenodo record:

1. **Dataset** (`dataset_3d.zip`, ~2.1 GB) - required for training and evaluation:
   - **Automatic**: Simply run any training script with an empty/non-existent `--dataset-dir` or `--root-dir` path, and the dataset will download automatically
   - **Manual**: Download from [Zenodo Record](https://zenodo.org/records/18341260/files/dataset_3d.zip?download=1) and extract to your preferred location (e.g., `./dataset_3d`)

2. **VAE Models** (`VAE's.zip`, ~1.7 GB) - required for latent diffusion:
   - Download from [Zenodo Record](https://zenodo.org/records/18341260/files/VAE%27s.zip?download=1)
   - Extract to `VAE_model/trained/` (creates `dual_vae_stage1_3d/` and `dual_vae_stage2_2d/` folders)

3. **Diffusion Model** (`20260120_unet_latent-diffusion_...zip`, ~2.2 GB) - for inference only, optional:
   - Download from [Zenodo Record](https://zenodo.org/records/18341260/files/20260120_unet_latent-diffusion_in-17-out-8-f-5-k-3-p-zeros-a-3..2-dr-0.0-wd-0.00e%2B00-b-2-lr-1.00e-03-ep-104.zip?download=1)
   - Extract to `Diffusion_model/trained/`

#### Directory Structure After Download

```
project_root/
├── data/
│   └── dataset_3d/
│       ├── x/                       
│       │   ├── domain.pt           # Microstructure masks
│       │   ├── U_2d.pt             # 2D velocity (vz=0)
│       │   ├── U.pt                # 3D velocity (target)
│       │   ├── p.pt                # Pressure fields
│       │   └── dxyz.pt             # Physical dimensions
│       │   
│       └── statistics.json         # Normalization statistics
│
├── VAE_model/
│   └── trained/
│       ├── dual_vae_stage1_3d/
│       │   ├── model.pt
│       │   ├── best_model.pt
│       │   └── vae_log.json
│       └── dual_vae_stage2_2d/
│           ├── model.pt
│           ├── best_model.pt
│           └── vae_log.json
│
└── Diffusion_model/
    └── trained/
        └── [model_folder]/
            ├── model.pt
            ├── best_model.pt
            └── log.json
```

### 2. Running the Code

All shell commands below use the hyperparameters from the final trained model. These can be adjusted to experiment with different configurations.

> **Note**: On Windows PowerShell, use backticks (`` ` ``) for line continuation. On Linux/macOS, use backslashes (`\`).

#### A. Data Requirements

**Important**: The model expects input microstructures of **256 × 256 pixels** in the (x, y) plane (corresponding to 50μm × 50μm physical dimensions). The default configuration uses **11 slices** along the z-axis (depth). The provided dataset is already in this format.

#### B. Training a VAE (Two-Stage Process)

All dataset loaders use a hardcoded **seed=2024** for the **70/15/15** train/validation/test split. This ensures consistent data splits across VAE training, diffusion training, evaluation, and inference.

**Stage 1: Train 3D VAE (Encoder E3D + Decoder D3D)**

```bash
cd VAE_model
```
```bash
python train_3d_vae_only.py `
  --dataset-dir ../dataset_3d `
  --save-dir trained/dual_vae_stage1_3d `
  --in-channels 3 `
  --latent-channels 8 `
  --batch-size 2 `
  --num-epochs 100 `
  --learning-rate 1e-4 `
  --per-component-norm
```

> **Note**: If `--dataset-dir` points to an empty or non-existent directory, the dataset will be **automatically downloaded** from Zenodo and extracted to that location.

**Stage 2: Train 2D VAE (Encoder E2D + Decoder D2D) with Alignment**

This loads the Stage 1 checkpoint and trains the 2D components with latent alignment and cross-reconstruction losses.

```bash
python train_2d_with_cross.py `
  --dataset-dir ../dataset_3d `
  --save-dir trained/dual_vae_stage2_2d `
  --stage1-checkpoint trained/dual_vae_stage1_3d `
  --in-channels 3 `
  --latent-channels 8 `
  --batch-size 1 `
  --num-epochs 100 `
  --learning-rate 1e-4 `
  --per-component-norm `
  --lambda-align 5 `
  --lambda-cross 50
```

**Outputs**: 
- Stage 1: `trained/dual_vae_stage1_3d/` — 3D VAE checkpoint (model.pt, best_model.pt, vae_log.json)
- Stage 2: `trained/dual_vae_stage2_2d/` — Complete dual-branch VAE (model.pt, best_model.pt, vae_log.json)

#### C. Training the Latent Diffusion Model

The dataset loader uses a hardcoded **seed=2024** for the **70/15/15** train/validation/test split, ensuring the same samples are used across VAE training, diffusion training, and inference. This prevents data leakage (i.e., the diffusion model never trains on samples the VAE hasn't seen during training).

```bash
cd ..
cd Diffusion_model
```
```bash
python train.py `
  --root-dir ../dataset_3d `
  --vae-encoder-path ../VAE_model/trained/dual_vae_stage2_2d `
  --vae-decoder-path ../VAE_model/trained/dual_vae_stage1_3d `
  --predictor-type latent-diffusion `
  --in-channels 17 `
  --out-channels 8 `
  --features 64 128 256 512 1024 `
  --batch-size 2 `
  --num-epochs 100 `
  --learning-rate 1e-3 `
  --weight-decay 0 `
  --use-3d True `
  --num-slices 11 `
  --attention '3..2'
```

> **Auto-Download**: If `--root-dir` points to an empty or non-existent directory, the dataset (~2.1 GB) will be **automatically downloaded** from Zenodo to that location. You can start training immediately without manual download.

**Key parameters:**
- `--in-channels 17`: Microstructure (1) + 2D velocity latent (8) + timestep embedding (8) = 17
- `--out-channels 8`: Must match VAE latent channels
- `--features`: U-Net depth (5 levels in this case)

**Output**: Saves to `trained/{timestamp}_unet_latent-diffusion_[params]/` with:
- `model.pt`: Final trained model weights
- `best_model.pt`: Best validation loss checkpoint
- `log.json`: Full configuration and training history

#### Grid Search (Hyperparameter Tuning)

The final model hyperparameters were selected via grid search. The search space is defined in [gridsearch_diffusion.py](Diffusion_model/gridsearch_diffusion.py).

**Grid Search Space**:
| Parameter | Values | Description |
|-----------|--------|-------------|
| Network depth (features) | 4 variants | `[64,128,256,512]`, `[64,128,256,512,1024]`, `[32,64,128,256,512]`, `[128,256,512,1024,2048]` |
| Kernel size | `[3]` | Fixed |
| Attention | `["3..2"]` | Attention from level 3, 2 heads |
| Learning rate | `[5e-5, 1e-4, 5e-4, 1e-3]` | 4 values |
| Dropout | `[0.0]` | Fixed (no dropout) |
| Time embedding dim | `[64]` | Fixed |

**Total**: 4 × 4 = **16 configurations**, each trained for **10 epochs** (seed=2024).

To run the grid search:

```bash
cd Diffusion_model
```
```bash
python gridsearch_diffusion.py `
  --root-dir ../dataset_3d `
  --vae-encoder-path ../VAE_model/trained/dual_vae_stage2_2d `
  --vae-decoder-path ../VAE_model/trained/dual_vae_stage1_3d `
  --output-dir ./trained/gridsearch/ `
  --num-epochs 10
```

**Output**:
- `results.csv`: All 16 runs with hyperparameters and validation metrics
- `top10.csv`: Top 10 configurations ranked by validation loss
- `summary.txt`: Best configuration details

**Best configuration found**: `features=[64, 128, 256, 512, 1024]` (depth 5) with `learning_rate=1e-3`.

#### D. Evaluation

**Basic Test Set Evaluation**

Compute quantitative metrics on the test set:

```bash
cd Diffusion_model
```
```bash
python evaluate.py trained/[timestamp]_unet_latent-diffusion_[params]
```

**Output**: Prints test loss and saves metrics to the model directory.

**Comprehensive End-to-End Evaluation**

For detailed metrics including per-component errors and physical consistency measures:

```bash
cd ..
python scripts/eval_testset_end2end.py `
  --diffusion-model-path Diffusion_model/trained/[model_folder] `
  --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d `
  --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d `
  --dataset-dir ../dataset_3d `
  --sampler ddim `
  --steps 50
```

**Metrics computed**:
- Per-component MAE/MSE/RMSE for $v_x$, $v_y$, $v_z$ velocity components
- Normalized metrics (using dataset statistics)
- Cosine similarity between predicted and ground truth velocity vectors
- IoU of top-k magnitude voxels (flow structure agreement)

**Options**:
| Flag | Description |
|------|-------------|
| `--sampler ddim` | Use DDIM sampler (faster, default) |
| `--sampler ddpm` | Use DDPM sampler (slower, more accurate) |
| `--steps N` | Number of diffusion steps (default: 50) |
| `--sanity-mode` | VAE-only reconstruction (bypasses diffusion) |

**Output**: JSON file with mean ± std metrics; optional CSV with per-sample results.

#### E. Inference

**Single-sample prediction:**

```bash
python Inference/inference.py `
  Diffusion_model/trained/[model_folder] `
  --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d `
  --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d `
  --dataset-dir path/to/dataset_3d `
  --index 0
```

- `--dataset-dir`: Path to dataset (for loading test samples)
- `--index`: Test sample index (default: 0)

**Output**: 
- `velocity_field_comparison.png`: Side-by-side visualization
- Interactive 3D visualization via Napari viewer

**VAE-only inference** (to verify VAE reconstruction quality):

```bash
cd VAE_model
```
```bash
python inference_vae.py `
  --model-path trained/dual_vae_stage2_2d `
  --dataset-dir ../dataset_3d `
  --index 0
```

Displays:
- Original vs reconstructed 2D velocity fields
- Original vs reconstructed 3D velocity fields
- Cross-reconstruction: 2D latent → 3D decoder output

#### F. Visualizing Training Progress

**Plot diffusion model training loss:**

```bash
cd ..
cd Diffusion_model
```
```bash
python scripts/plot_loss.py trained/[model_folder]
```

**Plot physics metrics** (if physics-informed training was used):

```bash
python scripts/plot_physics_metrics.py trained/[model_folder]
```

**Plot VAE training loss:**

```bash
cd ..
cd VAE_model
```
```bash
python plot_vae_loss.py trained/dual_vae_stage1_3d
```
```bash
# For stage 2 dual VAE
python plot_vae_loss.py trained/dual_vae_stage2_2d
```

**Output**: Generates loss curves showing training and validation metrics over epochs, including reconstruction loss, KL divergence, and (for stage 2) alignment and cross-reconstruction losses.

---

## Program Structure and Design

### File Structure Overview

```
project_root/
├── VAE_model/                         # Variational Autoencoder (Dual-Branch)
│   ├── train_3d_vae_only.py           # Stage 1: Train 3D VAE (E3D + D3D)
│   ├── train_2d_with_cross.py         # Stage 2: Train 2D VAE with alignment
│   ├── inference_vae.py               # VAE inference/visualization
│   ├── plot_vae_loss.py               # Plot training curves
│   ├── config/
│   │   └── vae.py                     # VAE configuration
│   ├── src/
│   │   ├── common.py                  # Shared utilities
│   │   ├── dual_vae/
│   │   │   └── model.py               # DualBranchVAE (E2D, D2D, E3D, D3D)
│   │   └── vae/
│   │       ├── autoencoder.py         # Standard VAE wrapper
│   │       ├── encoder.py             # Encoder with attention
│   │       ├── decoder.py             # Decoder (mirror of encoder)
│   │       └── blocks.py              # Residual and attention blocks
│   ├── utils/
│   │   ├── dataset.py                 # VAE dataset loader
│   │   ├── metrics.py                 # Loss functions
│   │   └── paired_sampler.py          # Paired 2D/3D sampling
│   └── trained/                       # Saved model checkpoints
│
├── Diffusion_model/                   # Latent Diffusion Model
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation script
│   ├── gridsearch_diffusion.py        # Hyperparameter search
│   ├── config.py                      # CLI argument parsing
│   ├── src/
│   │   ├── predictor.py               # LatentDiffusionPredictor class
│   │   ├── diffusion.py               # DDPM/DDIM scheduler
│   │   ├── physics.py                 # Physics-informed losses
│   │   ├── normalizer.py              # Data normalization
│   │   ├── helper.py                  # Training utilities
│   │   └── unet/
│   │       ├── models.py              # U-Net architecture
│   │       ├── blocks.py              # U-Net building blocks
│   │       └── metrics.py             # Loss functions
│   ├── utils/
│   │   ├── dataset.py                 # 3D microflow dataset loader
│   │   └── zenodo.py                  # Zenodo download utilities
│   ├── scripts/
│   │   ├── plot_loss.py               # Plot training loss
│   │   └── plot_physics_metrics.py    # Plot physics metrics
│   └── trained/                       # Saved model checkpoints
│
├── Inference/                         # Standalone inference
│   └── inference.py                   # Run predictions on test samples
│
├── scripts/                           # Root-level evaluation scripts
│   └── eval_testset_end2end.py        # Comprehensive end-to-end evaluation
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

### Architecture Design

#### Model workflow - Quick overview
Stage 1 - VAE 3D Pre-training: How the 3D encoder/decoder learns to compress and reconstruct 3D flows


Stage 2 - Dual-Branch VAE with Alignment: Weights from pre-training 3D are frozen. A new 2D encoder + decoder branch is trained with the previously learned information of the 3D VAE. The critical alignment and cross-reconstruction losses force the 2D encoder to learn vertical velocity information.


Stage 3: Training the Diffusion Model: Complete flow of adding noise, then denoising with U-net, and learning from microstructure context.


Stage 4: Inference Phase: New predictions are generated (from random noise → iterative denoising → final output)



#### 1. **Variational Autoencoder (VAE) - Dual-Branch Architecture**

**Purpose**: Learn aligned latent representations for 2D and 3D velocity fields, enabling cross-reconstruction from 2D inputs to 3D outputs.

**Architecture**:

- **2D Branch (E2D + D2D)**: Processes 2D velocity fields
  - Input: `(batch, slices, 3, H, W)` — 3 channels for $[v_x, v_y, v_z]$ where $v_z=0$
  - Latent: `(batch, latent_channels, slices, H/4, W/4)`
  - Uses 2D convolutions per slice

- **3D Branch (E3D + D3D)**: Processes 3D velocity fields
  - Input: `(batch, slices, 3, H, W)` — 3 channels for $[v_x, v_y, v_z]$ with non-zero $v_z$
  - Latent: `(batch, latent_channels, slices, H/4, W/4)`
  - Uses 3D convolutions across the slice dimension

> **Note**: Spatial dimensions (H, W) are downsampled by 4×, but the slice/depth dimension is preserved.

**Key Innovation**: Cross-reconstruction loss forces E2D to learn enough information for D3D to predict the vertical velocity component ($v_z$).

**Training Process (Two Stages)**:

| Stage | Components | Objective |
|-------|------------|-----------|  
| 1 | E3D + D3D | Reconstruct 3D velocity fields (establishes baseline) |
| 2 | E2D + D2D | Reconstruct 2D fields + align latents with E3D + cross-reconstruct via D3D |

**Stage 2 Loss Function**:
$$L = L_{\text{rec-2d}} + L_{\text{rec-3d}} + \lambda_{\text{align}} \cdot L_{\text{align}} + \lambda_{\text{cross}} \cdot L_{\text{cross}} + \beta_{\text{kl}} \cdot KL$$

**Loss Terms:**
| Term | Description |
|------|-------------|
| $L_{\text{rec-2d}}$ | MSE between original and reconstructed 2D velocity |
| $L_{\text{rec-3d}}$ | MSE between original and reconstructed 3D velocity |
| $L_{\text{align}}$ | MSE between E2D and E3D latent representations (paired samples) |
| $L_{\text{cross}}$ | MSE between ground truth 3D and D3D(E2D(2D input)) |
| $KL$ | KL divergence (sum of 2D and 3D terms) |

**Key Design Choices**:
- **Asymmetric padding** in stride-2 convolutions prevents checkerboard artifacts
- **Attention blocks** capture long-range dependencies in flow patterns  
- **KL weight** ($\beta_{kl}$): Set to `1e-3` to prioritize reconstruction quality
- **Per-component normalization**: Normalizes each velocity component ($v_x$, $v_y$, $v_z$) separately for better $v_z$ learning

#### 2. **Latent Diffusion Model**

**Purpose**: Generate 3D velocity fields from 2D velocity inputs via iterative denoising in VAE latent space.

**Inputs**:
- 2D microstructure: `(batch, num_slices, 1, H, W)` — binary mask (1=fluid, 0=solid)
- 2D velocity: `(batch, num_slices, 3, H, W)` — initial velocity where $v_z=0$

**Pipeline**:
```
Training:
  2D velocity → E2D (frozen) → latent z
  Add noise to z at timestep t → noisy latent z_t
  U-Net predicts noise ε given (z_t, microstructure, t)
  Loss: MSE(predicted ε, actual ε)

Inference:
  Start with random noise z_T
  For t = T, T-1, ..., 1:
    U-Net predicts noise ε
    z_{t-1} = denoise(z_t, ε)
  Decode: D3D(z_0) → 3D velocity field
```

**Key Design Choices**:
- **Frozen E2D/D3D**: Prevents catastrophic forgetting of VAE compression
- **Cross-branch decoding**: Uses E2D for encoding (trained on 2D) but D3D for decoding (outputs 3D), leveraging the stage 2 alignment loss
- **Multi-scale U-Net**: Features `[64, 128, 256, 512, 1024]` (5 levels) for capturing multi-scale flow patterns
- **Timestep embedding**: Sinusoidal positional encoding for diffusion step awareness
- **Conditional features**: Early concatenation of microstructure and velocity latents

#### 3. **Physics-Informed Losses** (Experimental)

> **Note**: Physics-informed losses were explored during development but the final model achieved best results **without** them. They are documented here for reference.

**Available physics constraints**:
| Loss | Parameter | Description |
|------|-----------|-------------|
| Divergence | `--lambda-div` | Enforces $\nabla \cdot \mathbf{u} = 0$ (mass conservation) |
| Flow-rate | `--lambda-flow` | Penalizes variation in volumetric flow rate $Q(x)$ |
| No-slip | `--lambda-laplacian` | Enforces $\mathbf{u} = 0$ in solid regions |
| Smoothness | `--lambda-smooth` | Tikhonov regularization for stable gradients |

**Implementation details**:
- Physics losses are computed on decoded velocity fields (through frozen D3D)
- Uses DDPM posterior estimate $\hat{x}_0$ for gradient stability
- Combined with noise prediction loss via weighted sum

---

### Design Rationale

#### Why Latent Diffusion?

1. **Computational efficiency**: Operating in compressed latent space (256x vs pixel space) reduces memory and compute
2. **Semantic understanding**: VAE learns meaningful flow patterns; diffusion refines in semantic space
3. **Stability**: Lower-dimensional space facilitates more stable training than pixel-space diffusion
4. **Flexibility**: Frozen VAE allows quick model updates without retraining compression


#### Why Conditional on 2D Flow?

1. **Better initialization**: 2D field provides strong signal for 3D reconstruction
2. **Practical relevance**: 2D fields are cheaper to compute or may be available from simplified models

### Main Difficulties and Solutions

#### 1. **VAE Training Stability and W-Component Learning**

**Problem**: Model could collapse to near-zero latents or struggle with w-component prediction.

**Solution**: 
- Two-stage training separates 3D baseline learning from 2D alignment
- Cross-reconstruction loss in stage 2 directly forces E2D to learn w-component information
- Per-component normalization option to give w-component more weight
- Reduced KL weight from 1e-0 to 1e-3 (prioritizing reconstruction)
- Applied scale factor normalization for stable gradients
- Used normalized MAE loss (divides by target magnitude) for scale-invariant training


#### 2. **Dataset Consistency**

**Problem**: VAE and diffusion models could use different data splits, causing train-test contamination.

**Solution**:
- **Hardcoded seed=2024** in all dataset loaders (`Diffusion_model/utils/dataset.py`, `VAE_model/utils/dataset.py`)
- Same 70/15/15 split used for VAE training, diffusion training, evaluation, and inference
- Ensures test samples were never seen during training of either model
- Generated shared `statistics.json` for normalization consistency

#### 3. **Path and Configuration Management**

**Problem**: Model paths broken when loading across different machines/directories.

**Solution**:
- Implemented auto-detection of absolute vs relative paths
- Store full config (including VAE path) in model's `log.json`
- Auto-fix VAE paths during model loading if file not found
- Support Zenodo URL detection and auto-download

### Limitations of the Design

#### 1. **Latent Space Bottleneck**

- **Limitation**: Compressing 3D velocity to 4-8 channels may lose fine-grained details
- **Mitigation**: Evaluate reconstruction quality on VAE; consider increasing latent channels if needed
- **Trade-off**: More latent channels → slower training and diffusion inference

#### 2. **Two-Stage VAE Training Complexity**

- **Limitation**: Requires sequential training of stage 1 (3D VAE) then stage 2 (2D VAE with alignment)
- **Mitigation**: Automate stage transitions; monitor alignment loss convergence
- **Benefit**: Allows proper learning of w-component without catastrophic forgetting

#### 3. **Fixed Depth Preservation**

- **Limitation**: VAE preserves depth (num_slices) without downsampling; only spatial dimensions reduced
- **Rationale**: Depth is typically small (11 slices) so cost is minimal
- **Impact**: Assumes depth information is important; may be sub-optimal for very large depths

#### 4. **Conditional Dependence on 2D Field**

- **Limitation**: Model always requires 2D velocity input; cannot generate purely from microstructure alone
- **Rationale**: 2D field provides strong anchor for physical realism
- **Limitation**: Assumes 2D field is accurate; errors compound to 3D prediction

#### 5. **Batch Normalization in Inference**

- **Limitation**: Batch norm layers use running statistics; single-sample inference may be suboptimal
- **Mitigation**: Use model in eval mode; consider layer norm alternative for better single-sample performance
- **Addressed via**: Inference script averages over stochastic samples


#### 6. **Limited to Steady-State Flows**

- **Limitation**: Model trained on steady-state, incompressible flows only
- **Scope**: Only handles Darcy-scale (low-Reynolds) flow in porous media
- **Future work**: Extension to transient or turbulent flows requires architectural changes

#### 7. **Dataset-Specific Normalization**

- **Limitation**: Normalization constants (scale_factor, max values) are dataset-specific
- **Risk**: Transfer to new datasets requires recomputing statistics
- **Mitigation**: `statistics.json` auto-generated; must be present for new datasets

---

## Computational Requirements

| Model | Hardware | Training Time |
|-------|----------|---------------|
| VAE (Stage 1 + 2) | 2× GPU (DelftBlue HPC) | ~10 hours |
| Diffusion Model | 1× RTX 5070 Ti (16GB) | ~1.5 hours |

**Memory requirements**: 
- VAE training: ~12 GB VRAM with batch size 2
- Diffusion training: ~30 GB VRAM with batch size 2

---

**Last Updated**: January 2026
