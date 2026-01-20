# Latent Diffusion Model for Microstructure-Based Flow Prediction

This project implements a physics-informed machine learning pipeline for predicting 3D resin flow fields (velocity and pressure) in fibrous composite microstructures. The main approach combines a Variational Autoencoder (VAE) with a latent diffusion process to predict complex 3D velocity fields from 2D microstructure images.

## Table of Contents

1. [Steps to Reproduce Results](#steps-to-reproduce-results)
2. [Program Structure and Design](#program-structure-and-design)
3. [Key Features](#key-features)

---

## Steps to Reproduce Results

### 1. Environment Setup

#### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, recommended)
- Git

#### Installation Steps

1. **Clone the repository** (if not already done):

Update: Check if bash is needed
   ```bash
   git clone <repository-url>
   cd Diffusion_model_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### 2. Downloading Pre-trained Models

Update

The project uses pre-trained models hosted on Zenodo. Models are automatically downloaded and cached when needed.

#### Option A: Automatic Download (Recommended)

Models are auto-downloaded on first use. The system caches them in `pretrained/` directory:

```bash
# During training or inference, models are automatically fetched:
python Diffusion_model/train.py --root-dir data/rve_5k_xy \
  --vae-path https://zenodo.org/records/XXXXX/files/vae_new_8.zip
```

#### Option B: Manual Download

1. **VAE Model** (required for latent diffusion):
   - Download from [Zenodo Record](https://doi.org/10.5281/zenodo.16940478)
   - Extract stage 1 checkpoint to `VAE_model/trained/dual_vae_stage1/`
   - Extract stage 2 checkpoint to `VAE_model/trained/dual_vae_stage2/`

2. **Diffusion Model** (for inference only):
   - Available at [Zenodo Record](https://doi.org/10.5281/zenodo.17306446)
   - Extract to `Diffusion_model/trained/`

3. **Dataset** (for training and evaluation):
   - Download from [Zenodo Record](https://doi.org/10.5281/zenodo.16940478)
   - Extract to `data/rve_5k_xy/`

#### Directory Structure After Download

```
project_root/
├── data/
│   └── rve_5k_xy/
│       ├── domain.pt
│       ├── U_2d.pt
│       ├── U.pt
│       ├── p.pt
│       └── statistics.json
├── VAE_model/
│   └── trained/
│       ├── dual_vae_stage1/
│       │   └── checkpoint_best.pt  # Stage 1: 3D VAE only
│       └── dual_vae_stage2/
│           ├── model.pt            # Stage 2: Complete dual-branch VAE
│           ├── checkpoint_best.pt
│           └── log.json
└── Diffusion_model/
    └── trained/
        └── [timestamp]_unet_latent-diffusion_[params]/
            ├── model.pt
            ├── best_model.pt
            └── log.json
```

### 3. Running the Code

In this section, all the shell code provided includes the parameters values used to obtain the final model that can be downloaded as described in the pre-trained model section. These can be changed to obtain different models.

#### A. Training a VAE (Prerequisite) - Two-Stage Process

Update: I don't think it is needed to actually train a whole VAE but check 
If training a new diffusion model, you must train a Dual-Branch VAE with a two-stage process.

**Stage 1: Train 3D VAE only**
Update parameters and add key parameters section
```bash
cd VAE_model
python train_3d_vae_only.py \
  --dataset-dir ../data/rve_5k_xy \
  --save-dir trained/dual_vae_stage1 \
  --in-channels 3 \
  --latent-channels 8 \
  --batch-size 2 \
  --num-epochs 100 \
  --learning-rate 1e-4 \
  --per-component-norm
```

**Stage 2: Train 2D VAE with alignment and cross-reconstruction**
Update parameters and add key parameters section
```bash
python train_2d_with_cross.py \
  --dataset-dir ../data/rve_5k_xy \
  --save-dir trained/dual_vae_stage2 \
  --stage1-checkpoint trained/dual_vae_stage1/checkpoint_best.pt \
  --in-channels 3 \
  --latent-channels 8 \
  --batch-size 1 \
  --num-epochs 100 \
  --learning-rate 5e-5 \
  --per-component-norm \
  --lambda-align 5 \
  --lambda-cross 50
```

**Output**: 
- Stage 1: Saves checkpoint to `trained/dual_vae_stage1/checkpoint_best.pt`
- Stage 2: Saves final model to `trained/dual_vae_stage2/` with complete dual-branch VAE

#### B. Training the Latent Diffusion Model
Update parameters and explain gridsearch

```bash
cd Diffusion_model
python train.py \
  --root-dir ../data/rve_5k_xy \
  --vae-path ../VAE_model/trained/dual_vae_stage2 \
  --predictor-type latent-diffusion \
  --in-channels 9 \
  --out-channels 8 \
  --features 64 128 256 512 \
  --batch-size 3 \
  --num-epochs 200 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --use-3d True \
  --num-slices 11 \
  --lambda-div 0.01 \
  --lambda-bc 0.1 \
  --lambda-flow 0.001 \
  --lambda-smooth 0.0001
```

**Key parameters**:
- `--vae-path`: Path to the stage 2 dual-branch VAE checkpoint
- `--in-channels 9`: Microstructure (1) + velocity latent (8) + time embedding
- `--out-channels 8`: Must match VAE latent channels (typically 4 or 8)
- `--use-3d True`: Loads 3D velocity datasets
- `--lambda-*`: Physics-informed loss weights (set to 0 to disable)

**Output**: Saves to `trained/{timestamp}_unet_latent-diffusion_[params]/` with:
- `model.pt`: Final trained model
- `best_model.pt`: Best validation checkpoint
- `log.json`: Full configuration and training history

#### C. Evaluating the Model

**Quantitative evaluation** on test set:

```bash
cd Diffusion_model
python evaluate.py \
  --dataset-dir ../data/rve_5k_xy \
  --vae-path ../VAE_model/trained/dual_vae_stage2 \
  --diffusion-path trained/[timestamp]_unet_latent-diffusion_[params]
```

**Visualization and physics metrics**:

Update Check this is actually doing what the title says

```bash
python scripts/evaluate_diffusion_vae.py \
  --dataset-dir ../data/rve_5k_xy \
  --vae-path ../VAE_model/trained/dual_vae_stage2 \
  --diffusion-path trained/[timestamp]_unet_latent-diffusion_[params]
```

#### D. Running Inference

Single-sample prediction using a trained model (index can be changed to infere a different sample):

```bash
python Inference/inference.py \
  Diffusion_model/trained/[timestamp]_unet_latent-diffusion_[params] \
  --vae-path VAE_model/trained/dual_vae_stage2 \
  --index 0
```
Update: Is this in the right section? Shouldn't it be C
**Output**: Saves predictions to `predictions/` directory with velocity fields and optional visualization.

#### E. Visualizing Training Progress

Plot training and validation losses:

```bash
cd Diffusion_model
python scripts/plot_loss.py \
  trained/[timestamp]_unet_latent-diffusion_[params]/log.json
```

Plot physics metrics (divergence, flow rate consistency, etc.):

```bash
python scripts/plot_physics_metrics.py \
  trained/[timestamp]_unet_latent-diffusion_[params]/log.json
```

---

## Program Structure and Design

### File Structure Overview

```
project_root/
├── data/                          # Dataset storage
│   └── rve_5k_xy/                # Main dataset (auto-downloaded)
│
├── VAE_model/                     # Variational Autoencoder component (Dual-Branch)
│   ├── train_3d_vae_only.py      # Stage 1: Train 3D VAE (E3D + D3D)
│   ├── train_2d_with_cross.py    # Stage 2: Train 2D VAE with alignment & cross-reconstruction
│   ├── inference_vae.py          # VAE inference utility
│   ├── config/
│   │   └── vae.py                # VAE configuration
│   ├── src/
│   │   ├── common.py             # Shared utilities
│   │   ├── dual_vae/
│   │   │   ├── __init__.py
│   │   │   └── model.py          # DualBranchVAE architecture (E2D, D2D, E3D, D3D)
│   │   └── vae/
│   │       ├── autoencoder.py    # Standard VAE (encoder + decoder)
│   │       ├── encoder.py        # 3D encoder with attention
│   │       ├── decoder.py        # Mirror decoder
│   │       └── blocks.py         # Residual and attention blocks
│   └── utils/
│       ├── dataset.py            # VAE dataset loader with 2D/3D support
│       ├── metrics.py            # Loss functions (normalized MAE, KL divergence)
│       └── paired_sampler.py     # Sampler for paired 2D/3D data
│
├── Diffusion_model/               # Main latent diffusion component
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation script
│   ├── config.py                 # Training configuration
│   ├── src/
│   │   ├── predictor.py          # LatentDiffusionPredictor class
│   │   ├── diffusion.py          # DDPM diffusion scheduler
│   │   ├── physics.py            # Physics-informed loss functions
│   │   ├── normalizer.py         # Data normalization utilities
│   │   ├── helper.py             # Training utilities
│   │   └── unet/
│   │       ├── models.py         # U-Net architecture for latent space
│   │       ├── blocks.py         # U-Net building blocks
│   │       └── metrics.py        # Loss functions
│   ├── utils/
│   │   ├── dataset.py            # 3D microflow dataset loader
│   │   └── zenodo.py             # Zenodo download utilities
│   ├── scripts/
│   │   ├── plot_loss.py          # Training loss visualization
│   │   ├── plot_physics_metrics.py  # Physics metrics visualization
│   │   ├── diagnose_noise.py     # Noise schedule diagnostics
│   │   └── diagnose_w_component.py  # Velocity component analysis
│   └── docs/
│       ├── PHYSICS_INFORMED_TRAINING.md  # Physics loss tuning guide
│       └── W_COMPONENT_FIX.md    # Notes on vertical velocity component
│
├── Inference/                     # Standalone inference pipeline
│   ├── inference.py              # Inference entry point
│   └── prompt.txt                # User interaction templates
│
├── src/                          # Root directory legacy models (2D prediction)
│   └── unet/
│
├── utils/                        # Shared utilities
│
├── requirements.txt              # Python dependencies
└── README.md, README_Final.md    # Documentation
```

### Architecture Design

#### 1. **Variational Autoencoder (VAE) - Dual-Branch Architecture**

**Purpose**: Learn separate latent representations for 2D and 3D velocity fields with alignment between them.

**Architecture**:
- **2D Branch**: The 2D branch consist of a 2-dimensional Encoder (E2D) and a 2-dimensional Decoder (D2D).  The structure of the 2D branch is as follows:


  - Input: 2D velocity field `(batch, 3 channels [Vx, Vy, Vz], height, width)` where Vz=0
  - Output: 2D latent `(batch, latent_channels, height/4, width/4)`


- **3D Branch**: Encoder E3D + Decoder D3D
  - Input: 3D velocity field `(batch, 3 channels [Vx, Vy, Vz], depth, height, width)` with non-zero Vz
  - Output: 3D latent `(batch, latent_channels, depth, height/4, width/4)`

  The depth, width and height represent the spatial dimensions of the measurement surface.

- **Key Innovation**:  Cross-reconstruction loss forces E2D to learn sufficient information for D3D to predict the w-component (vertical velocity)

**Training Process** (Two Stages):

*Stage 1*: Train 3D VAE (E3D + D3D) on 3D velocity fields only
- Loss: Reconstruction + KL divergence
- Establishes the 3D compression baseline

*Stage 2*: Train 2D VAE with alignment and cross-reconstruction
- Train E2D + D2D on 2D samples
- Alignment loss: Encourage E2D and E3D latents to be similar for paired data
- Cross-reconstruction loss: E2D → D3D (forces E2D to learn w-component information)
- Combined loss: `L_rec_2d + L_rec_3d + λ_align * L_align + λ_cross * L_cross + β_kl * KL`

Where: 
-L_rec_2d is the 2D reconstruction loss. The MSE between original and reconstructed 2D velocity fields
-L_rec_3d is the 3D reconstuction loss. The MSE between original and reconstruted 3D velocity fields
-λ_align is the weighting parameter for the latent alignment loss
-L_align is the latent alignment loss. The MSE between 2D AND 3D latent representation.
-λ_cross is the weighting parameter for the cross reconstruction loss.
-L_cross is the cross reconstruction loss. Recontruction between ground truth 3D and 3D reconstructed from 2D latent.
-β_kl is the weighting parameter for the KL divergence
-kl is the kullback-leibler Divergence. The sum of the 2D and 3D KL divergence.

**Key Design Choices**:
- **Asymmetric padding** in stride-2 convolutions prevents checkerboard artifacts
- **Attention blocks** capture long-range dependencies in flow patterns
- **Beta (KL weighting)**: Set to `1e-3` to focus on reconstruction quality over perfect posterior matching
- **Per-component normalization**: Optional per-channel statistics (u, v, w) for better w-component learning
__
**Training Loss** (Stage 1):
```
Loss = Reconstruction Loss + β_kl * KL Divergence
```

**Training Loss** (Stage 2): 
```
Loss = L_rec_2d + L_rec_3d + λ_align * L_align + λ_cross * L_cross + β_kl * (KL_2d + KL_3d)
```

#### 2. **Latent Diffusion Model**

**Purpose**: Generate realistic 3D velocity fields from 2D microstructures using iterative denoising in VAE latent space.

**Architecture**:
- **Input**: 
  - 2D microstructure slices: `(batch, num_slices, 1, H, W)` - binary domain image
  - 2D velocity field (initial guess): `(batch, num_slices, 3, H, W)` where vz=0

H and W here are the height and width spatial dimensions of the slices.
  
- **Processing Pipeline**:
  1. Encode 2D velocity slices → VAE E2D latent space (frozen encoder)
  2. Add random noise (forward diffusion process)
  3. Train U-Net to predict noise conditioned on:
     - Microstructure
     - 2D velocity latent features (from frozen VAE E2D applied to 2D input)
     - Timestep embedding
  4. During inference: iteratively denoise from random noise using trained U-Net
  5. Decode final latent → 3D velocity field using VAE D3D (frozen decoder)

**Key Design Choices**:
- **Frozen E2D/D3D**: Prevents catastrophic forgetting of VAE compression
- **Cross-branch decoding**: Uses E2D for encoding (trained on 2D) but D3D for decoding (outputs 3D), leveraging stage 2 alignment loss
- **Multi-scale U-Net**: Features `[64, 128, 256, 512]` for capturing multi-scale flow patterns
- **Timestep embedding**: Sinusoidal positional encoding for diffusion step awareness
- **Conditional features**: Early concatenation of microstructure and velocity latents

#### 3. **Physics-Informed Losses**

Optional training objective to incorporate physical constraints:


**Available physics losses**:
- **Divergence loss** (`lambda_div`): Enforces ∇·u = 0 in fluid (mass conservation)
- **Flow-rate loss** (`lambda_flow`): Penalizes variation in volumetric flow rate Q(x)
- **No-slip loss** (`lambda_bc`): Enforces u = 0 in solid regions (boundary condition)
- **Smoothness loss** (`lambda_smooth`): Tikhonov regularization for stable gradients

**Implementation**:
- Computed on decoded velocity fields (through frozen VAE decoder)
- Uses DDPM posterior estimate x̂₀ for gradient stability
- Combined with reconstruction loss via weighted sum

### Design Rationale

#### Why Latent Diffusion?

1. **Computational efficiency**: Operating in compressed latent space (256x vs pixel space) reduces memory and compute
2. **Semantic understanding**: VAE learns meaningful flow patterns; diffusion refines in semantic space
3. **Stability**: Lower-dimensional space facilitates more stable training than pixel-space diffusion
4. **Flexibility**: Frozen VAE allows quick model updates without retraining compression

#### Why Physics-Informed?

1. **Data efficiency**: Physics constraints reduce dependency on large training datasets
2. **Generalization**: Models trained with physics constraints generalize better to unseen conditions
3. **Physical validity**: Ensures predictions satisfy fundamental conservation laws
4. **Domain knowledge integration**: Embeds expertise directly into learning objective

#### Why Conditional on 2D Flow?

1. **Better initialization**: 2D field provides strong signal for 3D reconstruction
2. **Physics-guided**: 2D prediction already solves 2D flow problem; model focuses on adding vertical component
3. **Practical relevance**: 2D fields are cheaper to compute or may be available from simplified models

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
- Documented in [W_COMPONENT_FIX.md](Diffusion_model/docs/W_COMPONENT_FIX.md)
- Additional diagnostics in `scripts/diagnose_w_component.py`

#### 2. **Physics Loss Integration**

**Problem**: Physics constraints conflicted with data fitting; training became unstable.

**Solution**:
- Computed physics losses on decoded fields (not latents) for consistency
- Tuned loss weights carefully (recommended starting: `--lambda-div 0.01 --lambda-bc 0.1`)
- Used detached metrics for logging to avoid gradient contamination
- Documented tuning guide in [PHYSICS_INFORMED_TRAINING.md](Diffusion_model/docs/PHYSICS_INFORMED_TRAINING.md)

#### 3. **Dataset Consistency**

**Problem**: VAE and diffusion models used different data splits, causing train-test mismatch.

**Solution**:
- Unified split seed (seed=2024) across VAE and diffusion datasets
- Ensured same test samples used for both model evaluations
- Generated shared `statistics.json` for normalization consistency

#### 4. **Path and Configuration Management**

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

#### 6. **Physics Loss Uncertainty**

- **Limitation**: Physics loss weights are hyperparameters; no automated tuning mechanism
- **Mitigation**: Grid search and ablation studies recommended (see PHYSICS_INFORMED_TRAINING.md)
- **Cost**: Each weight combination requires full retraining

#### 7. **Limited to Steady-State Flows**

- **Limitation**: Model trained on steady-state, incompressible flows only
- **Scope**: Only handles Darcy-scale (low-Reynolds) flow in porous media
- **Future work**: Extension to transient or turbulent flows requires architectural changes

#### 8. **Dataset-Specific Normalization**

- **Limitation**: Normalization constants (scale_factor, max values) are dataset-specific
- **Risk**: Transfer to new datasets requires recomputing statistics
- **Mitigation**: `statistics.json` auto-generated; must be present for new datasets

---

## Key Features

### Physics-Informed Learning
- Differentiable physics constraints (divergence, flow rate, boundary conditions)
- Optional weighting to balance data fidelity and physical validity
- Comprehensive logging of physics metrics during training

### Efficient 3D Prediction
- Latent-space diffusion for computational efficiency
- Multi-scale U-Net capturing hierarchical flow patterns
- GPU acceleration with PyTorch

### Reproducibility
- Complete configuration logging in `log.json`
- Deterministic seeding for train/val/test splits
- Auto-download of datasets and pre-trained models from Zenodo

### Comprehensive Evaluation
- Quantitative metrics: MAE, RMSE, relative errors per component
- Qualitative visualization: flow field comparisons, error maps
- Physics metrics: divergence, flow rate consistency, boundary condition violations

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{diffusion_microflow_2025,
  author = {Your Name},
  title = {Latent Diffusion Model for Microstructure-Based Flow Prediction},
  year = {2025},
  url = {<repository-url>}
}
```

## Contact & Support

For questions, issues, or contributions, please open an issue on the repository.

---

**Last Updated**: January 2025
