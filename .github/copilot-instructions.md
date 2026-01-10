# Copilot Instructions: Arbitrary Microstructure Flow Prediction

## Project Overview

This is a **physics-informed ML project** for predicting resin flow fields (velocity and pressure) in fibrous composite microstructures. The repo contains three main approaches:

1. **Direct U-Net Prediction** (root directory): Two separate U-Net predictors for velocity/pressure with sliding window technique
2. **VAE Model** (`VAE_model/`): Variational autoencoder for learning latent representations of 3D velocity fields
3. **Latent Diffusion Model** (`Diffusion_model/`): Multi-step diffusion in VAE latent space for 3D velocity prediction from 2D microstructures

**Primary Active Development**: `Diffusion_model/` - combines VAE encoder with diffusion process for predicting 3D flow from 2D input.

## Critical Components

### Root Directory: Direct U-Net Flow Prediction

#### Dual Predictor System (`src/predictor.py`)
- **VelocityPredictor**: Predicts 2-channel (vx, vy) flow velocity from binary microstructure images
  - Input: 1-channel binary image (1=fluid, 0=fiber), optionally distance-transformed
  - Output: 2-channel normalized velocity field
- **PressurePredictor**: Predicts pressure field with physical length awareness
  - Input: 2 channels (microstructure + normalized inverse length `1/x_length`)
  - Output: 1-channel ρ-normalized pressure field
  - **Critical preprocessing**: multiplies first channel by fiber volume fraction, inverts second channel (see `pre_process`)

#### Physics-Based Post-Processing (`src/apps.py`)
- **Velocity correction** (`correct_velocity_field`): Enforces constant inlet flow rate across domain width
- **Pressure shifting** (`shift_pressure_fields`): Aligns overlapping window predictions using average pressure at boundaries
- These corrections are ESSENTIAL—predictions without them have significantly worse physics consistency



### VAE Model (`VAE_model/`)

#### Architecture (`src/vae/`)
- **Encoder** (`encoder.py`): 3-stage downsampling with ResidualBlocks + AttentionBlock, outputs mean/logvar
  - Input: `(B, 3, D, H, W)` → Latent: `(B, latent_channels, D, H/4, W/4)` - depth preserved with 3D convolutions
  - Uses asymmetric padding `(0,1,0,1)` for stride-2 convs
- **Decoder** (`decoder.py`): Mirror architecture with 2 upsampling stages
- **VariationalAutoencoder** (`autoencoder.py`): Combined encoder-decoder
  - Factory method: `VariationalAutoencoder.from_directory(folder, device, in_channels, latent_channels)`
  - Saves both `vae.pt` (weights) and `vae_log.json` (training history) to same directory

#### Training (`pretrain_vae.py`)
- **Loss composition**: `reconstruction_loss + 1e-3 * kl_loss`
- **Critical preprocessing**: Divide velocity targets by `scale_factor=0.004389363341033459` before training
- **Masking**: Predictions multiplied by microstructure mask (`preds * mask`) to zero out solid regions
- Uses `normalized_mae_loss()` for reconstruction (normalizes by target magnitude per sample)
- Config via `config/vae.py`: `--in-channels 3 --latent-channels 4/8 --batch-size 10`

### Latent Diffusion Model (`Diffusion_model/`)

#### LatentDiffusionPredictor (`src/predictor.py`)
**Core concept**: Predict 3D velocity fields from 2D microstructures using diffusion in VAE latent space

**Architecture**:
- **Input**: 
  - 2D microstructure slices: `(batch, num_slices, 1, H, W)` - binary domain (11 slices default)
  - 2D velocity field: `(batch, num_slices, 3, H, W)` - initial guess where vz=0
- **Target**: 3D velocity field `(batch, num_slices, 3, H, W)` with non-zero vz
- **Process**:
  1. Encode target 3D velocity → VAE latent space (frozen encoder)
  2. Train U-Net to denoise latents conditioned on microstructure + 2D velocity latents
  3. During inference: start from pure noise, iteratively denoise using trained U-Net
  4. Decode final latent → 3D velocity field (frozen decoder)

**Key methods** (`src/predictor.py:LatentDiffusionPredictor`):
- `forward()`: Training step - adds noise to target latents, predicts noise
- `predict()`: Inference - multi-step denoising from random noise
- `encode_target()`: Converts 3D velocity to latent representation via frozen VAE
- Requires `--vae-path` pointing to pre-trained VAE directory

#### Diffusion Scheduler (`src/diffusion.py`)
- **DiffusionScheduler**: Implements DDPM noise schedule
  - `q_sample()`: Add noise to clean data (forward diffusion)
  - `p_sample()`: Remove noise from noisy data (reverse diffusion)
  - Linear beta schedule: `beta_start=0.0001` to `beta_end=0.02` over 1000 timesteps
  - Must call `.to(device)` to move all tensors to GPU

#### Physics-Informed Training (`src/physics.py`)
The diffusion model supports physics-informed training with differentiable physics constraints:

**Available Physics Losses:**
- **Divergence Loss** (`lambda_div`): Mass conservation, penalizes non-zero ∇·u in fluid region
- **Flow-Rate Loss** (`lambda_flow`): Constant flux constraint, penalizes variation in Q(x)
- **No-Slip Loss** (`lambda_bc`): Boundary condition, penalizes non-zero velocity in solid
- **Smoothness Loss** (`lambda_smooth`): Tikhonov regularization, penalizes high-frequency noise

**Recommended starting values:**
- `--lambda-div 0.01 --lambda-flow 0.001 --lambda-bc 0.1 --lambda-smooth 0.0001`

**Implementation:** Physics losses are computed on decoded velocity fields (through frozen VAE decoder) using the DDPM posterior estimate x̂₀ = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t.

See `Diffusion_model/docs/PHYSICS_INFORMED_TRAINING.md` for detailed tuning guide.

#### Training Workflow (`train.py`)
```bash
python train.py \
  --root-dir data/rve_5k_xy \
  --vae-path VAE_model/trained/vae_new_8 \
  --predictor-type latent-diffusion \
  --in-channels 9 --out-channels 4 \
  --features 64 128 256 512 \
  --batch-size 3 --num-epochs 200 \
  --learning-rate 1e-4 --weight-decay 1e-4 \
  --use-3d True --num-slices 11 \
  --lambda-div 0.01 --lambda-bc 0.1
```

**Critical args**:
- `--in-channels`: Microstructure (1) + 2D velocity latent (4) + time embedding → 9 total per slice
- `--out-channels`: Must match VAE `latent_channels` (typically 4 or 8)
- `--use-3d True`: Loads `MicroFlowDataset` with 3D velocity fields
- `--vae-path`: Absolute or relative to project root (auto-converted to absolute)
- `--lambda-*`: Physics loss weights (set > 0 to enable physics-informed training)

**Training mechanics** (`src/helper.py:run_epoch`):
1. Load batch: `imgs` (microstructure), `velocity_2d` (input), `velocity` (target)
2. Encode target velocity → latents via VAE
3. Sample random timestep `t`, add noise to latents
4. Predict noise with U-Net conditioned on microstructure + velocity_2d latents + timestep
5. Loss: `normalized_mae_loss(predicted_noise, actual_noise)`

#### Inference (`Inference/inference.py`)
Standalone script for model deployment:
```bash
python Inference/inference.py \
  path/to/diffusion_model \
  --vae-path VAE_model/trained/vae_new_8 \
  --index 0
```

**Path resolution logic**:
- Loads config from `log.json` in model directory
- Auto-fixes VAE path if absolute path from different machine
- Searches for `statistics.json` in: dataset_root → default data path → sample directory
- Supports loading specific test sample via `--index` or custom `.pt` file

#### Dataset (`Diffusion_model/utils/dataset.py`)

**MicroFlowDataset with 3D support** (`use_3d=True`):
- **Required files** in dataset root:
  - `domain.pt`: 2D microstructure `(N, 1, H, W)` - binary mask
  - `U_2d.pt`: Input velocity `(N, num_slices, 3, H, W)` where vz=0
  - `U.pt`: Target velocity `(N, num_slices, 3, H, W)` with 3D flow
  - `p.pt`, `dxyz.pt`, `permeability.pt`
- **Data split**: 70/15/15 train/val/test with seed=2024 (matches VAE splitting)
- **Augmentation**: Horizontal flips with sign correction for vy component (if `augment=True`)

**Key difference from root dataset**:
- Root: 2D flow, single image per sample `(N, 1/3, H, W)`
- Diffusion: 3D flow, stacked slices `(N, num_slices, 3, H, W)`


## Data Flow & Normalization

### Root Directory Models
1. **Input normalization** (`src/normalizer.py:MaxNormalizer`):
   - Velocity model: no input normalization (distance transform only)
   - Pressure model: normalizes by `(1, max_length)` from `statistics.json`
2. **Output normalization**: All predictions normalized by max values during training
   - Velocity: `(max_vx, max_vy)` 
   - Pressure: `(max_p,)`
3. **Denormalization**: `.predict()` methods call `.inverse()` automatically

### Diffusion Model
1. **VAE normalization**: Velocity data divided by `scale_factor=0.004389363341033459` during VAE training
2. **Latent space**: U-Net operates on normalized latents `(batch, latent_channels, depth, H/4, W/4)`
3. **Output**: VAE decoder automatically reverses normalization when reconstructing velocity

**Key file**: `statistics.json` in dataset root—auto-generated by `MicroFlowDataset._save_statistics()`, required for proper normalization. Contains keys like `U`, `U_2d`, `U_3d`, `velocity`, `pressure` with `max`, `min`, `mean` values.


## Training & Evaluation Workflows

### Training VAE (Prerequisite for Diffusion Model)
```bash
cd VAE_model
python pretrain_vae.py --in-channels 3 --latent-channels 4 \
  --dataset-dir ../data/rve_5k_xy --batch-size 10 --num-epochs 100
```
- Saves to `trained/vae/` by default
- Creates `vae.pt` (weights) and `vae_log.json` (config + training history)

### Training Latent Diffusion Model
```bash
cd Diffusion_model
python train.py \
  --root-dir ../data/rve_5k_xy \
  --vae-path ../VAE_model/trained/vae_new_8 \
  --predictor-type latent-diffusion \
  --in-channels 9 --out-channels 4 \
  --batch-size 3 --num-epochs 200 \
  --use-3d True --num-slices 11
```
- Dataset auto-downloads from Zenodo if `--root-dir` empty
- Saves to timestamped folder: `trained/{date}_unet_latent-diffusion_{hyperparams}/`
- Outputs: `model.pt`, `best_model.pt`, `log.json` (contains full config + training history)

### Training Root Directory Models (Legacy)
```bash
# Velocity model
python train.py --root-dir data/dataset --predictor-type velocity --in-channels 1 --out-channels 2

# Pressure model  
python train.py --root-dir data/dataset --predictor-type pressure --in-channels 2 --out-channels 1 --distance-transform ''
```

### Model Loading Patterns
Three equivalent methods (see `src/predictor.py:Predictor`):
```python
# From local directory
predictor = Predictor.from_directory(folder, device)

# From Zenodo URL (auto-downloads & caches to `pretrained/`)
predictor = Predictor.from_url(url, device)

# Auto-detects local vs URL
predictor = Predictor.from_directory_or_url(directory_or_url, device)
```

### Evaluation

#### Diffusion Model Evaluation
```bash
cd Diffusion_model
python scripts/evaluate_diffusion_vae.py \
  --dataset-dir ../data/rve_5k_xy \
  --vae-path ../VAE_model/trained/vae_new_8 \
  --diffusion-path trained/20260109_unet_latent-diffusion_...
```

#### Root Model Evaluation
```bash
# Standard validation
python eval.py --root-dir data/dataset --split valid \
  --directory-or-url https://zenodo.org/records/17306446/files/velocity_model_base.zip?download=1

# Micrograph benchmark (uses sliding window)
python -m scripts.eval_micrograph --micrograph-dir path/to/micrographs \
  --velocity-model {url_or_path} --pressure-model {url_or_path}
```


## Project-Specific Conventions

### Tensor Shape Conventions
- **Microstructures**: `(batch, 1, height, width)` - binary with 1=fluid, 0=fiber
- **Velocity fields 2D**: `(batch, 2, height, width)` - channels are [vx, vy]
- **Velocity fields 3D**: `(batch, num_slices, 3, height, width)` - channels are [vx, vy, vz]
- **Pressure fields**: `(batch, 1, height, width)` - scalar field
- **Physical dimensions**: `(batch, 3)` - [dx, dy, dz] in meters

### Configuration via `config.py`
- Uses `argparse` groups (dataset, training, optimization) processed into nested dicts
- **Attention mechanism**: String format `"start.end.heads"` (e.g., `"3..2"` = levels 3-max with 2 heads)
  - Empty string `""` = no attention
  - Parsed by `src/unet/models.py:eval_expression()`
- **Distance transform**: Bool flag that applies `scipy.ndimage.distance_transform_edt` to inputs

### Loss Functions (`src/unet/metrics.py`)
- Primary: `normalized_mae_loss` - MAE divided by mean absolute target (scale-invariant)
- Alternative: `mae_loss` - standard MAE
- **Not used in training** but available: `mass_conservation_loss` for physics constraints

### Physics Losses (`Diffusion_model/src/physics.py`)
- **divergence_loss_masked**: Mass conservation (∇·u = 0) in fluid region
- **flow_rate_consistency_loss**: Constant flux constraint Q(x) = const
- **no_slip_loss**: Zero velocity in solid regions
- **smoothness_loss**: Tikhonov regularization |∇u|²
- **PhysicsLoss**: Combined class that computes all enabled constraints
- **compute_physics_metrics**: Detached metrics for logging

### File Organization
- **Root directory** (`src/`, `utils/`, `train.py`, `eval.py`): Legacy 2D U-Net models for direct prediction
- **VAE_model/**: Pre-training VAE for latent space learning
- **Diffusion_model/**: Main active development - latent diffusion for 3D prediction
- **Inference/**: Standalone deployment scripts
- Both root and Diffusion_model have separate `src/unet/` implementations (share common blocks)


## External Dependencies

- **Dataset**: Auto-downloaded from [Zenodo 16940478](https://doi.org/10.5281/zenodo.16940478) (square microstructures)
- **Pre-trained models**: [Zenodo 17306446](https://doi.org/10.5281/zenodo.17306446)
- **Micrograph benchmark**: [Zenodo 6611926](https://doi.org/10.5281/zenodo.6611926) - requires manual access request
- **Zenodo utilities** (`utils/zenodo.py`): Handles downloads, unzipping, URL detection

## Development Notes

### When Modifying Models
- **Never change normalization** without retraining—breaks pre-trained weights
- **UNet features list**: Must be power-of-2 progression for encoder/decoder symmetry
- **Attention levels**: 1-indexed in config string, converted to 0-indexed internally
- **VAE latent channels**: Must match diffusion model's `out_channels` (typically 4 or 8)

### When Adding New Predictors
- Inherit from `Predictor` (abstract base in `src/predictor.py`)
- Implement: `forward()`, `predict()`, `pre_process()`, `type` property
- Override `init_normalizer()` if custom normalization needed
- Update `src/helper.py:set_model()` and `get_model()` with new type

### Dataset Handling
- `MicroFlowDataset` (`utils/dataset.py`): Handles both x-flow and y-flow simulations
  - Y-flow data rotated 90° and velocity channels swapped
- Data augmentation: Vertical flips with sign correction for vy component
- **3D dataset variant**: Uses same permeability value across all slices
- **Critical**: VAE and Diffusion datasets use identical split (seed=2024) to ensure same test samples

### Cross-Component Dependencies
- **Diffusion model requires VAE**: Must train VAE before training latent diffusion
- **VAE path handling**: Automatically converts relative paths to absolute from project root
- **Import chain**: `Diffusion_model/src/predictor.py` imports `VAE_model/src/vae/autoencoder.py`
- **statistics.json**: Generated by dataset, required for normalization in both VAE and Diffusion

### Windows-Specific
- Tested on Ubuntu 22.04 LTS (development environment)
- PowerShell requires `;` for command chaining, not `&&`
- Use forward slashes or raw strings for paths in code

## Physics Context

**Darcy's Law**: `K = (Q * μ * L) / (A * ΔP)` where K=permeability, Q=flow rate, μ=viscosity (0.5 default), L=length, A=cross-section area, ΔP=pressure drop

**Flow rate calculation** (`src/physics.py:get_flow_rate`): Section-wise averaging of velocity × fluid area, not simple integration
