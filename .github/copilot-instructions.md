# Copilot Instructions for Diffusion Model Project

## Project Overview

This project implements **latent diffusion for 3D resin flow prediction** in fibrous microstructures. It predicts 3D velocity fields (with vertical component $v_z \neq 0$) from 2D velocity inputs ($v_z = 0$).

**Architecture Pipeline:**
1. **VAE Stage 1**: Train 3D encoder/decoder (E3D + D3D) on 3D velocity fields
2. **VAE Stage 2**: Train 2D encoder/decoder (E2D + D2D) with latent alignment to E3D
3. **Diffusion**: Train U-Net to denoise in VAE latent space, using frozen E2D (input) and D3D (output)

## Critical Conventions

### Working Directory
All scripts assume execution from **project root** (`Diffusion_model_project/`). Use module-style imports:
```bash
python -m Diffusion_model.train --root-dir ./dataset_3d ...
# OR cd into subdirectory first
cd Diffusion_model && python train.py --root-dir ../dataset_3d ...
```

### Dataset Split Consistency
**Hardcoded seed=2024** ensures identical 70/15/15 splits across all training/eval scripts. Never change this seed or data contamination will occur.

### Data Format
- Input shape: `(batch, num_slices, channels, H, W)` where num_slices=11, H=W=256
- Velocity: 3 channels ($v_x$, $v_y$, $v_z$)
- Microstructure: 1 channel binary mask (1=fluid, 0=solid)
- Statistics stored in `dataset_3d/statistics.json` for normalization

## Key File Patterns

### Configuration
- [Diffusion_model/config.py](Diffusion_model/config.py): All CLI args with defaults. Override any param via `--param-name value`
- [VAE_model/config/vae.py](VAE_model/config/vae.py): VAE-specific configuration

### Model Loading
```python
# Load VAE from directory (handles model.pt, best_model.pt, vae.pt)
from VAE_model.src.dual_vae.model import DualBranchVAE
vae = DualBranchVAE.from_directory("VAE_model/trained/dual_vae_stage2_2d")

# Diffusion predictor with frozen VAE branches
from Diffusion_model.src.predictor import LatentDiffusionPredictor
predictor = LatentDiffusionPredictor.from_directory(model_path, device=device)
```

### Training Outputs
Models save to `trained/` with structure:
```
trained/{timestamp}_unet_latent-diffusion_{params}/
├── model.pt          # Final weights
├── best_model.pt     # Best validation checkpoint
└── log.json          # Full config + training history
```

## Data Flow & Normalization

### Dataset Structure
- **statistics.json**: Auto-generated in dataset root by `MicroFlowDataset._save_statistics()`
  - Keys: `U`, `U_2d`, `U_3d`, `velocity`, `pressure` with `max`, `min`, `mean` per component
  - Required for proper normalization in both VAE and diffusion training
- **3D format** (`use_3d=True`):
  - `domain.pt`: 2D microstructure `(N, 1, H, W)` - binary mask
  - `U_2d.pt`: Input velocity `(N, num_slices, 3, H, W)` where $v_z=0$
  - `U.pt`: Target velocity `(N, num_slices, 3, H, W)` with non-zero $v_z$
  - `p.pt`, `dxyz.pt`, `permeability.pt`

### Normalization Chain
1. **VAE training**: Data **NOT** pre-normalized; VAE learns on raw velocity values
2. **Latent space**: U-Net operates on VAE latent representations `(B, 8, D, H/4, W/4)`
3. **Output**: VAE decoder automatically reconstructs original velocity scale

**Critical**: Never change normalization without retraining—breaks pre-trained weights

### Data Augmentation
- Horizontal flips applied to microstructure + velocity (if `--augment True`)
- **Sign correction**: $v_y$ component flipped when image is flipped vertically
- Y-flow simulations rotated 90° with velocity channels swapped during loading

## Training Workflows

### Stage 1: Train 3D VAE
```bash
python VAE_model/train_3d_vae_only.py \
  --dataset-dir ./dataset_3d \
  --save-dir VAE_model/trained/dual_vae_stage1_3d \
  --in-channels 3 --latent-channels 8 \
  --batch-size 2 --num-epochs 100 \
  --learning-rate 1e-4
```
**Outputs**: `model.pt`, `best_model.pt`, `vae_log.json` in save directory

**Loss function**: `normalized_mae_loss_per_channel` - each velocity channel's MAE is normalized by its mean absolute target value, making the loss scale-invariant without requiring input normalization.

### Stage 2: Train 2D VAE with Alignment
```bash
python VAE_model/train_2d_with_cross.py \
  --dataset-dir ./dataset_3d \
  --save-dir VAE_model/trained/dual_vae_stage2_2d \
  --stage1-checkpoint VAE_model/trained/dual_vae_stage1_3d \
  --in-channels 3 --latent-channels 8 \
  --batch-size 1 --num-epochs 100 \
  --learning-rate 1e-4 \
  --lambda-align 5 --lambda-cross 50
```
**Key losses**: Alignment (E2D ≈ E3D latents) + cross-reconstruction (E2D → D3D predicts 3D)

### Stage 3: Train Diffusion Model
```bash
python Diffusion_model/train.py \
  --root-dir ./dataset_3d \
  --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d \
  --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d \
  --predictor-type latent-diffusion \
  --in-channels 17 --out-channels 8 \
  --features 64 128 256 512 1024 \
  --batch-size 2 --num-epochs 100 \
  --learning-rate 1e-3 --weight-decay 0 \
  --use-3d True --num-slices 11 \
  --attention '3..2'
```
**Auto-download**: Empty `--root-dir` triggers Zenodo dataset download (~2.1 GB)

**Outputs**: `trained/{timestamp}_unet_latent-diffusion_{params}/` with `model.pt`, `best_model.pt`, `log.json`

### Training Mechanics (`src/helper.py:run_epoch`)
1. Load batch: `imgs` (microstructure), `velocity_2d` (input $v_z=0$), `velocity` (target 3D)
2. Encode target via frozen E3D → latents
3. Sample random timestep `t ∈ [0, 999]`, add noise to latents: `z_t = sqrt(ᾱ_t)·z_0 + sqrt(1-ᾱ_t)·ε`
4. U-Net predicts noise conditioned on: microstructure + E2D(velocity_2d) + timestep embedding
5. Loss: `normalized_mae_loss(predicted_ε, actual_ε)` - MAE normalized by mean absolute target

## Inference & Evaluation

### Single-Sample Inference
```bash
python Inference/inference.py \
  --diffusion-model-path Diffusion_model/trained/[model_folder] \
  --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d \
  --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d \
  --dataset-dir ./dataset_3d \
  --index 0
```
**Outputs**: Side-by-side comparison PNG + interactive Napari 3D viewer

**Path resolution**: Loads config from `log.json`, auto-fixes VAE paths if from different machine

### Test Set Evaluation
```bash
python scripts/eval_testset_end2end.py \
  --diffusion-model-path Diffusion_model/trained/[model] \
  --vae-encoder-path VAE_model/trained/dual_vae_stage2_2d \
  --vae-decoder-path VAE_model/trained/dual_vae_stage1_3d \
  --dataset-dir ./dataset_3d \
  --sampler ddim --steps 50
```
**Metrics**: Per-component MAE/MSE/RMSE, cosine similarity, IoU of top-k magnitude voxels

**Samplers**:
- `--sampler ddim`: DDIM (faster, default 50 steps)
- `--sampler ddpm`: DDPM (slower, more accurate, 1000 steps)
- `--sanity-mode`: VAE-only reconstruction (bypasses diffusion)

### Visualization
```bash
# Plot diffusion training loss
python Diffusion_model/scripts/plot_loss.py trained/[model_folder]

# Plot VAE training loss
python VAE_model/plot_vae_loss.py trained/dual_vae_stage1_3d
```

## Model Loading Patterns

### DualBranchVAE
```python
from VAE_model.src.dual_vae.model import DualBranchVAE

# Auto-detects model.pt, best_model.pt, or vae.pt
vae = DualBranchVAE.from_directory(
    "VAE_model/trained/dual_vae_stage2_2d",
    device="cuda",
    in_channels=3,
    latent_channels=8
)

# Loads config from vae_log.json if exists (auto-detects latent_channels)
```

### LatentDiffusionPredictor
```python
from Diffusion_model.src.predictor import LatentDiffusionPredictor

# Loads from directory with log.json
predictor = LatentDiffusionPredictor.from_directory(
    "Diffusion_model/trained/20260120_unet_...",
    device="cuda"
)

# Auto-loads VAE paths from log.json, converts relative to absolute
```

**Critical**: Both classes use `from_directory()` factory pattern, not direct `__init__()`

### Dual-Branch VAE Design
- **Spatial downsampling**: 4× in (H, W) only → latent shape `(B, latent_channels, D, H/4, W/4)`
- **Depth preservation**: Slice dimension unchanged (3D convolutions across depth)
- **Asymmetric padding**: `(0,1,0,1)` in stride-2 convs prevents checkerboard artifacts
- **Latent channels**: 8 (default), must match diffusion `--out-channels`
- **Stage 1 loss**: `reconstruction_loss + 1e-3 * kl_loss`
- **Stage 2 loss**: `L_rec_2d + L_rec_3d + λ_align*L_align + λ_cross*L_cross + β_kl*KL`
  - Cross-reconstruction loss (`λ_cross=50`) forces E2D to encode $v_z$ information
  - Alignment loss (`λ_align=5`) ensures E2D/E3D latents are similar for paired samples

### Diffusion U-Net Design
- **DDPM scheduler**: 1000 timesteps, linear beta `[0.0001, 0.02]`
- **U-Net features**: `[64, 128, 256, 512, 1024]` (5 levels, power-of-2 for encoder/decoder symmetry)
- **Attention format**: String `"start.end.heads"` (e.g., `"3..2"` = levels 3+ with 2 heads, `""` = no attention)
  - Parsed by `src/unet/models.py:eval_expression()`, 1-indexed in config → 0-indexed internally
- **Input channels**: `--in-channels 17` = microstructure(1) + E2D latent(8) + timestep embedding(8)
- **Output channels**: `--out-channels 8` (matches VAE latent_channels)
- **Frozen VAE**: E2D encoder and D3D decoder weights frozen during diffusion training

### Physics-Informed Training (Optional)
Available in `src/physics.py` but **not used in final model**:
- **Divergence loss** (`--lambda-div`): Mass conservation ∇·u = 0 in fluid
- **Flow-rate loss** (`--lambda-flow`): Constant flux Q(x) = const across slices
- **Smoothness loss** (`--lambda-smooth`): Tikhonov regularization |∇u|²
- **Laplacian loss** (`--lambda-laplacian`): No-slip boundary condition in solid

**Recommended starting values if experimenting**: `--lambda-div 0.01 --lambda-flow 0.001 --lambda-smooth 0.0001`

## Code Style & Conventions

### Loss Functions (`src/unet/metrics.py`)
- **Primary**: `normalized_mae_loss()` - MAE divided by mean absolute target (scale-invariant)
- **Alternative**: `mae_loss()` - standard MAE
- All diffusion losses use normalized MAE for stable training across different velocity magnitudes

### Tensor Shape Conventions
- **Microstructures**: `(batch, num_slices, 1, H, W)` - binary with 1=fluid, 0=fiber
- **2D velocity**: `(batch, num_slices, 3, H, W)` - channels [vx, vy, vz] where vz=0
- **3D velocity**: `(batch, num_slices, 3, H, W)` - channels [vx, vy, vz] with vz≠0
- **VAE latents**: `(batch, latent_channels, num_slices, H/4, W/4)` - depth preserved

### Configuration via Argparse
- Type hints used throughout (`typing.Tuple`, `torch.Tensor`)
- Grouped arguments: dataset, training, optimization, model
- Override any default: `--param-name value`
- Boolean flags: `--augment True` or `--per-component-norm` (store_true action)

### File Organization
- `config.py`: CLI args with defaults (e.g., `--features 64 128 256 512 1024`)
- `src/helper.py`: Training utilities (`run_epoch`, `set_model`, `make_log_folder`)
- `src/predictor.py`: Model classes with abstract `Predictor` base
- `utils/dataset.py`: `MicroFlowDataset` with `use_3d=True` for stacked slices
- `utils/zenodo.py`: Auto-download utilities, contains dataset URL constants

### Development Notes

**When Modifying Models**:
- Never change normalization without retraining—breaks pre-trained weights
- UNet features list must be power-of-2 progression for encoder/decoder symmetry
- Attention levels: 1-indexed in config string, converted to 0-indexed internally
- VAE latent_channels must match diffusion `--out-channels`

**When Adding Physics Losses**:
- Implement in `src/physics.py` following existing patterns
- Compute on decoded velocity fields (through frozen D3D)
- Use DDPM posterior estimate $\hat{x}_0$ for gradient stability
- Log detached metrics via `compute_physics_metrics()`

**Dataset Handling**:
- Y-flow data automatically rotated 90° with velocity channels swapped
- Augmentation: horizontal flips with sign correction for $v_y$ component
- Same permeability value used across all slices in 3D case

## Testing Changes

After modifications:
1. Verify imports: `python -c "from Diffusion_model.src.predictor import LatentDiffusionPredictor"`
2. Quick training test: Run 1-2 epochs with `--num-epochs 2 --batch-size 1`
3. Check `log.json` is written correctly with all hyperparameters
4. Validate loss curves: `python Diffusion_model/scripts/plot_loss.py trained/[model]`