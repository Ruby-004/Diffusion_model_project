# Fixing the W-Component (vz) Issue in Latent Diffusion Model

## Problem Summary

The w (vz) velocity component appears as a uniform "block/slab" instead of showing detailed flow structure, while u and v components look correct.

**Root Cause**: The w component is much sparser than u/v (90% of values are near-zero), but the VAE and diffusion model used **global normalization** which caused:
1. W component values to be "drowned out" after normalization by global max
2. VAE learns to predict near-zero for w since that minimizes average loss
3. U_2d (input) has w=0, so conditioning reinforces "predict zeros for w"

## Solutions Implemented

### 1. Per-Component Normalization for VAE

The VAE now uses separate normalization factors for each velocity component:
- `max_u` for vx
- `max_v` for vy  
- `max_w` for vz

This ensures each component is normalized to [0,1] range independently.

### 2. Conditional VAE Mode (Recommended)

The VAE now supports **conditional mode** where it receives a signal indicating whether the input is:
- **2D flow (U_2d)**: has w=0 everywhere → condition `is_3d=False`
- **3D flow (U)**: has w≠0 in flow regions → condition `is_3d=True`

This creates a **disentangled latent space**:
- One subspace for 2D flow representations (w=0)
- Another subspace for 3D flow representations (w≠0)

The decoder then specializes its output based on the condition, preventing the "always predict w=0" shortcut.

### 3. Component-Weighted Velocity Loss (Optional)

During diffusion training, you can boost the importance of the w component:
- `--weight-w 5.0` makes w-component errors count 5x more
- `--lambda-velocity 0.1` enables the component-weighted loss

## Training Instructions

### Step 1: Retrain the VAE with Conditional Mode + Per-Component Normalization

```bash
cd VAE_model

python pretrain_vae.py \
    --dataset-dir "C:\Users\alexd\Downloads\dataset_3d" \
    --save-dir trained/vae_conditional \
    --in-channels 3 \
    --latent-channels 8 \
    --batch-size 1 \
    --num-epochs 100 \
    --learning-rate 1e-6 \
    --per-component-norm \
    --conditional
```

**Key flags**:
- `--per-component-norm` (enabled by default) - separate normalization for u, v, w
- `--conditional` - enable conditional VAE mode for 2D/3D flow distinction

The VAE will save:
- `vae.pt` - Model weights
- `vae_log.json` - Contains:
  - `norm_factors: [max_u, max_v, max_w]` for per-component normalization
  - `conditional: true` to indicate conditional mode

### Step 2: Train Diffusion Model with New VAE

```bash
cd Diffusion_model

python train.py \
    --root-dir "C:\Users\alexd\Downloads\dataset_3d" \
    --vae-path "../VAE_model/trained/vae_conditional" \
    --predictor-type latent-diffusion \
    --in-channels 17 \
    --out-channels 8 \
    --batch-size 3 \
    --num-epochs 100 \
    --use-3d True \
    --device cuda
```

The diffusion model will automatically:
1. Load `norm_factors` from `vae_log.json` for per-component normalization
2. Load `conditional` flag from `vae_log.json`
3. Use correct conditions when encoding/decoding:
   - `is_3d=False` when encoding U_2d (input with w=0)
   - `is_3d=True` when encoding target U (output with w≠0)
   - `is_3d=True` when decoding predictions (we want 3D output)

### Step 3: (Optional) Boost W-Component Training

If w still appears weak after per-component normalization, add component weighting:

```bash
python train.py \
    --root-dir "C:\Users\alexd\Downloads\dataset_3d" \
    --vae-path "../VAE_model/trained/vae_conditional" \
    --predictor-type latent-diffusion \
    --in-channels 17 \
    --out-channels 8 \
    --batch-size 3 \
    --num-epochs 100 \
    --use-3d True \
    --lambda-velocity 0.1 \
    --weight-w 5.0
```

**Parameters**:
- `--lambda-velocity`: Weight for component-weighted velocity loss (0 = disabled)
- `--weight-u`: Weight for u component (default 1.0)
- `--weight-v`: Weight for v component (default 1.0)
- `--weight-w`: Weight for w component (default 1.0, try 3-10 to boost)

## Statistics Reference

From your dataset (`C:\Users\alexd\Downloads\dataset_3d\statistics.json`):

| Component | Max Value | Description |
|-----------|-----------|-------------|
| max_u (vx) | 0.00389 | X-direction velocity |
| max_v (vy) | 0.00190 | Y-direction velocity |
| max_w (vz) | 0.00401 | Z-direction velocity (sparse!) |

Note: While max_w is similar to max_u, the **median** of w is 10x smaller due to sparsity.

## Files Modified

### VAE_model/
- `pretrain_vae.py` - Added per-component normalization and conditional mode
- `config/vae.py` - Added `--per-component-norm` and `--conditional` flags
- `src/vae/encoder.py` - Added condition embedding for conditional VAE
- `src/vae/decoder.py` - Added condition embedding for conditional VAE
- `src/vae/autoencoder.py` - Updated to pass conditions through encode/decode

### Diffusion_model/
- `src/predictor.py` - Loads norm_factors and conditional flag from VAE log, uses correct conditions
- `src/helper.py` - Updated `get_norm_params()` for per-component stats
- `src/physics.py` - Added `component_weighted_velocity_loss()`
- `config.py` - Added `--weight-u/v/w` and `--lambda-velocity` args
- `train.py` - Integrated component weighting into training loop

## Verification

After training, run the diagnostic script:

```bash
cd Diffusion_model
python scripts/diagnose_w_component.py \
    --vae-path "../VAE_model/trained/vae_conditional" \
    --dataset-dir "C:\Users\alexd\Downloads\dataset_3d"
```

Expected improvements:
- VAE reconstruction ratio for w should be closer to 1.0 (was 0.043 before)
- Diffusion predictions should show w structure similar to ground truth
- With conditional VAE, the decoder knows to output non-zero w when condition=True

## How Conditional VAE Helps

The key insight is that the VAE trains on BOTH U_2d (w=0) and U (w≠0) samples. Without conditioning, this creates ambiguity in the latent space - the model learns to hedge by predicting small w values.

With conditional mode:
1. **Encoder** receives `is_3d` signal and creates different latent representations for 2D vs 3D flow
2. **Decoder** receives `is_3d` signal and specializes output:
   - `is_3d=False` → output w≈0 (appropriate for U_2d)
   - `is_3d=True` → output non-zero w (appropriate for U)
3. **During inference**, diffusion model always uses `is_3d=True` for decode, telling the VAE we want 3D output
