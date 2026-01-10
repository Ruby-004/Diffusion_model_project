# Physics-Informed Latent Diffusion Training

This document describes the physics-informed training approach for the latent diffusion model that predicts 3D fluid flow fields in fibrous microstructures.

## Overview

The latent diffusion model operates in VAE latent space and predicts noise to denoise velocity field representations. We add physics constraints computed on decoded velocity fields to improve physical consistency and generalization.

**Total Loss:**
$$L_{total} = L_{diffusion} + \sum_i \lambda_i L_{physics,i}$$

Where:
- $L_{diffusion}$ = noise prediction loss (normalized MAE)
- $L_{physics,i}$ = physics constraint losses

## Physics Constraints Implemented

### 1. Mass Conservation / Incompressibility (Divergence Loss)

For incompressible flow, the divergence of velocity must be zero everywhere in the fluid:

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

**Loss:**
$$L_{div} = \frac{1}{N_{fluid}} \sum_{fluid} |\nabla \cdot \mathbf{u}|^2$$

- Computed using central finite differences
- Applied only in fluid region (masked by microstructure)
- **Recommended weight:** `lambda_div = 0.01`

### 2. Flow Rate Consistency (Constant Flux)

For steady-state flow, the volumetric flow rate should be constant at all cross-sections:

$$Q(x) = \iint u \, dA = \text{constant}$$

**Loss:**
$$L_{flow} = \text{mean}\left(\left(\frac{Q(x)}{Q_{inlet}} - 1\right)^2\right)$$

- Computes flow rate at each x-position
- Normalizes by inlet flow rate
- Penalizes variation from inlet
- **Recommended weight:** `lambda_flow = 0.001`

### 3. No-Slip Boundary Condition

At solid boundaries, velocity must be zero:

$$\mathbf{u}|_{solid} = 0$$

**Loss:**
$$L_{bc} = \frac{1}{N_{solid}} \sum_{solid} |\mathbf{u}|^2$$

- Applied in solid regions (fiber locations)
- Ensures predictions respect no-slip condition
- **Recommended weight:** `lambda_bc = 0.1`

### 4. Smoothness Regularization (Tikhonov)

Penalizes high-frequency noise in velocity predictions:

$$L_{smooth} = \frac{1}{N_{fluid}} \sum_{fluid} |\nabla \mathbf{u}|^2$$

- Prevents overly noisy predictions
- Applied only in fluid region
- Keep weight small to avoid over-smoothing
- **Recommended weight:** `lambda_smooth = 0.0001`

## Implementation Details

### How Physics Losses Are Computed During Training

1. **Forward diffusion:** Add noise to target latents
2. **Predict noise:** U-Net predicts noise from noisy latents
3. **Reconstruct x̂₀:** Use DDPM posterior estimate to get clean latent estimate
4. **Decode:** Pass reconstructed latent through frozen VAE decoder → velocity field
5. **Compute physics losses:** Apply differentiable physics constraints on velocity
6. **Backpropagate:** Gradients flow through decoder and into U-Net

### Reconstruction from Noise Prediction

During training, we estimate the clean latent $x_0$ from the noisy latent $x_t$ and predicted noise $\epsilon_\theta$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

This estimate is decoded to velocity for physics loss computation.

### Efficiency Considerations

- `physics_loss_freq`: Compute physics loss every N batches (default: 1)
  - Set higher (e.g., 5-10) for faster training if physics computation is slow
- Physics losses are disabled during validation forward pass
- Metrics are computed on all validation batches

## Usage

### Training with Physics Constraints

```bash
cd Diffusion_model
python train.py \
    --root-dir ../data/rve_5k_xy \
    --vae-path ../VAE_model/trained/vae_new_8 \
    --predictor-type latent-diffusion \
    --in-channels 9 --out-channels 4 \
    --batch-size 3 --num-epochs 200 \
    --learning-rate 1e-4 \
    --lambda-div 0.01 \
    --lambda-flow 0.001 \
    --lambda-bc 0.1 \
    --lambda-smooth 0.0001 \
    --physics-loss-freq 1
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda-div` | 0.0 | Weight for divergence loss (mass conservation) |
| `--lambda-flow` | 0.0 | Weight for flow-rate consistency loss |
| `--lambda-bc` | 0.0 | Weight for no-slip boundary condition loss |
| `--lambda-smooth` | 0.0 | Weight for smoothness regularization |
| `--physics-loss-freq` | 1 | Compute physics loss every N batches |

### Viewing Physics Metrics

```bash
# Plot physics metrics from training log
python scripts/plot_physics_metrics.py trained/20260110_unet_latent-diffusion_.../

# Compare multiple runs
python scripts/plot_physics_metrics.py --compare run1/log.json run2/log.json
```

## Tuning Recipe for λ Weights

### Step-by-Step Guide

1. **Baseline Training (λ = 0)**
   - Train without physics constraints
   - Establish baseline reconstruction quality
   - Note physics metrics (they will be poor)

2. **Enable One Constraint at a Time**
   ```bash
   # Start with mass conservation only
   python train.py ... --lambda-div 0.001
   ```
   - Monitor both reconstruction loss AND physics metrics
   - If reconstruction loss increases > 20%, reduce λ

3. **Gradual Scaling**
   - Increase λ by factors of 3-10 until:
     - Physics metrics improve significantly
     - Reconstruction loss stays reasonable

4. **Combine Constraints**
   - Add constraints one at a time
   - Mass conservation is most important (add first)
   - No-slip is usually easy to satisfy
   - Flow-rate may conflict with other constraints

### Recommended Starting Configuration

```bash
# Conservative (safe start)
--lambda-div 0.001 --lambda-bc 0.01

# Moderate (recommended)
--lambda-div 0.01 --lambda-flow 0.001 --lambda-bc 0.1

# Aggressive (may need tuning)
--lambda-div 0.1 --lambda-flow 0.01 --lambda-bc 0.5 --lambda-smooth 0.001
```

### Signs of Good Tuning

✅ `div_mean` decreases over epochs → Mass conservation improving
✅ `flow_rate_cv` decreases → More consistent flow rate
✅ `vel_in_solid` → 0 → No-slip satisfied
✅ `val_loss` comparable to baseline → Not sacrificing reconstruction

### Signs of Poor Tuning

❌ `val_loss` much higher than baseline → λ too high
❌ Physics metrics not improving → λ too low
❌ Training unstable → Try gradient clipping or lower λ

## Physics Metrics Logged

| Metric | Meaning | Target |
|--------|---------|--------|
| `div_mean` | Mean absolute divergence in fluid | → 0 |
| `div_std` | Std dev of divergence | → 0 |
| `flow_rate_cv` | Coefficient of variation of flow rate | → 0 |
| `vel_in_solid` | RMS velocity in solid region | → 0 |
| `vel_mean_fluid` | Mean velocity magnitude in fluid | Physical value |
| `loss_divergence` | Divergence loss component | Monitoring |
| `loss_flow_rate` | Flow-rate loss component | Monitoring |
| `loss_no_slip` | No-slip loss component | Monitoring |
| `loss_smoothness` | Smoothness loss component | Monitoring |

## Expected Results

With proper tuning, you should see:

1. **Mass conservation:** 50-90% reduction in divergence compared to baseline
2. **Flow consistency:** Lower variation in flow rate across domain
3. **Boundary conditions:** Near-zero velocity in solid regions
4. **Generalization:** Better performance on out-of-distribution test samples

## Troubleshooting

### Physics loss is NaN or Inf
- Check for division by zero (add `eps`)
- Reduce learning rate
- Check if mask is properly defined (should be 1=fluid, 0=solid)

### Training is too slow
- Increase `physics_loss_freq` (e.g., 5 or 10)
- Reduce batch size
- Use smaller λ values (physics computation scales with gradient complexity)

### Reconstruction quality degraded
- Reduce all λ values by factor of 10
- Start with only one constraint (recommend `lambda_bc` first)
- Check if VAE decoder is properly frozen

### Physics metrics not improving
- Increase λ values
- Check that physics losses are being computed (print statements)
- Verify mask orientation matches velocity field

## Code Structure

```
Diffusion_model/
├── src/
│   ├── physics.py           # Physics loss functions and metrics
│   ├── helper.py            # run_epoch with physics integration
│   └── predictor.py         # LatentDiffusionPredictor
├── scripts/
│   └── plot_physics_metrics.py  # Visualization script
├── config.py                # Lambda parameters
└── train.py                 # Training loop with physics logging
```

## References

- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Physics-Informed Neural Networks: [Raissi et al. 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Latent Diffusion: [Rombach et al. 2022](https://arxiv.org/abs/2112.10752)
