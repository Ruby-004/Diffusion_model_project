import json
import matplotlib.pyplot as plt
import sys
import os

# Get the directory path from command line argument
if len(sys.argv) > 1:
    vae_dir = sys.argv[1]
    log_path = os.path.join(vae_dir, 'vae_log.json')
    output_path = os.path.join(vae_dir, 'vae_loss_per_epoch.png')
else:
    log_path = 'vae_log.json'
    output_path = 'vae_loss_per_epoch.png'

# Read the vae_log.json file
with open(log_path, 'r') as f:
    data = json.load(f)

# Extract the loss data - handle both standard and dual VAE formats
loss_data = data['loss']

# Detect format
has_cross_loss = False
has_align_loss = False

if 'recons_train' in loss_data:
    # Standard VAE format
    recons_train = loss_data['recons_train']
    recons_val = loss_data['recons_val']
    kl_train = loss_data['kl_train']
    kl_val = loss_data['kl_val']
    recons_test = loss_data['recons_test']
    kl_test = loss_data['kl_test']
    title_prefix = ''
elif 'recons_2d_train' in loss_data:
    # Dual VAE Stage 2 format (2D encoder)
    recons_train = loss_data['recons_2d_train']
    recons_val = loss_data['recons_2d_val']
    kl_train = loss_data['kl_2d_train']
    kl_val = loss_data['kl_2d_val']
    recons_test = loss_data['recons_2d_test']
    kl_test = loss_data['kl_2d_test']
    title_prefix = '2D '
    
    # Check for cross loss (alignment between E2D and frozen E3D)
    if 'cross_2d3d_train' in loss_data:
        has_cross_loss = True
        cross_train = loss_data['cross_2d3d_train']
        cross_val = loss_data.get('cross_2d3d_val', None)
        cross_test = loss_data.get('cross_2d3d_test', None)
    
    # Check for align loss
    if 'align_train' in loss_data:
        has_align_loss = True
        align_train = loss_data['align_train']
        align_val = loss_data.get('align_val', None)
        align_test = loss_data.get('align_test', None)
        
elif 'recons_3d_train' in loss_data:
    # Dual VAE Stage 1 format (3D decoder)
    recons_train = loss_data['recons_3d_train']
    recons_val = loss_data['recons_3d_val']
    kl_train = loss_data['kl_3d_train']
    kl_val = loss_data['kl_3d_val']
    recons_test = loss_data['recons_3d_test']
    kl_test = loss_data['kl_3d_test']
    title_prefix = '3D '
else:
    raise ValueError(f"Unknown VAE log format. Available keys: {list(loss_data.keys())}")

# Create epochs array
epochs = list(range(1, len(recons_train) + 1))

# Determine number of subplots based on available losses
num_plots = 1  # Always show reconstruction
# Show KL divergence for standard VAE (3D, 2D, or combined)
if 'recons_train' in loss_data:
    num_plots += 1
if has_cross_loss:
    num_plots += 1

# Create the plot
fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 5))
if num_plots == 1:
    axes = [axes]

plot_idx = 0

# Plot Reconstruction Loss - Training and Validation
axes[plot_idx].plot(epochs, recons_train, 'b-', linewidth=2, label='Training')
axes[plot_idx].plot(epochs, recons_val, 'r-', linewidth=2, label='Validation')
axes[plot_idx].axhline(y=recons_test, color='g', linestyle='--', linewidth=2, label=f'Test: {recons_test:.6f}')
axes[plot_idx].set_xlabel('Epoch', fontsize=12)
axes[plot_idx].set_ylabel('Reconstruction Loss', fontsize=12)
axes[plot_idx].set_title(f'{title_prefix}Reconstruction Loss', fontsize=14, fontweight='bold')
axes[plot_idx].legend(fontsize=11)
axes[plot_idx].grid(True, alpha=0.3)
plot_idx += 1

# Plot KL Loss for standard VAE
if 'recons_train' in loss_data:
    axes[plot_idx].plot(epochs, kl_train, 'b-', linewidth=2, label='Training')
    axes[plot_idx].plot(epochs, kl_val, 'r-', linewidth=2, label='Validation')
    axes[plot_idx].axhline(y=kl_test, color='g', linestyle='--', linewidth=2, label=f'Test: {kl_test:.6f}')
    axes[plot_idx].set_xlabel('Epoch', fontsize=12)
    axes[plot_idx].set_ylabel('KL Divergence Loss', fontsize=12)
    axes[plot_idx].set_title(f'{title_prefix}KL Divergence Loss', fontsize=14, fontweight='bold')
    axes[plot_idx].set_yscale('log')
    axes[plot_idx].legend(fontsize=11)
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

# Plot Cross Loss if available (E2D vs frozen E3D alignment)
if has_cross_loss:
    axes[plot_idx].plot(epochs, cross_train, 'b-', linewidth=2, label='Training')
    if cross_val is not None:
        axes[plot_idx].plot(epochs, cross_val, 'r-', linewidth=2, label='Validation')
    if cross_test is not None:
        axes[plot_idx].axhline(y=cross_test, color='g', linestyle='--', linewidth=2, label=f'Test: {cross_test:.6f}')
    axes[plot_idx].set_xlabel('Epoch', fontsize=12)
    axes[plot_idx].set_ylabel('Cross Loss (E2D↔E3D)', fontsize=12)
    axes[plot_idx].set_title('Cross Alignment Loss', fontsize=14, fontweight='bold')
    axes[plot_idx].legend(fontsize=11)
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Plot saved as {output_path}')
plt.show()
