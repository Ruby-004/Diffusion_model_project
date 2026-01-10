"""
Plot physics metrics from training log.

This script visualizes the physics-informed training metrics to assess
how well the model learns physically consistent flow predictions.

Usage:
    python scripts/plot_physics_metrics.py path/to/log.json
    python scripts/plot_physics_metrics.py path/to/trained_model_folder
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Plot physics metrics from log.json')
    parser.add_argument('path', nargs='?', default='log.json', 
                        help='Path to log.json file or directory containing it')
    parser.add_argument('--output', type=str, default='physics_metrics.png', 
                        help='Output filename for the plot')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='Additional log.json files to compare')
    args = parser.parse_args()

    log_file = args.path
    if os.path.isdir(log_file):
        log_file = os.path.join(log_file, 'log.json')

    if not os.path.exists(log_file):
        # Try looking in ../trained/
        if not os.path.isabs(args.path):
            potential_path = os.path.join('..', 'trained', args.path, 'log.json')
            if os.path.exists(potential_path):
                log_file = potential_path
            else:
                print(f"Error: {log_file} not found.")
                return
        else:
            print(f"Error: {log_file} not found.")
            return

    # Read the log.json file
    with open(log_file, 'r') as f:
        data = json.load(f)

    epochs = data['epoch']
    if epochs[0] == 0:
        epochs = [e + 1 for e in epochs]

    # Check if physics metrics exist
    if 'physics_metrics' not in data:
        print("No physics metrics found in log.json")
        print("Physics-informed training may not have been enabled.")
        return

    physics_metrics = data['physics_metrics']
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: Divergence (Mass Conservation)
    ax1 = axes[0, 0]
    if 'div_mean' in physics_metrics and len(physics_metrics['div_mean']) > 0:
        div_mean = physics_metrics['div_mean']
        ax1.plot(epochs[:len(div_mean)], div_mean, 'b-', linewidth=2, label='Mean |∇·u|')
        if 'div_std' in physics_metrics and len(physics_metrics['div_std']) > 0:
            div_std = physics_metrics['div_std']
            ax1.fill_between(epochs[:len(div_mean)], 
                           [m - s for m, s in zip(div_mean, div_std[:len(div_mean)])],
                           [m + s for m, s in zip(div_mean, div_std[:len(div_mean)])],
                           alpha=0.3, color='b')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Divergence', fontsize=11)
    ax1.set_title('Mass Conservation (Divergence)\nLower is better', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Flow Rate Consistency
    ax2 = axes[0, 1]
    if 'flow_rate_cv' in physics_metrics and len(physics_metrics['flow_rate_cv']) > 0:
        flow_cv = physics_metrics['flow_rate_cv']
        ax2.plot(epochs[:len(flow_cv)], flow_cv, 'g-', linewidth=2, label='Flow Rate CV')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Coefficient of Variation', fontsize=11)
    ax2.set_title('Flow Rate Consistency\nLower is better', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity in Solid Region (No-Slip BC)
    ax3 = axes[1, 0]
    if 'vel_in_solid' in physics_metrics and len(physics_metrics['vel_in_solid']) > 0:
        vel_solid = physics_metrics['vel_in_solid']
        ax3.plot(epochs[:len(vel_solid)], vel_solid, 'r-', linewidth=2, label='|u| in solid')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('RMS Velocity', fontsize=11)
    ax3.set_title('No-Slip BC Violation\nLower is better (should → 0)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Training & Validation Loss + Physics Loss Components
    ax4 = axes[1, 1]
    if 'train_loss' in data and len(data['train_loss']) > 0:
        ax4.plot(epochs[:len(data['train_loss'])], data['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'val_loss' in data and len(data['val_loss']) > 0:
        ax4.plot(epochs[:len(data['val_loss'])], data['val_loss'], 'r-', linewidth=2, label='Val Loss')
    
    # Add physics loss components if available
    loss_keys = ['loss_divergence', 'loss_flow_rate', 'loss_no_slip', 'loss_smoothness']
    loss_labels = ['Div Loss', 'Flow Loss', 'No-Slip Loss', 'Smooth Loss']
    loss_colors = ['c', 'm', 'y', 'k']
    
    for key, label, color in zip(loss_keys, loss_labels, loss_colors):
        if key in physics_metrics and len(physics_metrics[key]) > 0:
            values = physics_metrics[key]
            if any(v > 0 for v in values):  # Only plot if non-zero
                ax4.plot(epochs[:len(values)], values, f'{color}--', linewidth=1.5, alpha=0.7, label=label)
    
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Add summary statistics
    fig.suptitle('Physics-Informed Training Metrics', fontsize=14, fontweight='bold', y=1.02)
    
    # Print summary
    print("\n=== Physics Metrics Summary (Final Epoch) ===")
    for key, values in physics_metrics.items():
        if len(values) > 0 and values[-1] != 0:
            print(f"  {key}: {values[-1]:.6f}")
    
    # Print physics configuration from params
    if 'params' in data and 'training' in data['params']:
        train_params = data['params']['training']
        print("\n=== Physics Loss Weights ===")
        for key in ['lambda_div', 'lambda_flow', 'lambda_bc', 'lambda_smooth']:
            if key in train_params:
                print(f"  {key}: {train_params[key]}")

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved as {args.output}')


def compare_physics_metrics(log_files: list, output: str = 'physics_comparison.png'):
    """
    Compare physics metrics across multiple training runs.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for idx, log_file in enumerate(log_files):
        if os.path.isdir(log_file):
            log_file = os.path.join(log_file, 'log.json')
        
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping.")
            continue
            
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        epochs = data['epoch']
        if epochs[0] == 0:
            epochs = [e + 1 for e in epochs]
        
        # Get label from folder name
        label = os.path.basename(os.path.dirname(log_file))[:30]
        
        physics_metrics = data.get('physics_metrics', {})
        
        # Divergence
        if 'div_mean' in physics_metrics:
            axes[0, 0].plot(epochs[:len(physics_metrics['div_mean'])], 
                          physics_metrics['div_mean'], 
                          color=colors[idx], linewidth=2, label=label)
        
        # Flow rate CV
        if 'flow_rate_cv' in physics_metrics:
            axes[0, 1].plot(epochs[:len(physics_metrics['flow_rate_cv'])], 
                          physics_metrics['flow_rate_cv'], 
                          color=colors[idx], linewidth=2, label=label)
        
        # Velocity in solid
        if 'vel_in_solid' in physics_metrics:
            axes[1, 0].plot(epochs[:len(physics_metrics['vel_in_solid'])], 
                          physics_metrics['vel_in_solid'], 
                          color=colors[idx], linewidth=2, label=label)
        
        # Val loss
        if 'val_loss' in data:
            axes[1, 1].plot(epochs[:len(data['val_loss'])], 
                          data['val_loss'], 
                          color=colors[idx], linewidth=2, label=label)
    
    axes[0, 0].set_title('Divergence (Mass Conservation)')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_title('Flow Rate CV')
    axes[1, 0].set_title('Velocity in Solid')
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_title('Validation Loss')
    axes[1, 1].set_yscale('log')
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Comparison plot saved as {output}')


if __name__ == "__main__":
    main()
