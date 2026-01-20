#!/usr/bin/env python
"""
Grid Search for Latent Diffusion Model Hyperparameter Tuning.

This script performs a systematic grid search over a compact set of hyperparameters
for the latent diffusion model, running each combination for a fixed number of epochs
and recording validation performance.

Usage:
    python gridsearch_diffusion.py

Output:
    - results.csv: All runs with hyperparameters and metrics
    - top10.csv: Top 10 configurations by best validation loss
    - summary.txt: Best configuration details
"""

import os
import os.path as osp
import sys
import json
import time
import csv
import itertools
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.optim as optim

# Local imports
from utils.dataset import get_loader
from src.helper import set_model, run_epoch
from src.unet.metrics import cost_function
from src.predictor import LatentDiffusionPredictor


# =============================================================================
# CONFIGURATION: GRID SEARCH PARAMETERS
# =============================================================================

# Base configuration (center point)
BASE_CONFIG = {
    "name": "gridsearch",
    "mode": "train",
    "save_dir": "./trained/gridsearch_per_component/",
    
    "dataset": {
        "root_dir": r"C:\Users\alexd\Downloads\dataset_3d",
        "batch_size": 3,
        "augment": False,
        "shuffle": False,
        "k_folds": 5,
        "use_3d": True
    },
    
    "training": {
        "device": "cuda",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "scheduler": {"flag": False, "gamma": 0.95499},
        "num_epochs": 10,  # Grid search uses 10 epochs per combo
        "cost_function": "normalized_mse_loss_per_component",
        "lambda_div": 0.0,
        "lambda_flow": 0.0,
        "lambda_smooth": 0.0,
        "lambda_laplacian": 0.0,
        "physics_loss_freq": 0,
        "lambda_velocity": 0.0,
        "weight_u": 1.0,
        "weight_v": 1.0,
        "weight_w": 1.0,
        "velocity_loss_primary": False,
        "predictor_type": "latent-diffusion",
        
        "predictor": {
            "model_name": "UNet",
            "model_kwargs": {
                "in_channels": 17,
                "out_channels": 8,
                "features": [64, 128, 256, 512, 1024],  # depth 5 (baseline)
                "kernel_size": 3,
                "padding_mode": "zeros",
                "activation": "silu",
                "attention": "3..2",
                "dropout": 0.0,
                "time_embedding_dim": 64
            },
            "distance_transform": True,
            "vae_encoder_path": r"C:\Users\alexd\Documents\GitHub\Diffusion_model_project\VAE_model\trained\dual_vae_stage2_2d",
            "vae_decoder_path": r"C:\Users\alexd\Documents\GitHub\Diffusion_model_project\VAE_model\trained\dual_vae_stage1_3d",
            "num_slices": 11
        }
    }
}

# =============================================================================
# GRID SEARCH SPACE
# =============================================================================
# Total combos target: <= 55 (hard cap 60)
# 
# Selected ranges:
# - depth: {4, 5} = 2 values  (skip depth 6 due to memory concerns)
# - kernel_size: {3, 5} = 2 values
# - attention: {"3..2", "2..1"} = 2 values  
# - learning_rate: {5e-5, 1e-4, 2e-4} = 3 values
# - dropout: {0.0, 0.05} = 2 values
# - time_embedding_dim: {64} = 1 value (keep fixed to reduce combos)
#
# Total: 2 * 2 * 2 * 3 * 2 * 1 = 48 combinations (under 55 cap)

GRID = {
    # Depth/Features configurations
    # depth 4: smaller network, faster training
    # depth 5: baseline (current best guess)
    # depth 6: REMOVED - spatial dimensions collapse to 1x1 at bottleneck (64/2^6=1)
    "features": [
        [64, 128, 256, 512],           # depth 4
        [64, 128, 256, 512, 1024],     # depth 5 (baseline)
        [32, 64, 128, 256, 512],
        [128, 256, 512, 1024, 2048]
    ],
    
    # Kernel size
    # 3: baseline, faster
    # 5: larger receptive field, may capture longer-range patterns
    "kernel_size": [3],
    
    # Attention settings
    # "3..2": attention from level 3 to max, 2 heads (baseline)
    # "": no attention (lighter, faster)
    "attention": ["3..2"],
    
    # Learning rate
    # Center: 1e-4, explore 0.5x and 2x
    "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    
    # Dropout for regularization
    "dropout": [0.0],
    
    # Time embedding dimension (kept fixed to reduce search space)
    "time_embedding_dim": [64],
}

# Fixed random seed for reproducibility (must match VAE training seed for consistent data split)
RANDOM_SEED = 2024

# Number of epochs per grid search run
NUM_EPOCHS = 15

# Output directory for grid search results
OUTPUT_DIR = "./trained/gridsearchmse_comp/"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_run_name(params: Dict[str, Any]) -> str:
    """Generate a unique run name encoding all hyperparameters."""
    depth = len(params["features"])
    features_str = "-".join(map(str, params["features"]))
    
    name = (
        f"d{depth}_"
        f"f{features_str}_"
        f"k{params['kernel_size']}_"
        f"a{params['attention'].replace('.', 'd')}_"  # Replace dots for filename safety
        f"lr{params['learning_rate']:.0e}_"
        f"dr{params['dropout']}_"
        f"te{params['time_embedding_dim']}"
    )
    return name


def make_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a full config dict from grid search parameters."""
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    
    # Update model kwargs
    model_kwargs = config["training"]["predictor"]["model_kwargs"]
    model_kwargs["features"] = params["features"]
    model_kwargs["kernel_size"] = params["kernel_size"]
    model_kwargs["attention"] = params["attention"]
    model_kwargs["dropout"] = params["dropout"]
    model_kwargs["time_embedding_dim"] = params["time_embedding_dim"]
    
    # Update learning rate
    config["training"]["learning_rate"] = params["learning_rate"]
    
    # Update name
    config["name"] = generate_run_name(params)
    
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dry_run_forward_pass(predictor: LatentDiffusionPredictor, device: str, batch_size: int = 1) -> bool:
    """
    Perform a dry run forward pass to verify model works.
    
    Returns True if successful, False otherwise.
    """
    try:
        predictor.eval()
        with torch.no_grad():
            # Create dummy inputs
            # Shape: (batch, num_slices, 1, H, W) for microstructure
            # Shape: (batch, num_slices, 3, H, W) for velocity
            H, W = 128, 128  # Typical size
            num_slices = 11
            
            dummy_micro = torch.randn(batch_size, num_slices, 1, H, W, device=device)
            dummy_vel_2d = torch.randn(batch_size, num_slices, 3, H, W, device=device)
            dummy_target = torch.randn(batch_size, num_slices, 3, H, W, device=device)
            
            # Encode target
            target_latents = predictor.encode_target(dummy_target, dummy_vel_2d)
            
            # Forward pass
            preds, target_noise = predictor(dummy_micro, dummy_vel_2d, x_start=target_latents)
            
            # Check output shapes are valid
            assert preds.shape == target_noise.shape
            
        return True
    except Exception as e:
        print(f"  Dry run FAILED: {e}")
        return False


def train_single_config(
    config: Dict[str, Any],
    train_loader,
    val_loader,
    run_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Train a single configuration and return results.
    
    Returns dict with:
        - best_val_loss: minimum validation loss
        - best_epoch: epoch where best_val_loss was achieved
        - last_val_loss: validation loss at final epoch
        - wall_time: total training time in seconds
        - checkpoint_path: path to best model checkpoint
        - all_train_losses: list of train losses per epoch
        - all_val_losses: list of val losses per epoch
        - num_params: number of trainable parameters
        - success: whether training completed without errors
        - error_message: error message if failed
    """
    result = {
        "run_name": run_name,
        "best_val_loss": float("inf"),
        "best_epoch": -1,
        "last_val_loss": float("inf"),
        "wall_time": 0.0,
        "checkpoint_path": "",
        "all_train_losses": [],
        "all_val_losses": [],
        "num_params": 0,
        "success": False,
        "error_message": ""
    }
    
    try:
        # Create run directory
        run_dir = osp.join(output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Extract training params
        root_dir = config["dataset"]["root_dir"]
        train_dict = config["training"]
        
        device = train_dict["device"]
        learning_rate = train_dict["learning_rate"]
        weight_decay = train_dict["weight_decay"]
        scheduler_kwargs = train_dict["scheduler"]
        num_epochs = train_dict["num_epochs"]
        cost_function_name = train_dict["cost_function"]
        
        predictor_type = train_dict["predictor_type"]
        predictor_kwargs = train_dict["predictor"]
        
        # Create model
        predictor = set_model(
            type=predictor_type,
            kwargs=predictor_kwargs,
            norm_file=osp.join(root_dir, "statistics.json")
        )
        predictor.to(device)
        
        result["num_params"] = count_parameters(predictor)
        
        # Dry run validation
        print(f"  Running dry forward pass...")
        if not dry_run_forward_pass(predictor, device, batch_size=1):
            result["error_message"] = "Dry run forward pass failed"
            return result
        print(f"  Dry run passed. Model has {result['num_params']:,} parameters.")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            predictor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = None
        if scheduler_kwargs["flag"]:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=scheduler_kwargs["gamma"]
            )
        
        criterion = cost_function(cost_function_name)
        
        # Training loop
        best_loss = float("inf")
        best_model_path = osp.join(run_dir, "best_model.pt")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            avg_train_loss, avg_val_loss, physics_metrics = run_epoch(
                loaders=(train_loader, val_loader),
                predictor=predictor,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                lambda_div=train_dict.get("lambda_div", 0.0),
                lambda_flow=train_dict.get("lambda_flow", 0.0),
                lambda_smooth=train_dict.get("lambda_smooth", 0.0),
                lambda_laplacian=train_dict.get("lambda_laplacian", 0.0),
                physics_loss_freq=train_dict.get("physics_loss_freq", 1),
                lambda_velocity=train_dict.get("lambda_velocity", 0.0),
                weight_u=train_dict.get("weight_u", 1.0),
                weight_v=train_dict.get("weight_v", 1.0),
                weight_w=train_dict.get("weight_w", 1.0),
                velocity_loss_primary=train_dict.get("velocity_loss_primary", False)
            )
            
            epoch_time = time.time() - epoch_start
            
            result["all_train_losses"].append(avg_train_loss)
            result["all_val_losses"].append(avg_val_loss)
            
            # Track best
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                result["best_val_loss"] = avg_val_loss
                result["best_epoch"] = epoch
                torch.save(predictor.state_dict(), best_model_path)
            
            if scheduler is not None:
                scheduler.step()
            
            print(f"    Epoch {epoch+1}/{num_epochs}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f} ({epoch_time:.1f}s)")
        
        result["wall_time"] = time.time() - start_time
        result["last_val_loss"] = avg_val_loss
        result["checkpoint_path"] = best_model_path
        result["success"] = True
        
        # Save config to run directory
        config_path = osp.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save training log
        log_path = osp.join(run_dir, "log.json")
        log_dict = {
            "params": config,
            "epoch": list(range(num_epochs)),
            "train_loss": result["all_train_losses"],
            "val_loss": result["all_val_losses"],
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "wall_time": result["wall_time"]
        }
        with open(log_path, "w") as f:
            json.dump(log_dict, f, indent=2)
        
        # Cleanup to free GPU memory
        del predictor
        del optimizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        import traceback
        result["error_message"] = f"{str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR: {e}")
        torch.cuda.empty_cache()
    
    return result


def generate_all_combinations() -> List[Dict[str, Any]]:
    """Generate all hyperparameter combinations."""
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    
    combinations = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        combinations.append(params)
    
    return combinations


def save_results_csv(results: List[Dict[str, Any]], filepath: str):
    """Save results to CSV file."""
    if not results:
        return
    
    # Flatten nested params for CSV
    rows = []
    for r in results:
        row = {
            "run_name": r["run_name"],
            "depth": len(r.get("features", [])),
            "features": str(r.get("features", [])),
            "kernel_size": r.get("kernel_size", ""),
            "attention": r.get("attention", ""),
            "learning_rate": r.get("learning_rate", ""),
            "dropout": r.get("dropout", ""),
            "time_embedding_dim": r.get("time_embedding_dim", ""),
            "best_val_loss": r.get("best_val_loss", float("inf")),
            "best_epoch": r.get("best_epoch", -1),
            "last_val_loss": r.get("last_val_loss", float("inf")),
            "wall_time_sec": r.get("wall_time", 0.0),
            "num_params": r.get("num_params", 0),
            "success": r.get("success", False),
            "checkpoint_path": r.get("checkpoint_path", ""),
            "error_message": r.get("error_message", "")
        }
        rows.append(row)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def append_result_csv(result: Dict[str, Any], params: Dict[str, Any], filepath: str):
    """Append a single result to CSV file (for incremental saving)."""
    row = {
        "run_name": result["run_name"],
        "depth": len(params.get("features", [])),
        "features": str(params.get("features", [])),
        "kernel_size": params.get("kernel_size", ""),
        "attention": params.get("attention", ""),
        "learning_rate": params.get("learning_rate", ""),
        "dropout": params.get("dropout", ""),
        "time_embedding_dim": params.get("time_embedding_dim", ""),
        "best_val_loss": result.get("best_val_loss", float("inf")),
        "best_epoch": result.get("best_epoch", -1),
        "last_val_loss": result.get("last_val_loss", float("inf")),
        "wall_time_sec": result.get("wall_time", 0.0),
        "num_params": result.get("num_params", 0),
        "success": result.get("success", False),
        "checkpoint_path": result.get("checkpoint_path", ""),
        "error_message": result.get("error_message", "")
    }
    
    file_exists = osp.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def create_top10_report(results_csv: str, output_dir: str):
    """
    Create top 10 report from results CSV.
    
    Ranking: by best_val_loss ascending (lower is better).
    """
    import pandas as pd
    
    df = pd.read_csv(results_csv)
    
    # Filter successful runs only
    df_success = df[df["success"] == True].copy()
    
    if len(df_success) == 0:
        print("WARNING: No successful runs to rank!")
        return
    
    # Sort by best_val_loss (ascending = lower is better)
    df_sorted = df_success.sort_values("best_val_loss", ascending=True)
    
    # Top 10
    top10 = df_sorted.head(10)
    
    # Save top10.csv
    top10_path = osp.join(output_dir, "top10.csv")
    top10.to_csv(top10_path, index=False)
    print(f"\nTop 10 configurations saved to: {top10_path}")
    
    # Create summary.txt
    summary_path = osp.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GRID SEARCH SUMMARY - LATENT DIFFUSION MODEL\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total combinations tried: {len(df)}\n")
        f.write(f"Successful runs: {len(df_success)}\n")
        f.write(f"Failed runs: {len(df) - len(df_success)}\n")
        f.write(f"Epochs per run: {NUM_EPOCHS}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RANKING CRITERION: best_val_loss (lower is better)\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-" * 80 + "\n\n")
        
        for rank, (idx, row) in enumerate(top10.iterrows(), 1):
            f.write(f"Rank #{rank}\n")
            f.write(f"  Run name: {row['run_name']}\n")
            f.write(f"  Best val loss: {row['best_val_loss']:.6f}\n")
            f.write(f"  Best epoch: {row['best_epoch']}\n")
            f.write(f"  Last val loss: {row['last_val_loss']:.6f}\n")
            f.write(f"  Parameters: {row['num_params']:,}\n")
            f.write(f"  Wall time: {row['wall_time_sec']:.1f}s\n")
            f.write(f"  Checkpoint: {row['checkpoint_path']}\n")
            f.write(f"  Hyperparameters:\n")
            f.write(f"    - depth: {row['depth']}\n")
            f.write(f"    - features: {row['features']}\n")
            f.write(f"    - kernel_size: {row['kernel_size']}\n")
            f.write(f"    - attention: {row['attention']}\n")
            f.write(f"    - learning_rate: {row['learning_rate']}\n")
            f.write(f"    - dropout: {row['dropout']}\n")
            f.write(f"    - time_embedding_dim: {row['time_embedding_dim']}\n")
            f.write("\n")
        
        # Best configuration JSON dump
        best_row = top10.iloc[0]
        f.write("=" * 80 + "\n")
        f.write("WINNING CONFIGURATION (JSON)\n")
        f.write("=" * 80 + "\n\n")
        
        # Reconstruct config
        winning_params = {
            "features": eval(best_row["features"]),
            "kernel_size": int(best_row["kernel_size"]),
            "attention": best_row["attention"],
            "learning_rate": float(best_row["learning_rate"]),
            "dropout": float(best_row["dropout"]),
            "time_embedding_dim": int(best_row["time_embedding_dim"])
        }
        winning_config = make_config(winning_params)
        f.write(json.dumps(winning_config, indent=2))
        f.write("\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Print top 10 to console
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS (by best_val_loss, ascending)")
    print("=" * 80)
    print(top10[["run_name", "best_val_loss", "best_epoch", "num_params", "wall_time_sec"]].to_string(index=False))
    print("=" * 80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 80)
    print("LATENT DIFFUSION MODEL - GRID SEARCH")
    print("=" * 80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Epochs per run: {NUM_EPOCHS}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate all combinations
    all_combos = generate_all_combinations()
    total_combos = len(all_combos)
    print(f"\nTotal hyperparameter combinations: {total_combos}")
    
    if total_combos > 150:  # Increased cap to allow larger searches
        print(f"WARNING: Combination count {total_combos} exceeds hard cap of 150!")
        print("Consider reducing the search space.")
        return
    
    print("\nGrid Search Space:")
    for key, values in GRID.items():
        print(f"  {key}: {values}")
    
    # Load data once (all runs share the same data split)
    print("\nLoading dataset...")
    dataset_config = BASE_CONFIG["dataset"]
    
    train_loader, val_loader, test_loader = get_loader(
        root_dir=dataset_config["root_dir"],
        batch_size=dataset_config["batch_size"],
        shuffle=dataset_config["shuffle"],
        augment=dataset_config["augment"],
        k_folds=None,  # Single fold for speed
        num_workers=0,
        use_3d=dataset_config["use_3d"],
        seed=RANDOM_SEED
    )[0]
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print("\nFold strategy: Single fold (70/15/15 split with seed={})".format(RANDOM_SEED))
    
    # Results storage
    results_csv_path = osp.join(OUTPUT_DIR, "results.csv")
    all_results = []
    
    # Check for existing results to enable resumption
    completed_runs = set()
    if osp.exists(results_csv_path):
        import pandas as pd
        try:
            existing_df = pd.read_csv(results_csv_path)
            completed_runs = set(existing_df["run_name"].tolist())
            print(f"\nFound {len(completed_runs)} completed runs. Resuming...")
        except:
            pass
    
    # Run grid search
    print("\n" + "=" * 80)
    print("STARTING GRID SEARCH")
    print("=" * 80 + "\n")
    
    total_start_time = time.time()
    
    for idx, params in enumerate(all_combos):
        run_name = generate_run_name(params)
        
        print(f"\n[{idx+1}/{total_combos}] {run_name}")
        print("-" * 60)
        
        # Skip if already completed
        if run_name in completed_runs:
            print("  SKIPPED (already completed)")
            continue
        
        # Make config
        config = make_config(params)
        config["training"]["num_epochs"] = NUM_EPOCHS
        
        # Log hyperparameters
        print(f"  features: {params['features']} (depth={len(params['features'])})")
        print(f"  kernel_size: {params['kernel_size']}")
        print(f"  attention: {params['attention']}")
        print(f"  learning_rate: {params['learning_rate']}")
        print(f"  dropout: {params['dropout']}")
        print(f"  time_embedding_dim: {params['time_embedding_dim']}")
        
        # Train
        result = train_single_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            run_name=run_name,
            output_dir=OUTPUT_DIR
        )
        
        # Add params to result for CSV
        result.update(params)
        all_results.append(result)
        
        # Save incrementally
        append_result_csv(result, params, results_csv_path)
        
        # Status
        if result["success"]:
            print(f"  COMPLETED: best_val_loss={result['best_val_loss']:.6f} @ epoch {result['best_epoch']}")
            print(f"  Wall time: {result['wall_time']:.1f}s")
        else:
            print(f"  FAILED: {result['error_message'][:100]}...")
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print(f"GRID SEARCH COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print("=" * 80)
    
    # Create top 10 report
    print("\nGenerating Top-10 report...")
    create_top10_report(results_csv_path, OUTPUT_DIR)
    
    print(f"\nOutput files:")
    print(f"  - {results_csv_path}")
    print(f"  - {osp.join(OUTPUT_DIR, 'top10.csv')}")
    print(f"  - {osp.join(OUTPUT_DIR, 'summary.txt')}")


if __name__ == "__main__":
    main()
