import argparse
import sys
import os
import os.path as osp
import json
import torch
import numpy as np

# Add project root and Diffusion_model to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

diffusion_model_path = os.path.join(project_root, 'Diffusion_model')
if diffusion_model_path not in sys.path:
    sys.path.append(diffusion_model_path)

from Diffusion_model.src.helper import set_model
from Diffusion_model.utils.dataset import get_loader

def main():
    parser = argparse.ArgumentParser(description="Inference for Microstructure Flow Prediction")
    parser.add_argument('model_path', type=str, help='Path to the trained diffusion model (directory or .pt file)')
    parser.add_argument('sample_path', type=str, nargs='?', default=None, help='Path to the input sample (.pt file). If not provided, uses a sample from the test set.')
    parser.add_argument('--vae-path', type=str, default=None, help='Path to the trained VAE model directory. If not provided, uses VAE path from model config.')
    parser.add_argument('--dataset-dir', type=str, default=None, help='Path to the dataset directory (overrides config). Used to locate statistics.json and test samples.')
    parser.add_argument('--index', type=int, default=0, help='Index of the sample in the test set to use (only if sample_path is not provided). Default: 0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    device = args.device

    print(f"--- Setting up Inference on {device} ---")

    # --- 1. & 2. Import already trained VAE and Diffusion Model ---
    # We use the helper function set_model which handles loading the VAE internally 
    # within LatentDiffusionPredictor logic based on config.
    
    if os.path.isfile(args.model_path):
        model_dir = os.path.dirname(args.model_path)
        weights_file = args.model_path
    else:
        model_dir = args.model_path
        weights_file = os.path.join(model_dir, 'model.pt')
        
    print(f"Loading model configuration from {model_dir}")
    log_path = os.path.join(model_dir, 'log.json')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log.json not found in {model_dir}")
        
    with open(log_path, 'r') as f:
        log_try = json.load(f)
        if 'params' in log_try:
            config = log_try['params']
        else:
            config = log_try 

    predictor_kwargs = config['training']['predictor']
    dataset_root = config['dataset']['root_dir']
    # Override dataset root if provided via command line
    if args.dataset_dir:
        dataset_root = args.dataset_dir
        print(f"Using dataset directory from command-line argument: {dataset_root}")
    
    # Locate statistics.json for normalization
    # Priorities: 1. Dataset root from config/argsn
    # Priorities: 1. Dataset root from config, 2. Default data path, 3. Sample directory (if provided)
    norm_file = None
    if os.path.exists(os.path.join(dataset_root, 'statistics.json')):
        norm_file = os.path.join(dataset_root, 'statistics.json')
    elif os.path.exists(os.path.join(project_root, 'Diffusion_model', 'data', 'dataset', 'statistics.json')):
        norm_file = os.path.join(project_root, 'Diffusion_model', 'data', 'dataset', 'statistics.json')
    elif args.sample_path is not None:
        # Only try sample directory if sample_path was provided
        sample_dir_norm = os.path.join(os.path.dirname(args.sample_path), 'statistics.json')
        if os.path.exists(sample_dir_norm):
            norm_file = sample_dir_norm
    
    if norm_file is None:
        print("Warning: statistics.json not found. Fallback to None (may cause errors).")
    else:
        print(f"Using statistics from: {norm_file}")
            
    # Fix VAE path in config if it was absolute path from another machine
    if 'vae_path' in predictor_kwargs:
        vae_path = predictor_kwargs['vae_path']
        
        # Override with command-line argument if provided
        if args.vae_path:
            vae_path = args.vae_path
            print(f"Using VAE path from command-line argument: {vae_path}")
        
        if not os.path.exists(vae_path) and not os.path.isabs(vae_path):
             # Try absolute relative to project
             vae_path = os.path.join(project_root, vae_path)
        
        if not os.path.exists(vae_path) and 'VAE_model' in vae_path:
             # Heuristic: find VAE_model relative to project root
             idx = vae_path.find('VAE_model')
             vae_path = os.path.join(project_root, vae_path[idx:])
             
        predictor_kwargs['vae_path'] = vae_path
    elif args.vae_path:
        # VAE path not in config but provided via command line
        predictor_kwargs['vae_path'] = args.vae_path
        print(f"Using VAE path from command-line argument: {args.vae_path}")

    print("Initializing models...")
    # This initializes LatentDiffusionPredictor, which loads the VAE (Step 1)
    predictor = set_model(
        type='latent-diffusion',
        kwargs=predictor_kwargs,
        norm_file=norm_file
    )
    
    # Load Diffusion Model Weights (Step 2)
    print(f"Loading weights from {weights_file}")
    predictor.load_state_dict(torch.load(weights_file, map_location=device))
    predictor.to(device)
    predictor.eval()

    # --- Load Sample ---
    if args.sample_path:
        # Passing sample from file path 
        print(f"Loading sample from {args.sample_path}")
        sample_data = torch.load(args.sample_path, map_location=device)
        
        # Parse sample structure
        if isinstance(sample_data, dict):
            img = sample_data.get('microstructure')
            velocity_2d = sample_data.get('velocity_input')
            if velocity_2d is None:
                velocity_2d = sample_data.get('U_2d')
            
            target_velocity = sample_data.get('velocity')
            if target_velocity is None:
                target_velocity = sample_data.get('U')
        elif isinstance(sample_data, (list, tuple)):
            img = sample_data[0]
            velocity_2d = sample_data[1]
            target_velocity = sample_data[2] if len(sample_data) > 2 else None
        else:
            img = sample_data # Fallback if just tensor
            velocity_2d = None # Will fail if model needs it

        # Handle batch dimensions
        if img.dim() == 4: img = img.unsqueeze(0)
        if velocity_2d is not None and velocity_2d.dim() == 4: velocity_2d = velocity_2d.unsqueeze(0)
        if target_velocity is not None and target_velocity.dim() == 4: target_velocity = target_velocity.unsqueeze(0)
    
    else:
        # Use Test Set
        print(f"No sample path provided. Loading Test Set (seed=2024, index={args.index})...")
        
        # Use root_dir from config
        loaders = get_loader(
            root_dir=dataset_root,
            batch_size=1, # One sample per batch
            val_ratio=0.15,
            test_ratio=0.15,
            shuffle=False, # Important for consistent indexing inside the loader
            seed=2024,     # Enforce seed 2024
            augment=False,
            use_3d=True    # Defaulting to true as most complex case, usually inferred from logic but let's assume 3D based on project
        )
        
        # get_loader with default k_folds=None returns [(train, val, test)]
        train_loader, val_loader, test_loader = loaders[0]
        
        if len(test_loader) <= args.index:
            raise IndexError(f"Test set size ({len(test_loader)}) is smaller than requested index {args.index}")
            
        print(f"Fetching sample at index {args.index} from test set...")
        # Efficiently skip to index
        for i, data in enumerate(test_loader):
            if i == args.index:
                # data is a dict because use_3d=True usually or depends on dataset implementation
                # Based on dataset.py MicroFlowDataset returns dict
                img = data['microstructure']
                velocity_2d = data.get('velocity_input')
                if velocity_2d is None:
                    velocity_2d = data.get('U_2d')

                target_velocity = data.get('velocity')
                if target_velocity is None:
                    target_velocity = data.get('U')
                
                dxyz = data.get('dxyz')
                break
    
    img = img.to(device)
    if velocity_2d is not None:
        velocity_2d = velocity_2d.to(device)

    print(f"Input Microstructure shape: {img.shape}")
    if velocity_2d is not None:
        print(f"Input Velocity shape: {velocity_2d.shape}")

    # --- 3, 4, 5. Run Prediction Pipeline ---
    # The predictor.predict() method internally performs:
    # 3. Pass sample (velocity_2d) to VAE encoder (to get conditioning latents)
    # 4. Pass output of 3 (+ microstructure features) to Diffusion Model (denoising loop)
    # 5. Pass output of 4 (denoised latents) to VAE decoder
    
    print("Running prediction (VAE Encode -> Diffusion -> VAE Decode)...")
    with torch.no_grad():
        prediction = predictor.predict(img, velocity_2d)
    print("Prediction complete.")
    print(f"Prediction output shape: {prediction.shape}")

    # --- 6. Visualizer (Napari) ---
    try:
        import napari
        
        print("Starting Napari viewer...")
        viewer = napari.Viewer()
        
        # Calculate Scale based on user request (50x50x50 um volume, 5um z-spacing)
        # prediction shape is (Batch, Channels, Depth, Height, Width)
        depth = prediction.shape[2]
        height = prediction.shape[3]
        width = prediction.shape[4]
        
        # Z-spacing is explicitly 5.0 um
        # X and Y dimensions total 50.0 um
        z_scale = 5.0
        y_scale = 50.0 / height
        x_scale = 50.0 / width
        
        scale = [z_scale, y_scale, x_scale]
        print(f"Using physical scale: {scale} (z, y, x) for 50x50x50um volume")

        def to_numpy(t):
            return t[0].detach().cpu().numpy()

        # Add Microstructure (Slices, H, W)
        micro = to_numpy(img)
        # Check dims: (slices, 1, H, W) -> remove channel dim
        if micro.ndim == 4 and micro.shape[1] == 1:
            micro = micro[:, 0, :, :]
        
        viewer.add_image(micro, name='Microstructure', opacity=0.3, colormap='gray', rendering='translucent', depiction='volume', scale=scale)

        # Add Prediction (Slices, 3, H, W)
        pred = to_numpy(prediction)
        
        # Normalize each component independently for better visualization
        # Physical velocities are very small (~0.001-0.01 m/s) and x >> y,z
        pred_normalized = np.zeros_like(pred)
        
        print("\nVelocity ranges (physical, m/s):")
        for i, component in enumerate(['V_x', 'V_y', 'V_z'][:pred.shape[1]]):
            comp_data = pred[:, i]
            vmin, vmax = np.percentile(comp_data, [1, 99])
            print(f"  {component}: [{comp_data.min():.6f}, {comp_data.max():.6f}] (p1-p99: [{vmin:.6f}, {vmax:.6f}])")
            
            # Normalize each component independently
            if vmax > vmin:
                pred_normalized[:, i] = (comp_data - vmin) / (vmax - vmin)
            else:
                pred_normalized[:, i] = comp_data
        
        # Calculate Magnitude (use original physical values)
        pred_mag = np.sqrt(np.sum(pred**2, axis=1))
        # Normalize magnitude separately
        mag_min, mag_max = np.percentile(pred_mag, [1, 99])
        if mag_max > mag_min:
            pred_mag_normalized = (pred_mag - mag_min) / (mag_max - mag_min)
        else:
            pred_mag_normalized = pred_mag
        
        # Add components (each normalized to [0,1] independently)
        # Channels: 0=x, 1=y, 2=z
        viewer.add_image(pred_normalized[:, 0], name='Pred V_x', colormap='magma', rendering='iso', depiction='volume', scale=scale, blending='additive')
        viewer.add_image(pred_normalized[:, 1], name='Pred V_y', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale, blending='additive')
        if pred.shape[1] > 2:
            viewer.add_image(pred_normalized[:, 2], name='Pred V_z', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale, blending='additive')
            
        viewer.add_image(pred_mag_normalized, name='Pred Magnitude', colormap='turbo', visible=True, rendering='mip', depiction='volume', scale=scale, blending='additive')

        # Add Target if available
        if target_velocity is not None:
            if target_velocity.dim() == 4: target_velocity = target_velocity.unsqueeze(0)
            target = to_numpy(target_velocity)
            
            # Normalize each component independently (same as prediction)
            target_normalized = np.zeros_like(target)
            
            print("\nTarget velocity ranges (physical, m/s):")
            for i, component in enumerate(['V_x', 'V_y', 'V_z'][:target.shape[1]]):
                comp_data = target[:, i]
                vmin, vmax = np.percentile(comp_data, [1, 99])
                print(f"  {component}: [{comp_data.min():.6f}, {comp_data.max():.6f}] (p1-p99: [{vmin:.6f}, {vmax:.6f}])")
                
                if vmax > vmin:
                    target_normalized[:, i] = (comp_data - vmin) / (vmax - vmin)
                else:
                    target_normalized[:, i] = comp_data
            
            target_mag = np.sqrt(np.sum(target**2, axis=1))
            mag_min, mag_max = np.percentile(target_mag, [1, 99])
            if mag_max > mag_min:
                target_mag_normalized = (target_mag - mag_min) / (mag_max - mag_min)
            else:
                target_mag_normalized = target_mag
            
            viewer.add_image(target_normalized[:, 0], name='Target V_x', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
            viewer.add_image(target_normalized[:, 1], name='Target V_y', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
            if target.shape[1] > 2:
                viewer.add_image(target_normalized[:, 2], name='Target V_z', colormap='magma', visible=False, rendering='iso', depiction='volume', scale=scale)
                
            viewer.add_image(target_mag_normalized, name='Target Magnitude', colormap='turbo', visible=False, rendering='mip', depiction='volume', scale=scale)

        print("Opening napari window...")
        # Switch to 3D display mode
        viewer.dims.ndisplay = 3
        napari.run()
        
    except ImportError as e:
        print(f"Napari not found (ImportError: {e}). Skipping visualization. Try 'pip install napari[all]'")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main()
