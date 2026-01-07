import argparse
import json
import os
import os.path as osp
import sys

import torch

# Add current directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.dataset import get_loader
from src.helper import set_model, select_input_output
from src.unet.metrics import cost_function
from src.predictor import LatentDiffusionPredictor

def get_latest_model_dir(trained_dir):
    if not os.path.exists(trained_dir):
        raise ValueError(f"Trained directory does not exist: {trained_dir}")
        
    dirs = [d for d in os.listdir(trained_dir) if osp.isdir(osp.join(trained_dir, d))]
    if not dirs:
        raise ValueError(f"No model directories found in {trained_dir}")
    
    # Sort by name (which has timestamp) descending
    dirs.sort(reverse=True)
    return osp.join(trained_dir, dirs[0])

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument('model_path', nargs='?', type=str, default=None, 
                        help='Path to the model directory or model file (e.g. model.pt).')
    parser.add_argument('--model-dir', type=str, default=None, 
                        help='Directory of the trained model. Deprecated, use positional arg.')
    parser.add_argument('--save-dir', type=str, default='./trained', 
                        help='Root directory for trained models (default: ./trained)')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()

    # Determine model directory
    # Prefer positional arg, then flag, then latest
    target_path = args.model_path if args.model_path else args.model_dir
    
    weights_file = None

    if target_path is None:
        model_dir = get_latest_model_dir(args.save_dir)
        print(f"No model path specified. Using latest found: {model_dir}")
    else:
        # Check if it exists
        if not os.path.exists(target_path):
             # Try checking if it's a subfolder of save-dir
             tmp_path = osp.join(args.save_dir, target_path)
             if os.path.exists(tmp_path):
                 target_path = tmp_path
             else:
                 raise ValueError(f"Model path not found: {target_path}")
        
        # If it's a file (e.g. model.pt), get dirname
        if os.path.isfile(target_path):
            print(f"You provided a file: {target_path}. Using its parent directory.")
            model_dir = os.path.dirname(target_path)
            weights_file = target_path
        else:
            model_dir = target_path

    print(f"Evaluating model in: {model_dir}")
    
    # Load configuration
    log_path = osp.join(model_dir, 'log.json')
    if not osp.exists(log_path):
        raise ValueError(f"No log.json found in {model_dir}. Is the training started/finished?")

    with open(log_path, 'r') as f:
        log_dict = json.load(f)
    
    params = log_dict['params']
    dataset_params = params['dataset']
    training_params = params['training']

    # device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Parameters for Model
    predictor_type = training_params['predictor_type']
    predictor_kwargs = training_params['predictor']
    root_dir = dataset_params['root_dir']
    
    # Handle possible absolute path changes or missing keys
    if not os.path.exists(root_dir):
        print(f"Warning: Dataset root dir in log ({root_dir}) not found.")
        # Try to find it if it looks like the default
        if 'dataset_3d' in root_dir and os.path.exists("C:\\Users\\alexd\\Downloads\\dataset_3d"):
             root_dir = "C:\\Users\\alexd\\Downloads\\dataset_3d"
             print(f"Using alternate path: {root_dir}")
    
    dataset_params['root_dir'] = root_dir

    # Load Model Structure
    print("Initializing model...")
    def init_and_load(p_kwargs, strict=True):
        predictor = set_model(
            type=predictor_type,
            kwargs=p_kwargs,
            norm_file=osp.join(root_dir, 'statistics.json')
        )
        predictor.to(device)
        
        # Load Weights
        if weights_file:
             model_weights = weights_file
        else:
             # Try best_model.pt first, then model.pt
             model_weights = osp.join(model_dir, 'best_model.pt')
             if not osp.exists(model_weights):
                 print("best_model.pt not found, trying model.pt")
                 model_weights = osp.join(model_dir, 'model.pt')
        
        if not osp.exists(model_weights):
            raise FileNotFoundError(f"No model weights found in {model_dir}")
            
        print(f"Loading weights from {model_weights}")
        state_desc = torch.load(model_weights, map_location=device)
        predictor.load_state_dict(state_desc, strict=strict)
        predictor.eval()
        return predictor

    try:
        predictor = init_and_load(predictor_kwargs)
    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(e) and "time_mlp" in str(e):
            print("\nWarning: Model checkpoint missing time embeddings. Identifying as legacy model.")
            print("Retrying with time_embedding_dim=None...")
            
            # Create a deep copy or just modify depending on structure
            import copy
            p_kwargs_legacy = copy.deepcopy(predictor_kwargs)
            if 'model_kwargs' not in p_kwargs_legacy:
                p_kwargs_legacy['model_kwargs'] = {}
            p_kwargs_legacy['model_kwargs']['time_embedding_dim'] = None
            
            predictor = init_and_load(p_kwargs_legacy)
        else:
            raise e

    # Load Data
    print("Loading test dataset...")
    use_3d = dataset_params.get('use_3d', False)
    batch_size = dataset_params.get('batch_size', 1)
    
    loaders = get_loader(
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        k_folds=None,
        num_workers=0, # Use 0 to avoid potential multiprocessing issues in script
        use_3d=use_3d
    )
    
    # get_loader returns list of tuples
    train_loader, val_loader, test_loader = loaders[0]
    
    if test_loader is None:
        raise ValueError("Test loader is None. Check dataset splitting logic.")

    # Criterion
    criterion_name = training_params['cost_function']
    criterion = cost_function(criterion_name)
    print(f"Using cost function: {criterion_name}")

    # Evaluation Loop
    print("\nStarting evaluation on test set...")
    test_loss = 0
    num_test_batch = len(test_loader)
    
    # For Latent Diffusion, check if we provided VAE path and if it needs loading
    # The set_model should have initialized LatentDiffusionPredictor
    # which inside initializes its VAE.
    # predictor_kwargs has 'vae_path'.
    
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            print(f"\rBatch [{k+1}/{num_test_batch}]", end="")
            
            # Select input/output
            if predictor_type == 'latent-diffusion':
                 # Custom selection logic (borrowed from helper.select_input_output but inline if we want, or call it)
                 input_data, targets = select_input_output(data, predictor_type, device)
                 
                 img = input_data[0]
                 velocity_2d = input_data[1]
                 
                 # Encode target to latent
                 target_latents = predictor.encode_target(targets, velocity_2d)
                 
                 # Predict noise
                 # We pass x_start to calculate loss (forward process -> noise -> predict noise -> loss)
                 preds, target_noise = predictor(img, velocity_2d, x_start=target_latents)
                 
                 loss = criterion(output=preds, target=target_noise)
                 
            else:
                 input_data, targets = select_input_output(data, predictor_type, device)
                 preds = predictor(*input_data)
                 targets = predictor.normalizer['output'](targets)
                 loss = criterion(output=preds, target=targets)
            
            test_loss += loss.item()
            
    avg_test_loss = test_loss / num_test_batch
    print(f"\n\nTest Set Loss: {avg_test_loss:.6f}")
    
    # Save result to a file in the model dir for record
    result_file = osp.join(model_dir, 'test_result.txt')
    with open(result_file, 'w') as f:
        f.write(f"Test Loss: {avg_test_loss}\n")
    print(f"Result saved to {result_file}")

if __name__ == '__main__':
    main()
