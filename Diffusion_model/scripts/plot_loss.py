import json
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss from log.json')
    parser.add_argument('path', nargs='?', default='log.json', help='Path to log.json file or directory containing it')
    parser.add_argument('--output', type=str, default='loss_plot.png', help='Output filename for the plot')
    args = parser.parse_args()

    log_file = args.path
    if os.path.isdir(log_file):
        log_file = os.path.join(log_file, 'log.json')

    if not os.path.exists(log_file):
        # Try looking in ../trained/ if a folder name was given but not found locally
        if not os.path.exists(args.path) and not os.path.isabs(args.path):
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

    # Extract the loss data
    epochs = data['epoch']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    
    # Adjust epochs to be 1-based if they are 0-based
    if epochs[0] == 0:
        epochs = [e + 1 for e in epochs]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot Training and Validation Loss
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')

    # Plot Test Loss if available
    if 'test_loss' in data:
        test_loss = data['test_loss']
        plt.axhline(y=test_loss, color='g', linestyle='--', linewidth=2, label=f'Test Loss: {test_loss:.6f}')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Use log scale if loss varies significantly
    # plt.yscale('log') 

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Plot saved as {args.output}')
    # plt.show()

if __name__ == "__main__":
    main()
