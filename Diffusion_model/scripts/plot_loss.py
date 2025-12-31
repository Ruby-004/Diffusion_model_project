import json
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Plot training and validation loss from log.json')
    parser.add_argument('--log-file', type=str, default='log.json', help='Path to log.json file')
    parser.add_argument('--output', type=str, default='loss_plot.png', help='Output filename for the plot')
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: {args.log_file} not found.")
        return

    # Read the log.json file
    with open(args.log_file, 'r') as f:
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
