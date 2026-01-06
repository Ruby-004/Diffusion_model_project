import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset-dir',
    type=str,
    # required=True,
    default='C:/Users/alexd/Downloads/dataset_3d',
    help='Directory for dataset.'
)
parser.add_argument(
    '--save-dir',
    type=str,
    default='trained/vae',
    help='Directory where to save results.'
)

parser.add_argument(
    '--in-channels',
    type=int,
    default=3,
    help='Number of channels in input data (3 velocity components: vx, vy, vz).'
)
parser.add_argument(
    '--latent-channels',
    type=int,
    default=16,
    help='Number of channels in latent space.'
)
parser.add_argument(
    '--kernel-size',
    type=int,
    default=3,
    help='Kernel size for convolutional layers.'
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=1,
    help='Batch size (reduced to 1 for 3D Conv memory management).'
)
parser.add_argument(
    '--num-epochs',
    type=int,
    default=100,
    help='Number of epochs.'
)
parser.add_argument(
    '--augment',
    action='store_true',
    help='Whether to use data augmentation.'
)
parser.add_argument(
    '--device',
    type=str,
    default=None,
    help='Device (e.g., cpu, cuda) on which to train neural network.'
)
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-6,  # Reduced for stability with 3D model
    help='Learning rate.'
)
