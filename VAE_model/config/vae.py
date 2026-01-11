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
    default=8,
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
    default=False,
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
parser.add_argument(
    '--no-per-component-norm',
    dest='per_component_norm',
    action='store_false',
    default=True,
    help='Disable per-component normalization and use global max instead (legacy behavior). '
         'Per-component normalization is ENABLED by default and recommended for 3D flow.'
)

parser.add_argument(
    '--conditional',
    action='store_true',
    default=False,
    help='Enable conditional VAE mode. Injects is_3d condition signal to encoder/decoder '
         'to create separate latent representations for 2D flow (U_2d, w=0) vs 3D flow (U, w≠0). '
         'This helps the model learn to distinguish w=0 inputs from w≠0 outputs.'
)
