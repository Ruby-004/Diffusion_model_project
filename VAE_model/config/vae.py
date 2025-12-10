import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset-dir',
    type=str,
    # required=True,
    default='C:/Users/alexd/Downloads/sample',
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
    required=True,
    help='Number of channels in input data.'
)
parser.add_argument(
    '--latent-channels',
    type=int,
    default=4,
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
    default=10,
    help='Batch size.'
)
parser.add_argument(
    '--num-epochs',
    type=int,
    default=100,
    help='Number of epochs.'
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
    default=1e-5,  # Very low for large 3D model stability
    help='Learning rate.'
)
