import os
import os.path as osp
import shutil
from pathlib import Path
import json


import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    random_split
)
from torchvision.transforms import v2

from .zenodo import download_data, unzip_data



class MicroFlowDataset(Dataset):

    """
    Dataset for steady-state velocity flow field in 2D/3D microstructures.
    For 3D data: loads a 2D microstructure and 3D velocity field (stack of 2D slices).
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        use_3d: bool = False
    ):
        """
        Initialize dataset.

        Args:
            root_dir: directory where data is stored.
            augment: whether to augment the dataset by flipping the arrays.
            use_3d: whether to load 3D velocity data (stack of 2D slices).
        """
        self._download_url = 'https://zenodo.org/records/16940478/files/dataset.zip?download=1'

        self.root_dir = root_dir
        self.augment = augment
        self.use_3d = use_3d

        self.data: dict[str, torch.Tensor] = {}

        # Download dataset if needed
        if not osp.exists(self.root_dir):
            # make directory if it doesn't exist
            os.makedirs(self.root_dir)
        
        if os.listdir(self.root_dir) == []:
            # if directory is empty, download dataset
            self.download(url=self._download_url)

        # Load dataset
        self.process()


    def process(self):
        """Load datset."""

        if self.use_3d:
            # Required files for 3D case
            meta_dict = {
                'microstructure': 'domain.pt',     # 2D: (samples, 1, H, W)
                'velocity_input': 'U_2d.pt',       # Input: (samples, num_slices, 3, H, W) - 2D velocity where vz=0
                'velocity': 'U.pt',                # Target: (samples, num_slices, 3, H, W) - Full 3D velocity
                'pressure': 'p.pt',
                'dxyz': 'dxyz.pt',
            }
            # Optional files
            optional_dict = {
                'permeability': 'permeability.pt'
            }
        else:
            meta_dict = {
                'microstructure': 'domain.pt',
                'velocity': 'U.pt',                # 2D: (samples, 3, H, W)
                'pressure': 'p.pt',
                'dxyz': 'dxyz.pt',
            }
            optional_dict = {
                'permeability': 'permeability.pt'
            }

        # Read data
        # images have a shape of (samples, channels, height, width)
        _data_x = {}
        for key, val in meta_dict.items():
            file_path = osp.join(self.root_dir, 'x', val)
            if osp.exists(file_path):
                dta = torch.load(file_path)
                _data_x[key] = dta
            else:
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load optional files
        for key, val in optional_dict.items():
            file_path = osp.join(self.root_dir, 'x', val)
            if osp.exists(file_path):
                dta = torch.load(file_path)
                _data_x[key] = dta

        try:
            # try if there are simulations with flow in y-direction
            _data_y = {}
            has_y_data = True
            for key, val in meta_dict.items():
                file_path = osp.join(self.root_dir, 'y', val)
                if osp.exists(file_path):
                    dta = torch.load(file_path)

                    if key in ['microstructure', 'velocity', 'pressure']:
                        dta = self._rotate_y_field(dta)

                    _data_y[key] = dta
                else:
                    has_y_data = False
                    break
            
            # Load optional files for y direction
            if has_y_data:
                for key, val in optional_dict.items():
                    file_path = osp.join(self.root_dir, 'y', val)
                    if osp.exists(file_path):
                        dta = torch.load(file_path)
                        _data_y[key] = dta

            # Concatenate if we have y data
            if has_y_data:
                for key in list(meta_dict.keys()) + list(optional_dict.keys()):
                    if key in _data_x and key in _data_y:
                        self.data[key] = torch.cat(
                            (_data_x[key], _data_y[key]),
                            dim=0
                        )
                    elif key in _data_x:
                        self.data[key] = _data_x[key]
                
                print("Loaded simulations with flow in 'x' and 'y' directions.")
            else:
                self.data = _data_x
                print("Loaded simulations with flow in 'x' direction.")

        except Exception as e:
            print(f"Failed to load y direction: {e}")
            self.data = _data_x
            print("Loaded simulations with flow in 'x' direction.")

        if self.augment and not self.use_3d:
            # Flip (only for 2D data)
            for key in meta_dict.keys():
                if key in ['microstructure', 'pressure']:
                    self.data[key] = torch.cat(
                        (self.data[key], torch.from_numpy(self.data[key].numpy()[:, :, ::-1, :].copy()))
                    )
                elif key == 'velocity':
                    tmp = torch.from_numpy(self.data[key].numpy()[:, :, ::-1, :].copy()) # flip
                    tmp[:, 1, :, :] = - tmp[:, 1, :, :] # switch sign for y-velocity component

                    self.data[key] = torch.cat(
                        (self.data[key], tmp)
                    )
                else:
                    self.data[key] = torch.cat(
                        (self.data[key], self.data[key])
                    )
        
        # save statistics
        self._save_statistics()

    def download(self, url: str):
        """
        Download dataset.
        
        Args:
            url: URL of the dataset.
        """

        save_dir = Path(self.root_dir).parent
        # 1. Download dataset
        zip_path = download_data(url=url, save_dir=save_dir)
        
        # 2. Unzip data
        folder_path = unzip_data(zip_path=zip_path, save_dir=save_dir)
        
        # 3. Move folder
        dest_path = self.root_dir
        try:
            shutil.move(folder_path, dest_path)
            print(f'Moved "{folder_path}" to "{dest_path}".')

        except shutil.Error as e:
            print(f"Error during move operation: {e}")
        except FileNotFoundError:
            print(f"Destination path not found. Make sure the parent directory exists.")
    

    def __len__(self):
        num_data = self.data['microstructure'].shape[0]
        return num_data
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        if self.use_3d:
            # For 3D case: all fields already have shape (num_slices, channels, H, W)
            sample = {
                'microstructure': self.data['microstructure'][idx].float(),  # (num_slices, 1, H, W)
                'velocity': self.data['velocity'][idx].float(),  # (num_slices, 3, H, W)
                'pressure': self.data['pressure'][idx, :, :, :].float(),
                'dxyz': self.data['dxyz'][idx].float(),
            }
            
            # Add velocity_input for 3D latent diffusion
            if 'velocity_input' in self.data:
                sample['velocity_input'] = self.data['velocity_input'][idx].float()
        else:
            # For 2D case: velocity has shape (samples, 3, H, W), select only [vx, vy]
            sample = {
                'microstructure': self.data['microstructure'][idx, :, :, :].float(),
                'velocity': self.data['velocity'][idx, [0,1], :, :].float(),
                'pressure': self.data['pressure'][idx, :, :, :].float(),
                'dxyz': self.data['dxyz'][idx].float(),
            }
        
        # Add optional keys if they exist (for 2D case)
        if not self.use_3d:
            if 'permeability' in self.data:
                sample['permeability'] = self.data['permeability'][idx]
            
            if 'velocity_input' in self.data:
                sample['velocity_input'] = self.data['velocity_input'][idx].float()
        
        return sample

    @staticmethod
    def load_dataset(folder: str):
        """
        Load dataset.
        
        Args:
            folder: dataset folder.
        """

        meta_dict = {
            'microstructure': 'domain.pt',
            'velocity': 'U.pt',
            'pressure': 'p.pt',
            'dxyz': 'dxyz.pt',
            'permeability': 'permeability.pt'
        }
        cases_dict = {'x': None, 'y': None}
        
        def load_flow_results(folder) -> dict[str, torch.Tensor]:
            # load data from given folder
            out = {}
            for key, val in meta_dict.items():
                file_path = osp.join(folder, val)
                data = torch.load(file_path)
                out[key] = data
            return out
    
        # Read data for each case
        for case in cases_dict.keys():

            subfolder = osp.join(folder, case)
            if osp.exists(subfolder):
                cases_dict[case] = load_flow_results(subfolder)
        
        return 
    
    @staticmethod
    def augment_dataset(
        data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Augment dataset by flipping arrays.
        
        ``
        """
        pass

    def _save_statistics(self):
        """
        Save dataset statistics.

        """
        log_file = osp.join(self.root_dir, 'statistics.json')

        stats = {}
        
        # Only save statistics for data that exists
        if 'velocity' in self.data:
            stats['U'] = {
                'max': self.data['velocity'].abs().max().item()
            }
        
        if 'pressure' in self.data:
            stats['p'] = {
                'max': self.data['pressure'].abs().max().item()
            }
        
        if 'dxyz' in self.data:
            stats['dxyz'] = {
                'max': self.data['dxyz'].abs().max().item()
            }

        # save
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=0)
        
        print(f"Saved statistics to {log_file}: {list(stats.keys())}")

    @staticmethod
    def _rotate_y_field(x: torch.Tensor):
        """
        Rotate microstructure, velocity, and pressure fields
        for simulations with flow in the y-direction.

        
        """
        _, num_channels, _, _ = x.shape

        # rotate
        x = torch.rot90(x, k=1, dims=(-2,-1))

        if num_channels != 1:
            # swap channel order
            x = x[:, [1, 0, 2], :, :]

            # change sign of new y-velocity
            x[:, 1, :, :] = - x[:, 1, :, :]
            
        return x


class BlindDataset(Dataset):
    """
    Dataset for blind prediction (no target values).
    """

    def __init__(self, data: dict[str, torch.Tensor]):
        """
        Initialize dataset.
        
        Args:
            data: dictionary of data tensors.
        """

        _keys = ['microstructure', 'dxyz']
        data_keys = data.keys()
        for key in _keys:
            if key not in data_keys:
                raise ValueError(f'Missing key `{key}` in data dictionary.')
            
        self.data: dict = data
        
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        out = {
            key: val[idx]
            for (key, val) in self.data.items()
        }
        return out
    
    def __len__(self):
        num_data = len(self.data['microstructure'])
        return num_data


class MicroFlowDataset3D(MicroFlowDataset):

    """
    Dataset for steady-state velocity flow field in slices of a 3D microstructure.
    """

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        sample = {
            'microstructure': self.data['microstructure'][idx, :, :, :].float(),
            'velocity': self.data['velocity'][idx, [0,1], :, :].float(),
            'pressure': self.data['pressure'][idx, :, :, :].float(),
            'dxyz': self.data['dxyz'][idx].float(),
            'permeability': self.data['permeability'][0] # there's a single permeability value
        }
        return sample




def get_loader(
    root_dir,
    augment=False,
    train_ratio=0.7,  # 70% train
    val_ratio=0.15,   # 15% validation
    test_ratio=0.15,  # 15% test
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    seed=2024,
    k_folds: int = None,
    use_3d: bool = False
) -> list[tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Load dataset with 70/15/15 train/val/test split.

    Args:
        root_dir: directory where data is stored.
        use_3d: whether to load 3D velocity data.
    
    Returns:
        List of tuples containing (train_loader, val_loader, test_loader).
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else seed

    # Dataset
    dataset = MicroFlowDataset(root_dir, augment=augment, use_3d=use_3d)

    # Split data: 70/15/15
    if k_folds is None:
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        lengths = [train_size, val_size, test_size]
        
        print(f"Dataset split: train={train_size}, val={val_size}, test={test_size} (total={len(dataset)})")

        train_set, val_set, test_set = random_split(
            dataset,
            lengths,
            generator=generator
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory
        )

        out = [(train_loader, val_loader, test_loader)]

    else:
        kf = KFold(
            n_splits=k_folds,
            shuffle=True,
            random_state=seed
        )

        out = []
        for i, (train_idx, test_idx) in enumerate(kf.split(dataset)):

            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(train_idx, generator=generator),
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory
            )

            val_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(test_idx, generator=generator),
                num_workers=num_workers,
                shuffle=False,
                pin_memory=pin_memory
            )
            # For k-fold CV, test_loader is same as val_loader (no separate test set)
            out.append(
                (train_loader, val_loader, val_loader)
            )

    return out


def load_VirtualPermeabilityBenchmark(
        folder: str # './Benchmark package/Stack of segmented images/'
    ) -> dict[str, torch.Tensor]:
    """
    Load micrograph data from the Virtual Permeability Benchmark hosted here
        https://doi.org/10.5281/zenodo.6611926

    and used in this paper:
        Syerko, Elena, et al.
        "Benchmark exercise on image-based permeability determination of engineering textiles: Microscale predictions."
        Composites Part A: Applied Science and Manufacturing 167 (2023): 107397.

    To get access to the data, please visit the link above and request access from the authors.

    Args:
        folder: The folder contains (.tif) images representing microstructure cross-sections obtained through an X-ray microscope.
    """
    VOXEL_SIZE = 0.521 * 1e-6 # 0.521 microns/voxel

    img_paths = os.listdir(folder)
    # sort paths
    img_paths = sorted(img_paths)
    # full paths
    img_paths = [osp.join(folder, _pth) for _pth in img_paths]

    """1. Microstructure images"""
    img_list = []
    for path in img_paths:

        im = Image.open(path)

        # convert to binary
        im = im.convert('1')

        im = np.array(im) # 2D array

        # invert so that there is 0 in fiber regions, and 1 elsewhere
        im = np.invert(im)

        # 4D tensor (batch, channels, height, width)
        img_tens = torch.from_numpy(im).unsqueeze(0).unsqueeze(0)
        img_list.append(img_tens)

    # concatenate
    microstructure = torch.cat(img_list, dim=0)


    """2. Microstructure dimensions"""

    dx = microstructure.shape[-1] * VOXEL_SIZE
    dy = microstructure.shape[-2] * VOXEL_SIZE
    dz = VOXEL_SIZE

    num_slices = microstructure.shape[0]
    dxyz = torch.tensor([[dx, dy, dz]]).expand(num_slices, -1)


    out = {
        'microstructure_original': microstructure.float(),
        'dxyz': dxyz
    }
    return out


def resize_image(
    img: torch.Tensor,
    target_height: int = 256
):
    """
    Resize image `img` to a height of `target_height`.
    
    Args:
        img: input image tensor, with shape (*, H, W).
        target_height: target height.
    """
    assert img.dim() > 2, "Input image must have more than 2 dimensions."
    
    # original image size
    orig_size = img.shape[-2:]
    orig_height, orig_width = orig_size

    factor = target_height / orig_height
    target_width = int(orig_width * factor)

    new_size = (target_height, target_width)

    # Resize images
    img = v2.Resize(
        size=new_size,
        antialias=True
    )(img)

    return img
