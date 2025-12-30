import os
import os.path as osp
import json

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)
from torchvision.transforms.functional import hflip, vflip




class MicroFlowDataset(Dataset):

    """
    Dataset for steady-state velocity flow field in 2D microstructures.
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False
    ):
        self.root_dir = root_dir
        self.augment = augment

        self.data: dict[str, torch.Tensor] = {}

        self.process()


    def process(self):
        """Load datset."""

        meta_dict = {
            'microstructure': 'domain.pt',
            'velocity_input': 'U_2d.pt',  # 2D flow field (input)
            'velocity': 'U.pt',            # 3D flow field (target)
            'pressure': 'p.pt',
            'dxyz': 'dxyz.pt',
        }

        # Read data from x directory
        # For 3D data: shape is (samples, depth, channels, height, width)
        _data_x = {}
        for key, val in meta_dict.items():
            file_path = osp.join(self.root_dir, 'x', val)
            if osp.exists(file_path):
                dta = torch.load(file_path)
                _data_x[key] = dta

        # Check if y directory exists and has data
        y_dir_exists = osp.exists(osp.join(self.root_dir, 'y'))
        
        if y_dir_exists:
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

                # Concatenate if we successfully loaded y data
                if has_y_data:
                    for key in meta_dict.keys():
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
                print(f"Failed to load y direction data: {e}")
                self.data = _data_x
                print("Loaded simulations with flow in 'x' direction.")
        else:
            self.data = _data_x
            print("Loaded simulations with flow in 'x' direction.")

        # No augmentation for 3D data for now
        # if self.augment:
        #     # Flip operations would need to be updated for 3D
        #     pass
        
        # save statistics
        self._save_statistics()

    def __len__(self):
        num_data = self.data['microstructure'].shape[0]
        return num_data
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:

        # For 3D data with shape [samples, depth, channels, H, W]
        # Reshape to [samples, channels, depth, H, W] for Conv3d compatibility
        # Pad depth to 12 (divisible by 4 for two stride-2 downsamples)
        microstructure = self.data['microstructure'][idx].permute(1, 0, 2, 3).float()  # [C, D, H, W]
        
        # Input: 2D flow field (all 3 components, z is null)
        velocity_input = self.data['velocity_input'][idx].permute(1, 0, 2, 3).float()  # [3, D, H, W]
        
        # Target: 3D flow field (all 3 components)
        velocity_target = self.data['velocity'][idx].permute(1, 0, 2, 3).float()  # [3, D, H, W]
        
        pressure = self.data['pressure'][idx].permute(1, 0, 2, 3).float()  # [C, D, H, W]
        
        # Pad depth dimension from 11 to 12 by replicating the last slice
        import torch.nn.functional as F
        microstructure = F.pad(microstructure, (0, 0, 0, 0, 0, 1), mode='replicate')  # [C, 12, H, W]
        velocity_input = F.pad(velocity_input, (0, 0, 0, 0, 0, 1), mode='replicate')  # [3, 12, H, W]
        velocity_target = F.pad(velocity_target, (0, 0, 0, 0, 0, 1), mode='replicate')  # [3, 12, H, W]
        pressure = F.pad(pressure, (0, 0, 0, 0, 0, 1), mode='replicate')  # [C, 12, H, W]
        
        sample = {
            'microstructure': microstructure,
            'velocity_input': velocity_input,
            'velocity': velocity_target,
            'pressure': pressure,
            'dxyz': self.data['dxyz'][idx].float(),
            'permeability': self.data['permeability'][idx] if 'permeability' in self.data else torch.tensor(0.0)
        }
        return sample

    @staticmethod
    def load_dataset(folder: str):
        """
        Load dataset.
        
        `folder`: dataset folder.
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
        stats = {
            'U_2d': {
                'max': self.data.get('velocity_input', torch.tensor(0.0)).abs().max().item()
            },
            'U': {
                'max': self.data.get('velocity', torch.tensor(0.0)).abs().max().item()
            },
            'p': {
                'max': self.data.get('pressure', torch.tensor(0.0)).abs().max().item()
            },
            'dxyz': {
                'max': self.data.get('dxyz', torch.tensor(0.0)).abs().max().item()
            },
        }

        # save
        log_file = osp.join(self.root_dir, 'statistics.json')
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=0)

    @staticmethod
    def _rotate_y_field(x: torch.Tensor):
        """
        Rotate microstructure, velocity, and pressure fields
        for simulations with flow in the y-direction.
        For 3D data: (samples, depth, channels, height, width)
        """
        num_samples, num_depth, num_channels, _, _ = x.shape

        # rotate in the H-W plane for each depth slice
        x = torch.rot90(x, k=1, dims=(-2,-1))

        if num_channels != 1:
            # swap channel order for velocity
            x = x[:, :, [1, 0, 2], :, :]

        # change sign of new y-velocity
        if num_channels > 1:
            x[:, :, 1, :, :] = - x[:, :, 1, :, :]
            
        return x



class MicroFlowDatasetVAE(Dataset):
    """
    VAE dataset that treats U_2d and U as SEPARATE samples.
    This doubles the dataset size: each sample index maps to either a 2D or 3D flow field.
    """
    
    def __init__(
        self,
        root_dir: str,
        augment: bool = False
    ):
        self.root_dir = root_dir
        self.augment = augment
        self.data: dict[str, torch.Tensor] = {}
        
        self.process()
    
    def process(self):
        """Load dataset."""
        
        meta_dict = {
            'microstructure': 'domain.pt',
            'velocity_2d': 'U_2d.pt',
            'velocity_3d': 'U.pt',
            'pressure': 'p.pt',
            'dxyz': 'dxyz.pt',
        }
        
        # Read data from x directory
        _data_x = {}
        for key, val in meta_dict.items():
            file_path = osp.join(self.root_dir, 'x', val)
            if osp.exists(file_path):
                dta = torch.load(file_path)
                _data_x[key] = dta
        
        # Store both velocity fields separately
        self.data = _data_x
        self.num_samples_per_field = self.data['microstructure'].shape[0]
        
        print(f"Loaded {self.num_samples_per_field} samples from 'x' directory.")
        print(f"Total VAE samples (2D + 3D): {2 * self.num_samples_per_field}")
        
        # Save statistics
        self._save_statistics()
    
    def _save_statistics(self):
        """Save dataset statistics."""
        stats = {
            'U_2d': {
                'max': self.data.get('velocity_2d', torch.tensor(0.0)).abs().max().item()
            },
            'U': {
                'max': self.data.get('velocity_3d', torch.tensor(0.0)).abs().max().item()
            },
            'p': {
                'max': self.data.get('pressure', torch.tensor(0.0)).abs().max().item()
            },
            'dxyz': {
                'max': self.data.get('dxyz', torch.tensor(0.0)).abs().max().item()
            },
        }
        
        # Save
        log_file = osp.join(self.root_dir, 'statistics.json')
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=0)
    
    def __len__(self):
        # Return DOUBLE the number of samples (one for each U_2d, one for each U)
        return 2 * self.num_samples_per_field
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        import torch.nn.functional as F
        
        # First half: U_2d samples (indices 0 to num_samples-1)
        # Second half: U samples (indices num_samples to 2*num_samples-1)
        is_2d = idx < self.num_samples_per_field
        actual_idx = idx if is_2d else idx - self.num_samples_per_field
        
        # Get microstructure (same for both)
        microstructure = self.data['microstructure'][actual_idx].permute(1, 0, 2, 3).float()  # [C, D, H, W]
        
        # Get appropriate velocity field
        if is_2d:
            velocity = self.data['velocity_2d'][actual_idx].permute(1, 0, 2, 3).float()  # [3, D, H, W]
        else:
            velocity = self.data['velocity_3d'][actual_idx].permute(1, 0, 2, 3).float()  # [3, D, H, W]
        
        pressure = self.data['pressure'][actual_idx].permute(1, 0, 2, 3).float()  # [C, D, H, W]
        
        # Pad depth dimension from 11 to 12
        microstructure = F.pad(microstructure, (0, 0, 0, 0, 0, 1), mode='replicate')  # [C, 12, H, W]
        velocity = F.pad(velocity, (0, 0, 0, 0, 0, 1), mode='replicate')  # [3, 12, H, W]
        pressure = F.pad(pressure, (0, 0, 0, 0, 0, 1), mode='replicate')  # [C, 12, H, W]
        
        sample = {
            'microstructure': microstructure,
            'velocity': velocity,  # Either U_2d or U depending on idx
            'pressure': pressure,
            'dxyz': self.data['dxyz'][actual_idx].float(),
            'is_2d': torch.tensor(is_2d, dtype=torch.bool),  # Flag to identify which type
            'original_idx': torch.tensor(actual_idx)  # Original index in the base data
        }
        return sample


class MicroFlowDataset3D(MicroFlowDataset):

    """
    Dataset for steady-state velocity flow field in slices of a 3D microstructure.
    """

    def __getitem__(self, idx):

        img  = self.input[idx]
        U = self.target_U[idx]
        p = self.target_p[idx]

        dxdydz = self.dxyz[idx]

        # there's a single value,
        # representing the permeability of the 3D microstructure
        k = self.permeab[0]

        sample = {
            'microstructure': img,
            'velocity': U,
            'pressure': p,
            'dxyz': dxdydz,
            'permeability': k

        }
        if self.transform:
            sample = self.transform(sample)

        return sample



class DatasetTransform:
    """
    Normalize velocity, pressure, and dimension values in dataset.
    """

    def __init__(self, input_var: str | dict) -> None:


        if isinstance(input_var, str):
            # the input is directory of dataset,
            # compute statistics
            root_dir = input_var
            
            # velocity, pressure, and dimensions
            target_U: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'U.pt'))
            # target_U_y: torch.Tensor = torch.load(osp.join(root_dir, 'y', 'U.pt'))
            target_p: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'p.pt'))
            dxyz: torch.Tensor = torch.load(osp.join(root_dir, 'x', 'dxyz.pt'))

            # target_U = torch.cat((target_U_x, target_U_y[:, [1,0,2]]))
            self._max_U = target_U.abs().max().item()
            self._max_p = target_p.max().item()
            self._max_d = dxyz.max().item()

            # save statistics
            self._params = {
                'U': {'max': self._max_U},
                'p': {'max': self._max_p},
                'd': {'max': self._max_d},
            }

            # write
            log_file = osp.join(root_dir, 'statistics.json')
            with open(log_file, 'w') as f:
                json.dump(self._params, f, indent=0)

        elif isinstance(input_var, dict):
            
            self._params = input_var

            # Directly use statistics that have already been computed
            self._max_U = self._params['U']['max']
            self._max_p = self._params['p']['max']
            self._max_d = self._params['d']['max']

        print(f'Statistics: {self._params}')

    def __call__(
        self,
        data: dict[str, torch.Tensor]
    ):

        velocity = data['velocity']
        pressure = data['pressure']
        dxyz = data['dxyz']

        # transform
        velocity_new = self.transform_U(velocity)
        pressure_new = self.transform_p(pressure)
        dxyz_new = self.transform_d(dxyz)

        data['velocity'] = velocity_new
        data['pressure'] = pressure_new
        data['dxyz'] = dxyz_new

        return data

    def inverse_transform(
        self,
        data: dict[str, torch.Tensor]
    ):

        velocity = data['velocity']
        pressure = data['pressure']
        dxyz = data['dxyz']

        # inverse-transform
        velocity_new = self.inverse_transform_U(velocity)
        pressure_new = self.inverse_transform_p(pressure)
        dxyz_new = self.inverse_transform_d(dxyz)

        data['velocity'] = velocity_new
        data['pressure'] = pressure_new
        data['dxyz'] = dxyz_new

        return data

    def transform_U(self, data: torch.Tensor):
        """Transform velocity"""

        # transform
        data = data / self._max_U

        return data
    
    def transform_p(self, data: torch.Tensor):
        """Transform pressure"""

        # transform
        data = data / self._max_p

        return data
    
    def transform_d(self, data: torch.Tensor):
        """Transform dimension"""

        # transform
        data = data / self._max_d

        return data
    
    def inverse_transform_U(self, data: torch.Tensor):
        """Inverse-transform velocity"""

        # inverse-transform
        data = data * self._max_U

        return data
    
    def inverse_transform_p(self, data: torch.Tensor):
        """Inverse-transform pressure"""

        # inverse-transform
        data = data * self._max_p

        return data

    def inverse_transform_d(self, data: torch.Tensor):
        """Inverse-transform dimension"""

        # inverse-transform
        data = data * self._max_d

        return data



def get_loader(
    root_dir,
    augment=False,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32,
    num_workers=0,  # Changed default to 0 for Windows compatibility
    shuffle=True,
    pin_memory=False,  # Disabled to save memory
    seed=2024,
    use_vae_dataset=True  # New parameter to use VAE dataset that separates 2D and 3D
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load dataset with stratified train/val/test split.

    `root_dir`: directory where data is stored.
    `use_vae_dataset`: If True, uses MicroFlowDatasetVAE which treats U_2d and U as separate samples.
    Returns: (train_loader, val_loader, test_loader)
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else seed

    # Dataset
    if use_vae_dataset:
        dataset = MicroFlowDatasetVAE(
            root_dir,
            augment=augment
        )
    else:
        dataset = MicroFlowDataset(
            root_dir,
            augment=augment
        )

    # Paired split for VAE dataset: keep 2D and 3D from same microstructure together
    if use_vae_dataset and hasattr(dataset, 'num_samples_per_field'):
        num_per_field = dataset.num_samples_per_field
        
        # Split base microstructure indices (each microstructure has both 2D and 3D)
        train_size_base = int(train_ratio * num_per_field)
        val_size_base = int(val_ratio * num_per_field)
        test_size_base = num_per_field - train_size_base - val_size_base
        
        # Shuffle base indices
        import random
        base_indices = list(range(num_per_field))
        rng = random.Random(seed)
        rng.shuffle(base_indices)
        
        # Split base indices
        train_base = base_indices[:train_size_base]
        val_base = base_indices[train_size_base:train_size_base + val_size_base]
        test_base = base_indices[train_size_base + val_size_base:]
        
        # Create paired indices (2D at idx i, 3D at idx i+num_per_field)
        train_indices = train_base + [i + num_per_field for i in train_base]
        val_indices = val_base + [i + num_per_field for i in val_base]
        test_indices = test_base + [i + num_per_field for i in test_base]
        
        print(f"Paired split: train={len(train_indices)} ({train_size_base} microstructures × 2 flow types)")
        print(f"              val={len(val_indices)} ({val_size_base} microstructures × 2 flow types)")
        print(f"              test={len(test_indices)} ({test_size_base} microstructures × 2 flow types)")
        
        # Create Subset datasets
        from torch.utils.data import Subset
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)
    else:
        # Non-stratified split for regular dataset
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        lengths = [train_size, val_size, test_size]
        print(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")

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
        shuffle=shuffle,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Don't shuffle test set
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

