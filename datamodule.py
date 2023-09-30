import os
import glob
import numpy as np

from typing import Callable, Optional, Sequence

from argparse import ArgumentParser

import monai 
from monai.data import Dataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import first, set_determinism, get_seed, MAX_SEED
from monai.transforms import (
    # apply_transform, ensure_channel_first=True, 
    # AddChanneld,
    Compose, 
    OneOf, 
    HistogramNormalized,
    LoadImaged, 
    Spacingd, 
    Lambdad,
    Orientationd, 
    DivisiblePadd, 
    RandFlipd, 
    RandZoomd, 
    RandScaleCropd, 
    CropForegroundd,
    RandAffined,
    Resized, 
    Rotate90d, 
    ScaleIntensityd,
    ScaleIntensityRanged, 
    ToTensord,
)
from pytorch_lightning import LightningDataModule

class UnpairedDataset(Dataset, monai.transforms.Randomizable):
    def __init__(
        self,
        keys: Sequence, 
        data: Sequence, 
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None, 
        batch_size: int = 32, 
    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        
    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else: 
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.data):
            rand_idx = self.R.randint(0, len(dataset)) 
            data[key] = dataset[rand_idx]
        
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class CustomDataModule(LightningDataModule):
    def __init__(self, 
        train_image2d_folders: str = "path/to/folder", 
        val_image2d_folders: str = "path/to/folder", 
        test_image2d_folders: str = "path/to/dir", 
        shape: int = 256,
        batch_size: int = 32, 
        train_samples: int = 4000,
        val_samples: int = 800,
        test_samples: int = 800,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup() 
        self.train_image2d_folders = train_image2d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image2d_folders = test_image2d_folders
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_image2d_files = glob_files(folders=train_image2d_folders, extension='**/*.png')
        self.val_image2d_files = glob_files(folders=val_image2d_folders, extension='**/*.png')
        self.test_image2d_files = glob_files(folders=test_image2d_folders, extension='**/*.png')


    def setup(self, seed: int=42, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image2d"], ensure_channel_first=True),
                # AddChanneld(keys=["image2d"],),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                # CropForegroundd(keys=["image2d"], source_key="image", select_fn=(lambda x: x > 0), margin=0),
                # RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image2d"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.train_image2d_files], 
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image2d"], ensure_channel_first=True),
                # AddChanneld(keys=["image2d"],),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,), 
                # CropForegroundd(keys=["image2d"], source_key="image", select_fn=(lambda x: x > 0), margin=0),
                HistogramNormalized(keys=["image2d"], min=0.0, max=1.0,),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image2d"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.val_image2d_files], 
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
        )
        
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--shape", type=int, default=256, help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()
    # Create data module
    
    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_label2d_folders = [
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_image2d_folders = val_image2d_folders

    datamodule = CustomDataModule(
        train_image2d_folders = train_image2d_folders, 
        val_image2d_folders = val_image2d_folders, 
        test_image2d_folders = test_image2d_folders, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup(seed=hparams.seed)