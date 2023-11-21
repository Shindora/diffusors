import os
import glob

from typing import Optional, Union, List, Dict, Sequence, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import wandb

from argparse import ArgumentParser

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import first, set_determinism, get_seed, MAX_SEED
from monai.transforms import (
    apply_transform,
    Compose,
    LoadImaged,
    DivisiblePadd,
    RandFlipd,
    Resized,
    HistogramNormalized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ToTensord,
)

# from data import CustomDataModule
# from cdiff import *
from diffusers import UNet2DModel, DDPMScheduler
from loss_function.dice_loss import dice_coef_loss


class PairedAndUnsupervisedDataset(monai.data.Dataset, monai.transforms.Randomizable):
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
        # for key, dataset in zip(self.keys, self.data):
        #     rand_idx = self.R.randint(0, len(dataset))
        #     data[key] = dataset[rand_idx]
        rand_idx = self.R.randint(0, len(self.data[0]))
        data[self.keys[0]] = self.data[0][rand_idx]  # image
        data[self.keys[1]] = self.data[1][rand_idx]  # label
        rand_idy = self.R.randint(0, len(self.data[2]))
        data[self.keys[2]] = self.data[2][rand_idy]  # unsup

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data


class PairedAndUnsupervisedDataModule(LightningDataModule):
    def __init__(
            self,
            train_image_dirs: str = "path/to/dir",
            train_label_dirs: str = "path/to/dir",
            train_unsup_dirs: str = "path/to/dir",
            val_image_dirs: str = "path/to/dir",
            val_label_dirs: str = "path/to/dir",
            val_unsup_dirs: str = "path/to/dir",
            test_image_dirs: str = "path/to/dir",
            test_label_dirs: str = "path/to/dir",
            test_unsup_dirs: str = "path/to/dir",
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
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.train_unsup_dirs = train_unsup_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.val_unsup_dirs = val_unsup_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs
        self.test_unsup_dirs = test_unsup_dirs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str = None, extension: str = "*.nii.gz"):
            assert folders is not None
            paths = [
                glob.glob(os.path.join(folder, extension), recursive=True)
                for folder in folders
            ]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.train_image_files = glob_files(
            folders=train_image_dirs, extension="**/*.png"
        )
        self.train_label_files = glob_files(
            folders=train_label_dirs, extension="**/*.png"
        )
        self.train_unsup_files = glob_files(
            folders=train_unsup_dirs, extension="**/*.png"
        )
        self.val_image_files = glob_files(folders=val_image_dirs, extension="**/*.png")
        self.val_label_files = glob_files(folders=val_label_dirs, extension="**/*.png")
        self.val_unsup_files = glob_files(folders=val_unsup_dirs, extension="**/*.png")
        self.test_image_files = glob_files(
            folders=test_image_dirs, extension="**/*.png"
        )
        self.test_label_files = glob_files(
            folders=test_label_dirs, extension="**/*.png"
        )
        self.test_unsup_files = glob_files(
            folders=test_unsup_dirs, extension="**/*.png"
        )

    def setup(self, seed: int = 42, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"], ensure_channel_first=True),
                # AddChanneld(keys=["image", "label", "unsup"],),
                ScaleIntensityRanged(
                    keys=["label"], a_min=0, a_max=128, b_min=0, b_max=1, clip=True
                ),
                ScaleIntensityd(
                    keys=["image", "label", "unsup"],
                    minv=0.0,
                    maxv=1.0,
                ),
                # CropForegroundd(keys=["image", "label", "unsup"], source_key="image", select_fn=(lambda x: x>0), margin=0),
                HistogramNormalized(
                    keys=["image", "unsup"],
                    min=0.0,
                    max=1.0,
                ),
                # RandZoomd(keys=["image", "label", "unsup"], prob=1.0, min_zoom=0.9, max_zoom=1.1, padding_mode='constant', mode=["area", "nearest", "area"]),
                RandFlipd(keys=["image", "label", "unsup"], prob=0.5, spatial_axis=0),
                # RandAffined(keys=["image", "label", "unsup"], prob=1.0, rotate_range=0.1, translate_range=10, scale_range=0.01, padding_mode='zeros', mode=["bilinear", "nearest", "bilinear"]),
                Resized(
                    keys=["image", "label", "unsup"],
                    spatial_size=256,
                    size_mode="longest",
                    mode=["area", "nearest", "area"],
                ),
                DivisiblePadd(
                    keys=["image", "label", "unsup"],
                    k=256,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensord(
                    keys=["image", "label", "unsup"],
                ),
            ]
        )

        self.train_datasets = PairedAndUnsupervisedDataset(
            keys=["image", "label", "unsup"],
            data=[
                self.train_image_files,
                self.train_label_files,
                self.train_unsup_files,
            ],
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=True,
            persistent_workers=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"], ensure_channel_first=True),
                # AddChanneld(keys=["image", "label", "unsup"],),
                ScaleIntensityRanged(
                    keys=["label"], a_min=0, a_max=128, b_min=0, b_max=1, clip=True
                ),
                ScaleIntensityd(
                    keys=["image", "label", "unsup"],
                    minv=0.0,
                    maxv=1.0,
                ),
                # CropForegroundd(keys=["image", "label", "unsup"], source_key="image", select_fn=(lambda x: x>0), margin=0),
                HistogramNormalized(
                    keys=["image", "unsup"],
                    min=0.0,
                    max=1.0,
                ),
                Resized(
                    keys=["image", "label", "unsup"],
                    spatial_size=256,
                    size_mode="longest",
                    mode=["area", "nearest", "area"],
                ),
                DivisiblePadd(
                    keys=["image", "label", "unsup"],
                    k=256,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensord(
                    keys=["image", "label", "unsup"],
                ),
            ]
        )

        self.val_datasets = PairedAndUnsupervisedDataset(
            keys=["image", "label", "unsup"],
            data=[self.val_image_files, self.val_label_files, self.val_unsup_files],
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
        )

        self.val_loader = DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=True,
            persistent_workers=True,
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"], ensure_channel_first=True),
                # AddChanneld(keys=["image", "label", "unsup"],),
                ScaleIntensityRanged(
                    keys=["label"], a_min=0, a_max=128, b_min=0, b_max=1, clip=False
                ),
                ScaleIntensityd(
                    keys=["image", "label", "unsup"],
                    minv=0.0,
                    maxv=1.0,
                ),
                # CropForegroundd(keys=["image", "label", "unsup"], source_key="image", select_fn=(lambda x: x>0), margin=0),
                HistogramNormalized(
                    keys=["image", "unsup"],
                    min=0.0,
                    max=1.0,
                ),
                Resized(
                    keys=["image", "label", "unsup"],
                    spatial_size=256,
                    size_mode="longest",
                    mode=["area", "nearest", "area"],
                ),
                DivisiblePadd(
                    keys=["image", "label", "unsup"],
                    k=256,
                    mode="constant",
                    constant_values=0,
                ),
                ToTensord(
                    keys=["image", "label", "unsup"],
                ),
            ]
        )

        self.test_datasets = PairedAndUnsupervisedDataset(
            keys=["image", "label", "unsup"],
            data=[self.test_image_files, self.test_label_files, self.test_unsup_files],
            transform=self.test_transforms,
            length=self.test_samples,
            batch_size=self.batch_size,
        )

        self.test_loader = DataLoader(
            self.test_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=False,
            persistent_workers=True,
        )
        return self.test_loader


class DDMMLightningModule(LightningModule):
    def __init__(self, hparams, *kwargs) -> None:
        super().__init__()
        self.lr = hparams.lr
        self.epochs = hparams.epochs
        self.weight_decay = hparams.weight_decay
        self.num_timesteps = hparams.timesteps
        self.batch_size = hparams.batch_size
        self.shape = hparams.shape
        self.is_use_cycle = hparams.is_use_cycle

        self.num_classes = 2
        self.timesteps = hparams.timesteps

        # Create a scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.timesteps, beta_schedule="squaredcos_cap_v2"
        )

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.diffusion_image = UNet2DModel(
            sample_size=self.shape,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                128,
                128,
                256,
                256,
                512,
                512,
            ),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.diffusion_label = UNet2DModel(
            sample_size=self.shape,  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(
                128,
                128,
                256,
                256,
                512,
                512,
            ),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        if self.is_use_cycle:
            self.diffusion_from_image_to_label = UNet2DModel(
                sample_size=self.shape,  # the target image resolution
                in_channels=1,  # the number of input channels, 3 for RGB images
                out_channels=1,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(
                    128,
                    128,
                    256,
                    256,
                    512,
                    512,
                ),  # the number of output channes for each UNet block
                down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

            self.diffusion_from_label_to_image = UNet2DModel(
                sample_size=self.shape,  # the target image resolution
                in_channels=1,  # the number of input channels, 3 for RGB images
                out_channels=1,  # the number of output channels
                layers_per_block=2,  # how many ResNet layers to use per UNet block
                block_out_channels=(
                    128,
                    128,
                    256,
                    256,
                    512,
                    512,
                ),  # the number of output channes for each UNet block
                down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )

        self.l1_loss = nn.SmoothL1Loss(reduction="mean", beta=0.02)
        self.dice_loss = dice_coef_loss
        self.save_hyperparameters()

    def _common_step(
            self, batch, batch_idx, optimizer_idx, stage: Optional[str] = "common"
    ):
        image, label, unsup = batch["image"], batch["label"], batch["unsup"]
        _device = image.device

        rng_p = torch.randn_like(image)
        rng_u = torch.randn_like(unsup)

        bs = image.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bs,), device=_device
        ).long()

        # 1st pass, supervised
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        mid_i = self.noise_scheduler.add_noise(image * 2.0 - 1.0, rng_p, timesteps)
        mid_l = self.noise_scheduler.add_noise(label * 2.0 - 1.0, rng_p, timesteps)

        est_i = self.diffusion_image.forward(mid_i, timesteps).sample
        est_l = self.diffusion_label.forward(mid_l, timesteps).sample

        super_loss = (
                self.l1_loss(est_i, rng_p)
                + self.l1_loss(est_l, rng_p)
                + self.l1_loss(est_i, image)
                + self.l1_loss(est_l, label)
        )

        if self.is_use_cycle:
            pred_label = self.diffusion_from_image_to_label.forward(mid_i, torch.zeros_like(timesteps)).sample
            pred_image = self.diffusion_from_label_to_image.forward(mid_l, torch.zeros_like(timesteps)).sample
            super_loss += (
                    self.l1_loss(pred_image, mid_i)
                    + self.l1_loss(pred_label, mid_l)
                    + self.l1_loss(pred_image, image)
                    + self.l1_loss(pred_label, label)

            )

        # 2nd pass, unsupervised
        mid_u = self.noise_scheduler.add_noise(unsup * 2.0 - 1.0, rng_u, timesteps)
        est_u = self.diffusion_image.forward(mid_u, timesteps).sample
        unsup_loss = (
                self.l1_loss(est_u, rng_u)
                + self.l1_loss(est_u, unsup)
        )

        self.log(
            f"{stage}_super_loss",
            super_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            f"{stage}_unsup_loss",
            unsup_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        loss = super_loss + unsup_loss

        if batch_idx == 0:
            with torch.no_grad():
                rng = torch.randn_like(image)
                sam_i = rng.clone().detach()
                sam_l = rng.clone().detach()
                for i, t in enumerate(self.noise_scheduler.timesteps):
                    res_i = self.diffusion_image.forward(sam_i, t).sample
                    res_l = self.diffusion_label.forward(sam_l, t).sample

                    if self.is_use_cycle:
                        cycle_i = self.diffusion_from_label_to_image(sam_l, t).sample
                        cycle_l = self.diffusion_from_image_to_label(sam_i, t).sample

                    # Update sample with step
                    res_i = res_i.to(device=sam_i.device)
                    res_l = res_l.to(device=sam_l.device)
                    sam_i = self.noise_scheduler.step(res_i, t, sam_i).prev_sample
                    sam_l = self.noise_scheduler.step(res_l, t, sam_l).prev_sample

                sam_i = sam_i * 0.5 + 0.5
                sam_l = sam_l * 0.5 + 0.5

            if self.is_use_cycle:
                viz2d = torch.cat(
                    [image, label, sam_i, sam_l, cycle_i, cycle_l, unsup], dim=-1
                ).transpose(2, 3)
            else:
                viz2d = torch.cat(
                    [image, label, sam_i, sam_l, unsup], dim=-1
                ).transpose(2, 3)
            grid = torchvision.utils.make_grid(
                viz2d, normalize=False, scale_each=False, nrow=8, padding=0
            )

            # Convert the PyTorch tensor to a PIL Image
            grid_image = torchvision.transforms.ToPILImage()(grid.clamp(0.0, 1.0))
            wandb_log = self.logger.experiment
            wandb_log.log(
                {f"{stage}__samples": [wandb.Image(grid_image)]},
                step=self.global_step // 10,
            )

        info = {f"loss": loss}
        return info

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage="validation")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage="test")

    def _common_epoch_end(self, outputs, stage: Optional[str] = "common"):
        loss = torch.stack([x[f"loss"] for x in outputs]).mean()
        self.log(
            f"{stage}_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage="train")

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage="validation")

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100, help="timesteps")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--shape", type=int, default=256, help="spatial size of the tensor"
    )
    parser.add_argument(
        "--train_samples", type=int, default=40000, help="training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=8000, help="validation samples"
    )
    parser.add_argument("--test_samples", type=int, default=4000, help="test samples")

    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")

    parser.add_argument("--epochs", type=int, default=31, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--is_use_cycle", type=bool, default=True, help="Use cycle prediction")

    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="accelerator instances"
    )
    parser.add_argument("--devices", type=str, default="auto", help="number of devices")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="Strategy controls the model distribution across training",
    )
    parser.add_argument("--precision", type=int, default=32)

    # parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()
    # Create data module

    train_image_dirs = [
        os.path.join(hparams.datadir, "data/JSRT/processed/images/"),
        os.path.join(hparams.datadir, "data/ChinaSet/processed/images/"),
        os.path.join(hparams.datadir, "data/Montgomery/processed/images/"),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'),
    ]
    train_label_dirs = [
        os.path.join(hparams.datadir, "data/JSRT/processed/labels/"),
        os.path.join(hparams.datadir, "data/ChinaSet/processed/labels/"),
        os.path.join(hparams.datadir, "data/Montgomery/processed/labels/"),
    ]

    train_unsup_dirs = [
        os.path.join(hparams.datadir, "data/VinDR/train/"),
    ]

    val_image_dirs = [
        os.path.join(hparams.datadir, "data/JSRT/processed/images/"),
        os.path.join(hparams.datadir, "data/ChinaSet/processed/images/"),
        os.path.join(hparams.datadir, "data/Montgomery/processed/images/"),
    ]

    val_label_dirs = [
        os.path.join(hparams.datadir, "data/JSRT/processed/labels/"),
        os.path.join(hparams.datadir, "data/ChinaSet/processed/labels/"),
        os.path.join(hparams.datadir, "data/Montgomery/processed/labels/"),
    ]

    val_unsup_dirs = [
        os.path.join(hparams.datadir, "data/VinDR/test/"),
    ]
    test_image_dirs = val_image_dirs
    test_label_dirs = val_label_dirs
    test_unsup_dirs = val_unsup_dirs

    datamodule = PairedAndUnsupervisedDataModule(
        train_image_dirs=train_image_dirs,
        train_label_dirs=train_label_dirs,
        train_unsup_dirs=train_unsup_dirs,
        val_image_dirs=val_image_dirs,
        val_label_dirs=val_label_dirs,
        val_unsup_dirs=val_unsup_dirs,
        test_image_dirs=test_image_dirs,
        test_label_dirs=test_label_dirs,
        test_unsup_dirs=test_unsup_dirs,
        train_samples=hparams.train_samples,
        val_samples=hparams.val_samples,
        test_samples=hparams.test_samples,
        batch_size=hparams.batch_size,
        shape=hparams.shape,
        # keys = ["image", "label", "unsup"]
    )

    datamodule.setup(seed=hparams.seed)

    # debug_data = first(datamodule.val_dataloader())
    # image, label, unsup = debug_data["image"], \
    #                       debug_data["label"], \
    #                       debug_data["unsup"]
    # print(image.shape, label.shape, unsup.shape)

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DDMMLightningModule(hparams=hparams)

    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename="{epoch:02d}-{validation_loss_epoch:.2f}",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="validation_loss_epoch",  # The quantity to be monitored
        min_delta=0.00,  # Minimum change in the monitored quantity to qualify as an improvement
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,  # Whether to print logs in stdout
        mode="min",  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
    )
    # Logger
    wandb.init(project="cycle-consistent-DDMM", entity="diffusors", dir=hparams.logsdir)
    wandb_logger = WandbLogger(
        save_dir=hparams.logsdir, log_model=True, project="diffusor"
    )
    # Init model with callbacks
    trainer = Trainer(
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        max_epochs=hparams.epochs,
        logger=[wandb_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
            early_stop_callback
        ],
        # accumulate_grad_batches=4,
        # strategy=hparams.strategy, #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=hparams.precision,  # if hparams.use_amp else 32,
        # amp_backend='apex',
        # amp_level='O1', # see https://nvidia.github.io/apex/amp.html#opt-levels
        # stochastic_weight_avg=True,
        auto_scale_batch_size=True,
        # gradient_clip_val=5,
        # gradient_clip_algorithm='norm', #'norm', #'value'
        # track_grad_norm=2,
        # detect_anomaly=True,
        # benchmark=None,
        # deterministic=False,
        # profiler="simple",
    )

    trainer.fit(
        model,
        datamodule,  # ,
        ckpt_path=hparams.ckpt
        if hparams.ckpt is not None
        else None,  # "some/path/to/my_checkpoint.ckpt"
    )

    # test
    trainer.test(
        model, datamodule, ckpt_path=hparams.ckpt if hparams.ckpt is not None else None
    )

    # serve
