import os
import glob

from typing import Optional, Union, List, Dict, Sequence, Callable
import torch
import torch.nn.functional as F
import wandb

# Finish the current wandb run if any
wandb.finish()
wandb.login()

import torchvision

from argparse import ArgumentParser

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import first, set_determinism, get_seed, MAX_SEED
from monai.transforms import (
    apply_transform,
    Randomizable,
    AddChanneld,
    Compose,
    OneOf,
    LoadImaged,
    Spacingd,
    Orientationd,
    DivisiblePadd,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandScaleCropd,
    CropForegroundd,
    Resized,
    Rotate90d,
    HistogramNormalized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ToTensord,
)

# from data import CustomDataModule
from model import *


def select_fn(x):
    return x > 0


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
                LoadImaged(keys=["image", "label", "unsup"]),
                AddChanneld(
                    keys=["image", "label", "unsup"],
                ),
                HistogramNormalized(
                    keys=["image", "unsup"],
                    min=0.0,
                    max=1.0,
                ),
                CropForegroundd(
                    keys=["image", "label", "unsup"],
                    source_key="image",
                    select_fn=select_fn,
                    margin=0,
                ),
                ScaleIntensityd(
                    keys=["image", "label", "unsup"],
                    minv=0.0,
                    maxv=1.0,
                ),
                # RandZoomd(keys=["image", "label", "unsup"], prob=1.0, min_zoom=0.9, max_zoom=1.1, padding_mode='constant', mode=["area", "nearest", "area"]),
                RandAffined(
                    keys=["image", "label", "unsup"],
                    prob=1.0,
                    rotate_range=0.1,
                    translate_range=10,
                    scale_range=0.1,
                    padding_mode="zeros",
                    mode=["bilinear", "nearest", "bilinear"],
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
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"]),
                AddChanneld(
                    keys=["image", "label", "unsup"],
                ),
                HistogramNormalized(
                    keys=["image", "unsup"],
                    min=0.0,
                    max=1.0,
                ),
                CropForegroundd(
                    keys=["image", "label", "unsup"],
                    source_key="image",
                    select_fn=select_fn,
                    margin=0,
                ),
                ScaleIntensityd(
                    keys=["image", "label", "unsup"],
                    minv=0.0,
                    maxv=1.0,
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
        )
        return self.val_loader


class DDMMLightningModule(LightningModule):
    def __init__(self, hparams, *kwargs) -> None:
        super().__init__()
        self.lr = hparams.lr
        self.epochs = hparams.epochs
        self.weight_decay = hparams.weight_decay
        self.num_timesteps = hparams.timesteps
        self.batch_size = hparams.batch_size
        model_image = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )

        model_label = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )

        self.diffusion_image = GaussianDiffusion(
            model_image,
            image_size=hparams.shape,
            timesteps=hparams.timesteps,  # number of steps
            loss_type="L1",  # L1 or L2 or smooth L1,
            objective="pred_x0",
        )

        self.diffusion_label = GaussianDiffusion(
            model_label,
            image_size=hparams.shape,
            timesteps=hparams.timesteps,  # number of steps
            loss_type="L1",  # L1 or L2 or smooth L1
            objective="pred_noise",
        )
        self.val_outputs = []
        self.train_outputs = []
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        # return torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizers = [
            torch.optim.RAdam(
                [{"params": self.diffusion_image.parameters()}],
                lr=1e0 * (self.lr or self.learning_rate),
            ),
            torch.optim.RAdam(
                [{"params": self.diffusion_label.parameters()}],
                lr=1e0 * (self.lr or self.learning_rate),
            ),
        ]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optimizers[0],
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.epochs,
            ),
            torch.optim.lr_scheduler.LinearLR(
                optimizers[1],
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.epochs,
            ),
        ]

        return optimizers, schedulers

    def _common_step(
        self, batch, batch_idx, optimizer_idx, stage: Optional[str] = "common"
    ):
        image, label, unsup = batch["image"], batch["label"], batch["unsup"]

        noise_p = torch.randn_like(image).to(torch.float32)
        noise_u = torch.randn_like(unsup).to(torch.float32)
        t_p = (
            torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device)
            .long()
            .to(torch.float32)
        )
        t_u = (
            torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device)
            .long()
            .to(torch.float32)
        )
        loss_image = self.diffusion_image.forward(
            torch.cat([image, unsup], dim=0),
            torch.cat([t_p, t_u], dim=0),
            torch.cat([noise_p, noise_u], dim=0),
        )
        # loss_label = self.diffusion_label.forward(torch.cat([label, label], dim=0),
        #                                           torch.cat([t_p, t_p], dim=0),
        #                                           torch.cat([noise_p, noise_p], dim=0)) #(label, t_p, noise_p)
        loss_label = self.diffusion_label.forward(label, t_p, noise_p)
        if batch_idx == 0:
            noise_samples = torch.randn_like(unsup)
            image_samples = self.diffusion_image.sample(
                batch_size=self.batch_size, img=noise_samples
            )
            label_samples = self.diffusion_label.sample(
                batch_size=self.batch_size, img=noise_samples
            )
            viz2d = torch.cat(
                [image, label, image_samples, label_samples], dim=-1
            ).transpose(2, 3)
            grid = torchvision.utils.make_grid(
                viz2d, normalize=False, scale_each=False, nrow=8, padding=0
            )
            wandb_log = self.logger.experiment
            grid_permuted = grid.permute(1, 2, 0)
            wandb_log.log({f"{stage}_samples": wandb.Image(grid_permuted.cpu().clamp(0.0, 1.0))})

        # loss = loss_image + loss_label
        # info = {"loss": loss}

        if optimizer_idx == 0:  # forward picture
            info = {f"loss": loss_image}
        elif optimizer_idx == 1:  # forward density
            info = {f"loss": loss_label}
        else:
            info = {f"loss": loss_image + loss_label}
        return info

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        for opt in optimizers:
            opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        for opt in optimizers:
            opt.step()


    def validation_step(self, batch, batch_idx):
        output = self._common_step(
            batch, batch_idx, optimizer_idx=-1, stage="validation"
        )
        self.val_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=-1, stage="test")

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

    def on_train_epoch_end(self):
        self._common_epoch_end(self.train_outputs, stage="train")
        self.train_outputs = []

    def on_validation_epoch_end(self):
        self._common_epoch_end(self.val_outputs, stage="validation")
        self.val_outputs = []

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage="test")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100, help="timesteps")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--shape", type=int, default=256, help="spatial size of the tensor"
    )
    parser.add_argument(
        "--train_samples", type=int, default=4000, help="training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=800, help="validation samples"
    )
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")

    parser.add_argument("--logsdir", type=str, default="logs", help="logging directory")
    parser.add_argument("--datadir", type=str, default="data", help="data directory")

    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--accelerator", type=str, default="mps", help="accelerator instances"
    )
    parser.add_argument("--devices", type=str, default="auto", help="number of devices")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="Strategy controls the model distribution across training",
    )

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
    model = (
        model.load_from_checkpoint(hparams.ckpt, strict=False)
        if hparams.ckpt is not None
        else model
    )

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename="{epoch:02d}-{validation_loss_epoch:.2f}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    # Logger
    # Initialize wandb
    wandb.init(project="cycle-consistent-DDMM", entity="diffusors")
    wandb_logger = WandbLogger(
        save_dir=hparams.logsdir, log_model="all", project="diffusor"
    )

    # Init model with callbacks
    trainer = Trainer(
        devices=hparams.devices,
        accelerator=hparams.accelerator,
        max_epochs=hparams.epochs,
        logger=[wandb_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
        ],
        # accumulate_grad_batches=4,
        # strategy=hparams.strategy,  # "fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=32,  # if hparams.use_amp else 32,
        # amp_backend='apex',
        # amp_level='O1', # see https://nvidia.github.io/apex/amp.html#opt-levels
        # stochastic_weight_avg=True,
        # auto_scale_batch_size=True,
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
        datamodule,
    )

    # test

    # serve
