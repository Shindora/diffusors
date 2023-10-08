import os
import glob

from typing import Optional, Union, List, Dict, Sequence, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from argparse import ArgumentParser
from tqdm import tqdm
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import monai 
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import pad_list_data_collate, decollate_batch
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
    Resized, Rotate90d, HistogramNormalized,
    ScaleIntensityd,
    ScaleIntensityRanged, 
    ToTensord,
)

class PairedAndUnpairedDataset(Dataset, Randomizable):
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
        
        rand_idx = self.R.randint(0, len(self.data[0])) 
        data[self.keys[0]] = self.data[0][rand_idx] # image
        data[self.keys[1]] = self.data[1][rand_idx] # label
        
        rand_idy = self.R.randint(0, len(self.data[2])) 
        data[self.keys[2]] = self.data[2][rand_idy] # unsup
        rand_idz = self.R.randint(0, len(self.data[3]))
        data[self.keys[3]] = self.data[3][rand_idz] # unsup

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class PairedAndUnpairedDataModule(LightningDataModule):
    def __init__(self, 
        train_ssource_dirs: List[str] = ["path/to/dir"], 
        train_starget_dirs: List[str] = ["path/to/dir"], 
        train_usource_dirs: List[str] = ["path/to/dir"], 
        train_utarget_dirs: List[str] = ["path/to/dir"], 
        val_ssource_dirs: List[str] = ["path/to/dir"], 
        val_starget_dirs: List[str] = ["path/to/dir"], 
        val_usource_dirs: List[str] = ["path/to/dir"], 
        val_utarget_dirs: List[str] = ["path/to/dir"], 
        test_ssource_dirs: List[str] = ["path/to/dir"], 
        test_starget_dirs: List[str] = ["path/to/dir"], 
        test_usource_dirs: List[str] = ["path/to/dir"], 
        test_utarget_dirs: List[str] = ["path/to/dir"], 
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
        self.train_ssource_dirs = train_ssource_dirs
        self.train_starget_dirs = train_starget_dirs
        self.train_usource_dirs = train_usource_dirs
        self.train_utarget_dirs = train_utarget_dirs
        self.val_ssource_dirs = val_ssource_dirs
        self.val_starget_dirs = val_starget_dirs
        self.val_usource_dirs = val_usource_dirs
        self.val_utarget_dirs = val_utarget_dirs
        self.test_ssource_dirs = test_ssource_dirs
        self.test_starget_dirs = test_starget_dirs
        self.test_usource_dirs = test_usource_dirs
        self.test_utarget_dirs = test_utarget_dirs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: List[str]=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_ssource_files = glob_files(folders=train_ssource_dirs, extension='**/*.png')
        self.train_starget_files = glob_files(folders=train_starget_dirs, extension='**/*.png')
        self.train_usource_files = glob_files(folders=train_usource_dirs, extension='**/*.png')
        self.train_utarget_files = glob_files(folders=train_utarget_dirs, extension='**/*.png')

        self.val_ssource_files = glob_files(folders=val_ssource_dirs, extension='**/*.png')
        self.val_starget_files = glob_files(folders=val_starget_dirs, extension='**/*.png')
        self.val_usource_files = glob_files(folders=val_usource_dirs, extension='**/*.png')
        self.val_utarget_files = glob_files(folders=val_utarget_dirs, extension='**/*.png')
        
        self.test_ssource_files = glob_files(folders=test_ssource_dirs, extension='**/*.png')
        self.test_starget_files = glob_files(folders=test_starget_dirs, extension='**/*.png')
        self.test_usource_files = glob_files(folders=test_usource_dirs, extension='**/*.png')
        self.test_utarget_files = glob_files(folders=test_utarget_dirs, extension='**/*.png')

    def setup(self, seed: int=42, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["source", "target", "images", "labels"], ensure_channel_first=True),
                # AddChanneld(keys=["source", "target", "images", "labels"]),
                ScaleIntensityRanged(keys=["target", "labels"], a_min=0, a_max=128, b_min=0, b_max=1, clip=True),
                ScaleIntensityd(keys=["source", "target", "images", "labels"], minv=0.0, maxv=1.0,),
                HistogramNormalized(keys=["source", "images"], min=0.0, max=1.0,), # type: ignore
                RandFlipd(keys=["source", "target", "images", "labels"], prob=0.5, spatial_axis=0),
                Resized(keys=["source", "target", "images", "labels"], spatial_size=256, size_mode="longest", mode=["area", "nearest", "area", "nearest"]),
                DivisiblePadd(keys=["source", "target", "images", "labels"], k=256, mode="constant", constant_values=0.0),
                ToTensord(keys=["source", "target", "images", "labels"],),
            ]
        )

        self.train_datasets = PairedAndUnpairedDataset(
            keys=["source", "target", "images", "labels"],
            data=[self.train_ssource_files, self.train_starget_files, self.train_usource_files, self.train_utarget_files],
            transform=self.train_transforms,
            length=self.train_samples, # type: ignore
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=pad_list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["source", "target", "images", "labels"], ensure_channel_first=True),
                # AddChanneld(keys=["source", "target", "images", "labels"]),
                ScaleIntensityRanged(keys=["target", "labels"], a_min=0, a_max=128, b_min=0, b_max=1, clip=True),
                ScaleIntensityd(keys=["source", "target", "images", "labels"], minv=0.0, maxv=1.0,),
                HistogramNormalized(keys=["source", "images"], min=0.0, max=1.0,),  # type: ignore
                RandFlipd(keys=["source", "target", "images", "labels"], prob=0.5, spatial_axis=0),
                Resized(keys=["source", "target", "images", "labels"], spatial_size=256, size_mode="longest", mode=["area", "nearest", "area", "nearest"]),
                DivisiblePadd(keys=["source", "target", "images", "labels"], k=256, mode="constant", constant_values=0.0),
                ToTensord(keys=["source", "target", "images", "labels"],),
            ]
        )

        self.val_datasets = PairedAndUnpairedDataset(
            keys=["source", "target", "images", "labels"],
            data=[self.val_ssource_files, self.val_starget_files, self.val_usource_files, self.val_utarget_files],
            transform=self.val_transforms,
            length=self.val_samples,  # type: ignore
            batch_size=self.batch_size,
        )
        
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=pad_list_data_collate,
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
        self.shape = hparams.shape
        self.num_classes = 2
        self.timesteps = hparams.timesteps

        self.noise2space = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )
        
        self.image2label = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )
        
        self.label2image = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 128, 256, 512),
            attention_levels=(False, False, True, True),
            num_res_blocks=1,
            num_head_channels=256,
        )

        self.scheduler = DDPMScheduler(num_train_timesteps=hparams.timesteps)
        self.inferer = DiffusionInferer(self.scheduler)
        self.loss_func = nn.L1Loss()
        
    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='common'): 
        source, target, images, labels = batch["source"], batch["target"], batch["images"], batch["labels"]
        _device = source.device
        batches = source.shape[0]
        # print(source.shape, target.shape, images.shape, labels.shape)
        
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (batches,), device=_device).long() # type: ignore
           
        # Sample noise to add to the images
        class_source = torch.zeros(batches, device=_device)
        class_target = torch.ones(batches, device=_device)
        
        noise = torch.randn_like(source)
        noisy_source = self.scheduler.add_noise(original_samples=source, noise=noise, timesteps=timesteps)
        noisy_target = self.scheduler.add_noise(original_samples=target, noise=noise, timesteps=timesteps)
        super_loss = 0

        # Get model prediction
        noise_source_pred = self.noise2space.forward(noisy_source, timesteps=timesteps, class_labels=class_source)
        noise_target_pred = self.noise2space.forward(noisy_target, timesteps=timesteps, class_labels=class_target)
        
        super_loss += self.loss_func(noise, noise_source_pred)
        super_loss += self.loss_func(noise, noise_target_pred)
       
        # Implement end2end denoiser
        class_prob = torch.rand(batches, device=_device)
        class_prev = (class_prob * self.num_timesteps).int()
        class_next = class_prev + 1
        sample_prev = noise.clone()
        sample_next = noise.clone()
        # Implent visualization
        sample_source = noise.clone()
        sample_target = noise.clone()
        
        with torch.no_grad():
            for t in tqdm(range(self.num_timesteps)):
                output_source = self.noise2space.forward(sample_source, timesteps=torch.Tensor((t,)).to(_device), class_labels=class_source)
                output_target = self.noise2space.forward(sample_target, timesteps=torch.Tensor((t,)).to(_device), class_labels=class_target)
                sample_source, _ = self.scheduler.step(output_source, t, sample_source)   
                sample_target, _ = self.scheduler.step(output_target, t, sample_target)   
                
                output_prev = self.noise2space.forward(sample_prev, timesteps=torch.Tensor((t,)).to(_device), class_labels=class_prev)
                output_next = self.noise2space.forward(sample_next, timesteps=torch.Tensor((t,)).to(_device), class_labels=class_next)
                sample_prev, _ = self.scheduler.step(output_prev, t, sample_prev)   
                sample_next, _ = self.scheduler.step(output_next, t, sample_next)  
        
        unsup_loss = 0
        # Get model prediction
        sample_prev_pred = self.image2label.forward(sample_prev, timesteps=class_prev)
        sample_next_pred = self.label2image.forward(sample_next, timesteps=class_next)
        
        unsup_loss += self.loss_func(sample_prev, sample_prev_pred)
        unsup_loss += self.loss_func(sample_next, sample_next_pred)
        
        self.log(f'{stage}_super_loss', super_loss, on_step=(stage == 'train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f'{stage}_unsup_loss', unsup_loss, on_step=(stage == 'train'), prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size)
        loss = super_loss + unsup_loss
        
        
        sample_prev = sample_prev * 0.5 + 0.5 # type: ignore
        sample_next = sample_next * 0.5 + 0.5 # type: ignore
        sample_prev_pred = sample_prev_pred * 0.5 + 0.5 # type: ignore
        sample_next_pred = sample_next_pred * 0.5 + 0.5 # type: ignore
            
        sample_source = sample_source * 0.5 + 0.5 # type: ignore
        sample_target = sample_target * 0.5 + 0.5 # type: ignore
        
        if stage == 'train' and batch_idx % 10 == 0:
            # print(source, target, sample_source, sample_target)
            viz2d = torch.Tensor(torch.cat([source, target, 
                                            sample_source, 
                                            sample_target, 
                                            sample_prev, 
                                            sample_next, 
                                            sample_prev_pred, 
                                            sample_next_pred
                                            ], dim=-1).transpose(2, 3))
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=8, padding=0)
            tensorboard = self.logger.experiment # type: ignore
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.global_step // 10)
            
        info = {f'loss': loss} 
        return info

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str] = 'common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100, help="timesteps")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--train_samples", type=int, default=40000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=8000, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=4000, help="test samples")
    
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    
    parser.add_argument("--epochs", type=int, default=31, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args() # type: ignore
    # Create data module
    
    train_ssource_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_starget_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/labels'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_usource_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_utarget_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/labels'), 
    ]


    val_ssource_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    val_starget_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/labels'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    val_usource_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/images'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    val_utarget_dirs = [
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62022/20220501/raw/labels'), 
        os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/labels'), 
    ]
    
    test_ssource_dirs = val_ssource_dirs
    test_starget_dirs = val_starget_dirs
    test_usource_dirs = val_usource_dirs
    test_utarget_dirs = val_utarget_dirs

    datamodule = PairedAndUnpairedDataModule(
        train_ssource_dirs = train_ssource_dirs, 
        train_starget_dirs = train_starget_dirs, 
        train_usource_dirs = train_usource_dirs, 
        train_utarget_dirs = train_utarget_dirs, 
        val_ssource_dirs = val_ssource_dirs, 
        val_starget_dirs = val_starget_dirs, 
        val_usource_dirs = val_usource_dirs, 
        val_utarget_dirs = val_utarget_dirs, 
        test_ssource_dirs=test_ssource_dirs,
        test_starget_dirs=test_starget_dirs,
        test_usource_dirs=test_usource_dirs,
        test_utarget_dirs=test_utarget_dirs,
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
        batch_size = hparams.batch_size, 
        shape = hparams.shape,
        # keys = ["source", "target", "images", "labels"]
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

    model = DDMMLightningModule(
        hparams = hparams
    )

    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

     # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
        ],
        # accumulate_grad_batches=4, 
        strategy="ddp_sharded", #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        # strategy="fsdp", #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        # precision=16,  #if hparams.use_amp else 32,
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
        datamodule, # , 
        ckpt_path=hparams.ckpt if hparams.ckpt is not None else None, # "some/path/to/my_checkpoint.ckpt"
    )

    # test

    # serve