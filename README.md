# Diffusors

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://monai.io/"><img alt="MONAI" src="https://img.shields.io/badge/Project-MONAI-blue"></a>
<a href="https://github.com/wandb/wandb"><img alt="wandb" src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge.svg"></a>


## Introduction

This project is folked from [diffusors](https://github.com/tmquan/diffusors)

## Setup

1. Install the required libraries:

```bash
pip install pytorch_lightning wandb
```

2. Log in to Weights & Biases:

```bash
wandb login
```

## Usage

Replace the TensorBoard logger with the Weights & Biases logger as follows:

```python
import wandb
from pytorch_lightning.loggers import WandbLogger

# Initialize wandb
wandb.init(project='my-project', entity='my-entity')

# Create a WandbLogger
wandb_logger = WandbLogger()
```
Please replace 'my-project' and 'my-entity' with your actual project and entity names.

3. Build Docker

```dockerfile
docker build -t root .
```

```docker
docker run -it root python segmentation_diffuser_two.py --train_samples 600 --val_samples 100 --test_samples 200 --datadir . --epochs 25 
--devices 2 --strategy ddp --batch_size 2 --accelerator gpu --logsdir . --precision 16 --wandb_tags diffusion_all
```

## License
MIT