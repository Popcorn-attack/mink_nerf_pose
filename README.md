# Direct-PoseNet

This repository contains the implementation of Direct-PoseNet, a novel approach for camera pose estimation using NeRF and MinkowskiEngine.

## Requirements

- Python 3.9+
- PyTorch
- MinkowskiEngine
- CUDA (for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/direct-posenet.git
cd direct-posenet
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
direct-posenet/
├── script/
│   ├── train_mink_depth_nerf.py  # Main training script
│   ├── options.py                # Configuration options
│   ├── models/                   # Model definitions
│   ├── dataset_loaders/         # Data loading utilities
│   └── layers/                  # Custom neural network layers
├── requirements.txt             # Project dependencies
└── README.md                   # This file
```

## Usage

1. Prepare your dataset according to the 7Scenes format.

2. Train the model:
```bash
python script/train_mink_depth_nerf.py --config script/config_mink.txt
```

3. For evaluation:
```bash
python script/train_mink_depth_nerf.py --config script/config_mink.txt --render_test
```

## Configuration

The main configuration options can be found in `script/options.py`. Key parameters include:

- `--datadir`: Path to the dataset
- `--basedir`: Directory for saving logs and checkpoints
- `--model_name`: Name of the model for saving
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--max_epochs`: Maximum number of training epochs

## License

[Your chosen license]

## Citation

If you use this code in your research, please cite:
```
@inproceedings{chen2021direct,
  title={Direct-PoseNet: Absolute pose regression with photometric consistency},
  author={Chen, Shuai and Wang, Zirui and Prisacariu, Victor},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={1175--1185},
  year={2021},
  organization={IEEE}
}
```
