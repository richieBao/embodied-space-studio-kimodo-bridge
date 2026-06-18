# Installation With Virtual Environment

> Note: the repo was tested with Python 3.10+ and PyTorch 2.0+.

## Create Enviroment
We recommend setting up a separate virtual environment for Kimodo to avoid dependency conflicts.

### Using venv
```bash
python -m venv venv
source venv/bin/activate
```

### Using Conda
```bash
conda create -n kimodo python=3.10
conda activate kimodo
```

## Install Dependencies

### Install PyTorch
First, make sure to install a version of [PyTorch](https://pytorch.org/get-started/locally/) that works with your system and CUDA version. We suggest anything over PyTorch 2.0. We strongly suggest using a GPU-capable version of PyTorch to generate motions in a reasonable amount of time.

### Install Bundled Modified Viser Library
The interactive demo relies on the modified Viser library included in this release repo under `./kimodo-viser`. Install it from the local copy:
```bash
pip install -e ./kimodo-viser
```

### Install Kimodo
Next, install Kimodo run this command from the base of repo:
```bash
pip install -e .
```
This results in a single editable install for Kimodo and the MotionCorrection package.

If you plan to use the demo, you can instead run:
```bash
pip install -e ./kimodo-viser
pip install -e ".[soma]"
```
This will install the bundled Viser fork and the [SOMA body model](https://github.com/NVlabs/SOMA-X).

Next, head over to the [Quick Start](quick_start.md) page to test out your installation by generating some motions.
