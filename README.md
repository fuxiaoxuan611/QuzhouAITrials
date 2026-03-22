# NUE_PCSE-Gym

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](#installation)
[![uv](https://img.shields.io/badge/package_manager-uv-6e56cf.svg)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Computers%20and%20Electronics%20in%20Agriculture-orange.svg)](https://www.sciencedirect.com/science/article/pii/S016816992500660X)

An extended version of [_CropGym_](https://cropgym.ai), containing the code used in the paper:

**[_Adaptive fertilizer management for optimizing nitrogen use efficiency with constrained reinforcement learning_](https://www.sciencedirect.com/science/article/pii/S016816992500660X)**

This repository provides a reinforcement learning environment for adaptive fertilizer management, with a focus on optimizing **nitrogen use efficiency (NUE)** under agronomic and environmental constraints.

---

## Overview

This project builds on CropGym and PCSE to support experiments on fertilizer management with reinforcement learning. It includes:

- Simulations with [PCSE](https://github.com/ajwdewit/pcse.git)
- training scripts for constrained RL agents (LagrangianPPO with Stable-Baselines3)
- support for random weather training scenarios
- implementations used in the published NUE paper

---

## Repository structure

Below is an example overview of the repository structure.

```text
NUE_PCSE-Gym/
├── pcse_gym/
│   ├── agents/
│   ├── envs/
│   ├── utils/
│   │   └── weather_utils/
│   │       └── random_weather_csv/
│   └── ...
├── train_winterwheat.py
├── pyproject.toml
├── README.md
└── ...
```

---

## Installation

This project should be installed with **Python 3.11**.

### 1. Clone PCSE

First, clone the [PCSE repository](https://github.com/ajwdewit/pcse.git):

```bash
git clone https://github.com/ajwdewit/pcse.git
```

### 2. Clone this repository

```bash
git clone https://github.com/WUR-AI/NUE_PCSE-Gym.git
cd NUE_PCSE-Gym
```

### 3. Create a virtual environment
Using [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 4. Install dependencies
From the root of this repository:
```bash
uv sync
```

### 5. Install PCSE in editable mode

From the pcse root:

```bash
cd ../pcse
uv pip install -e .
```

### 6. Install rllte-core

Install rllte-core without dependencies:
```bash
uv pip install --no-deps rllte-core
```
rllte-core is installed this way because some of its dependencies are outdated and no longer install cleanly in tandem with other packages. The dependencies used in this repository have already been handled through uv sync.

### 7. Download the random weather file

Download the CSV file from Zenodo:

https://doi.org/10.5281/zenodo.15267400

Place the downloaded `.csv` file in:

```text
pcse_gym/utils/weather_utils/random_weather_csv/
```

Create the folder if it does not already exist.

---

## Quick start

An example command to train a model using:
- NUE as the reward function
- LagrangianPPO as the agent
- E3B as the intrinsic reward
- random weather and random initialization enabled

```bash
python train_winterwheat.py \
  --reward NUE \
  --environment 2 \
  --agent LagPPO \
  --seed 4 \
  --nsteps 3000000 \
  --random-weather \
  --random-init \
  --irs E3B \
  --no-comet
```


## Citation
If you find this work useful, please consider citing:

```bibtex
@article{baja2025nue,
  title={Adaptive fertilizer management for optimizing nitrogen use efficiency with constrained reinforcement learning},
  author={Baja, Hilmy and Kallenberg, Michiel GJ and Berghuijs, Herman NC and Athanasiadis, Ioannis N},
  journal={Computers and Electronics in Agriculture},
  volume={237},
  pages={110554},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compag.2025.110554},
}
```

### Acknowledgements

This work was carried out as part of the EU Horizon project [Smart Droplets](https://github.com/Smart-Droplets-Project), [https://smartdroplets.eu/](https://smartdroplets.eu/).

