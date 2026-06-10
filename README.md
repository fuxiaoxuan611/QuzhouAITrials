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

The YAML-driven entrypoint is:

```bash
python train.py --config configs/experiments/NL-Wheat.yaml
```

You can override individual settings with repeated `--override key=value` flags:

```bash
python train.py \
  --config configs/experiments/CN-Maize.yaml \
  --override experiment.seed=3 \
  --override agent.seed=3 \
  --override training.total_timesteps=2000000 \
  --override logging.comet.enabled=false
```

For a fast build check without learning:

```bash
python train.py --config configs/experiments/debug.yaml --dry-run
```

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

## Lagrangian PPO constraints

`LagPPO` uses a Stable-Baselines3-compatible Lagrangian PPO implementation in
`pcse_gym/agent/ppo_mod.py`. The algorithm is generic: it does not contain
crop-specific fertilization rules. Instead, environments or wrappers expose a
scalar cost at every step:

```python
info["cost"] = 1.0
info["cost_components"] = {
    "too_many_nonzero_actions": 0.0,
    "nue_violation": 1.0,
    "n_surplus_violation": 0.0,
}
```

For the built-in agronomic soft costs, wrap the environment with
`ConstraintCostWrapper` from `pcse_gym.envs.constraints`. This wrapper does not
alter actions; it only reports costs. The older `ActionConstrainer` remains
available for experiments that intentionally need hard action modification.

Important `LagPPO` settings exposed by the training scripts:

```bash
--lag-cost-limit 0.0
--lag-lambda-init 1.0
--lag-lambda-lr 0.05
--lag-lambda-max 3.0
--lag-cost-vf-coef 0.7
--lag-cost-key cost
```

The policy objective uses the OmniSafe-style penalized advantage:

```text
A_lag = (A_reward - lambda * A_cost) / (1 + lambda)
```

`RecurrentLagrangianPPO` is also available for LSTM policies and can be selected
in the training scripts with `--agent RecurrentLagPPO`. It adapts
`sb3_contrib.RecurrentPPO` with recurrent cost storage, cost advantages, and a
shared recurrent reward/cost critic feature path. Current limitations are that
the algorithm consumes one aggregate scalar cost and the recurrent v1 shares
critic features between reward and cost value heads.

`RecurrentCUP` adds a recurrent, SB3-native implementation of CUP for the same
cost interface. It first runs the recurrent reward PPO update, then applies a
second actor-only cost projection step using the stored recurrent cost
advantages. CUP requires `agent.gamma < 1.0`; configs using `gamma: 1.0` must
override it, for example `--override agent.gamma=0.99`.

`RecurrentFOCOPS` adds a recurrent, SB3-native FOCOPS policy update for the same
cost interface. It uses the Lagrangian-combined reward/cost advantage with a
KL-gated actor loss controlled by `focops_eta` and `focops_lam`. Unlike CUP,
FOCOPS does not require overriding configs that use `agent.gamma: 1.0`.

## Quzhou Maize Calibration

The winter-wheat RL training path is kept as the current working baseline in `train_winterwheat.py`, because the wheat trials still need that line of work. For the China maize field-trial work, the calibrated Quzhou crop parameter file is included at:

```text
pcse_gym/envs/configs/crop/maize.yaml
```

It registers `maize` in `pcse_gym/envs/configs/crop/crops.yaml` and provides both `Grain_maize_201` and `Quzhou_maize_2025_Opt` varieties. These parameters are the crop-side overrides from `../quzhou_maize/scripts/run_final_model.py`; the calibrated site, soil, agromanagement, and weather inputs still live under `../quzhou_maize/inputs/`.

The matching Quzhou site and soil inputs have also been copied into the RL config tree:

```text
pcse_gym/envs/configs/site/quzhou_maize_site_2025.yaml
pcse_gym/envs/configs/soil/quzhou_maize_4layer_whcns_2025.yaml
pcse_gym/envs/configs/agro/quzhou_maize_2025.yaml
pcse_gym/envs/configs/weather/quzhou_maize_2025_power.xlsx
```

These are direct copies of the final Quzhou inputs used by `../quzhou_maize/scripts/run_final_model.py`, except that the copied agromanagement file is stored in the raw list format used by CropGym configs.

Use the maize constructor with:

```python
from pcse_gym.envs.maize import Maize

env = Maize(reward="GRO")
```

By default, `Maize` preserves the calibrated Quzhou site N profile and fixed 2025 weather, keeps the timed irrigation events, and removes the timed N fertilizer events so the RL action controls N applications. Passing `remove_timed_n=False` keeps the original scheduled fertilizer events and reproduces the calibrated `run_final_model.py` replay with zero RL action.


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
