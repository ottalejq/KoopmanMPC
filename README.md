# KoopmanMPC

Implementation of a hierarchical Koopman Model Predictive Control (MPC) framework for nonlinear control of the Cart-Pole system.

The approach combines two data-driven Koopman prediction models at different temporal resolutions:

- **H-step model** for long-horizon global planning
- **1-step model** for high-resolution local tracking

The Koopman models use a bilinear state-input representation and Fourier feature lifting. Both Random Fourier Features and learned Fourier features are considered.

## Hierarchical MPC

The controller consists of three stages:

1. **Global planning** — an H-step Koopman model generates a coarse long-horizon trajectory.
2. **Reference generation** — the global trajectory is expanded to the original sampling resolution using the 1-step model.
3. **Local tracking** — a local MPC tracks the resulting state and input references.

This separates long-horizon planning from short-horizon feedback control and reduces the number of recursive model evaluations required for global prediction.

## Structure

```text
KoopmanMPC/
├── main.py          # Main simulation
├── experiment.py    # Multi-seed evaluation
├── models.py        # Koopman models and MPC controllers
├── system.py        # Cart-Pole system and simulation
├── utils.py         # State encoding
└── requirements.txt # Python dependencies
```

## Installation

The project requires Python and [acados](https://github.com/acados/acados). Linux or WSL with Ubuntu is recommended.

### 1. Install acados

Install the required system packages:

```bash
sudo apt update
sudo apt install git cmake build-essential python3-pip python3-venv
```

Clone and build acados:

```bash
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```

### 2. Clone this repository

```bash
cd ~
git clone https://github.com/ottalejq/KoopmanMPC.git
cd KoopmanMPC
```

### 3. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ~/acados/interfaces/acados_template
```

### 4. Configure acados

Set the acados installation path:

```bash
export ACADOS_SOURCE_DIR=$HOME/acados
```

This command must be run in each new terminal before running the project. To set it permanently, add it to `~/.bashrc`.

### 5. Run

Run the main simulation:

```bash
python main.py
```
