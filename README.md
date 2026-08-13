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
