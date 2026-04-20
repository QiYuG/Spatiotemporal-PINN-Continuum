# Spatiotemporal-PINN-Continuum
# Physics-Informed Neural Networks for Continuum Robot Dynamics

This repository contains the official PyTorch implementation for the dynamic modeling of flexible continuum robots using Physics-Informed Neural Networks (PINNs). 

By embedding spatial-algebraic constraints (Cholesky decomposition) and temporal evolution loss (Runge-Kutta 4th order integration), this project provides a robust and physically consistent framework for both 4-DOF and 6-DOF continuum robots under complex environments (e.g., non-modeled friction and sensor noise).

## 📌 Project Structure

The training pipeline is designed progressively from low-dimensional ablation to high-dimensional generalization:

```text
├── dataset/                  # Directory for generated 4-DOF/6-DOF datasets
├── models/
│   ├── basic_model.py        # Core MLP architectures (mass_net, damp_net, potential_net)
│   └── PINN_Residual.py      # PINN loss functions and Cholesky hard-constraints
├── scripts/
│   ├── <DATA_GENERATION_SCRIPT>.py  # Generates Clean/Noisy forward dynamic datasets
│   ├── tune.py               # Hyperparameter Pareto optimization (4-DOF)
│   ├── train.py              # Pre-training for base physical skeleton (4-DOF)
│   ├── tune_evo.py           # Evolution loss fine-tuning (4-DOF)
│   ├── <TRAIN_6DOF_SCRIPT>.py       # Zero-shot transfer base training (6-DOF)
│   └── <TRAIN_6DOF_EVO_SCRIPT>.py   # Evolution loss fine-tuning (6-DOF)
├── requirements.txt
└── README.md
