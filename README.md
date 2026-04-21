# Spatiotemporal-PINN-Continuum
# Physics-Informed Neural Networks for Continuum Robot Dynamics

This repository contains the official PyTorch implementation for the dynamic modeling of flexible continuum robots using Physics-Informed Neural Networks (PINNs). 

By embedding spatial-algebraic constraints (Cholesky decomposition) and temporal evolution loss (Runge-Kutta 4th order integration), this project provides a robust and physically consistent framework for both 4-DOF and 6-DOF continuum robots under complex environments (e.g., non-modeled friction and sensor noise).

## 📌 Project Structure

The training pipeline is designed progressively from low-dimensional ablation to high-dimensional generalization:

```text
├── 📊 Data Generation & Datasets
│   ├── data.py                        # Generate 4-DOF datasets (clean/noisy)
│   ├── data_6dof.py                   # Generate 6-DOF datasets (clean/noisy)
│   ├── dataset/                       # 4-DOF dataset directory
│   │   ├── dataset_clean_train.pt
│   │   ├── dataset_clean_test.pt
│   │   ├── dataset_noisy_train.pt
│   │   └── dataset_noisy_test.pt
│   └── dataset_6dof/                  # 6-DOF dataset directory
│       ├── dataset_clean_train.pt
│       ├── dataset_clean_test.pt
│       ├── dataset_noisy_train.pt
│       └── dataset_noisy_test.pt
│
├── 🏗️ Model Architecture (model/)
│   ├── basic_model.py                 # Core MLP architectures (mass_net, damp_net, potential_net)
│   ├── PINN_Residual.py               # PINN loss functions and Cholesky hard-constraints
│   └── PINN_Tau.py                    # Tau-related PINN implementations
│
├── 🔧 Training & Hyperparameter Optimization
│   ├── train.py                       # Base training (4-DOF)
│   ├── tune.py                        # Hyperparameter optimization (4-DOF)
│   ├── tune_evo.py                    # Evolution loss fine-tuning (4-DOF)
│   ├── tune_evo_update.py             # Evolution loss update variants (4-DOF)
│   ├── train_6dof.py                  # Base training (6-DOF)
│   └── train_6dof_evo.py              # Evolution loss training (6-DOF)
│
├── ✅ Validation & Analysis (Testing Scripts)
│   ├── close_loop.py                  # Closed-loop control analysis (base)
│   ├── close_loop_4dof.py             # Closed-loop analysis (4-DOF)
│   ├── close_loop_6dof.py             # Closed-loop analysis (6-DOF)
│   ├── close_loop_adaptive.py         # Adaptive closed-loop control
│   ├── forward_rollout.py             # Forward rollout verification
│   ├── robustness.py                  # Robustness evaluation
│   ├── robust_4dof.py                 # Robustness testing (4-DOF)
│   ├── robust_6dof.py                 # Robustness testing (6-DOF)
│   ├── passivity.py                   # Passivity constraint verification
│   ├── free_fall.py                   # Free-fall test scenario
│   └── M_condition.py                 # Mass matrix condition number analysis
│
├── 📈 Visualization & Analysis
│   ├── M_min_eigen_4dof.py            # 4-DOF eigenvalue visualization
│   └── M_min_eigenvalue_visualization.py  # General eigenvalue visualization
│
├── 🧠 Trained Models
│   ├── models/                        # Base trained models (4-DOF)
│   ├── models_6dof/                   # Trained models (6-DOF)
│   ├── models_6dof_evo/               # Evolution-trained models (6-DOF)
│   └── models_evo_final/              # Final evolution trained models
│
├── 📁 Results & Analysis Outputs
│   ├── condition_results/             # Condition number analysis results (.npy files)
│   ├── control_results/               # Control analysis results
│   ├── free_fall_results/             # Free-fall test results
│   ├── min_eig_results/               # Minimum eigenvalue results (.npy files)
│   ├── passivity_results/             # Passivity verification results
│   ├── robustness_results/            # Robustness evaluation results
│   └── rollout_results/               # Forward rollout verification results
│
├── 🛠 Utility Functions (utils/)
│   ├── condition_number_regularization.py  # Condition number regularization
│   ├── skew_structure_loss.py              # Skew-symmetric structure losses
│   └── spectual_margin_loss.py            # Spectral margin loss functions
│
└── 📄 Documentation
    └── README.md

