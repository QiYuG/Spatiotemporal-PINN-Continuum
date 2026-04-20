import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model.PINN_Tau import PINN_Tau
from model.PINN_Residual import PINN_Residual

torch.set_default_dtype(torch.float32)
DIM = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_TRAIN_TYPE = "noisy"   # 模型是在哪个数据集上训练的 (保持 noisy 不变)
TEST_DATA_TYPE = "clean"     # 🌟 重点！测试集必须用干净的物理基准数据
TEST_PATH = f"dataset/dataset_{TEST_DATA_TYPE}_test.pt"

# 真实系统的阻尼参数 (4-DOF)
TRUE_DAMPING = torch.tensor([0.8, 0.8, 0.8, 0.8], device=DEVICE)

MODEL_DICT = {
    "Residual_margin": PINN_Residual,
    "Residual_condition": PINN_Residual,
    "Tau_margin": PINN_Tau,
    "Tau_condition": PINN_Tau
}

COLORS = {
    "Residual_margin": "#d62728",    # 红色
    "Residual_condition": "#2ca02c", # 绿色
    "Tau_margin": "#1f77b4",         # 蓝色
    "Tau_condition": "#9467bd"       # 紫色
}

def load_model_safely(model_name, model_class, is_evo):
    model = model_class(DIM=DIM, device=DEVICE).to(DEVICE)
    if is_evo:
        path = f"models_evo_final/final_evo_{model_name}_{MODEL_TRAIN_TYPE}.pth"
    else:
        # 兼容不同的单步模型保存命名
        path = f"models/final_{model_name}_{MODEL_TRAIN_TYPE}.pth"
        if not os.path.exists(path):
            path = f"models/best_{model_name}_{MODEL_TRAIN_TYPE}.pth"
            
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    else:
        print(f"[警告] 找不到模型权重: {path}")
        return None

if __name__ == "__main__":
    print("🚀 开始进行能量无源性与李雅普诺夫稳定性分析...")
    
    # 1. 加载测试轨迹数据 (取第一条轨迹的前 5 秒 / 1000 步)
    test_data = torch.load(TEST_PATH, weights_only=False)
    STEPS = 1000
    q_seq = test_data["q"][:STEPS].to(DEVICE)
    dq_seq = test_data["dq"][:STEPS].to(DEVICE)
    time_axis = np.arange(STEPS) * 0.005

    # 2. 计算 Ground Truth 的真实耗散功率: P_diss = dq^T * D_true * dq
    P_diss_true = torch.sum(TRUE_DAMPING * (dq_seq ** 2), dim=1).cpu().numpy()

    # 3. 准备绘图 (左右对比)
    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle("System Passivity & Lyapunov Stability Analysis ($P_{diss} = \dot{q}^T D(q,\dot{q}) \dot{q}$)", fontsize=20, fontweight='bold')

    # ========== 子图 1: 未加 Evo Loss ==========
    ax1 = axs[0]
    ax1.plot(time_axis, P_diss_true, 'k--', linewidth=3, zorder=10, label="Ground Truth (True Damping)")
    
    for name, model_class in MODEL_DICT.items():
        model = load_model_safely(name, model_class, is_evo=False)
        if model:
            with torch.no_grad():
                D_matrix = model.D(q_seq, dq_seq)
                # 计算预测的耗散功率
                P_diss_pred = torch.bmm(dq_seq.unsqueeze(1), torch.bmm(D_matrix, dq_seq.unsqueeze(2))).squeeze().cpu().numpy()
            ax1.plot(time_axis, P_diss_pred, color=COLORS[name], linewidth=2.5, alpha=0.8, label=name)
            
    ax1.set_title("Without Evolution Loss (Single-step PINN)", fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.set_ylabel("Dissipation Power (Watts)", fontsize=14)
    ax1.axhline(0, color='red', linestyle=':', linewidth=2, label="Passivity Bound (0 W)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11, loc='upper right')

    # ========== 子图 2: 加入 Evo Loss ==========
    ax2 = axs[1]
    ax2.plot(time_axis, P_diss_true, 'k--', linewidth=3, zorder=10, label="Ground Truth (True Damping)")
    
    for name, model_class in MODEL_DICT.items():
        model = load_model_safely(name, model_class, is_evo=True)
        if model:
            with torch.no_grad():
                D_matrix = model.D(q_seq, dq_seq)
                P_diss_pred = torch.bmm(dq_seq.unsqueeze(1), torch.bmm(D_matrix, dq_seq.unsqueeze(2))).squeeze().cpu().numpy()
            ax2.plot(time_axis, P_diss_pred, color=COLORS[name], linewidth=2.5, alpha=0.8, label=name)

    ax2.set_title("With Evolution Loss (Our Full Framework)", fontsize=16)
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.axhline(0, color='red', linestyle=':', linewidth=2, label="Passivity Bound (0 W)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11, loc='upper right')

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_name = "Passivity_Lyapunov_Analysis_final.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ 无源性分析图表已生成: {save_name}")

# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# from model.PINN_Tau import PINN_Tau
# from model.PINN_Residual import PINN_Residual

# torch.set_default_dtype(torch.float32)

# # ==============================
# # 1. 核心升维：6-DOF
# # ==============================
# DIM = 6
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODEL_TRAIN_TYPE = "noisy"   # 模型是在哪个数据集上训练的 (保持 noisy 不变)
# TEST_DATA_TYPE = "clean"     # 🌟 重点！测试集必须用干净的物理基准数据
# # 指向 6-DOF 测试集
# TEST_PATH = f"dataset_6dof/dataset_{TEST_DATA_TYPE}_test.pt"

# # 真实系统的阻尼参数 (6-DOF)，全关节设定为 0.8
# TRUE_DAMPING = torch.tensor([0.8, 0.8, 0.8, 0.8, 0.8, 0.8], device=DEVICE)

# MODEL_DICT = {
#     "Residual_margin": PINN_Residual,
#     "Residual_condition": PINN_Residual,
#     "Tau_margin": PINN_Tau,
#     "Tau_condition": PINN_Tau
# }

# COLORS = {
#     "Residual_margin": "#d62728",    # 红色
#     "Residual_condition": "#2ca02c", # 绿色
#     "Tau_margin": "#1f77b4",         # 蓝色
#     "Tau_condition": "#9467bd"       # 紫色
# }

# # ==============================
# # 2. 强制只加载 Evo 模型
# # ==============================
# def load_model_safely(model_name, model_class):
#     model = model_class(DIM=DIM, device=DEVICE).to(DEVICE)
#     path = f"models_6dof_evo/final_evo_{model_name}_{MODEL_TRAIN_TYPE}_new.pth"
            
#     if os.path.exists(path):
#         model.load_state_dict(torch.load(path, map_location=DEVICE))
#         model.eval()
#         return model
#     else:
#         print(f"[警告] 找不到模型权重: {path}")
#         return None

# if __name__ == "__main__":
#     print("🚀 开始进行 6-DOF 能量无源性与李雅普诺夫稳定性分析...")
    
#     # 1. 加载测试轨迹数据 (取第一条轨迹的前 5 秒 / 1000 步)
#     try:
#         test_data = torch.load(TEST_PATH, map_location=DEVICE, weights_only=False)
#     except FileNotFoundError:
#         print(f"❌ 找不到测试数据集: {TEST_PATH}")
#         exit()
        
#     STEPS = 1000
#     q_seq = test_data["q"][:STEPS].to(DEVICE)
#     dq_seq = test_data["dq"][:STEPS].to(DEVICE)
#     time_axis = np.arange(STEPS) * 0.005

#     # 2. 计算 Ground Truth 的真实耗散功率: P_diss = dq^T * D_true * dq
#     P_diss_true = torch.sum(TRUE_DAMPING * (dq_seq ** 2), dim=1).cpu().numpy()

#     # 3. 准备绘图 (重构为单图)
#     fig, ax = plt.subplots(figsize=(12, 7))
#     fig.suptitle("6-DOF System Passivity & Lyapunov Stability Analysis\n($P_{diss} = \dot{q}^T D(q,\dot{q}) \dot{q}$ with Evo Loss)", fontsize=18, fontweight='bold')

#     # 绘制 Ground Truth
#     ax.plot(time_axis, P_diss_true, 'k--', linewidth=3, zorder=10, label="Ground Truth (True Damping)")
    
#     for name, model_class in MODEL_DICT.items():
#         model = load_model_safely(name, model_class)
#         if model:
#             with torch.no_grad():
#                 D_matrix = model.D(q_seq, dq_seq)
#                 # 计算预测的耗散功率
#                 P_diss_pred = torch.bmm(dq_seq.unsqueeze(1), torch.bmm(D_matrix, dq_seq.unsqueeze(2))).squeeze().cpu().numpy()
#             ax.plot(time_axis, P_diss_pred, color=COLORS[name], linewidth=2.5, alpha=0.8, label=name)

#     ax.set_title("Evolutionary Fine-Tuned Models on 6-DOF Dynamics", fontsize=15)
#     ax.set_xlabel("Time (s)", fontsize=14)
#     ax.set_ylabel("Dissipation Power (Watts)", fontsize=14)
#     ax.axhline(0, color='red', linestyle=':', linewidth=2, label="Passivity Bound (0 W)")
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.legend(fontsize=12, loc='upper right')

#     # 调整布局并保存
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     save_name = "Passivity_Lyapunov_Analysis_6DOF_Evo_new_new.png"
#     plt.savefig(save_name, dpi=300)
#     print(f"✅ 6-DOF 无源性分析图表已生成: {save_name}")