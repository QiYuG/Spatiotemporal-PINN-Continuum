import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from model.PINN_Residual import PINN_Residual
from model.PINN_Tau import PINN_Tau

torch.set_default_dtype(torch.float32)
# DIM = 4
DIM = 6
GRID_SIZE = 200  # 分辨率
Q_RANGE = np.pi  # [-pi, pi]
DATASET_TYPE = "noisy" # 你的模型后缀名

# ============================================================
# 在单个 GPU 上执行批量计算的 Worker
# ============================================================
def compute_grid_worker(gpu_id, model_name, model_class, weights_path):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] 正在加载并计算 {model_name}...")
    
    # 加载模型
    model = model_class(DIM=DIM, device=device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 在 CPU 上生成网格
    q1_vals = np.linspace(-Q_RANGE, Q_RANGE, GRID_SIZE)
    q2_vals = np.linspace(-Q_RANGE, Q_RANGE, GRID_SIZE)
    Q1, Q2 = np.meshgrid(q1_vals, q2_vals)

    # 展平并推入 GPU
    q1_flat = torch.tensor(Q1.flatten(), dtype=torch.float32)
    q2_flat = torch.tensor(Q2.flatten(), dtype=torch.float32)
    q3_flat = torch.zeros_like(q1_flat)
    q4_flat = torch.zeros_like(q1_flat)
    q5_flat = torch.zeros_like(q1_flat)  # 新增 q5
    q6_flat = torch.zeros_like(q1_flat)  # 新增 q6

    # 构建 Batch (40000, 6)
    q_batch = torch.stack([q1_flat, q2_flat, q3_flat, q4_flat, q5_flat, q6_flat], dim=1).to(device)
    # q_batch = torch.stack([q1_flat, q2_flat, q3_flat, q4_flat], dim=1).to(device)

    with torch.no_grad():
        # 瞬间完成 40000 个 M 矩阵的推理
        M_batch = model.M(q_batch) 
        
        # 强制对称并计算特征值
        M_batch = 0.5 * (M_batch + M_batch.transpose(1, 2))
        eigvals = torch.linalg.eigvalsh(M_batch)
        
        # 提取最小特征值
        min_eig_flat = eigvals[:, 0].cpu().numpy()

    print(f"[GPU {gpu_id}] {model_name} 计算完毕！")
    return model_name, min_eig_flat.reshape(Q1.shape), Q1, Q2

# ============================================================
# 主调度与可视化
# ============================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if not os.path.exists("min_eig_results"):
        os.makedirs("min_eig_results")

    print(f"========== 开启 4卡并行: 质量矩阵最小特征值空间扫描 ==========")
    start_time = time.time()

    # 分配 4 个模型到 4 张显卡
    # tasks = [
    #     (0, "Residual_margin", PINN_Residual, f"models/final_Residual_margin_{DATASET_TYPE}.pth"),
    #     (1, "Residual_condition", PINN_Residual, f"models/final_Residual_condition_{DATASET_TYPE}.pth"),
    #     (2, "Tau_margin", PINN_Tau, f"models/final_Tau_margin_{DATASET_TYPE}.pth"),
    #     (3, "Tau_condition", PINN_Tau, f"models/final_Tau_condition_{DATASET_TYPE}.pth")
    # ]
    # tasks = [
    #     (0, "Residual_margin", PINN_Residual, f"models_evo_final/final_evo_Residual_margin_{DATASET_TYPE}.pth"),
    #     (1, "Residual_condition", PINN_Residual, f"models_evo_final/final_evo_Residual_condition_{DATASET_TYPE}.pth"),
    #     (2, "Tau_margin", PINN_Tau, f"models_evo_final/final_evo_Tau_margin_{DATASET_TYPE}.pth"),
    #     (3, "Tau_condition", PINN_Tau, f"models_evo_final/final_evo_Tau_condition_{DATASET_TYPE}.pth")
    # ]
    tasks = [
        (0, "Residual_margin", PINN_Residual, f"models_6dof_evo/final_evo_Residual_margin_{DATASET_TYPE}_new.pth"),
        (1, "Residual_condition", PINN_Residual, f"models_6dof_evo/final_evo_Residual_condition_{DATASET_TYPE}_new.pth"),
        (2, "Tau_margin", PINN_Tau, f"models_6dof_evo/final_evo_Tau_margin_{DATASET_TYPE}_new.pth"),
        (3, "Tau_condition", PINN_Tau, f"models_6dof_evo/final_evo_Tau_condition_{DATASET_TYPE}_new.pth")
    ]

    with mp.Pool(processes=4) as pool:
        results = pool.starmap(compute_grid_worker, tasks)
        
    print(f"所有特征值扫描完成！耗时: {time.time() - start_time:.2f} 秒。正在生成图像...")

    # 整理结果字典
    results_dict = {res[0]: res[1] for res in results}
    Q1, Q2 = results[0][2], results[0][3]

    # 绘图
    plt.figure(figsize=(16, 12))
    plot_order = ["Residual_margin", "Residual_condition", "Tau_margin", "Tau_condition"]
    
    for i, name in enumerate(plot_order):
        plt.subplot(2, 2, i + 1)
        grid_data = results_dict[name]
        
        # 统一颜色映射范围，以便公平对比
        contour = plt.contourf(Q1, Q2, grid_data, levels=100, cmap='viridis')
        plt.colorbar(contour)
        
        plt.title(f"{name} Model\nMass Matrix Min Eigenvalue", fontsize=14, fontweight='bold')
        plt.xlabel("Joint Angle $q_1$ (rad)", fontsize=12)
        plt.ylabel("Joint Angle $q_2$ (rad)", fontsize=12)

    plt.tight_layout()
    plt.savefig("min_eig_results/MinEigenvalue_Spatial_Distribution_6dof_evo_last.png", dpi=300, bbox_inches='tight')

    # 保存矩阵数据
    for name in plot_order:
        safe_name = name.replace(" ", "_")
        np.save(f"min_eig_results/min_eig_{safe_name}_grid_6dof_evo_last.npy", results_dict[name])

    print("✅ 可视化完成并已保存至 min_eig_results/ 目录。")