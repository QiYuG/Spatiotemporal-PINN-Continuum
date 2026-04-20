# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os
# from model.PINN_Residual import PINN_Residual
# from model.PINN_Tau import PINN_Tau

# torch.set_default_dtype(torch.float32)
# DEVICE = torch.device("cpu")
# DIM = 4
# NUM_SAMPLES = 5000

# # ============================================================
# # 加载模型
# # ============================================================
# model_residual_margin = PINN_Residual(DIM=DIM, device=DEVICE)
# model_residual_margin.load_state_dict(torch.load("/models/best_Residual_margin.pth", map_location=DEVICE))
# model_residual_margin.eval()

# model_residual_condition = PINN_Residual(DIM=DIM, device=DEVICE)
# model_residual_condition.load_state_dict(torch.load("/models/best_Residual_condition.pth", map_location=DEVICE))
# model_residual_condition.eval()

# model_tau_margin = PINN_Tau(DIM=DIM, device=DEVICE)
# model_tau_margin.load_state_dict(torch.load("/models/best_Tau_margin.pth", map_location=DEVICE))
# model_tau_margin.eval()

# model_tau_condition = PINN_Tau(DIM=DIM, device=DEVICE)
# model_tau_condition.load_state_dict(torch.load("/models/best_Tau_condition.pth", map_location=DEVICE))
# model_tau_condition.eval()


# # ============================================================
# # 随机采样 q
# # ============================================================
# def sample_q(num_samples):

#     # 假设关节角范围 [-pi, pi]
#     return (2 * np.pi) * torch.rand(num_samples, DIM) - np.pi


# # ============================================================
# # 条件数计算
# # ============================================================
# @torch.no_grad()
# # def compute_condition_statistics(model, q_samples):

# #     cond_list = []
# #     min_eig_list = []
# #     max_eig_list = []
    
# #     pbar = tqdm(range(q_samples.shape[0]))

# #     for i in pbar:

# #         q = q_samples[i].unsqueeze(0)

# #         M = model.M(q).squeeze(0)

# #         # M = 0.5 * (M + M.T)

# #         eigvals = torch.linalg.eigvalsh(M)

# #         min_eig = eigvals[0].item()
# #         max_eig = eigvals[-1].item()

# #         # 防止除零
# #         cond = max_eig / (min_eig + 1e-12)

# #         cond_list.append(cond)
# #         min_eig_list.append(min_eig)
# #         max_eig_list.append(max_eig)
        
# #         pbar.set_description(f"Sample {i+1}/{q_samples.shape[0]}")

# #     return (
# #         np.array(cond_list),
# #         np.array(min_eig_list),
# #         np.array(max_eig_list)
# #     )
# def compute_condition_statistics(model, q_samples):
#     # q_samples 已经是形状为 (5000, 4) 的 Tensor
#     q_samples = q_samples.to(DEVICE)
    
#     # 1. 一次性计算 5000 个质量矩阵 M
#     M_batch = model.M(q_samples)  # (5000, 4, 4)
    
#     # 2. 批量计算特征值
#     eigvals = torch.linalg.eigvalsh(M_batch)  # (5000, 4)
    
#     # 3. 提取最小和最大特征值
#     min_eig = eigvals[:, 0].cpu().numpy()
#     max_eig = eigvals[:, -1].cpu().numpy()
    
#     # 4. 向量化计算条件数
#     cond = max_eig / (min_eig + 1e-12)
    
#     return cond, min_eig, max_eig


# # ============================================================
# # 主程序
# # ============================================================
# if __name__ == "__main__":
    
#     if not os.path.exists("condition_results"):
#         os.makedirs("condition_results")

#     print("========== Mass Matrix Condition Number Analysis ==========")

#     q_samples = sample_q(NUM_SAMPLES)

#     print("\nResidual Margin Model Analysis")
#     cond_residual_margin, min_residual_margin, max_residual_margin = compute_condition_statistics(model_residual_margin, q_samples)

#     print("\nResidual Condition Model Analysis")
#     cond_residual_condition, min_residual_condition, max_residual_condition = compute_condition_statistics(model_residual_condition, q_samples)

#     print("\nTau Margin Model Analysis")
#     cond_tau_margin, min_tau_margin, max_tau_margin = compute_condition_statistics(model_tau_margin, q_samples)

#     print("\nTau Condition Model Analysis")
#     cond_tau_condition, min_tau_condition, max_tau_condition = compute_condition_statistics(model_tau_condition, q_samples)

#     # ============================================================
#     # 统计输出
#     # ============================================================
#     def print_stats(name, cond, min_eig):
#         print(f"\n{name} Statistics:")
#         print("Mean cond:", np.mean(cond))
#         print("Max cond:", np.max(cond))
#         print("95% percentile cond:", np.percentile(cond, 95))
#         print("Min eigenvalue mean:", np.mean(min_eig))
#         print("Min eigenvalue min:", np.min(min_eig))

#     print_stats("Residual Margin", cond_residual_margin, min_residual_margin)
#     print_stats("Residual Condition", cond_residual_condition, min_residual_condition)
#     print_stats("Tau Margin", cond_tau_margin, min_tau_margin)
#     print_stats("Tau Condition", cond_tau_condition, min_tau_condition)

#     # ============================================================
#     # 绘图
#     # ============================================================
#     plt.figure(figsize=(16,10))
    
#     # Residual Margin
#     plt.subplot(2, 2, 1)
#     plt.hist(cond_residual_margin, bins=100, alpha=0.6, label="Residual Margin Model")
#     plt.xlabel("Condition Number")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()
    
#     # Residual Condition
#     plt.subplot(2, 2, 2)
#     plt.hist(cond_residual_condition, bins=100, alpha=0.6, label="Residual Condition Model")
#     plt.xlabel("Condition Number")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()
    
#     # Tau Margin
#     plt.subplot(2, 2, 3)
#     plt.hist(cond_tau_margin, bins=100, alpha=0.6, label="Tau Margin Model")
#     plt.xlabel("Condition Number")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()
    
#     # Tau Condition
#     plt.subplot(2, 2, 4)
#     plt.hist(cond_tau_condition, bins=100, alpha=0.6, label="Tau Condition Model")
#     plt.xlabel("Condition Number")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid()
    
#     plt.suptitle("Mass Matrix Condition Number Distribution", fontsize=16, y=1.02)
#     plt.tight_layout()
#     plt.savefig("condition_results/Mass_Matrix_Condition_Distribution.png", dpi=300)

#     # ============================================================
#     # 保存数据
#     # ============================================================
#     np.save("condition_results/cond_residual_margin_M.npy", cond_residual_margin)
#     np.save("condition_results/cond_residual_condition_M.npy", cond_residual_condition)
#     np.save("condition_results/cond_tau_margin_M.npy", cond_tau_margin)
#     np.save("condition_results/cond_tau_condition_M.npy", cond_tau_condition)

#     print("\nAnalysis Completed.")


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
NUM_SAMPLES = 5000
DATASET_TYPE = "noisy" 

# ============================================================
# 随机采样生成器 (运行在 CPU 上)
# ============================================================
def sample_q(num_samples):
    # 假设关节角范围 [-pi, pi]
    return (2 * np.pi) * torch.rand(num_samples, DIM) - np.pi

# ============================================================
# 在单个 GPU 上批量计算条件数的 Worker
# ============================================================
def compute_cond_worker(gpu_id, model_name, model_class, weights_path, q_samples):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[GPU {gpu_id}] 正在分析 {model_name} 的 {NUM_SAMPLES} 个样本...")
    
    # 加载模型
    model = model_class(DIM=DIM, device=device).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 推入指定 GPU
    q_batch = q_samples.to(device)

    with torch.no_grad():
        M_batch = model.M(q_batch)
        M_batch = 0.5 * (M_batch + M_batch.transpose(1, 2))
        
        # 批量并行求解特征值
        eigvals = torch.linalg.eigvalsh(M_batch)
        
        min_eig = eigvals[:, 0].cpu().numpy()
        max_eig = eigvals[:, -1].cpu().numpy()

    # 向量化计算条件数
    cond = max_eig / (min_eig + 1e-12)

    print(f"[GPU {gpu_id}] {model_name} 分析完毕！")
    return model_name, cond, min_eig, max_eig

# ============================================================
# 主调度与直方图绘制
# ============================================================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if not os.path.exists("condition_results"):
        os.makedirs("condition_results")

    print(f"========== 开启 4卡并行: 质量矩阵条件数蒙特卡洛统计 ==========")
    start_time = time.time()

    # 生成全局统一的 5000 个随机样本，保证四个模型面对的是绝对相同的测试姿态
    q_samples_global = sample_q(NUM_SAMPLES)

    # 任务分配
    # tasks = [
    #     (0, "Residual_margin", PINN_Residual, f"models/final_Residual_margin_{DATASET_TYPE}.pth", q_samples_global),
    #     (1, "Residual_condition", PINN_Residual, f"models/final_Residual_condition_{DATASET_TYPE}.pth", q_samples_global),
    #     (2, "Tau_margin", PINN_Tau, f"models/final_Tau_margin_{DATASET_TYPE}.pth", q_samples_global),
    #     (3, "Tau_condition", PINN_Tau, f"models/final_Tau_condition_{DATASET_TYPE}.pth", q_samples_global)
    # ]
    # tasks = [
    #     (0, "Residual_margin", PINN_Residual, f"models_evo_final/final_evo_Residual_margin_{DATASET_TYPE}.pth", q_samples_global),
    #     (1, "Residual_condition", PINN_Residual, f"models_evo_final/final_evo_Residual_condition_{DATASET_TYPE}.pth", q_samples_global),
    #     (2, "Tau_margin", PINN_Tau, f"models_evo_final/final_evo_Tau_margin_{DATASET_TYPE}.pth", q_samples_global),
    #     (3, "Tau_condition", PINN_Tau, f"models_evo_final/final_evo_Tau_condition_{DATASET_TYPE}.pth", q_samples_global)
    # ]
    tasks = [
        (0, "Residual_margin", PINN_Residual, f"models_6dof_evo/final_evo_Residual_margin_{DATASET_TYPE}_new.pth", q_samples_global),
        (1, "Residual_condition", PINN_Residual, f"models_6dof_evo/final_evo_Residual_condition_{DATASET_TYPE}_new.pth", q_samples_global),
        (2, "Tau_margin", PINN_Tau, f"models_6dof_evo/final_evo_Tau_margin_{DATASET_TYPE}_new.pth", q_samples_global),
        (3, "Tau_condition", PINN_Tau, f"models_6dof_evo/final_evo_Tau_condition_{DATASET_TYPE}_new.pth", q_samples_global)
    ]

    with mp.Pool(processes=4) as pool:
        results = pool.starmap(compute_cond_worker, tasks)
        
    print(f"\n所有条件数计算完成！耗时: {time.time() - start_time:.2f} 秒。\n")

    # 提取结果并统计
    results_dict = {res[0]: {'cond': res[1], 'min_eig': res[2], 'max_eig': res[3]} for res in results}
    plot_order = ["Residual_margin", "Residual_condition", "Tau_margin", "Tau_condition"]

    print("=" * 60)
    print(f"{'模型名称':<22} | {'平均条件数':<12} | {'95%分位条件数':<15} | {'最小特征值平均':<15}")
    print("-" * 60)
    for name in plot_order:
        r = results_dict[name]
        mean_cond = np.mean(r['cond'])
        p95_cond = np.percentile(r['cond'], 95)
        mean_min = np.mean(r['min_eig'])
        print(f"{name:<22} | {mean_cond:<14.4f} | {p95_cond:<18.4f} | {mean_min:.6e}")
    print("=" * 60)

    # 绘图
    plt.figure(figsize=(16, 12))
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd']
    
    for i, (name, color) in enumerate(zip(plot_order, colors)):
        plt.subplot(2, 2, i + 1)
        cond_data = results_dict[name]['cond']
        
        plt.hist(cond_data, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.title(f"{name} Model\nCondition Number Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Condition Number $\kappa(M)$", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("condition_results/Mass_Matrix_Condition_Distribution_6dof_evo_last.png", dpi=300, bbox_inches='tight')

    # 保存原始数据
    for name in plot_order:
        safe_name = name.replace(" ", "_")
        np.save(f"condition_results/cond_{safe_name}_M_6dof_evo_last.npy", results_dict[name]['cond'])

    print("✅ 统计完成并已保存图像至 condition_results/ 目录。")