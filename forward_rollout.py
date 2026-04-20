# import torch
# import torch.multiprocessing as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import time
# from tqdm import tqdm

# from model.PINN_Tau import PINN_Tau
# from model.PINN_Residual import PINN_Residual

# torch.set_default_dtype(torch.float32)
# DIM = 4

# # ==============================
# # 实验配置
# # ==============================
# MODEL_TRAIN_TYPE = "noisy"   # 模型是在哪个数据集上训练的 (保持 noisy 不变)
# TEST_DATA_TYPE = "clean"     # 测试集必须用干净的物理基准数据
# DT = 0.005                 # 数据集的采样间隔 (与生成数据时一致)
# ROLLOUT_STEPS = 1000       # 向前预测的步数 (1000步 = 5秒)
# START_IDX = 0              # 从测试集的哪个索引开始测试 (通常 0 就是第一条轨迹的起点)

# # ==============================
# # 加载模型辅助函数
# # ==============================
# def load_model(model_class, path, device):
#     model = model_class(DIM=DIM, device=device).to(device)
#     # 允许在没有找到模型时报错，提醒检查路径
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()
#     return model

# # ==============================
# # 动力学加速度求解与 RK4 积分
# # ==============================
# def get_ddq(model, q_in, dq_in, tau_in):
#     # 必须赋予梯度以便计算 C 矩阵和势能梯度
#     q = q_in.clone().detach().requires_grad_(True)
#     dq = dq_in.clone().detach()
#     tau = tau_in.clone().detach()

#     M = model.M(q)
#     C = model.C(q, dq)
#     D = model.D(q, dq)
#     V = model.potential_net(q)
    
#     # 计算势能引发的保守力 (重力/弹性力)
#     gradV = torch.autograd.grad(V.sum(), q, create_graph=False)[0]

#     # 内部动态力组合
#     tau_internal = (
#         torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + 
#         torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + 
#         gradV
#     )
    
#     # 求解加速度 ddq
#     ddq = torch.linalg.solve(M, tau - tau_internal)
    
#     return ddq.detach() # 截断计算图，防止 OOM

# def rk4_step(model, q, dq, tau, dt):
#     # 张量维度的 RK4 积分
#     ddq1 = get_ddq(model, q, dq, tau)
    
#     q2 = q + 0.5 * dt * dq
#     dq2 = dq + 0.5 * dt * ddq1
#     ddq2 = get_ddq(model, q2, dq2, tau)
    
#     q3 = q + 0.5 * dt * dq2
#     dq3 = dq + 0.5 * dt * ddq2
#     ddq3 = get_ddq(model, q3, dq3, tau)
    
#     q4 = q + dt * dq3
#     dq4 = dq + dt * ddq3
#     ddq4 = get_ddq(model, q4, dq4, tau)

#     q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
#     dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
    
#     return q_next, dq_next

# # ==============================
# # 能量计算 (网络眼中的能量)
# # ==============================
# def compute_energy(model, q_in, dq_in):
#     q = q_in.clone().detach()
#     dq = dq_in.clone().detach()
    
#     M = model.M(q)
#     V = model.potential_net(q)
    
#     dq_vec = dq.unsqueeze(-1)
#     kinetic_energy = 0.5 * torch.bmm(dq_vec.transpose(1, 2), torch.bmm(M, dq_vec)).squeeze(-1).squeeze(-1)
    
#     total_energy = kinetic_energy + V
#     return total_energy.detach().cpu().numpy()[0]

# # ==============================
# # 单个 GPU 的 Rollout 工作流
# # ==============================
# def rollout_worker(gpu_id, model_name, model_class, weights_path):
#     device = torch.device(f"cuda:{gpu_id}")
    
#     # 1. 加载测试集数据
#     test_path = f"dataset/dataset_{TEST_DATA_TYPE}_test.pt"
#     test_data = torch.load(test_path, map_location=device, weights_only=False)
    
#     # 提取一段完整的 Ground Truth 轨迹
#     q_true = test_data["q"][START_IDX : START_IDX + ROLLOUT_STEPS]
#     dq_true = test_data["dq"][START_IDX : START_IDX + ROLLOUT_STEPS]
#     tau_seq = test_data["tau"][START_IDX : START_IDX + ROLLOUT_STEPS]
    
#     # 2. 加载模型
#     model = load_model(model_class, weights_path, device)
    
#     # 3. 初始化状态
#     q_curr = q_true[0:1].clone()
#     dq_curr = dq_true[0:1].clone()
    
#     q_traj = []
#     energy_traj = []
    
#     # 设置专属进度条
#     desc_str = f"[GPU {gpu_id}] 仿真 {model_name:<18}"
#     pbar = tqdm(range(ROLLOUT_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
#     # 4. 执行时域演化
#     for t in pbar:
#         q_traj.append(q_curr.squeeze(0).cpu().numpy())
#         energy_traj.append(compute_energy(model, q_curr, dq_curr))
        
#         tau_curr = tau_seq[t:t+1]
#         q_curr, dq_curr = rk4_step(model, q_curr, dq_curr, tau_curr, DT)
        
#     pbar.close()
    
#     # 返回预测轨迹、能量演化以及真实的轨迹数据（只返回一次真值即可）
#     return {
#         'model_name': model_name,
#         'q_pred': np.array(q_traj),
#         'energy_pred': np.array(energy_traj),
#         'q_true': q_true.cpu().numpy(),
#         'tau_seq': tau_seq.cpu().numpy()
#     }

# # ==============================
# # 主调度与绘图程序
# # ==============================
# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
    
#     if not os.path.exists("rollout_results"):
#         os.makedirs("rollout_results")
        
#     print("\033[2J\033[H", end="")
#     print("============================================================")
#     print(f"🚀 开启 4 卡并行 Forward Rollout | 数据集: {TEST_DATA_TYPE.upper()}")
#     print("============================================================\n")

#     # 定义四张卡的任务
#     tasks = [
#         (0, "Residual_margin", PINN_Residual, f"models/final_Residual_margin_{MODEL_TRAIN_TYPE}.pth"),
#         (1, "Residual_condition", PINN_Residual, f"models/final_Residual_condition_{MODEL_TRAIN_TYPE}.pth"),
#         (2, "Tau_margin", PINN_Tau, f"models/final_Tau_margin_{MODEL_TRAIN_TYPE}.pth"),
#         (3, "Tau_condition", PINN_Tau, f"models/final_Tau_condition_{MODEL_TRAIN_TYPE}.pth")
#     ]
#     # tasks = [
#     #     (0, "Residual_margin", PINN_Residual, f"models_evo_final/final_evo_Residual_margin_{MODEL_TRAIN_TYPE}.pth"),
#     #     (1, "Residual_condition", PINN_Residual, f"models_evo_final/final_evo_Residual_condition_{MODEL_TRAIN_TYPE}.pth"),
#     #     (2, "Tau_margin", PINN_Tau, f"models_evo_final/final_evo_Tau_margin_{MODEL_TRAIN_TYPE}.pth"),
#     #     (3, "Tau_condition", PINN_Tau, f"models_evo_final/final_evo_Tau_condition_{MODEL_TRAIN_TYPE}.pth")
#     # ]
    
#     start_time = time.time()
    
#     # 开启进程池并发仿真
#     with mp.Pool(processes=4) as pool:
#         results = pool.starmap(rollout_worker, tasks)
        
#     print("\n\n🎉 仿真计算全部完成！总耗时: {:.2f} 秒".format(time.time() - start_time))
#     print("正在生成高分辨率物理演化图表...")

#     # 数据解析
#     time_axis = np.arange(ROLLOUT_STEPS) * DT
#     q_true = results[0]['q_true'] # 基准真实轨迹
    
#     results_q = {r['model_name']: r['q_pred'] for r in results}
#     results_energy = {r['model_name']: r['energy_pred'] for r in results}
    
#     colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd'] # 红, 绿, 蓝, 紫
    
#     # ============================================================
#     # 可视化 1: 轨迹追踪对比图
#     # ============================================================
#     plt.figure(figsize=(18, 12))
#     for i in range(DIM):
#         plt.subplot(2, 2, i+1)
#         # 绘制真实轨迹
#         plt.plot(time_axis, q_true[:, i], 'k--', linewidth=3, label="Ground Truth", zorder=10)
        
#         # 绘制四个模型的预测轨迹
#         for (name, q_pred), color in zip(results_q.items(), colors):
#             plt.plot(time_axis, q_pred[:, i], color=color, alpha=0.85, linewidth=2, label=name)
            
#         plt.title(f"Joint $q_{i+1}$ Trajectory Tracking", fontsize=14, fontweight='bold')
#         plt.xlabel("Time (s)", fontsize=12)
#         plt.ylabel("Angle/Position", fontsize=12)
#         plt.legend(loc='best', framealpha=0.9)
#         plt.grid(True, linestyle=':', alpha=0.7)
        
#     plt.tight_layout()
#     plt.savefig(f"rollout_results/Rollout_Trajectories_{TEST_DATA_TYPE}_4dof_last.png", dpi=300, bbox_inches='tight')

#     # ============================================================
#     # 可视化 2: 系统总能量演化图
#     # ============================================================
#     plt.figure(figsize=(12, 7))
#     for (name, E_pred), color in zip(results_energy.items(), colors):
#         plt.plot(time_axis, E_pred, color=color, linewidth=2.5, label=name)
        
#     plt.title(f"System Total Energy Evolution ($E = K + V$) - {TEST_DATA_TYPE.upper()} Data", fontsize=15, fontweight='bold')
#     plt.xlabel("Time (s)", fontsize=12)
#     plt.ylabel("Energy (Joules)", fontsize=12)
#     plt.legend(loc='best', fontsize=11)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(f"rollout_results/Rollout_Energy_{TEST_DATA_TYPE}_4dof_last.png", dpi=300, bbox_inches='tight')

#     # ============================================================
#     # 误差统计与评估报告
#     # ============================================================
#     print("\n" + "="*70)
#     print(f"📊 连续 {ROLLOUT_STEPS} 步 ( {ROLLOUT_STEPS*DT} 秒 ) 积分误差报告")
#     print("="*70)
#     print(f"{'模型名称':<20} | {'全局轨迹 MSE':<15} | {'最终点偏移 MSE':<15}")
#     print("-" * 70)
    
#     for name, q_pred in results_q.items():
#         mse_error = np.mean((q_pred - q_true)**2)
#         final_error = np.mean((q_pred[-1] - q_true[-1])**2)
#         print(f"{name:<20} | {mse_error:<18.6e} | {final_error:<18.6e}")
#     print("=" * 70)
#     print(f"图像已成功保存至 rollout_results/ 目录下。")

import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from model.PINN_Tau import PINN_Tau
from model.PINN_Residual import PINN_Residual

torch.set_default_dtype(torch.float32)

# ==============================
# 1. 核心升维：6-DOF
# ==============================
DIM = 6

# ==============================
# 实验配置
# ==============================
MODEL_TRAIN_TYPE = "noisy"   # 模型是在哪个数据集上训练的 (保持 noisy 不变)
TEST_DATA_TYPE = "noisy"     # 🌟 重点！测试集必须用干净的物理基准数据
DT = 0.005                 # 数据集的采样间隔 (与生成数据时一致)
ROLLOUT_STEPS = 1000       # 向前预测的步数 (1000步 = 5秒)
START_IDX = 0              # 从测试集的哪个索引开始测试 

# ==============================
# 加载模型辅助函数
# ==============================
def load_model(model_class, path, device):
    model = model_class(DIM=DIM, device=device).to(device)
    # 允许在没有找到模型时报错，提醒检查路径
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# ==============================
# 动力学加速度求解与 RK4 积分
# ==============================
def get_ddq(model, q_in, dq_in, tau_in):
    # 必须赋予梯度以便计算 C 矩阵和势能梯度
    q = q_in.clone().detach().requires_grad_(True)
    dq = dq_in.clone().detach()
    tau = tau_in.clone().detach()

    M = model.M(q)
    C = model.C(q, dq)
    D = model.D(q, dq)
    V = model.potential_net(q)
    
    # 计算势能引发的保守力 (重力/弹性力)
    gradV = torch.autograd.grad(V.sum(), q, create_graph=False)[0]

    # 内部动态力组合
    tau_internal = (
        torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + 
        torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + 
        gradV
    )
    
    # 求解加速度 ddq
    ddq = torch.linalg.solve(M, tau - tau_internal)
    
    return ddq.detach() # 截断计算图，防止 OOM

def rk4_step(model, q, dq, tau, dt):
    # 张量维度的 RK4 积分
    ddq1 = get_ddq(model, q, dq, tau)
    
    q2 = q + 0.5 * dt * dq
    dq2 = dq + 0.5 * dt * ddq1
    ddq2 = get_ddq(model, q2, dq2, tau)
    
    q3 = q + 0.5 * dt * dq2
    dq3 = dq + 0.5 * dt * ddq2
    ddq3 = get_ddq(model, q3, dq3, tau)
    
    q4 = q + dt * dq3
    dq4 = dq + dt * ddq3
    ddq4 = get_ddq(model, q4, dq4, tau)

    q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
    dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
    
    return q_next, dq_next

# ==============================
# 能量计算 (网络眼中的能量)
# ==============================
def compute_energy(model, q_in, dq_in):
    q = q_in.clone().detach()
    dq = dq_in.clone().detach()
    
    M = model.M(q)
    V = model.potential_net(q)
    
    dq_vec = dq.unsqueeze(-1)
    kinetic_energy = 0.5 * torch.bmm(dq_vec.transpose(1, 2), torch.bmm(M, dq_vec)).squeeze(-1).squeeze(-1)
    
    total_energy = kinetic_energy + V
    return total_energy.detach().cpu().numpy()[0]

# ==============================
# 单个 GPU 的 Rollout 工作流
# ==============================
def rollout_worker(gpu_id, model_name, model_class, weights_path):
    device = torch.device(f"cuda:{gpu_id}")
    
    # 2. 修改数据集路径：确保指向 6-DOF 的测试集
    test_path = f"dataset_6dof/dataset_{TEST_DATA_TYPE}_test.pt"
    
    try:
        test_data = torch.load(test_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[GPU {gpu_id}] ❌ 找不到测试数据集: {test_path}")
        return None
    
    # 提取一段完整的 Ground Truth 轨迹
    q_true = test_data["q"][START_IDX : START_IDX + ROLLOUT_STEPS]
    dq_true = test_data["dq"][START_IDX : START_IDX + ROLLOUT_STEPS]
    tau_seq = test_data["tau"][START_IDX : START_IDX + ROLLOUT_STEPS]
    
    # 加载模型
    model = load_model(model_class, weights_path, device)
    
    # 初始化状态
    q_curr = q_true[0:1].clone()
    dq_curr = dq_true[0:1].clone()
    
    q_traj = []
    energy_traj = []
    
    # 设置专属进度条
    desc_str = f"[GPU {gpu_id}] 仿真 {model_name:<18}"
    pbar = tqdm(range(ROLLOUT_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
    # 执行时域演化
    for t in pbar:
        q_traj.append(q_curr.squeeze(0).cpu().numpy())
        energy_traj.append(compute_energy(model, q_curr, dq_curr))
        
        tau_curr = tau_seq[t:t+1]
        q_curr, dq_curr = rk4_step(model, q_curr, dq_curr, tau_curr, DT)
        
    pbar.close()
    
    return {
        'model_name': model_name,
        'q_pred': np.array(q_traj),
        'energy_pred': np.array(energy_traj),
        'q_true': q_true.cpu().numpy(),
        'tau_seq': tau_seq.cpu().numpy()
    }

# ==============================
# 主调度与绘图程序
# ==============================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    if not os.path.exists("rollout_results"):
        os.makedirs("rollout_results")
        
    print("\033[2J\033[H", end="")
    print("============================================================")
    print(f"🚀 开启 4 卡并行: 6-DOF Forward Rollout | 数据集: {TEST_DATA_TYPE.upper()}")
    print("============================================================\n")

    # 3. 强制指向 6dof_evo 权重文件夹
    tasks = [
        (0, "Residual_margin", PINN_Residual, f"models_6dof_evo/final_evo_Residual_margin_{MODEL_TRAIN_TYPE}_new.pth"),
        (1, "Residual_condition", PINN_Residual, f"models_6dof_evo/final_evo_Residual_condition_{MODEL_TRAIN_TYPE}_new.pth"),
        (2, "Tau_margin", PINN_Tau, f"models_6dof_evo/final_evo_Tau_margin_{MODEL_TRAIN_TYPE}_new.pth"),
        (3, "Tau_condition", PINN_Tau, f"models_6dof_evo/final_evo_Tau_condition_{MODEL_TRAIN_TYPE}_new.pth")
    ]
    
    start_time = time.time()
    
    # 开启进程池并发仿真
    with mp.Pool(processes=4) as pool:
        results = pool.starmap(rollout_worker, tasks)
        
    results = [r for r in results if r is not None]
    if not results:
        print("❌ 仿真失败，请检查数据集路径和模型权重路径！")
        exit()
        
    print("\n\n🎉 仿真计算全部完成！总耗时: {:.2f} 秒".format(time.time() - start_time))
    print("正在生成高分辨率物理演化图表...")

    # 数据解析
    time_axis = np.arange(ROLLOUT_STEPS) * DT
    q_true = results[0]['q_true'] # 基准真实轨迹
    
    results_q = {r['model_name']: r['q_pred'] for r in results}
    results_energy = {r['model_name']: r['energy_pred'] for r in results}
    
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd'] # 红, 绿, 蓝, 紫
    
    # ============================================================
    # 可视化 1: 轨迹追踪对比图 (4. 升级为 3x2 画布)
    # ============================================================
    plt.figure(figsize=(18, 18)) # 拉高画布以容纳 3 行
    for i in range(DIM):
        plt.subplot(3, 2, i+1)
        # 绘制真实轨迹
        plt.plot(time_axis, q_true[:, i], 'k--', linewidth=3, label="Ground Truth", zorder=10)
        
        # 绘制四个模型的预测轨迹
        for (name, q_pred), color in zip(results_q.items(), colors):
            plt.plot(time_axis, q_pred[:, i], color=color, alpha=0.85, linewidth=2, label=name)
            
        plt.title(f"Joint $q_{i+1}$ Trajectory Tracking", fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Angle/Position", fontsize=12)
        if i == 0:
            plt.legend(loc='best', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(f"rollout_results/Rollout_Trajectories_{TEST_DATA_TYPE}_6dof_last.png", dpi=300, bbox_inches='tight')

    # ============================================================
    # 可视化 2: 系统总能量演化图
    # ============================================================
    plt.figure(figsize=(12, 7))
    for (name, E_pred), color in zip(results_energy.items(), colors):
        plt.plot(time_axis, E_pred, color=color, linewidth=2.5, label=name)
        
    plt.title(f"6-DOF System Total Energy Evolution ($E = K + V$) - {TEST_DATA_TYPE.upper()} Data", fontsize=15, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Energy (Joules)", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"rollout_results/Rollout_Energy_{TEST_DATA_TYPE}_6dof_last.png", dpi=300, bbox_inches='tight')

    # ============================================================
    # 误差统计与评估报告
    # ============================================================
    print("\n" + "="*70)
    print(f"📊 连续 {ROLLOUT_STEPS} 步 ( {ROLLOUT_STEPS*DT} 秒 ) 积分误差报告")
    print("="*70)
    print(f"{'模型名称':<20} | {'全局轨迹 MSE':<15} | {'最终点偏移 MSE':<15}")
    print("-" * 70)
    
    for name, q_pred in results_q.items():
        mse_error = np.mean((q_pred - q_true)**2)
        final_error = np.mean((q_pred[-1] - q_true[-1])**2)
        print(f"{name:<20} | {mse_error:<18.6e} | {final_error:<18.6e}")
    print("=" * 70)
    print(f"图像已成功保存至 rollout_results/ 目录下。")