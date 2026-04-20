# import torch
# import torch.multiprocessing as mp
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import time
# from tqdm import tqdm

# from model.PINN_Tau import PINN_Tau
# from model.PINN_Residual import PINN_Residual

# torch.set_default_dtype(torch.float32)
# DIM = 4

# # ==============================
# # 测试配置
# # ==============================
# DATASET_TYPE = "noisy"
# TEST_PATH = f"dataset/dataset_{DATASET_TYPE}_test.pt"

# DT = 0.005
# TOTAL_STEPS = 1000       # 仿真 5 秒
# DISTURBANCE_STEP = 500   # 在 t = 2.5s 注入扰动

# # 💥 巨大的瞬时速度脉冲扰动 (足以将状态推向边缘)
# DQ_DISTURBANCE = [2.0, -2.0, 2.0, -2.0]  

# # 模型路径配置
# NO_EVO_DIR = "models"
# EVO_DIR = "models_evo_final"

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
# # 极其强壮的安全积分器 (防爆版)
# # ==============================
# def rk4_step_robust(model, q_in, dq_in, tau_in, dt):
#     q = q_in.clone().detach()
#     dq = dq_in.clone().detach()
#     tau = tau_in.clone().detach()

#     def get_ddq(q_t, dq_t):
#         with torch.enable_grad():
#             q_g = q_t.clone().detach().requires_grad_(True)
#             V = model.potential_net(q_g)
#             gradV = torch.autograd.grad(V.sum(), q_g)[0]
            
#             M = model.M(q_g)
#             C = model.C(q_g, dq_t)
#             D = model.D(q_g, dq_t)
            
#             tau_internal = torch.bmm(C, dq_t.unsqueeze(-1)).squeeze(-1) + \
#                            torch.bmm(D, dq_t.unsqueeze(-1)).squeeze(-1) + gradV
            
#             # 🛡️ 尝试求解线性方程组，如果矩阵奇异则抛出异常
#             try:
#                 ddq = torch.linalg.solve(M, tau - tau_internal)
#             except RuntimeError:
#                 raise ValueError("Matrix Singular")
#         return ddq.detach()

#     try:
#         ddq1 = get_ddq(q, dq)
#         q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
#         ddq2 = get_ddq(q2, dq2)
#         q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
#         ddq3 = get_ddq(q3, dq3)
#         q4, dq4 = q + dt * dq3, dq + dt * ddq3
#         ddq4 = get_ddq(q4, dq4)

#         q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
#         dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
        
#         # 🛡️ 防止数值溢出爆炸
#         if torch.any(torch.isnan(q_next)) or torch.max(torch.abs(q_next)) > 1e4:
#             raise ValueError("Numerical Explosion")
            
#         return q_next, dq_next
#     except ValueError:
#         return None, None

# # ==============================
# # 并行评测 Worker (加入了炫酷的多进程进度条)
# # ==============================
# def evaluate_model(args):
#     model_name, is_evo, device_id = args
#     device = torch.device(f"cuda:{device_id}")
    
#     # 1. 寻找并加载模型权重
#     model_class = MODEL_DICT[model_name]
#     model = model_class(DIM=DIM, device=device).to(device)
    
#     if is_evo:
#         path = f"{EVO_DIR}/final_evo_{model_name}_{DATASET_TYPE}.pth"
#     else:
#         path = f"{NO_EVO_DIR}/final_{model_name}_{DATASET_TYPE}.pth"
#         if not os.path.exists(path):
#             path = f"{NO_EVO_DIR}/best_{model_name}_{DATASET_TYPE}.pth" # Fallback
            
#     if not os.path.exists(path):
#         return args, None
        
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()
    
#     # 2. 准备测试数据 (取第0条轨迹)
#     test_data = torch.load(TEST_PATH, weights_only=False)
#     STEPS_PER_TRAJ = 2000
#     q_true = test_data["q"][:STEPS_PER_TRAJ].to(device)
#     dq_true = test_data["dq"][:STEPS_PER_TRAJ].to(device)
#     tau_seq = test_data["tau"][:STEPS_PER_TRAJ].to(device)
    
#     q_curr = q_true[0:1].clone()
#     dq_curr = dq_true[0:1].clone()
    
#     trajectory = []
#     status = "Stable"
    
#     # 📊 精美进度条设置 (利用 device_id 分配到不同的终端行)
#     evo_tag = "Evo" if is_evo else "NoEvo"
#     desc_str = f"[GPU {device_id}] {model_name[:15]:<15} | {evo_tag:<5}"
#     pbar = tqdm(total=TOTAL_STEPS, desc=desc_str, position=device_id, leave=True)
    
#     # 3. 开启受扰动的前向仿真
#     for step in range(TOTAL_STEPS):
#         # 💥 注入扰动
#         if step == DISTURBANCE_STEP:
#             dq_dist = torch.tensor([DQ_DISTURBANCE], device=device)
#             dq_curr = dq_curr + dq_dist
#             pbar.set_postfix({'Event': 'Disturbance Injected!'})
            
#         q_curr, dq_curr = rk4_step_robust(model, q_curr, dq_curr, tau_seq[step:step+1], DT)
        
#         if q_curr is None:
#             # 模型爆炸了，后续全部填充 NaN
#             trajectory.extend([[np.nan]*DIM] * (TOTAL_STEPS - step))
#             status = "💥 EXPLODED"
#             pbar.set_postfix({'Status': status})
#             pbar.update(TOTAL_STEPS - step) # 进度条瞬间拉满
#             break
            
#         trajectory.append(q_curr.squeeze().cpu().numpy())
#         pbar.update(1)
        
#     if status == "Stable":
#         pbar.set_postfix({'Status': '✅ Survived'})
        
#     pbar.close()
#     return args, np.array(trajectory)

# # ==============================
# # 绘图函数
# # ==============================
# def plot_results(results_dict, is_evo, ground_truth):
#     fig, axs = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle(f"Robustness Test against Impulse Disturbance at t=2.5s\n({'WITH' if is_evo else 'WITHOUT'} Evolution Loss)", fontsize=18, fontweight='bold')
    
#     time_axis = np.arange(TOTAL_STEPS) * DT
    
#     for i in range(DIM):
#         row, col = i // 2, i % 2
#         ax = axs[row, col]
        
#         # 画真实轨迹参考线
#         ax.plot(time_axis, ground_truth[:TOTAL_STEPS, i], 'k--', linewidth=2.5, label='Ground Truth (Unperturbed)')
        
#         # 标记扰动点
#         ax.axvline(x=2.5, color='red', linestyle=':', alpha=0.7, label='Impulse Disturbance')
        
#         for model_name, traj in results_dict.items():
#             if traj is not None:
#                 ax.plot(time_axis, traj[:, i], color=COLORS[model_name], linewidth=2.0, alpha=0.8, label=model_name)
                
#         ax.set_title(f"Joint $q_{i+1}$ Trajectory", fontsize=14)
#         ax.set_xlabel("Time (s)", fontsize=12)
#         ax.set_ylabel("Angle (rad)", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.5)
#         if i == 0:
#             ax.legend(loc='upper right', fontsize=10)
            
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     suffix = "Evo" if is_evo else "NoEvo"
#     plt.savefig(f"Robustness_Test_{suffix}.png", dpi=300)
#     print(f"\n✅ 图表已保存为: Robustness_Test_{suffix}.png")

# # ==============================
# # 主调度程序
# # ==============================
# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
    
#     # 构建 8 个任务 (充分分配给 4 张 GPU)
#     tasks = []
#     gpu_id = 0
#     for is_evo in [False, True]:
#         for model_name in MODEL_DICT.keys():
#             tasks.append((model_name, is_evo, gpu_id % 4))
#             gpu_id += 1
            
#     print("\033[2J\033[H", end="")
#     print("=====================================================================")
#     print("🚀 开启 4 卡并行: 系统抗扰动鲁棒性极限测试 (带有实时状态监测)")
#     print("=====================================================================\n")
    
#     start_time = time.time()
    
#     with mp.Pool(processes=4) as pool:
#         results_raw = pool.map(evaluate_model, tasks)
        
#     print(f"\n\n🎉 评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
#     # 分离数据并绘图
#     res_no_evo = {}
#     res_evo = {}
    
#     for args, traj in results_raw:
#         model_name, is_evo, _ = args
#         if traj is not None:
#             if is_evo:
#                 res_evo[model_name] = traj
#             else:
#                 res_no_evo[model_name] = traj
            
#     # 获取无扰动的真实基准轨迹用于参考
#     test_data = torch.load(TEST_PATH, weights_only=False)
#     q_truth = test_data["q"][:TOTAL_STEPS].cpu().numpy()
            
#     print("正在渲染最终绝杀对比图...")
#     plot_results(res_no_evo, is_evo=False, ground_truth=q_truth)
#     plot_results(res_evo, is_evo=True, ground_truth=q_truth)

import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
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
# 测试配置
# ==============================
DATASET_TYPE = "noisy"
# ⚠️ 注意：请确保这里的路径指向你真正的 6-DOF 测试集！
TEST_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_test.pt"

DT = 0.005
TOTAL_STEPS = 1000       # 仿真 5 秒
DISTURBANCE_STEP = 500   # 在 t = 2.5s 注入极强扰动

# 💥 6维巨大的瞬时速度脉冲扰动 (足以将高维状态推向奇异边缘)
DQ_DISTURBANCE = [2.0, -2.0, 2.0, -2.0, 2.0, -2.0]  

# 只加载 6-DOF Evo 终极模型
EVO_DIR = "models_6dof_evo"

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

# ==============================
# 极其强壮的安全积分器 (防爆版)
# ==============================
def rk4_step_robust(model, q_in, dq_in, tau_in, dt):
    q = q_in.clone().detach()
    dq = dq_in.clone().detach()
    tau = tau_in.clone().detach()

    def get_ddq(q_t, dq_t):
        with torch.enable_grad():
            q_g = q_t.clone().detach().requires_grad_(True)
            V = model.potential_net(q_g)
            gradV = torch.autograd.grad(V.sum(), q_g)[0]
            
            M = model.M(q_g)
            C = model.C(q_g, dq_t)
            D = model.D(q_g, dq_t)
            
            tau_internal = torch.bmm(C, dq_t.unsqueeze(-1)).squeeze(-1) + \
                           torch.bmm(D, dq_t.unsqueeze(-1)).squeeze(-1) + gradV
            
            # 🛡️ 尝试求解线性方程组，如果矩阵奇异则抛出异常
            try:
                ddq = torch.linalg.solve(M, tau - tau_internal)
            except RuntimeError:
                raise ValueError("Matrix Singular")
        return ddq.detach()

    try:
        ddq1 = get_ddq(q, dq)
        q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
        ddq2 = get_ddq(q2, dq2)
        q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
        ddq3 = get_ddq(q3, dq3)
        q4, dq4 = q + dt * dq3, dq + dt * ddq3
        ddq4 = get_ddq(q4, dq4)

        q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
        dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
        
        # 🛡️ 防止数值溢出爆炸
        if torch.any(torch.isnan(q_next)) or torch.max(torch.abs(q_next)) > 1e4:
            raise ValueError("Numerical Explosion")
            
        return q_next, dq_next
    except ValueError:
        return None, None

# ==============================
# 并行评测 Worker (6-DOF 专属)
# ==============================
def evaluate_model(args):
    model_name, device_id = args
    device = torch.device(f"cuda:{device_id}")
    
    # 1. 寻找并加载模型权重
    model_class = MODEL_DICT[model_name]
    model = model_class(DIM=DIM, device=device).to(device)
    
    path = f"{EVO_DIR}/final_evo_{model_name}_{DATASET_TYPE}_new.pth"
            
    if not os.path.exists(path):
        return args, None
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    # 2. 准备测试数据 (取第0条轨迹)
    try:
        test_data = torch.load(TEST_PATH, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[GPU {device_id}] ❌ 找不到测试数据集: {TEST_PATH}")
        return args, None

    STEPS_PER_TRAJ = 2000
    q_true = test_data["q"][:STEPS_PER_TRAJ]
    dq_true = test_data["dq"][:STEPS_PER_TRAJ]
    tau_seq = test_data["tau"][:STEPS_PER_TRAJ]
    
    q_curr = q_true[0:1].clone()
    dq_curr = dq_true[0:1].clone()
    
    trajectory = []
    status = "Stable"
    
    desc_str = f"[GPU {device_id}] {model_name[:15]:<15}"
    pbar = tqdm(total=TOTAL_STEPS, desc=desc_str, position=device_id, leave=True)
    
    # 3. 开启受扰动的前向仿真
    for step in range(TOTAL_STEPS):
        # 💥 注入扰动
        if step == DISTURBANCE_STEP:
            dq_dist = torch.tensor([DQ_DISTURBANCE], device=device)
            dq_curr = dq_curr + dq_dist
            pbar.set_postfix({'Event': 'Disturbance Injected!'})
            
        q_curr, dq_curr = rk4_step_robust(model, q_curr, dq_curr, tau_seq[step:step+1], DT)
        
        if q_curr is None:
            trajectory.extend([[np.nan]*DIM] * (TOTAL_STEPS - step))
            status = "💥 EXPLODED"
            pbar.set_postfix({'Status': status})
            pbar.update(TOTAL_STEPS - step) 
            break
            
        trajectory.append(q_curr.squeeze().cpu().numpy())
        pbar.update(1)
        
    if status == "Stable":
        pbar.set_postfix({'Status': '✅ Survived'})
        
    pbar.close()
    return args, np.array(trajectory)

# ==============================
# 绘图函数 (升级为 3x2 画布)
# ==============================
def plot_results(results_dict, ground_truth):
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f"6-DOF Robustness Test against Extreme Impulse Disturbance at t=2.5s\n(Validation of Evolutionary Models)", fontsize=20, fontweight='bold')
    
    time_axis = np.arange(TOTAL_STEPS) * DT
    
    for i in range(DIM):
        row, col = i // 2, i % 2
        ax = axs[row, col]
        
        # 画真实轨迹参考线
        ax.plot(time_axis, ground_truth[:TOTAL_STEPS, i], 'k--', linewidth=2.5, label='Ground Truth (Unperturbed)')
        
        # 标记扰动点
        ax.axvline(x=2.5, color='red', linestyle=':', alpha=0.7, label='Impulse Disturbance')
        
        for model_name, traj in results_dict.items():
            if traj is not None:
                ax.plot(time_axis, traj[:, i], color=COLORS[model_name], linewidth=2.0, alpha=0.8, label=model_name)
                
        ax.set_title(f"Joint $q_{i+1}$ Trajectory", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (rad)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "Robustness_Test_6DOF_Evo_new.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 6-DOF 极限抗压图表已保存为: {save_path}")

# ==============================
# 主调度程序
# ==============================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 构建 4 个任务 (分配给 4 张 GPU)
    tasks = []
    for idx, model_name in enumerate(MODEL_DICT.keys()):
        tasks.append((model_name, idx % 4))
            
    print("\033[2J\033[H", end="")
    print("=====================================================================")
    print("🚀 开启 4 卡并行: 6-DOF 纯 Evo 模型极限抗扰动鲁棒性测试")
    print("=====================================================================\n")
    
    start_time = time.time()
    
    with mp.Pool(processes=4) as pool:
        results_raw = pool.map(evaluate_model, tasks)
        
    print(f"\n\n🎉 评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
    # 分离数据
    res_evo = {}
    for args, traj in results_raw:
        model_name, _ = args
        if traj is not None:
            res_evo[model_name] = traj
            
    if not res_evo:
        print("❌ 所有模型都没找到或全部在第一步崩溃了，请检查权重路径！")
        exit()
            
    # 获取无扰动的真实基准轨迹用于参考
    test_data = torch.load(TEST_PATH, map_location='cpu', weights_only=False)
    q_truth = test_data["q"][:TOTAL_STEPS].numpy()
            
    print("正在渲染最终绝杀对比图...")
    plot_results(res_evo, ground_truth=q_truth)