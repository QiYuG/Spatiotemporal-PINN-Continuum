# import torch
# import torch.multiprocessing as mp
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import time

# from model.PINN_Tau import PINN_Tau
# from model.PINN_Residual import PINN_Residual

# torch.set_default_dtype(torch.float32)
# DIM = 4

# # ==============================
# # 测试配置
# # ==============================
# DATASET_TYPE = "noisy"
# DT = 0.005
# TOTAL_STEPS = 1000  # 观察 5 秒的自由落体衰减

# MODEL_DICT = {
#     "Residual_margin": PINN_Residual,
#     "Residual_condition": PINN_Residual,
#     "Tau_margin": PINN_Tau,
#     "Tau_condition": PINN_Tau
# }

# COLORS = {
#     "Residual_margin": "#d62728",    
#     "Residual_condition": "#2ca02c", 
#     "Tau_margin": "#1f77b4",         
#     "Tau_condition": "#9467bd"       
# }

# # ==============================
# # 鲁棒积分器 (处理 tau=0 的自由演化)
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
        
#         if torch.any(torch.isnan(q_next)) or torch.max(torch.abs(q_next)) > 1e3:
#             raise ValueError("Numerical Explosion")
#         return q_next, dq_next
#     except ValueError:
#         return None, None

# # ==============================
# # 4卡并行 Worker
# # ==============================
# def evaluate_free_fall(args):
#     model_name, is_evo, gpu_id = args
#     device = torch.device(f"cuda:{gpu_id}")
    
#     # 加载模型
#     model_class = MODEL_DICT[model_name]
#     model = model_class(DIM=DIM, device=device).to(device)
    
#     if is_evo:
#         path = f"models_evo_final/final_evo_{model_name}_{DATASET_TYPE}.pth"
#     else:
#         path = f"models/final_{model_name}_{DATASET_TYPE}.pth"
#         if not os.path.exists(path):
#             path = f"models/best_{model_name}_{DATASET_TYPE}.pth"
            
#     if not os.path.exists(path):
#         print(f"[GPU {gpu_id}] ⚠️ 找不到模型权重: {path}")
#         return args, None, None
        
#     model.load_state_dict(torch.load(path, map_location=device))
#     model.eval()
    
#     # 🌟 初始状态设定：抬高到一个初始姿态，初始速度为 0，力矩为 0
#     q_curr = torch.tensor([[0.5, -0.5, 0.4, -0.4]], device=device)
#     dq_curr = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
#     zero_tau = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    
#     traj_q, traj_dq = [], []
#     evo_tag = "Evo" if is_evo else "No-Evo"
#     print(f"[GPU {gpu_id}] 🚀 正在仿真: {model_name:<20} | {evo_tag}")
    
#     for _ in range(TOTAL_STEPS):
#         q_curr, dq_curr = rk4_step_robust(model, q_curr, dq_curr, zero_tau, DT)
#         if q_curr is None:
#             break
#         traj_q.append(q_curr.squeeze().cpu().numpy())
#         traj_dq.append(dq_curr.squeeze().cpu().numpy())
        
#     if len(traj_q) == TOTAL_STEPS:
#         print(f"[GPU {gpu_id}] ✅ 仿真完成: {model_name:<20} | {evo_tag}")
#         return args, np.array(traj_q), np.array(traj_dq)
#     else:
#         print(f"[GPU {gpu_id}] ❌ 仿真崩溃: {model_name:<20} | {evo_tag}")
#         return args, None, None

# # ==============================
# # 主调度程序
# # ==============================
# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
#     print("\033[2J\033[H", end="")
#     print("=====================================================================")
#     print("🚀 开启 4 卡并行: 零力矩自由衰减对比极限测试 (Free-Fall)")
#     print("=====================================================================\n")
    
#     # 构建 8 个任务，轮询分配给 GPU 0~3
#     tasks = []
#     gpu_id = 0
#     for is_evo in [False, True]:
#         for model_name in MODEL_DICT.keys():
#             tasks.append((model_name, is_evo, gpu_id % 4))
#             gpu_id += 1
            
#     start_time = time.time()
    
#     with mp.Pool(processes=4) as pool:
#         results_raw = pool.map(evaluate_free_fall, tasks)
        
#     print(f"\n🎉 4卡并行评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
#     # 整理结果用于绘图
#     results_q = {False: {}, True: {}}
#     results_dq = {False: {}, True: {}}
    
#     for args, traj_q, traj_dq in results_raw:
#         model_name, is_evo, _ = args
#         if traj_q is not None:
#             results_q[is_evo][model_name] = traj_q
#             results_dq[is_evo][model_name] = traj_dq

#     # ==============================
#     # 绘制 2x2 绝美对比图
#     # ==============================
#     print("正在渲染自由衰减相平面图...")
#     fig, axs = plt.subplots(2, 2, figsize=(16, 12))
#     fig.suptitle("Unforced Free-Fall Dynamics (Zero-Torque Response)\nAblation on Evolution Loss", fontsize=20, fontweight='bold')
    
#     time_axis = np.arange(TOTAL_STEPS) * DT
#     row_titles = ["Without Evolution Loss (Single-step PINN)", "With Evolution Loss (Our Full Framework)"]
    
#     for row, is_evo in enumerate([False, True]):
#         ax_time = axs[row, 0]
#         for model_name, traj in results_q[is_evo].items():
#             ax_time.plot(time_axis, traj[:, 0], color=COLORS[model_name], linewidth=2.5, alpha=0.8, label=model_name)
        
#         ax_time.set_title(f"[{row_titles[row]}]\nTime Domain: Joint $q_1$ Damped Oscillation", fontsize=14)
#         ax_time.set_xlabel("Time (s)", fontsize=12)
#         ax_time.set_ylabel("Angle $q_1$ (rad)", fontsize=12)
#         ax_time.grid(True, linestyle='--', alpha=0.5)
#         ax_time.legend(loc="upper right")
        
#         ax_phase = axs[row, 1]
#         for model_name in results_q[is_evo].keys():
#             q_traj = results_q[is_evo][model_name][:, 0]
#             dq_traj = results_dq[is_evo][model_name][:, 0]
#             ax_phase.plot(q_traj, dq_traj, color=COLORS[model_name], linewidth=2.0, alpha=0.8, label=model_name)
#             ax_phase.scatter(q_traj[0], dq_traj[0], color='black', marker='*', s=150, zorder=10)
            
#         # 🐛 Bug Fix: 在 f-string 中使用 {{q}} 来正确渲染 LaTeX 的 \dot{q}
#         ax_phase.set_title(f"[{row_titles[row]}]\nPhase Portrait: Joint 1 ($q_1$ vs $\dot{{q}}_1$)", fontsize=14)
#         ax_phase.set_xlabel("Angle $q_1$ (rad)", fontsize=12)
        
#         # 这里的不是 f-string，所以可以直接用 \dot{q}
#         ax_phase.set_ylabel(r"Angular Velocity $\dot{q}_1$ (rad/s)", fontsize=12)
#         ax_phase.axhline(0, color='gray', linestyle=':', linewidth=1.5)
#         ax_phase.axvline(0, color='gray', linestyle=':', linewidth=1.5)
#         ax_phase.grid(True, linestyle='--', alpha=0.5)
#         ax_phase.legend(loc="upper right")
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig("Free_Fall_Comparison.png", dpi=300)
#     print("✅ 完美！自由衰减对比图已保存为 Free_Fall_Comparison.png")

import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from model.PINN_Tau import PINN_Tau
from model.PINN_Residual import PINN_Residual

torch.set_default_dtype(torch.float32)
DIM = 6  # 确认 6-DOF

# ==============================
# 测试配置
# ==============================
DATASET_TYPE = "noisy"
DT = 0.005
TOTAL_STEPS = 1000  # 观察 5 秒的自由落体衰减

MODEL_DICT = {
    "Residual_margin": PINN_Residual,
    "Residual_condition": PINN_Residual,
    "Tau_margin": PINN_Tau,
    "Tau_condition": PINN_Tau
}

COLORS = {
    "Residual_margin": "#d62728",    
    "Residual_condition": "#2ca02c", 
    "Tau_margin": "#1f77b4",         
    "Tau_condition": "#9467bd"       
}

# ==============================
# 鲁棒积分器 (处理 tau=0 的自由演化)
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
        
        if torch.any(torch.isnan(q_next)) or torch.max(torch.abs(q_next)) > 1e3:
            raise ValueError("Numerical Explosion")
        return q_next, dq_next
    except ValueError:
        return None, None

# ==============================
# 4卡并行 Worker (只读取 evo 模型)
# ==============================
def evaluate_free_fall(args):
    model_name, gpu_id = args
    device = torch.device(f"cuda:{gpu_id}")
    
    # 加载模型
    model_class = MODEL_DICT[model_name]
    model = model_class(DIM=DIM, device=device).to(device)
    
    # 强制指向 6dof evo 文件夹
    path = f"models_6dof_evo/final_evo_{model_name}_{DATASET_TYPE}_new.pth"
            
    if not os.path.exists(path):
        print(f"[GPU {gpu_id}] ⚠️ 找不到模型权重: {path}")
        return args, None, None
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    # 6-DOF 初始状态设定
    q_curr = torch.tensor([[0.5, -0.5, 0.4, -0.4, 0.3, -0.3]], device=device)
    dq_curr = torch.zeros((1, 6), device=device)
    zero_tau = torch.zeros((1, 6), device=device)
    
    traj_q, traj_dq = [], []
    print(f"[GPU {gpu_id}] 🚀 正在仿真: {model_name:<20}")
    
    for _ in range(TOTAL_STEPS):
        q_curr, dq_curr = rk4_step_robust(model, q_curr, dq_curr, zero_tau, DT)
        if q_curr is None:
            break
        traj_q.append(q_curr.squeeze().cpu().numpy())
        traj_dq.append(dq_curr.squeeze().cpu().numpy())
        
    if len(traj_q) == TOTAL_STEPS:
        print(f"[GPU {gpu_id}] ✅ 仿真完成: {model_name:<20}")
        return args, np.array(traj_q), np.array(traj_dq)
    else:
        print(f"[GPU {gpu_id}] ❌ 仿真崩溃: {model_name:<20}")
        return args, None, None

# ==============================
# 主调度程序
# ==============================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("\033[2J\033[H", end="")
    print("=====================================================================")
    print("🚀 开启并行: 6-DOF 纯 Evo 模型零力矩自由衰减极限测试")
    print("=====================================================================\n")
    
    # 构建 4 个任务，分配给 GPU 0~3
    tasks = [(model_name, idx % 4) for idx, model_name in enumerate(MODEL_DICT.keys())]
            
    start_time = time.time()
    
    with mp.Pool(processes=4) as pool:
        results_raw = pool.map(evaluate_free_fall, tasks)
        
    print(f"\n🎉 评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
    # 整理结果用于绘图
    results_q = {}
    results_dq = {}
    
    for args, traj_q, traj_dq in results_raw:
        model_name, _ = args
        if traj_q is not None:
            results_q[model_name] = traj_q
            results_dq[model_name] = traj_dq

    # ==============================
    # 绘制 1x2 绝美对比图
    # ==============================
    print("正在渲染自由衰减相平面图...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Unforced Free-Fall Dynamics (6-DOF Zero-Torque Response with Evo Loss)", fontsize=18, fontweight='bold')
    
    time_axis = np.arange(TOTAL_STEPS) * DT
    
    # 想要观察的关节索引 (0 对应 q1, 5 对应末端 q6)
    JOINT_IDX = 0 
    
    # 左图：时间域衰减曲线
    ax_time = axs[0]
    for model_name, traj in results_q.items():
        ax_time.plot(time_axis, traj[:, JOINT_IDX], color=COLORS[model_name], linewidth=2.5, alpha=0.8, label=model_name)
    
    ax_time.set_title(f"Time Domain: Joint $q_{JOINT_IDX+1}$ Damped Oscillation", fontsize=14)
    ax_time.set_xlabel("Time (s)", fontsize=12)
    ax_time.set_ylabel(f"Angle $q_{JOINT_IDX+1}$ (rad)", fontsize=12)
    ax_time.grid(True, linestyle='--', alpha=0.5)
    ax_time.legend(loc="upper right")
    
    # 右图：相平面图
    ax_phase = axs[1]
    for model_name in results_q.keys():
        q_traj = results_q[model_name][:, JOINT_IDX]
        dq_traj = results_dq[model_name][:, JOINT_IDX]
        ax_phase.plot(q_traj, dq_traj, color=COLORS[model_name], linewidth=2.0, alpha=0.8, label=model_name)
        ax_phase.scatter(q_traj[0], dq_traj[0], color='black', marker='*', s=150, zorder=10)
        
    ax_phase.set_title(f"Phase Portrait: Joint {JOINT_IDX+1} ($q_{JOINT_IDX+1}$ vs $\dot{{q}}_{JOINT_IDX+1}$)", fontsize=14)
    ax_phase.set_xlabel(f"Angle $q_{JOINT_IDX+1}$ (rad)", fontsize=12)
    ax_phase.set_ylabel(rf"Angular Velocity $\dot{{q}}_{JOINT_IDX+1}$ (rad/s)", fontsize=12)
    ax_phase.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    ax_phase.axvline(0, color='gray', linestyle=':', linewidth=1.5)
    ax_phase.grid(True, linestyle='--', alpha=0.5)
    ax_phase.legend(loc="upper right")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"Free_Fall_Comparison_6DOF_q{JOINT_IDX+1}_new.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 完美！自由衰减图已保存为 {save_path}")