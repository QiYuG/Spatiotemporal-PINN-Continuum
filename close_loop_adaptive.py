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
# # 自适应控制实验配置
# # ==============================
# DT = 0.005
# TOTAL_STEPS = 1000  # 运行 5 秒钟
# DATASET_TYPE = "noisy"

# # 🌟 自适应 PD 增益基准与学习率 (Adaptive Gains)
# KP_BASE = 80.0     # 基础刚度 (调低基础刚度，让自适应律发挥作用)
# KD_BASE = 10.0     # 基础阻尼
# GAMMA_P = 400.0    # Kp 的自适应增长率 (对误差的敏感度)
# GAMMA_D = 50.0     # Kd 的自适应增长率
# KP_MAX = 500.0     # 增益上限，防止数值爆炸
# KD_MAX = 80.0

# CTRL_TASKS = [
#     (0, "PID_Only"),
#     (1, "CTC_Residual_margin"),
#     (2, "CTC_Tau_margin"),
#     (3, "CTC_Tau_condition")
# ]

# COLORS = {
#     "PID_Only": "#7f7f7f",             
#     "CTC_Residual_margin": "#d62728",  
#     "CTC_Tau_margin": "#1f77b4",       
#     "CTC_Tau_condition": "#9467bd"     
# }

# # ==============================
# # 🎯 期望轨迹生成器
# # ==============================
# def get_desired_trajectory(t, device):
#     A = torch.tensor([[0.5, -0.4, 0.3, -0.2]], device=device)
#     f = 0.5 
#     omega = 2 * np.pi * f
#     t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    
#     q_d = A * torch.sin(omega * t_tensor)
#     dq_d = A * omega * torch.cos(omega * t_tensor)
#     ddq_d = -A * (omega**2) * torch.sin(omega * t_tensor)
#     return q_d, dq_d, ddq_d

# # ==============================
# # 🤖 真实机器人解析动力学仿真器 (The Real Plant)
# # ==============================
# class RealRobotSimulator:
#     def __init__(self, device):
#         self.device = device
#         self.m1, self.m2 = 0.5, 0.5
#         self.J1, self.J2 = 0.02, 0.02
#         self.kappa_stiff = 3.0
#         self.damping = 0.8
#         self.FRICTION_MU = 0.5  
#         self.inject_friction = True 

#     def M_true(self, q):
#         k1, p1, k2, p2 = q[:,0], q[:,1], q[:,2], q[:,3]
#         batch = q.shape[0]
#         M = torch.zeros(batch, 4, 4, device=self.device)
#         M[:,0,0] = self.m1 + 0.2 * torch.cos(k1)
#         M[:,1,1] = self.J1 + 0.1 * torch.sin(k1)**2
#         M[:,2,2] = self.m2 + 0.2 * torch.cos(k2)
#         M[:,3,3] = self.J2 + 0.1 * torch.sin(k2)**2
#         M[:,0,2] = 0.05 * torch.cos(k1-k2)
#         M[:,2,0] = M[:,0,2]
#         return M

#     def C_true(self, q, dq):
#         q_clone = q.clone().detach().requires_grad_(True)
#         M = self.M_true(q_clone)
#         batch = q.shape[0]
#         C = torch.zeros(batch, 4, 4, device=self.device)
#         for k in range(4):
#             for j in range(4):
#                 for i in range(4):
#                     dM_ik_dqj = torch.autograd.grad(M[:,i,k].sum(), q_clone, retain_graph=True)[0][:,j]
#                     dM_ij_dqk = torch.autograd.grad(M[:,i,j].sum(), q_clone, retain_graph=True)[0][:,k]
#                     dM_jk_dqi = torch.autograd.grad(M[:,j,k].sum(), q_clone, retain_graph=True)[0][:,i]
#                     C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]
#         return C

#     def D_true(self, q, dq):
#         return self.damping * torch.eye(4, device=self.device).unsqueeze(0).repeat(q.shape[0], 1, 1)

#     def gradV_true(self, q):
#         grad = torch.zeros_like(q)
#         grad[:,0] = self.kappa_stiff * q[:,0]
#         grad[:,2] = self.kappa_stiff * q[:,2]
#         return grad

#     def get_ddq(self, q, dq, tau):
#         M = self.M_true(q)
#         C = self.C_true(q, dq)
#         D = self.D_true(q, dq)
#         gV = self.gradV_true(q)

#         tau_internal = torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + \
#                        torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + gV
        
#         if self.inject_friction:
#             tau_friction = self.FRICTION_MU * torch.tanh(10.0 * dq)
#             tau_internal += tau_friction
            
#         ddq = torch.linalg.solve(M, tau - tau_internal)
#         return ddq.detach()

#     def step(self, q, dq, tau, dt):
#         ddq1 = self.get_ddq(q, dq, tau)
#         q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
#         ddq2 = self.get_ddq(q2, dq2, tau)
#         q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
#         ddq3 = self.get_ddq(q3, dq3, tau)
#         q4, dq4 = q + dt * dq3, dq + dt * ddq3
#         ddq4 = self.get_ddq(q4, dq4, tau)

#         q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
#         dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
#         return q_next, dq_next

# # ==============================
# # 🎮 并行控制任务 Worker (带自适应律)
# # ==============================
# def evaluate_controller(args):
#     gpu_id, ctrl_name = args
#     device = torch.device(f"cuda:{gpu_id}")
    
#     robot = RealRobotSimulator(device)
    
#     model = None
#     if "CTC" in ctrl_name:
#         model_class = PINN_Tau if "Tau" in ctrl_name else PINN_Residual
#         model = model_class(DIM=DIM, device=device).to(device)
#         core_name = ctrl_name.replace("CTC_", "")
#         path = f"models_evo_final/final_evo_{core_name}_{DATASET_TYPE}.pth"
#         if not os.path.exists(path):
#             print(f"[GPU {gpu_id}] ❌ 找不到模型: {path}")
#             return args, None, None, None
#         model.load_state_dict(torch.load(path, map_location=device))
#         model.eval()

#     traj_q, traj_q_des, traj_tau = [], [], []
    
#     q_curr = torch.zeros(1, DIM, device=device)
#     dq_curr = torch.zeros(1, DIM, device=device)
    
#     desc_str = f"[GPU {gpu_id}] {ctrl_name[:20]:<20}"
#     pbar = tqdm(range(TOTAL_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
#     for step in pbar:
#         t = step * DT
#         q_d, dq_d, ddq_d = get_desired_trajectory(t, device)
        
#         e = q_d - q_curr
#         de = dq_d - dq_curr
        
#         # 🌟 核心：计算自适应 PD 增益 (矩阵维度独立调节)
#         Kp_dyn = KP_BASE + GAMMA_P * torch.abs(e)
#         Kd_dyn = KD_BASE + GAMMA_D * torch.abs(de)
        
#         # 限制自适应增益上限，防止积分发散
#         Kp_dyn = torch.clamp(Kp_dyn, max=KP_MAX)
#         Kd_dyn = torch.clamp(Kd_dyn, max=KD_MAX)
        
#         if ctrl_name == "PID_Only":
#             tau_cmd = Kp_dyn * e + Kd_dyn * de
#         else:
#             with torch.enable_grad():
#                 q_g = q_curr.clone().detach().requires_grad_(True)
#                 V = model.potential_net(q_g)
#                 gradV = torch.autograd.grad(V.sum(), q_g)[0]
                
#                 M_hat = model.M(q_g)
#                 C_hat = model.C(q_g, dq_curr)
#                 D_hat = model.D(q_g, dq_curr)
                
#                 # 融入动态自适应增益
#                 v_control = ddq_d + Kp_dyn * e + Kd_dyn * de
                
#                 tau_feedforward = torch.bmm(C_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + \
#                                   torch.bmm(D_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + gradV
                                  
#                 tau_cmd = torch.bmm(M_hat, v_control.unsqueeze(-1)).squeeze(-1) + tau_feedforward
#                 tau_cmd = tau_cmd.detach()

#         # 电机保护机制：夹断极端力矩输出
#         tau_cmd = torch.clamp(tau_cmd, min=-20.0, max=20.0)

#         q_curr, dq_curr = robot.step(q_curr, dq_curr, tau_cmd, DT)
        
#         if torch.any(torch.isnan(q_curr)) or torch.max(torch.abs(q_curr)) > 5.0:
#             pbar.set_postfix({'Status': "💥 Diverged"})
#             break
            
#         traj_q.append(q_curr.squeeze().cpu().numpy())
#         traj_q_des.append(q_d.squeeze().cpu().numpy())
#         traj_tau.append(tau_cmd.squeeze().cpu().numpy())

#     pbar.close()
#     return args, np.array(traj_q), np.array(traj_q_des), np.array(traj_tau)

# # ==============================
# # 主调度与绘图引擎
# # ==============================
# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
#     print("\033[2J\033[H", end="")
#     print("=====================================================================")
#     print("🚀 开启 4 卡并行: 自适应闭环轨迹跟踪控制 (Adaptive PD-CTC)")
#     print("=====================================================================\n")
    
#     start_time = time.time()
#     with mp.Pool(processes=4) as pool:
#         results_raw = pool.map(evaluate_controller, CTRL_TASKS)
        
#     print(f"\n🎉 4卡并行评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
#     results = {}
#     q_des_global = None
#     for args, t_q, t_q_des, t_tau in results_raw:
#         ctrl_name = args[1]
#         if t_q is not None and len(t_q) > 0:
#             results[ctrl_name] = {"q": t_q, "tau": t_tau}
#             q_des_global = t_q_des

#     if q_des_global is None:
#         print("致命错误：所有控制器在第一步就发散了！")
#         exit()

#     time_axis = np.arange(len(q_des_global)) * DT
    
#     # 绘图 1: 轨迹追踪对比
#     fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12))
#     fig1.suptitle("Adaptive Closed-Loop Trajectory Tracking Performance\n(Robustness Validation under Dynamic Gains)", fontsize=18, fontweight='bold')
    
#     for i in range(DIM):
#         row, col = i // 2, i % 2
#         ax = axs1[row, col]
#         ax.plot(time_axis, q_des_global[:, i], 'k--', linewidth=3.5, label="Reference ($q_d$)")
#         for ctrl_name, data in results.items():
#             length = len(data["q"])
#             ax.plot(time_axis[:length], data["q"][:, i], color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
#         ax.set_title(f"Joint $q_{i+1}$ Adaptive Tracking", fontsize=14)
#         ax.set_xlabel("Time (s)", fontsize=12)
#         ax.set_ylabel("Angle (rad)", fontsize=12)
#         ax.grid(True, linestyle='--', alpha=0.5)
#         if i == 0:
#             ax.legend(loc="upper right", fontsize=10)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig("Adaptive_Control_Tracking.png", dpi=300)

#     # 绘图 2: 控制力矩消耗
#     fig2 = plt.figure(figsize=(12, 6))
#     for ctrl_name, data in results.items():
#         length = len(data["tau"])
#         tau_norm = np.linalg.norm(data["tau"], axis=1)
#         plt.plot(time_axis[:length], tau_norm, color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
#     plt.title("Adaptive Control Effort (Actuation Torque Norm $||\\tau||_2$)", fontsize=16, fontweight='bold')
#     plt.xlabel("Time (s)", fontsize=14)
#     plt.ylabel("Torque Norm (Nm)", fontsize=14)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.savefig("Adaptive_Control_Effort.png", dpi=300)
    
#     print("✅ 自适应控制实验图表已保存为 Adaptive_Control_Tracking.png 和 Adaptive_Control_Effort.png")

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
# 1. 核心升维修改：系统自由度改为 6
# ==============================
DIM = 6

# ==============================
# 自适应控制实验配置
# ==============================
DT = 0.005
TOTAL_STEPS = 1000  # 运行 5 秒钟
DATASET_TYPE = "noisy"

# 🌟 自适应 PD 增益基准与学习率 (Adaptive Gains)
KP_BASE = 80.0     # 基础刚度
KD_BASE = 10.0     # 基础阻尼
GAMMA_P = 400.0    # Kp 的自适应增长率 (对误差的敏感度)
GAMMA_D = 50.0     # Kd 的自适应增长率
KP_MAX = 500.0     # 增益上限，防止数值爆炸
KD_MAX = 80.0

# 测试任务：1 个纯 PID 基准 + 4 个 6-DOF Evo 模型
CTRL_TASKS = [
    (0, "PID_Only"),
    (1, "CTC_Residual_margin"),
    (2, "CTC_Residual_condition"),
    (3, "CTC_Tau_margin"),
    (0, "CTC_Tau_condition") # 第 5 个任务分配给 GPU 0，多进程池会自动排队
]

COLORS = {
    "PID_Only": "#7f7f7f",             
    "CTC_Residual_margin": "#d62728",
    "CTC_Residual_condition": "#2ca02c",
    "CTC_Tau_margin": "#1f77b4",       
    "CTC_Tau_condition": "#9467bd"     
}

# ==============================
# 🎯 期望轨迹生成器 (扩展为 6 维)
# ==============================
def get_desired_trajectory(t, device):
    # 为 6 个关节设定不同的振幅与反相，模拟复杂的连续体蛇形运动
    A = torch.tensor([[0.5, -0.4, 0.4, -0.3, 0.3, -0.2]], device=device)
    f = 0.5 
    omega = 2 * np.pi * f
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    
    q_d = A * torch.sin(omega * t_tensor)
    dq_d = A * omega * torch.cos(omega * t_tensor)
    ddq_d = -A * (omega**2) * torch.sin(omega * t_tensor)
    return q_d, dq_d, ddq_d

# ==============================
# 🤖 真实机器人解析动力学仿真器 (扩展为 6-DOF)
# ==============================
class RealRobotSimulator:
    def __init__(self, device):
        self.device = device
        # 增加第三段的质量与惯量
        self.m1, self.m2, self.m3 = 0.5, 0.5, 0.5
        self.J1, self.J2, self.J3 = 0.02, 0.02, 0.02
        self.kappa_stiff = 3.0
        self.damping = 0.8
        self.FRICTION_MU = 0.5  
        self.inject_friction = True 

    def M_true(self, q):
        k1, p1, k2, p2, k3, p3 = q[:,0], q[:,1], q[:,2], q[:,3], q[:,4], q[:,5]
        batch = q.shape[0]
        M = torch.zeros(batch, 6, 6, device=self.device)
        
        # 对角线主惯量
        M[:,0,0] = self.m1 + 0.2 * torch.cos(k1)
        M[:,1,1] = self.J1 + 0.1 * torch.sin(k1)**2
        M[:,2,2] = self.m2 + 0.2 * torch.cos(k2)
        M[:,3,3] = self.J2 + 0.1 * torch.sin(k2)**2
        M[:,4,4] = self.m3 + 0.2 * torch.cos(k3)
        M[:,5,5] = self.J3 + 0.1 * torch.sin(k3)**2
        
        # 跨段非线性耦合 (基部、中部、末端之间的互相影响)
        M[:,0,2] = 0.05 * torch.cos(k1-k2)
        M[:,2,0] = M[:,0,2]
        M[:,2,4] = 0.05 * torch.cos(k2-k3)
        M[:,4,2] = M[:,2,4]
        M[:,0,4] = 0.02 * torch.cos(k1-k3) # 头尾微弱耦合
        M[:,4,0] = M[:,0,4]
        
        return M

    def C_true(self, q, dq):
        q_clone = q.clone().detach().requires_grad_(True)
        M = self.M_true(q_clone)
        batch = q.shape[0]
        C = torch.zeros(batch, 6, 6, device=self.device)
        # 自动扩展为 6x6x6 的克里斯托费尔符号计算
        for k in range(6):
            for j in range(6):
                for i in range(6):
                    dM_ik_dqj = torch.autograd.grad(M[:,i,k].sum(), q_clone, retain_graph=True)[0][:,j]
                    dM_ij_dqk = torch.autograd.grad(M[:,i,j].sum(), q_clone, retain_graph=True)[0][:,k]
                    dM_jk_dqi = torch.autograd.grad(M[:,j,k].sum(), q_clone, retain_graph=True)[0][:,i]
                    C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]
        return C

    def D_true(self, q, dq):
        return self.damping * torch.eye(6, device=self.device).unsqueeze(0).repeat(q.shape[0], 1, 1)

    def gradV_true(self, q):
        grad = torch.zeros_like(q)
        grad[:,0] = self.kappa_stiff * q[:,0]
        grad[:,2] = self.kappa_stiff * q[:,2]
        grad[:,4] = self.kappa_stiff * q[:,4] # 新增第三段重力/弹性恢复力
        return grad

    def get_ddq(self, q, dq, tau):
        M = self.M_true(q)
        C = self.C_true(q, dq)
        D = self.D_true(q, dq)
        gV = self.gradV_true(q)

        tau_internal = torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + \
                       torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + gV
        
        if self.inject_friction:
            # 6 个关节全量注入高频非线性摩擦
            tau_friction = self.FRICTION_MU * torch.tanh(10.0 * dq)
            tau_internal += tau_friction
            
        ddq = torch.linalg.solve(M, tau - tau_internal)
        return ddq.detach()

    def step(self, q, dq, tau, dt):
        ddq1 = self.get_ddq(q, dq, tau)
        q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
        ddq2 = self.get_ddq(q2, dq2, tau)
        q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
        ddq3 = self.get_ddq(q3, dq3, tau)
        q4, dq4 = q + dt * dq3, dq + dt * ddq3
        ddq4 = self.get_ddq(q4, dq4, tau)

        q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
        dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
        return q_next, dq_next

# ==============================
# 🎮 并行控制任务 Worker (带自适应律)
# ==============================
def evaluate_controller(args):
    gpu_id, ctrl_name = args
    device = torch.device(f"cuda:{gpu_id}")
    
    robot = RealRobotSimulator(device)
    
    model = None
    if "CTC" in ctrl_name:
        model_class = PINN_Tau if "Tau" in ctrl_name else PINN_Residual
        model = model_class(DIM=DIM, device=device).to(device)
        core_name = ctrl_name.replace("CTC_", "")
        
        # 指向 6dof_evo 权重路径
        path = f"models_6dof_evo/final_evo_{core_name}_{DATASET_TYPE}_new.pth"
        if not os.path.exists(path):
            print(f"[GPU {gpu_id}] ❌ 找不到模型: {path}")
            return args, None, None, None
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

    traj_q, traj_q_des, traj_tau = [], [], []
    
    q_curr = torch.zeros(1, DIM, device=device)
    dq_curr = torch.zeros(1, DIM, device=device)
    
    desc_str = f"[GPU {gpu_id}] {ctrl_name[:20]:<20}"
    pbar = tqdm(range(TOTAL_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
    for step in pbar:
        t = step * DT
        q_d, dq_d, ddq_d = get_desired_trajectory(t, device)
        
        e = q_d - q_curr
        de = dq_d - dq_curr
        
        # 🌟 核心：计算自适应 PD 增益 (矩阵维度独立调节)
        Kp_dyn = KP_BASE + GAMMA_P * torch.abs(e)
        Kd_dyn = KD_BASE + GAMMA_D * torch.abs(de)
        
        # 限制自适应增益上限，防止积分发散
        Kp_dyn = torch.clamp(Kp_dyn, max=KP_MAX)
        Kd_dyn = torch.clamp(Kd_dyn, max=KD_MAX)
        
        if ctrl_name == "PID_Only":
            tau_cmd = Kp_dyn * e + Kd_dyn * de
        else:
            with torch.enable_grad():
                q_g = q_curr.clone().detach().requires_grad_(True)
                V = model.potential_net(q_g)
                gradV = torch.autograd.grad(V.sum(), q_g)[0]
                
                M_hat = model.M(q_g)
                C_hat = model.C(q_g, dq_curr)
                D_hat = model.D(q_g, dq_curr)
                
                # 融入动态自适应增益
                v_control = ddq_d + Kp_dyn * e + Kd_dyn * de
                
                tau_feedforward = torch.bmm(C_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + \
                                  torch.bmm(D_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + gradV
                                  
                tau_cmd = torch.bmm(M_hat, v_control.unsqueeze(-1)).squeeze(-1) + tau_feedforward
                tau_cmd = tau_cmd.detach()

        # 电机保护机制：夹断极端力矩输出
        tau_cmd = torch.clamp(tau_cmd, min=-20.0, max=20.0)

        q_curr, dq_curr = robot.step(q_curr, dq_curr, tau_cmd, DT)
        
        if torch.any(torch.isnan(q_curr)) or torch.max(torch.abs(q_curr)) > 5.0:
            pbar.set_postfix({'Status': "💥 Diverged"})
            break
            
        traj_q.append(q_curr.squeeze().cpu().numpy())
        traj_q_des.append(q_d.squeeze().cpu().numpy())
        traj_tau.append(tau_cmd.squeeze().cpu().numpy())

    pbar.close()
    return args, np.array(traj_q), np.array(traj_q_des), np.array(traj_tau)

# ==============================
# 主调度与绘图引擎
# ==============================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("\033[2J\033[H", end="")
    print("=====================================================================")
    print("🚀 开启 4 卡并行: 6-DOF 高维自适应闭环轨迹跟踪控制 (Adaptive PD-CTC)")
    print("=====================================================================\n")
    
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        results_raw = pool.map(evaluate_controller, CTRL_TASKS)
        
    print(f"\n🎉 评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
    results = {}
    q_des_global = None
    for args, t_q, t_q_des, t_tau in results_raw:
        ctrl_name = args[1]
        if t_q is not None and len(t_q) > 0:
            results[ctrl_name] = {"q": t_q, "tau": t_tau}
            q_des_global = t_q_des

    if q_des_global is None:
        print("致命错误：所有控制器在第一步就发散了！")
        exit()

    time_axis = np.arange(len(q_des_global)) * DT
    
    # 绘图 1: 6-DOF 轨迹追踪对比 (3行2列)
    fig1, axs1 = plt.subplots(3, 2, figsize=(16, 18))
    fig1.suptitle("6-DOF Adaptive Closed-Loop Trajectory Tracking Performance\n(Validation of Evolutionary High-Dimensional Models)", fontsize=20, fontweight='bold')
    
    for i in range(DIM):
        row, col = i // 2, i % 2
        ax = axs1[row, col]
        ax.plot(time_axis, q_des_global[:, i], 'k--', linewidth=3.5, label="Reference ($q_d$)")
        for ctrl_name, data in results.items():
            length = len(data["q"])
            ax.plot(time_axis[:length], data["q"][:, i], color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
        ax.set_title(f"Joint $q_{i+1}$ Adaptive Tracking", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (rad)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Adaptive_Control_Tracking_6DOF_new.png", dpi=300)

    # 绘图 2: 控制力矩消耗
    fig2 = plt.figure(figsize=(12, 6))
    for ctrl_name, data in results.items():
        length = len(data["tau"])
        tau_norm = np.linalg.norm(data["tau"], axis=1)
        plt.plot(time_axis[:length], tau_norm, color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
    plt.title("6-DOF Adaptive Control Effort (Actuation Torque Norm $||\\tau||_2$)", fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Torque Norm (Nm)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("Adaptive_Control_Effort_6DOF_new.png", dpi=300)
    
    print("✅ 6-DOF 闭环自适应控制实验图表已保存为 Adaptive_Control_Tracking_6DOF.png 和 Adaptive_Control_Effort_6DOF.png")