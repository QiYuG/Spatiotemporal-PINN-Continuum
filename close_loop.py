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
DIM = 4

# ==============================
# 控制实验配置
# ==============================
DT = 0.005
TOTAL_STEPS = 1000  # 运行 5 秒钟的控制追踪
DATASET_TYPE = "noisy"

# PD 反馈增益 (调高以保证底层追踪性能)
KP = 150.0  # 比例增益
KD = 25.0   # 微分增益

# 注意这里统一采用了小写的 margin/condition 以匹配你之前生成的权重文件名
CTRL_TASKS = [
    (0, "PID_Only"),
    (1, "CTC_Residual_margin"),
    (2, "CTC_Tau_margin"),
    (3, "CTC_Tau_condition")
]

COLORS = {
    "PID_Only": "#7f7f7f",             
    "CTC_Residual_margin": "#d62728",  
    "CTC_Tau_margin": "#1f77b4",       
    "CTC_Tau_condition": "#9467bd"     
}

# ==============================
# 🎯 期望轨迹生成器 (平滑 Lissajous / 正弦波)
# ==============================
def get_desired_trajectory(t, device):
    """
    生成要求机器人在 5 秒内跟踪的光滑手术探查轨迹
    """
    A = torch.tensor([[0.5, -0.4, 0.3, -0.2]], device=device)
    f = 0.5 # 0.5 Hz 的平稳运动
    omega = 2 * np.pi * f
    
    # 【修复关键点】：将普通的 Python float 转换为 torch.Tensor
    t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
    
    q_d = A * torch.sin(omega * t_tensor)
    dq_d = A * omega * torch.cos(omega * t_tensor)
    ddq_d = -A * (omega**2) * torch.sin(omega * t_tensor)
    
    return q_d, dq_d, ddq_d

# ==============================
# 🤖 真实机器人解析动力学仿真器 (The Real Plant)
# ==============================
class RealRobotSimulator:
    def __init__(self, device):
        self.device = device
        # 物理常数 (从你的 data.py 完美移植)
        self.m1, self.m2 = 0.5, 0.5
        self.J1, self.J2 = 0.02, 0.02
        self.kappa_stiff = 3.0
        self.damping = 0.8
        self.FRICTION_MU = 0.5  
        
        # 🌟 开启环境干摩擦注入，考验控制器的抗扰动能力！
        self.inject_friction = True 

    def M_true(self, q):
        k1, p1, k2, p2 = q[:,0], q[:,1], q[:,2], q[:,3]
        batch = q.shape[0]
        M = torch.zeros(batch, 4, 4, device=self.device)
        M[:,0,0] = self.m1 + 0.2 * torch.cos(k1)
        M[:,1,1] = self.J1 + 0.1 * torch.sin(k1)**2
        M[:,2,2] = self.m2 + 0.2 * torch.cos(k2)
        M[:,3,3] = self.J2 + 0.1 * torch.sin(k2)**2
        M[:,0,2] = 0.05 * torch.cos(k1-k2)
        M[:,2,0] = M[:,0,2]
        return M

    def C_true(self, q, dq):
        # 批量求偏导必须 retain_graph
        q_clone = q.clone().detach().requires_grad_(True)
        M = self.M_true(q_clone)
        batch = q.shape[0]
        C = torch.zeros(batch, 4, 4, device=self.device)
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    dM_ik_dqj = torch.autograd.grad(M[:,i,k].sum(), q_clone, retain_graph=True)[0][:,j]
                    dM_ij_dqk = torch.autograd.grad(M[:,i,j].sum(), q_clone, retain_graph=True)[0][:,k]
                    dM_jk_dqi = torch.autograd.grad(M[:,j,k].sum(), q_clone, retain_graph=True)[0][:,i]
                    C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]
        return C

    def D_true(self, q, dq):
        return self.damping * torch.eye(4, device=self.device).unsqueeze(0).repeat(q.shape[0], 1, 1)

    def gradV_true(self, q):
        grad = torch.zeros_like(q)
        grad[:,0] = self.kappa_stiff * q[:,0]
        grad[:,2] = self.kappa_stiff * q[:,2]
        return grad

    def get_ddq(self, q, dq, tau):
        M = self.M_true(q)
        C = self.C_true(q, dq)
        D = self.D_true(q, dq)
        gV = self.gradV_true(q)

        tau_internal = torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + \
                       torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + gV
        
        # 引入未建模的高频干摩擦干扰
        if self.inject_friction:
            tau_friction = self.FRICTION_MU * torch.tanh(10.0 * dq)
            tau_internal += tau_friction
            
        ddq = torch.linalg.solve(M, tau - tau_internal)
        return ddq.detach()

    def step(self, q, dq, tau, dt):
        """ RK4 高保真时间积分推演 """
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
# 🎮 并行控制任务 Worker
# ==============================
def evaluate_controller(args):
    gpu_id, ctrl_name = args
    device = torch.device(f"cuda:{gpu_id}")
    
    # 实例化真实的物理反馈环境
    robot = RealRobotSimulator(device)
    
    # 加载 PINN 大脑 (CTC 前馈控制器)
    model = None
    if "CTC" in ctrl_name:
        model_class = PINN_Tau if "Tau" in ctrl_name else PINN_Residual
        model = model_class(DIM=DIM, device=device).to(device)
        
        core_name = ctrl_name.replace("CTC_", "")
        path = f"models_evo_final/final_evo_{core_name}_{DATASET_TYPE}.pth"
        if not os.path.exists(path):
            print(f"[GPU {gpu_id}] ❌ 找不到模型: {path}")
            return args, None, None, None
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

    traj_q, traj_q_des, traj_tau = [], [], []
    
    # 从零状态起步
    q_curr = torch.zeros(1, DIM, device=device)
    dq_curr = torch.zeros(1, DIM, device=device)
    
    desc_str = f"[GPU {gpu_id}] {ctrl_name[:20]:<20}"
    pbar = tqdm(range(TOTAL_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
    status = "✅ Stable"
    
    for step in pbar:
        t = step * DT
        # 1. 轨迹规划层
        q_d, dq_d, ddq_d = get_desired_trajectory(t, device)
        
        # 2. 误差计算层
        e = q_d - q_curr
        de = dq_d - dq_curr
        
        # 3. 控制解算层
        if ctrl_name == "PID_Only":
            tau_cmd = KP * e + KD * de
        else:
            # CTC 控制律: tau = M_hat*(ddq_d + Kp*e + Kd*de) + C_hat*dq + D_hat*dq + g_hat
            with torch.enable_grad():
                q_g = q_curr.clone().detach().requires_grad_(True)
                V = model.potential_net(q_g)
                gradV = torch.autograd.grad(V.sum(), q_g)[0]
                
                M_hat = model.M(q_g)
                C_hat = model.C(q_g, dq_curr)
                D_hat = model.D(q_g, dq_curr)
                
                v_control = ddq_d + KP * e + KD * de
                
                tau_feedforward = torch.bmm(C_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + \
                                  torch.bmm(D_hat, dq_curr.unsqueeze(-1)).squeeze(-1) + gradV
                                  
                tau_cmd = torch.bmm(M_hat, v_control.unsqueeze(-1)).squeeze(-1) + tau_feedforward
                tau_cmd = tau_cmd.detach()

        # 电机保护机制：夹断极端力矩输出
        tau_cmd = torch.clamp(tau_cmd, min=-15.0, max=15.0)

        # 4. 物理反馈层：送入真实机器人执行
        q_curr, dq_curr = robot.step(q_curr, dq_curr, tau_cmd, DT)
        
        if torch.any(torch.isnan(q_curr)) or torch.max(torch.abs(q_curr)) > 5.0:
            status = "💥 Diverged"
            pbar.set_postfix({'Status': status})
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
    print("🚀 开启 4 卡并行: 闭环轨迹跟踪控制实验 (基于真实非线性动力学交互)")
    print("=====================================================================\n")
    
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        results_raw = pool.map(evaluate_controller, CTRL_TASKS)
        
    print(f"\n🎉 4卡并行控制评测完成！总耗时: {(time.time() - start_time):.2f} 秒")
    
    # 整理绘图数据
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
    
    # ==============================
    # 绝杀图表 1: 轨迹追踪对比
    # ==============================
    print("正在渲染闭环控制轨迹追踪图...")
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle("Closed-Loop Trajectory Tracking Performance\n(Using True Physics Plant with Friction Disturbances)", fontsize=18, fontweight='bold')
    
    for i in range(DIM):
        row, col = i // 2, i % 2
        ax = axs1[row, col]
        
        # 画黑色虚线：我们要追的目标
        ax.plot(time_axis, q_des_global[:, i], 'k--', linewidth=3.5, label="Reference ($q_d$)")
        
        for ctrl_name, data in results.items():
            length = len(data["q"])
            ax.plot(time_axis[:length], data["q"][:, i], color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
            
        ax.set_title(f"Joint $q_{i+1}$ Tracking", fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Angle (rad)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Control_Tracking.png", dpi=300)

    # ==============================
    # 绝杀图表 2: 控制力矩消耗 (暴露黑盒劣质模型的照妖镜)
    # ==============================
    print("正在渲染控制输入力矩对比图...")
    fig2 = plt.figure(figsize=(12, 6))
    
    for ctrl_name, data in results.items():
        length = len(data["tau"])
        # 计算力矩的 L2 范数
        tau_norm = np.linalg.norm(data["tau"], axis=1)
        plt.plot(time_axis[:length], tau_norm, color=COLORS[ctrl_name], linewidth=2.0, alpha=0.8, label=ctrl_name.replace("CTC_", ""))
        
    plt.title("Control Effort (Actuation Torque Norm $||\\tau||_2$)", fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Torque Norm (Nm)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("Control_Effort.png", dpi=300)
    
    print("✅ 完美！控制实验绝杀图表已保存为 Control_Tracking.png 和 Control_Effort.png")