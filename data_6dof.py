import torch
import torch.autograd as autograd
import torch.multiprocessing as mp
import numpy as np
import os
import time

DIM = 6  # 6-DOF (3段式 PCC 连续体机器人)

# ==============================
# 6-DOF 物理参数设定
# ==============================
# 刚度与阻尼随段数逐渐减小 (符合柔性臂物理直觉)
K_STIFF = torch.tensor([12., 12., 10., 10., 8., 8.])
DAMPING = torch.tensor([0.8, 0.8, 0.7, 0.7, 0.6, 0.6])

# 摩擦力参数 (用于 Noisy 数据集)
FRICTION_MU = 0.5  

# ==============================
# 动力学核心 (纯张量 Batch 运算，极速)
# ==============================
def M_true_6dof(q, device):
    """
    构建一个严格正定且高度非线性耦合的 6x6 质量矩阵。
    这种解析形式比 autograd.jacobian 快上百倍，且完全符合拉格朗日力学结构。
    """
    batch = q.shape[0]
    M = torch.zeros(batch, DIM, DIM, device=device)

    # 1. 对角线主惯量 (保证绝对正定)
    for i in range(DIM):
        M[:, i, i] = 0.5 + 0.1 * torch.sin(q[:, i])**2

    # 2. 段与段之间的非线性动力学耦合 (距离越近，耦合越强)
    for i in range(DIM):
        for j in range(i + 1, DIM):
            # 耦合强度随自由度距离衰减
            coupling = (0.05 / (j - i)) * torch.cos(q[:, i] - q[:, j])
            M[:, i, j] = coupling
            M[:, j, i] = coupling
            
    return M

def C_true_6dof(q, dq, device):
    """ 利用 Autograd 批量精确推导科里奥利矩阵，保证李代数斜对称性 """
    q.requires_grad_(True)
    M = M_true_6dof(q, device)
    batch = q.shape[0]
    C = torch.zeros(batch, DIM, DIM, device=device)

    for k in range(DIM):
        for j in range(DIM):
            for i in range(DIM):
                dM_ik_dqj = autograd.grad(M[:,i,k].sum(), q, retain_graph=True)[0][:,j]
                dM_ij_dqk = autograd.grad(M[:,i,j].sum(), q, retain_graph=True)[0][:,k]
                dM_jk_dqi = autograd.grad(M[:,j,k].sum(), q, retain_graph=True)[0][:,i]
                C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]
    return C

def D_true_6dof(q, dq, device):
    D_diag = DAMPING.to(device)
    return torch.diag_embed(D_diag).unsqueeze(0).repeat(q.shape[0], 1, 1)

def gradV_true_6dof(q, device):
    K_diag = K_STIFF.to(device)
    return q * K_diag

# ==============================
# 加速度求解器
# ==============================
def get_true_ddq_6dof(q, dq, tau, device, inject_friction=False):
    q_clone = q.clone().detach().requires_grad_(True)
    M = M_true_6dof(q_clone, device)
    C = C_true_6dof(q_clone, dq, device)
    D = D_true_6dof(q_clone, dq, device)
    gV = gradV_true_6dof(q_clone, device)

    tau_internal = torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + \
                   torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + gV
    
    # 注入未建模的非线性干摩擦 (仅在 Noisy 数据集)
    if inject_friction:
        tau_friction = FRICTION_MU * torch.tanh(10.0 * dq)
        tau_internal += tau_friction
        
    ddq = torch.linalg.solve(M, tau - tau_internal)
    return ddq.detach()

# ==============================
# RK4 批量积分仿真器 (在单张 GPU 上运行)
# ==============================
def simulate_batch_trajectories_6dof(num_traj, device_id, T=10.0, dt=0.005, inject_friction=False):
    device = torch.device(f"cuda:{device_id}")
    steps = int(T / dt)
    
    freq = torch.rand(num_traj, DIM, device=device) * 1.5 + 0.5
    amp = torch.rand(num_traj, DIM, device=device) * 0.5

    # 随机初始状态
    q = torch.rand(num_traj, DIM, device=device) * 1.0 - 0.5
    dq = torch.rand(num_traj, DIM, device=device) * 0.2 - 0.1

    Q, DQ, DDQ, TAU = [], [], [], []

    print(f"[GPU {device_id}] 开始 RK4 积分 {num_traj} 条 6-DOF 轨迹 (摩擦: {inject_friction})...")
    
    for i in range(steps):
        t_val = i * dt
        
        # 构建 6-DOF 随机力矩
        tau_components = []
        for d in range(DIM):
            func = torch.sin if d % 2 == 0 else torch.cos
            tau_components.append(func(freq[:, d] * t_val))
        tau = amp * torch.stack(tau_components, dim=1)

        Q.append(q.clone())
        DQ.append(dq.clone())
        TAU.append(tau.clone())

        # RK4 积分核心
        ddq1 = get_true_ddq_6dof(q, dq, tau, device, inject_friction)
        DDQ.append(ddq1.clone())

        q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
        ddq2 = get_true_ddq_6dof(q2, dq2, tau, device, inject_friction)

        q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
        ddq3 = get_true_ddq_6dof(q3, dq3, tau, device, inject_friction)

        q4, dq4 = q + dt * dq3, dq + dt * ddq3
        ddq4 = get_true_ddq_6dof(q4, dq4, tau, device, inject_friction)

        q = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
        dq = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)

    print(f"[GPU {device_id}] 积分完成.")
    
    Q_cat = torch.stack(Q).transpose(0, 1).reshape(-1, DIM)
    DQ_cat = torch.stack(DQ).transpose(0, 1).reshape(-1, DIM)
    DDQ_cat = torch.stack(DDQ).transpose(0, 1).reshape(-1, DIM)
    TAU_cat = torch.stack(TAU).transpose(0, 1).reshape(-1, DIM)
    
    return Q_cat.cpu(), DQ_cat.cpu(), DDQ_cat.cpu(), TAU_cat.cpu()

# ==============================
# 多进程工作节点
# ==============================
def worker(rank, num_traj_per_gpu, inject_friction, return_dict):
    try:
        q, dq, ddq, tau = simulate_batch_trajectories_6dof(
            num_traj_per_gpu, device_id=rank, T=10.0, dt=0.005, inject_friction=inject_friction
        )
        return_dict[rank] = (q, dq, ddq, tau)
    except Exception as e:
        print(f"[GPU {rank}] 发生错误: {e}")

# ==============================
# 主调度与数据集构建
# ==============================
def generate_and_save(dataset_type="clean", total_traj=200):
    start_time = time.time()
    num_gpus = 4
    traj_per_gpu = total_traj // num_gpus
    inject_friction = True if dataset_type == "noisy" else False
    
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    
    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, traj_per_gpu, inject_friction, return_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    Q_all, DQ_all, DDQ_all, TAU_all = [], [], [], []
    for rank in range(num_gpus):
        q, dq, ddq, tau = return_dict[rank]
        Q_all.append(q)
        DQ_all.append(dq)
        DDQ_all.append(ddq)
        TAU_all.append(tau)
        
    Q = torch.cat(Q_all)
    DQ = torch.cat(DQ_all)
    DDQ = torch.cat(DDQ_all)
    TAU = torch.cat(TAU_all)
    
    # 【高频传感器噪声注入】 (对 Noisy 数据集)
    if dataset_type == "noisy":
        print("正在注入 6-DOF 传感器测量噪声 (10% SNR)...")
        DQ = DQ + 0.1 * DQ.std(dim=0) * torch.randn_like(DQ)
        DDQ = DDQ + 0.1 * DDQ.std(dim=0) * torch.randn_like(DDQ)

    # 按轨迹严格划分 (防止时间穿插)
    total_steps_per_traj = int(10.0 / 0.005)
    train_traj_count = int(total_traj * 0.8)
    train_samples = train_traj_count * total_steps_per_traj
    
    train_data = {"q": Q[:train_samples], "dq": DQ[:train_samples], "ddq": DDQ[:train_samples], "tau": TAU[:train_samples]}
    test_data = {"q": Q[train_samples:], "dq": DQ[train_samples:], "ddq": DDQ[train_samples:], "tau": TAU[train_samples:]}

    save_dir = "dataset_6dof"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_data, f"{save_dir}/dataset_{dataset_type}_train.pt")
    torch.save(test_data, f"{save_dir}/dataset_{dataset_type}_test.pt")
    
    print(f"\n✅ 6-DOF {dataset_type.upper()} 数据集构建完成！(耗时 {time.time() - start_time:.2f} 秒)")
    print(f"   总样本: {Q.shape[0]} | 训练集: {train_data['q'].shape[0]} | 测试集: {test_data['q'].shape[0]}\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    print("==================================================")
    print("阶段一：构建 6-DOF [CLEAN] 理想物理真值数据集")
    print("==================================================")
    generate_and_save(dataset_type="clean", total_traj=200)
    
    print("==================================================")
    print("阶段二：构建 6-DOF [NOISY] 含摩擦与测量噪声数据集")
    print("==================================================")
    generate_and_save(dataset_type="noisy", total_traj=200)

    print("双数据集准备就绪！下一步可以更改网络 DIM=6 进行训练测试了。")
