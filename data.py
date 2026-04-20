import torch
import torch.autograd as autograd
import torch.multiprocessing as mp
import numpy as np
import os
import time

DIM = 4

# ==============================
# 物理参数
# ==============================
m1, m2 = 0.5, 0.5
J1, J2 = 0.02, 0.02
kappa_stiff = 3.0
damping = 0.8

# 未建模动态参数 (用于 Noisy 数据集)
FRICTION_MU = 0.5  

# ==============================
# 动力学核心 (支持全维度 Batch)
# ==============================
def M_true(q, device):
    k1, p1, k2, p2 = q[:,0], q[:,1], q[:,2], q[:,3]
    batch = q.shape[0]
    M = torch.zeros(batch, 4, 4, device=device)

    M[:,0,0] = m1 + 0.2*torch.cos(k1)
    M[:,1,1] = J1 + 0.1*torch.sin(k1)**2
    M[:,2,2] = m2 + 0.2*torch.cos(k2)
    M[:,3,3] = J2 + 0.1*torch.sin(k2)**2

    # 耦合项
    M[:,0,2] = 0.05*torch.cos(k1-k2)
    M[:,2,0] = M[:,0,2]
    return M

def C_true(q, dq, device):
    q.requires_grad_(True)
    M = M_true(q, device)
    batch = q.shape[0]
    C = torch.zeros(batch, DIM, DIM, device=device)

    for k in range(DIM):
        for j in range(DIM):
            for i in range(DIM):
                # 批量求导，极大利用 GPU 并行能力
                dM_ik_dqj = autograd.grad(M[:,i,k].sum(), q, retain_graph=True)[0][:,j]
                dM_ij_dqk = autograd.grad(M[:,i,j].sum(), q, retain_graph=True)[0][:,k]
                dM_jk_dqi = autograd.grad(M[:,j,k].sum(), q, retain_graph=True)[0][:,i]
                C[:,i,j] += 0.5*(dM_ik_dqj + dM_ij_dqk - dM_jk_dqi)*dq[:,k]
    return C

def D_true(q, dq, device):
    return damping * torch.eye(4, device=device).unsqueeze(0).repeat(q.shape[0], 1, 1)

def gradV_true(q, device):
    grad = torch.zeros_like(q)
    grad[:,0] = kappa_stiff * q[:,0]
    grad[:,2] = kappa_stiff * q[:,2]
    return grad

# ==============================
# 加速度求解器
# ==============================
def get_true_ddq(q, dq, tau, device, inject_friction=False):
    q_clone = q.clone().detach().requires_grad_(True)
    M = M_true(q_clone, device)
    C = C_true(q_clone, dq, device)
    D = D_true(q_clone, dq, device)
    gV = gradV_true(q_clone, device)

    # 组装内部保守与耗散力
    tau_internal = torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + \
                   torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + gV
    
    # 【噪声注入逻辑】：如果是 Noisy 数据集，隐式加入干摩擦力
    if inject_friction:
        # 使用 tanh 作为平滑的干摩擦近似，防止 RK4 数值震荡
        tau_friction = FRICTION_MU * torch.tanh(10.0 * dq)
        tau_internal += tau_friction
        
    ddq = torch.linalg.solve(M, tau - tau_internal)
    return ddq.detach()

# ==============================
# RK4 批量积分仿真器 (在单张 GPU 上运行)
# ==============================
def simulate_batch_trajectories(num_traj, device_id, T=10.0, dt=0.005, inject_friction=False):
    device = torch.device(f"cuda:{device_id}")
    steps = int(T / dt)
    
    # 随机初始化所有轨迹的频率和幅值
    freq = torch.rand(num_traj, 4, device=device) * 1.5 + 0.5
    amp = torch.rand(num_traj, 4, device=device) * 0.5

    # 随机初始状态
    q = torch.rand(num_traj, 4, device=device) * 1.0 - 0.5
    dq = torch.rand(num_traj, 4, device=device) * 0.2 - 0.1

    Q, DQ, DDQ, TAU = [], [], [], []

    print(f"[GPU {device_id}] 开始积分 {num_traj} 条轨迹 (摩擦注入: {inject_friction})...")
    
    for i in range(steps):
        t_val = i * dt
        # 批量生成力矩
        tau = amp * torch.stack([
            torch.sin(freq[:,0]*t_val),
            torch.cos(freq[:,1]*t_val),
            torch.sin(freq[:,2]*t_val),
            torch.cos(freq[:,3]*t_val)
        ], dim=1)

        Q.append(q.clone())
        DQ.append(dq.clone())
        TAU.append(tau.clone())

        # RK4 积分核心
        ddq1 = get_true_ddq(q, dq, tau, device, inject_friction)
        DDQ.append(ddq1.clone())

        q2 = q + 0.5 * dt * dq
        dq2 = dq + 0.5 * dt * ddq1
        ddq2 = get_true_ddq(q2, dq2, tau, device, inject_friction)

        q3 = q + 0.5 * dt * dq2
        dq3 = dq + 0.5 * dt * ddq2
        ddq3 = get_true_ddq(q3, dq3, tau, device, inject_friction)

        q4 = q + dt * dq3
        dq4 = dq + dt * ddq3
        ddq4 = get_true_ddq(q4, dq4, tau, device, inject_friction)

        q = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
        dq = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)

    print(f"[GPU {device_id}] 积分完成.")
    
    # 转换维度为 (steps*num_traj, 4) 以适配之前的格式
    Q_cat = torch.stack(Q).transpose(0, 1).reshape(-1, 4)
    DQ_cat = torch.stack(DQ).transpose(0, 1).reshape(-1, 4)
    DDQ_cat = torch.stack(DDQ).transpose(0, 1).reshape(-1, 4)
    TAU_cat = torch.stack(TAU).transpose(0, 1).reshape(-1, 4)
    
    return Q_cat.cpu(), DQ_cat.cpu(), DDQ_cat.cpu(), TAU_cat.cpu()

# ==============================
# 多进程工作节点
# ==============================
def worker(rank, num_traj_per_gpu, inject_friction, return_dict):
    try:
        q, dq, ddq, tau = simulate_batch_trajectories(
            num_traj_per_gpu, 
            device_id=rank, 
            T=10.0, 
            dt=0.005, 
            inject_friction=inject_friction
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
    
    # 启动 4 个进程，分发到 4 张卡
    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, traj_per_gpu, inject_friction, return_dict))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    # 合并结果
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
    
    # 【高频传感器噪声注入】 (仅对 Noisy 数据集)
    if dataset_type == "noisy":
        print("正在注入传感器测量噪声 (10% SNR)...")
        # 根据标准差注入 10% 强度的高斯噪声
        DQ_noise = 0.1 * DQ.std(dim=0) * torch.randn_like(DQ)
        DDQ_noise = 0.1 * DDQ.std(dim=0) * torch.randn_like(DDQ)
        DQ = DQ + DQ_noise
        DDQ = DDQ + DDQ_noise

    # 按轨迹划分 (避免数据泄露)
    total_steps_per_traj = int(10.0 / 0.005)
    train_traj_count = int(total_traj * 0.8)
    train_samples = train_traj_count * total_steps_per_traj
    
    train_data = {
        "q": Q[:train_samples],
        "dq": DQ[:train_samples],
        "ddq": DDQ[:train_samples],
        "tau": TAU[:train_samples]
    }
    test_data = {
        "q": Q[train_samples:],
        "dq": DQ[train_samples:],
        "ddq": DDQ[train_samples:],
        "tau": TAU[train_samples:]
    }

    os.makedirs("dataset", exist_ok=True)
    torch.save(train_data, f"dataset/dataset_{dataset_type}_train.pt")
    torch.save(test_data, f"dataset/dataset_{dataset_type}_test.pt")
    
    time_cost = time.time() - start_time
    print(f"\n✅ {dataset_type.upper()} 数据集构建完成！(耗时 {time_cost:.2f} 秒)")
    print(f"   总样本数: {Q.shape[0]} | 训练集: {train_data['q'].shape[0]} | 测试集: {test_data['q'].shape[0]}\n")


if __name__ == "__main__":
    # 必须使用 spawn 模式启动进程以支持 CUDA
    mp.set_start_method('spawn', force=True)
    
    # 生成各 200 条轨迹 (共约 40万 样本)
    print("========================================")
    print("阶段一：构建 [CLEAN] 理想物理真值数据集")
    print("========================================")
    generate_and_save(dataset_type="clean", total_traj=200)
    
    print("========================================")
    print("阶段二：构建 [NOISY] 含摩擦与测量噪声数据集")
    print("========================================")
    generate_and_save(dataset_type="noisy", total_traj=200)

    print("双数据集准备就绪！现在可以执行抗干扰对照实验了。")