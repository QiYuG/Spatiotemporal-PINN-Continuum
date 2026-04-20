# import torch
# import torch.multiprocessing as mp
# import os
# import time
# from tqdm import tqdm
# import numpy as np

# from model.PINN_Tau import PINN_Tau
# from model.PINN_Residual import PINN_Residual
# from utils.spectual_margin_loss import spectral_margin_loss
# from utils.skew_structure_loss import skew_structure_loss
# from utils.condition_number_regularization import condition_number_regularization

# torch.set_default_dtype(torch.float32)

# # ==============================
# # 6-DOF 演化微调参数配置
# # ==============================
# DIM = 6
# DATASET_TYPE = "noisy"
# TRAIN_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_train.pt"
# TEST_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_test.pt"

# # ⚠️ 极度关键：6-DOF 的 BPTT 计算图包含成千上万次 autograd.grad
# # 为防 RTX 4090 显存溢出 (OOM)，Batch Size 必须严格下调至 16 或 32
# BATCH_SIZE = 512     
# TRAIN_STEPS = 1500    
# DT = 0.005
# H_STEPS = 5           
# STEPS_PER_TRAJ = 2000 

# # 沿用 4-DOF 最优物理正则化权重 (Zero-shot Hyperparameter Transfer)
# FINAL_OPTIMAL_PARAMS = {
#     "Residual_margin":    {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 10.0},
#     "Residual_condition": {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 10.0},
#     "Tau_margin":         {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 1.0}, 
#     "Tau_condition":      {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 1.0}  
# }

# # ==============================
# # 1. BPTT 专用的全计算图可微积分器 (修复 Leaf Variable 崩溃)
# # ==============================
# def get_ddq_diff(model, q, dq, tau):
#     M = model.M(q)
    
#     # 🛡️ 安全的科里奥利矩阵计算：避开原始代码中的 q.requires_grad_(True)
#     # 因为在 BPTT 中，q 是上一步积分的输出，属于非叶子节点 (Non-leaf Tensor)
#     batch = q.shape[0]
#     C = torch.zeros(batch, DIM, DIM, device=q.device)
#     for k in range(DIM):
#         for j in range(DIM):
#             for i in range(DIM):
#                 # 利用 create_graph=True 保留高阶导数计算图，允许时间反向传播
#                 dM_ik_dqj = torch.autograd.grad(M[:,i,k].sum(), q, create_graph=True)[0][:,j]
#                 dM_ij_dqk = torch.autograd.grad(M[:,i,j].sum(), q, create_graph=True)[0][:,k]
#                 dM_jk_dqi = torch.autograd.grad(M[:,j,k].sum(), q, create_graph=True)[0][:,i]
#                 C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]

#     D = model.D(q, dq)
#     V = model.potential_net(q)
    
#     gradV = torch.autograd.grad(V.sum(), q, create_graph=True)[0]

#     tau_internal = (
#         torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + 
#         torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + 
#         gradV
#     )
#     return torch.linalg.solve(M, tau - tau_internal)

# def rk4_step_diff(model, q, dq, tau, dt):
#     ddq1 = get_ddq_diff(model, q, dq, tau)
#     q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
#     ddq2 = get_ddq_diff(model, q2, dq2, tau)
#     q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
#     ddq3 = get_ddq_diff(model, q3, dq3, tau)
#     q4, dq4 = q + dt * dq3, dq + dt * ddq3
#     ddq4 = get_ddq_diff(model, q4, dq4, tau)

#     q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
#     dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
#     return q_next, dq_next

# # ==============================
# # 2. 测试专用的内存隔离积分器 (防止显存泄漏)
# # ==============================
# def rk4_step_eval(model, q_in, dq_in, tau_in, dt):
#     # 彻底切断与过去历史的梯度连接
#     q = q_in.clone().detach()
#     dq = dq_in.clone().detach()
#     tau = tau_in.clone().detach()

#     def get_ddq_eval(q_t, dq_t):
#         # 局部开启梯度计算以获取偏导数
#         with torch.enable_grad():
#             q_t_grad = q_t.clone().detach().requires_grad_(True)
#             V = model.potential_net(q_t_grad)
#             gradV = torch.autograd.grad(V.sum(), q_t_grad)[0]
            
#             M = model.M(q_t_grad)
            
#             # 这里可以直接调用模型的 C() 因为 q_t_grad 是我们刚刚创建的叶子节点
#             C = model.C(q_t_grad, dq_t)
#             D = model.D(q_t_grad, dq_t)
            
#             tau_internal = torch.bmm(C, dq_t.unsqueeze(-1)).squeeze(-1) + \
#                            torch.bmm(D, dq_t.unsqueeze(-1)).squeeze(-1) + gradV
#             ddq = torch.linalg.solve(M, tau - tau_internal)
#         # 强制截断，决不能将图传给下一个步骤
#         return ddq.detach()

#     ddq1 = get_ddq_eval(q, dq)
#     q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
#     ddq2 = get_ddq_eval(q2, dq2)
#     q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
#     ddq3 = get_ddq_eval(q3, dq3)
#     q4, dq4 = q + dt * dq3, dq + dt * ddq3
#     ddq4 = get_ddq_eval(q4, dq4)

#     q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
#     dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
#     return q_next.detach(), dq_next.detach()

# # ==============================
# # 核心演化微调任务 Worker
# # ==============================
# def train_worker(args):
#     gpu_id, model_name, model_class, params = args
#     device = torch.device(f"cuda:{gpu_id}")
    
#     os.makedirs("models_6dof_evo", exist_ok=True)
    
#     train_data_raw = torch.load(TRAIN_PATH, weights_only=False)
#     test_data_raw = torch.load(TEST_PATH, weights_only=False)
    
#     num_train_traj = train_data_raw["q"].shape[0] // STEPS_PER_TRAJ
#     num_test_traj = test_data_raw["q"].shape[0] // STEPS_PER_TRAJ
    
#     train_traj = {k: v.view(num_train_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in train_data_raw.items()}
#     test_traj = {k: v.view(num_test_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in test_data_raw.items()}
    
#     model = model_class(DIM=DIM, device=device).to(device)
    
#     # 加载预训练的基准权重地基
#     base_path = f"models_6dof/base_{model_name}_{DATASET_TYPE}.pth"
#     if os.path.exists(base_path):
#         model.load_state_dict(torch.load(base_path, map_location=device))
#     else:
#         raise FileNotFoundError(f"[GPU {gpu_id}] 致命错误：未找到 6-DOF 预训练权重 {base_path}！")

#     optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
#     desc_str = f"[GPU {gpu_id}] 6-DOF Evo | {model_name[:15]:<15}"
#     pbar = tqdm(total=TRAIN_STEPS, desc=desc_str, position=gpu_id, leave=True)
    
#     for step in range(TRAIN_STEPS):
#         model.train()
        
#         traj_idx = torch.randint(0, num_train_traj, (BATCH_SIZE,), device=device)
#         step_idx = torch.randint(0, STEPS_PER_TRAJ - H_STEPS, (BATCH_SIZE,), device=device)
        
#         # 这里的 q_curr 是第一步的源头，可以安全设置为 requires_grad_(True)
#         q_curr = train_traj["q"][traj_idx, step_idx].clone().requires_grad_(True)
#         dq_curr = train_traj["dq"][traj_idx, step_idx].clone()
#         ddq_curr = train_traj["ddq"][traj_idx, step_idx].clone()
#         tau_curr = train_traj["tau"][traj_idx, step_idx].clone()
        
#         optimizer.zero_grad()
        
#         # 1. 拟合单步代数约束
#         if "Residual" in model_name:
#             residual, M_matrix, _ = model(q_curr, dq_curr, ddq_curr, tau_curr)
#             loss_single = torch.mean(residual**2)
#         else:
#             tau_pred = model(q_curr, dq_curr, ddq_curr)
#             loss_single = torch.mean((tau_pred - tau_curr)**2)
#             M_matrix = model.M(q_curr)
            
#         loss_skew = skew_structure_loss(model, q_curr, dq_curr, DIM)
#         loss_reg = spectral_margin_loss(M_matrix, margin=0.05) if "margin" in model_name else condition_number_regularization(M_matrix)

#         # 2. 拟合多步时域约束 (BPTT 极其消耗显存，依赖于 get_ddq_diff 的计算图连接)
#         loss_evo = 0.0
#         q_pred, dq_pred = q_curr, dq_curr
        
#         for h in range(1, H_STEPS + 1):
#             tau_h = train_traj["tau"][traj_idx, step_idx + h - 1]
#             q_true_h = train_traj["q"][traj_idx, step_idx + h]
#             q_pred, dq_pred = rk4_step_diff(model, q_pred, dq_pred, tau_h, DT)
#             loss_evo = loss_evo + torch.mean((q_pred - q_true_h)**2)

#         loss = loss_single + params['weight_reg']*loss_reg + params['weight_skew']*loss_skew + params['weight_evo']*loss_evo
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         if step % 5 == 0:
#             pbar.set_postfix({'Evo_Loss': f"{loss_evo.item():.4e}"})
#         pbar.update(1)

#     pbar.close()

#     # --- 100 步时域演化验证 ---
#     model.eval()
#     val_loss_evo_total = 0.0
#     with torch.no_grad():
#         for t_idx in range(5):  
#             q_curr = test_traj["q"][t_idx, 0:1].clone()
#             dq_curr = test_traj["dq"][t_idx, 0:1].clone()
            
#             q_pred, dq_pred = q_curr, dq_curr
#             traj_evo_err = 0.0
#             for h in range(1, 100):
#                 tau_h = test_traj["tau"][t_idx, h - 1:h]
#                 q_true_h = test_traj["q"][t_idx, h:h+1]
#                 q_pred, dq_pred = rk4_step_eval(model, q_pred, dq_pred, tau_h, DT)
#                 traj_evo_err += torch.mean((q_pred - q_true_h)**2).item()
#             val_loss_evo_total += traj_evo_err / 100.0

#     final_val_mse = val_loss_evo_total / 5.0
    
#     save_path = f"models_6dof_evo/final_evo_{model_name}_{DATASET_TYPE}.pth"
#     torch.save(model.state_dict(), save_path)
    
#     return model_name, final_val_mse

# if __name__ == "__main__":
#     mp.set_start_method('spawn', force=True)
    
#     tasks = [
#         (0, "Residual_margin", PINN_Residual, FINAL_OPTIMAL_PARAMS["Residual_margin"]),
#         (1, "Residual_condition", PINN_Residual, FINAL_OPTIMAL_PARAMS["Residual_condition"]),
#         (2, "Tau_margin", PINN_Tau, FINAL_OPTIMAL_PARAMS["Tau_margin"]),
#         (3, "Tau_condition", PINN_Tau, FINAL_OPTIMAL_PARAMS["Tau_condition"])
#     ]
    
#     print("\033[2J\033[H", end="")
#     print(f"============================================================")
#     print(f"🚀 开启 4 卡并行: 6-DOF 终极演化微调 (Evolution Fine-Tuning)")
#     print(f"============================================================\n")
    
#     start_time = time.time()
    
#     results = {}
#     with mp.Pool(processes=4) as pool:
#         for model_name, mse in pool.imap_unordered(train_worker, tasks):
#             results[model_name] = mse
            
#     print(f"\n\n🎉 6-DOF 演化微调全部完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
    
#     print("\n" + "="*60)
#     print("🏆 6-DOF 最终演化微调评估报告 (100-step Rollout MSE)")
#     print("="*60)
#     for model_name, mse in results.items():
#         print(f"  ➜ {model_name:<20}: {mse:.6e}")
        
#     print("\n所有终极权重均已安全存入 models_6dof_evo/ 文件夹！准备开展高维控制测试。")

import torch
import torch.multiprocessing as mp
import os
import time
from tqdm import tqdm
import numpy as np

from model.PINN_Tau import PINN_Tau
from model.PINN_Residual import PINN_Residual
from utils.spectual_margin_loss import spectral_margin_loss
from utils.skew_structure_loss import skew_structure_loss
from utils.condition_number_regularization import condition_number_regularization

# ==============================
# 🚀 性能榨干设置 (RTX 4090 专属)
# ==============================
torch.set_default_dtype(torch.float32)
# 开启 TensorFloat-32 (TF32)，在 4090 上矩阵运算速度提升 2~3 倍！
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 提升 CuDNN 自动优化器寻找最优卷积/矩阵乘算法的能力
torch.backends.cudnn.benchmark = True  

# ==============================
# 6-DOF 演化微调参数配置
# ==============================
DIM = 6
DATASET_TYPE = "noisy"
TRAIN_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_train.pt"
TEST_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_test.pt"

# 🚀 既然显存空虚，直接拉大 Batch Size 让 GPU 并行计算！(如果 128 爆显存，改回 64)
BATCH_SIZE = 256      
TRAIN_STEPS = 2000    
DT = 0.005
H_STEPS = 5           
STEPS_PER_TRAJ = 2000 

FINAL_OPTIMAL_PARAMS = {
    "Residual_margin":    {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 10.0},
    "Residual_condition": {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 10.0},
    "Tau_margin":         {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 1.0}, 
    "Tau_condition":      {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 1.0}  
}

# ==============================
# 1. 可微积分器
# ==============================
def get_ddq_diff(model, q, dq, tau):
    M = model.M(q)
    batch = q.shape[0]
    C = torch.zeros(batch, DIM, DIM, device=q.device)
    for k in range(DIM):
        for j in range(DIM):
            for i in range(DIM):
                dM_ik_dqj = torch.autograd.grad(M[:,i,k].sum(), q, create_graph=True)[0][:,j]
                dM_ij_dqk = torch.autograd.grad(M[:,i,j].sum(), q, create_graph=True)[0][:,k]
                dM_jk_dqi = torch.autograd.grad(M[:,j,k].sum(), q, create_graph=True)[0][:,i]
                C[:,i,j] += 0.5 * (dM_ik_dqj + dM_ij_dqk - dM_jk_dqi) * dq[:,k]

    D = model.D(q, dq)
    V = model.potential_net(q)
    gradV = torch.autograd.grad(V.sum(), q, create_graph=True)[0]

    tau_internal = (
        torch.bmm(C, dq.unsqueeze(-1)).squeeze(-1) + 
        torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1) + 
        gradV
    )
    return torch.linalg.solve(M, tau - tau_internal)

def rk4_step_diff(model, q, dq, tau, dt):
    ddq1 = get_ddq_diff(model, q, dq, tau)
    q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
    ddq2 = get_ddq_diff(model, q2, dq2, tau)
    q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
    ddq3 = get_ddq_diff(model, q3, dq3, tau)
    q4, dq4 = q + dt * dq3, dq + dt * ddq3
    ddq4 = get_ddq_diff(model, q4, dq4, tau)

    q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
    dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
    return q_next, dq_next

# ==============================
# 2. 隔离验证积分器
# ==============================
def rk4_step_eval(model, q_in, dq_in, tau_in, dt):
    q = q_in.clone().detach()
    dq = dq_in.clone().detach()
    tau = tau_in.clone().detach()

    def get_ddq_eval(q_t, dq_t):
        with torch.enable_grad():
            q_t_grad = q_t.clone().detach().requires_grad_(True)
            V = model.potential_net(q_t_grad)
            gradV = torch.autograd.grad(V.sum(), q_t_grad)[0]
            M = model.M(q_t_grad)
            C = model.C(q_t_grad, dq_t)
            D = model.D(q_t_grad, dq_t)
            
            tau_internal = torch.bmm(C, dq_t.unsqueeze(-1)).squeeze(-1) + \
                           torch.bmm(D, dq_t.unsqueeze(-1)).squeeze(-1) + gradV
            ddq = torch.linalg.solve(M, tau - tau_internal)
        return ddq.detach()

    ddq1 = get_ddq_eval(q, dq)
    q2, dq2 = q + 0.5 * dt * dq, dq + 0.5 * dt * ddq1
    ddq2 = get_ddq_eval(q2, dq2)
    q3, dq3 = q + 0.5 * dt * dq2, dq + 0.5 * dt * ddq2
    ddq3 = get_ddq_eval(q3, dq3)
    q4, dq4 = q + dt * dq3, dq + dt * ddq3
    ddq4 = get_ddq_eval(q4, dq4)

    q_next = q + (dt / 6.0) * (dq + 2*dq2 + 2*dq3 + dq4)
    dq_next = dq + (dt / 6.0) * (ddq1 + 2*ddq2 + 2*ddq3 + ddq4)
    return q_next.detach(), dq_next.detach()

# ==============================
# 核心演化微调任务 Worker
# ==============================
def train_worker(args):
    gpu_id, model_name, model_class, params = args
    device = torch.device(f"cuda:{gpu_id}")
    
    os.makedirs("models_6dof_evo", exist_ok=True)
    
    train_data_raw = torch.load(TRAIN_PATH, weights_only=False)
    test_data_raw = torch.load(TEST_PATH, weights_only=False)
    
    num_train_traj = train_data_raw["q"].shape[0] // STEPS_PER_TRAJ
    num_test_traj = test_data_raw["q"].shape[0] // STEPS_PER_TRAJ
    
    train_traj = {k: v.view(num_train_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in train_data_raw.items()}
    test_traj = {k: v.view(num_test_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in test_data_raw.items()}
    
    model = model_class(DIM=DIM, device=device).to(device)
    
    # 🚀 PyTorch 2.0 终极魔法：编译模型
    # 如果你的 PyTorch 版本 >= 2.0，这行代码能把零碎的 216 次循环融合成底层的超大 CUDA Graph！
    # 如果运行报错，可以安全地将下面这行注释掉。
    try:
        model = torch.compile(model)
        print(f"[GPU {gpu_id}] ✅ torch.compile 编译加速已开启！")
    except Exception as e:
        pass

    base_path = f"models_6dof/base_{model_name}_{DATASET_TYPE}.pth"
    if os.path.exists(base_path):
        # 处理被 torch.compile 包装后的 key 命名变化 (_orig_mod.)
        state_dict = torch.load(base_path, map_location=device)
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"[GPU {gpu_id}] 致命错误：未找到 6-DOF 预训练权重 {base_path}！")

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    desc_str = f"[GPU {gpu_id}] 6-DOF Evo | {model_name[:15]:<15}"
    pbar = tqdm(total=TRAIN_STEPS, desc=desc_str, position=gpu_id, leave=True)
    
    for step in range(TRAIN_STEPS):
        model.train()
        
        traj_idx = torch.randint(0, num_train_traj, (BATCH_SIZE,), device=device)
        step_idx = torch.randint(0, STEPS_PER_TRAJ - H_STEPS, (BATCH_SIZE,), device=device)
        
        q_curr = train_traj["q"][traj_idx, step_idx].clone().requires_grad_(True)
        dq_curr = train_traj["dq"][traj_idx, step_idx].clone()
        ddq_curr = train_traj["ddq"][traj_idx, step_idx].clone()
        tau_curr = train_traj["tau"][traj_idx, step_idx].clone()
        
        optimizer.zero_grad()
        
        if "Residual" in model_name:
            residual, M_matrix, _ = model(q_curr, dq_curr, ddq_curr, tau_curr)
            loss_single = torch.mean(residual**2)
        else:
            tau_pred = model(q_curr, dq_curr, ddq_curr)
            loss_single = torch.mean((tau_pred - tau_curr)**2)
            M_matrix = model.M(q_curr)
            
        loss_skew = skew_structure_loss(model, q_curr, dq_curr, DIM)
        loss_reg = spectral_margin_loss(M_matrix, margin=0.05) if "margin" in model_name else condition_number_regularization(M_matrix)

        loss_evo = 0.0
        q_pred, dq_pred = q_curr, dq_curr
        
        for h in range(1, H_STEPS + 1):
            tau_h = train_traj["tau"][traj_idx, step_idx + h - 1]
            q_true_h = train_traj["q"][traj_idx, step_idx + h]
            q_pred, dq_pred = rk4_step_diff(model, q_pred, dq_pred, tau_h, DT)
            loss_evo = loss_evo + torch.mean((q_pred - q_true_h)**2)

        loss = loss_single + params['weight_reg']*loss_reg + params['weight_skew']*loss_skew + params['weight_evo']*loss_evo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 5 == 0:
            pbar.set_postfix({'Evo_Loss': f"{loss_evo.item():.4e}"})
        pbar.update(1)

    pbar.close()

    model.eval()
    val_loss_evo_total = 0.0
    with torch.no_grad():
        for t_idx in range(5):  
            q_curr = test_traj["q"][t_idx, 0:1].clone()
            dq_curr = test_traj["dq"][t_idx, 0:1].clone()
            
            q_pred, dq_pred = q_curr, dq_curr
            traj_evo_err = 0.0
            for h in range(1, 100):
                tau_h = test_traj["tau"][t_idx, h - 1:h]
                q_true_h = test_traj["q"][t_idx, h:h+1]
                q_pred, dq_pred = rk4_step_eval(model, q_pred, dq_pred, tau_h, DT)
                traj_evo_err += torch.mean((q_pred - q_true_h)**2).item()
            val_loss_evo_total += traj_evo_err / 100.0

    final_val_mse = val_loss_evo_total / 5.0
    
    # 保存权重时，如果是 compile 后的模型，需要保存底层原模型的权重
    save_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    save_path = f"models_6dof_evo/final_evo_{model_name}_{DATASET_TYPE}_new.pth"
    torch.save(save_state, save_path)
    
    return model_name, final_val_mse

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    tasks = [
        (0, "Residual_margin", PINN_Residual, FINAL_OPTIMAL_PARAMS["Residual_margin"]),
        (1, "Residual_condition", PINN_Residual, FINAL_OPTIMAL_PARAMS["Residual_condition"]),
        (2, "Tau_margin", PINN_Tau, FINAL_OPTIMAL_PARAMS["Tau_margin"]),
        (3, "Tau_condition", PINN_Tau, FINAL_OPTIMAL_PARAMS["Tau_condition"])
    ]
    
    print("\033[2J\033[H", end="")
    print(f"============================================================")
    print(f"🚀 开启 4 卡并行: 6-DOF 终极演化微调 (TF32 + 大Batch加速版)")
    print(f"============================================================\n")
    
    start_time = time.time()
    
    results = {}
    with mp.Pool(processes=4) as pool:
        for model_name, mse in pool.imap_unordered(train_worker, tasks):
            results[model_name] = mse
            
    print(f"\n\n🎉 6-DOF 演化微调全部完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
    print("\n" + "="*60)
    print("🏆 6-DOF 最终演化微调评估报告 (100-step Rollout MSE)")
    print("="*60)
    for model_name, mse in results.items():
        print(f"  ➜ {model_name:<20}: {mse:.6e}")