import torch
import torch.multiprocessing as mp
import os
import time
from tqdm import tqdm

from model.PINN_Tau import PINN_Tau
from model.PINN_Residual import PINN_Residual
from utils.spectual_margin_loss import spectral_margin_loss
from utils.skew_structure_loss import skew_structure_loss
from utils.condition_number_regularization import condition_number_regularization

torch.set_default_dtype(torch.float32)

# ==============================
# 6-DOF 预训练配置
# ==============================
DIM = 6
DATASET_TYPE = "noisy"
TRAIN_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_train.pt"
TEST_PATH = f"dataset_6dof/dataset_{DATASET_TYPE}_test.pt"

# ⚠️ 注意：由于 6-DOF 科里奥利计算图暴增(O(N^3))，为防OOM，Batch Size设为 256 或 512
BATCH_SIZE = 256      
TRAIN_STEPS = 5000     # 6-DOF 映射更复杂，基础步数加到 5000 确保物理骨架稳定

# 白嫖 4-DOF 证实的最优物理正则化权重
FIXED_PARAMS = {
    "Residual_margin":    {'lr': 0.001, 'weight_reg': 10.0, 'weight_skew': 10.0},
    "Residual_condition": {'lr': 0.001, 'weight_reg': 0.1,  'weight_skew': 10.0},
    "Tau_margin":         {'lr': 0.001, 'weight_reg': 10.0, 'weight_skew': 10.0}, 
    "Tau_condition":      {'lr': 0.001, 'weight_reg': 0.1,  'weight_skew': 10.0}  
}

def train_worker(args):
    gpu_id, model_name, model_class, params = args
    device = torch.device(f"cuda:{gpu_id}")
    
    # 确保保存目录存在
    os.makedirs("models_6dof", exist_ok=True)
    
    # 加载 6-DOF 数据
    data_raw = torch.load(TRAIN_PATH, weights_only=False)
    # 打平数据，单步训练不需要维持时间序列(H_STEPS)
    q_all = data_raw["q"].view(-1, DIM).to(device)
    dq_all = data_raw["dq"].view(-1, DIM).to(device)
    ddq_all = data_raw["ddq"].view(-1, DIM).to(device)
    tau_all = data_raw["tau"].view(-1, DIM).to(device)
    
    total_samples = q_all.shape[0]
    
    # 传入 DIM=6，自动初始化高维网络
    model = model_class(DIM=DIM, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    desc_str = f"[GPU {gpu_id}] 6-DOF {model_name}"
    pbar = tqdm(range(TRAIN_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
    for step in pbar:
        model.train()
        
        # 随机采样
        idx = torch.randint(0, total_samples, (BATCH_SIZE,), device=device)
        q_batch = q_all[idx].clone().requires_grad_(True)
        dq_batch = dq_all[idx]
        ddq_batch = ddq_all[idx]
        tau_batch = tau_all[idx]
        
        optimizer.zero_grad()
        
        if "Residual" in model_name:
            residual, M_matrix, _ = model(q_batch, dq_batch, ddq_batch, tau_batch)
            loss_mse = torch.mean(residual**2)
        else:
            tau_pred = model(q_batch, dq_batch, ddq_batch)
            loss_mse = torch.mean((tau_pred - tau_batch)**2)
            M_matrix = model.M(q_batch)
            
        loss_skew = skew_structure_loss(model, q_batch, dq_batch, DIM)
        
        if "margin" in model_name:
            loss_reg = spectral_margin_loss(M_matrix, margin=0.05)
        else:
            loss_reg = condition_number_regularization(M_matrix)

        # 仅拟合物理方程，无演化损失
        loss = loss_mse + params['weight_reg'] * loss_reg + params['weight_skew'] * loss_skew
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0:
            pbar.set_postfix({'MSE': f"{loss_mse.item():.4e}"})

    pbar.close()

    # 保存 6-DOF 预训练底座
    save_path = f"models_6dof/base_{model_name}_{DATASET_TYPE}.pth"
    torch.save(model.state_dict(), save_path)
    return model_name

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 构建 4 个核心模型的预训练任务
    tasks = [
        (0, "Residual_margin", PINN_Residual, FIXED_PARAMS["Residual_margin"]),
        (1, "Residual_condition", PINN_Residual, FIXED_PARAMS["Residual_condition"]),
        (2, "Tau_margin", PINN_Tau, FIXED_PARAMS["Tau_margin"]),
        (3, "Tau_condition", PINN_Tau, FIXED_PARAMS["Tau_condition"])
    ]
    
    print("\033[2J\033[H", end="")
    print(f"============================================================")
    print(f"🚀 开启 4 卡并行: 6-DOF 基础物理模型单步预训练")
    print(f"⚠️ 警告: 6-DOF 将面临 O(N^3) 科里奥利计算图，风扇准备起飞！")
    print(f"============================================================\n")
    
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        pool.map(train_worker, tasks)
        
    print(f"\n🎉 6-DOF 基础底座训练完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
    print("权重已保存至 models_6dof/。下一步可以直接挂上演化微调！")