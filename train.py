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

DIM = 4
BATCH_SIZE = 256
TRAIN_STEPS = 1500

# ==============================
# 全局配置
# ==============================
DATASET_TYPE = "noisy"  # 可选: "clean" 或 "noisy"
TRAIN_PATH = f"dataset/dataset_{DATASET_TYPE}_train.pt"
TEST_PATH = f"dataset/dataset_{DATASET_TYPE}_test.pt"

# 请将你网格搜索得出的四个模型的最优超参数填入这里
# (此处 Residual 模型的参数已经填为你之前实验得出的帕累托最优解)
BEST_PARAMS = {
    "Residual_margin":    {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0},
    "Residual_condition": {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0},
    "Tau_margin":         {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0}, # 请替换为实际最优值
    "Tau_condition":      {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0}  # 请替换为实际最优值
}

# ==============================
# 核心独立训练 Worker (跑在单张 GPU 上)
# ==============================
def train_worker(gpu_id, model_name, model_class, params):
    # 1. 初始化专属设备并加载数据
    device = torch.device(f"cuda:{gpu_id}")
    
    train_data_raw = torch.load(TRAIN_PATH, weights_only=False)
    test_data_raw = torch.load(TEST_PATH, weights_only=False)
    
    train_data = {k: v.to(device) for k, v in train_data_raw.items()}
    test_data = {k: v.to(device) for k, v in test_data_raw.items()}
    N_train = train_data["q"].shape[0]

    # 2. 初始化模型与优化器
    model = model_class(DIM=DIM, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    # 3. 设置多行专属进度条 (position=gpu_id 可以让 4 个进度条固定在不同行)
    desc_str = f"[GPU {gpu_id}] {model_name:<20}"
    pbar = tqdm(range(TRAIN_STEPS), desc=desc_str, position=gpu_id, leave=True)
    
    # 4. 正式训练循环
    for step in pbar:
        model.train()
        idx = torch.randint(0, N_train, (BATCH_SIZE,), device=device)
        
        q = train_data["q"][idx]
        dq = train_data["dq"][idx]
        ddq = train_data["ddq"][idx]
        tau = train_data["tau"][idx]
        
        optimizer.zero_grad()
        
        # 前向传播
        if "Residual" in model_name:
            residual, M_matrix, _ = model(q, dq, ddq, tau)
            loss_dyn = torch.mean(residual**2)
        else: # Tau model
            tau_pred = model(q, dq, ddq)
            loss_dyn = torch.mean((tau_pred - tau)**2)
            M_matrix = model.M(q)
            
        # 结构正则化
        loss_skew = skew_structure_loss(model, q, dq, DIM)
        if "margin" in model_name:
            loss_reg = spectral_margin_loss(M_matrix, margin=params.get('spd_margin', 0.05))
        elif "condition" in model_name:
            loss_reg = condition_number_regularization(M_matrix)
        else:
            loss_reg = torch.tensor(0.0, device=device)

        loss = loss_dyn + params['weight_reg'] * loss_reg + params['weight_skew'] * loss_skew
        loss.backward()
        optimizer.step()

        # 实时刷新进度条后缀
        if step % 20 == 0:
            pbar.set_postfix({
                'MSE': f"{loss_dyn.item():.4f}", 
                'Cond_L': f"{loss_reg.item():.4f}"
            })

    pbar.close()

    # 5. 训练结束，截取测试集前1024步进行最终属性评估
    model.eval()
    q_val = test_data["q"][:1024].clone()
    dq_val = test_data["dq"][:1024].clone()
    ddq_val = test_data["ddq"][:1024].clone()
    tau_val = test_data["tau"][:1024].clone()
    
    q_val.requires_grad_(True)
    
    with torch.set_grad_enabled(True):
        if "Residual" in model_name:
            residual, M_matrix, _ = model(q_val, dq_val, ddq_val, tau_val)
            val_loss_dyn = torch.mean(residual**2).item()
        else:
            tau_pred = model(q_val, dq_val, ddq_val)
            val_loss_dyn = torch.mean((tau_pred - tau_val)**2).item()
            M_matrix = model.M(q_val)
            
        val_cond = condition_number_regularization(M_matrix).item()
        val_skew = skew_structure_loss(model, q_val, dq_val, DIM).item()

    # 6. 保存最终模型权重
    save_path = f"models/final_{model_name}_{DATASET_TYPE}.pth"
    torch.save(model.state_dict(), save_path)
    
    return {
        'model_name': model_name,
        'dyn_mse': val_loss_dyn,
        'condition_number': val_cond,
        'skew_error': val_skew,
        'save_path': save_path
    }

# ==============================
# 主调度程序 (分配 4 个任务到 4 张卡)
# ==============================
if __name__ == "__main__":
    # 必须使用 spawn，以支持 CUDA 的多进程并行
    mp.set_start_method('spawn', force=True)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # 定义 4 个任务映射
    tasks = [
        (0, "Residual_margin", PINN_Residual, BEST_PARAMS["Residual_margin"]),
        (1, "Residual_condition", PINN_Residual, BEST_PARAMS["Residual_condition"]),
        (2, "Tau_margin", PINN_Tau, BEST_PARAMS["Tau_margin"]),
        (3, "Tau_condition", PINN_Tau, BEST_PARAMS["Tau_condition"])
    ]
    
    # 提前清空一下终端，让多行进度条显示更美观
    print("\033[2J\033[H", end="")
    print(f"============================================================")
    print(f"🚀 开启 4 卡并行最终训练 | 当前数据集: {DATASET_TYPE.upper()}")
    print(f"============================================================\n")
    
    start_time = time.time()
    
    # 使用 Pool 异步执行 4 个任务
    with mp.Pool(processes=4) as pool:
        # starmap 用于向函数传递多个参数
        results = pool.starmap(train_worker, tasks)
        
    # 等待所有进度条完成后，稍微隔开一点距离打印结果
    print("\n\n")
    print(f"🎉 4 个模型训练全部完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # ==============================
    # 最终结果打榜输出
    # ==============================
    print("\n" + "="*85)
    print(f"✨ 最终模型属性对比报告 (基于 {DATASET_TYPE.upper()} Test 集前 1024 个样本)")
    print("="*85)
    print(f"{'模型名称':<22} | {'动力学 MSE (Dyn)':<16} | {'M 矩阵条件数':<15} | {'斜对称误差 (Skew)':<15}")
    print("-" * 85)
    
    for res in results:
        print(f"{res['model_name']:<22} | {res['dyn_mse']:<18.6e} | {res['condition_number']:<15.4f} | {res['skew_error']:.6e}")
    print("=" * 85)
    print("权重均已安全落盘！下一步可直接运行 Forward Rollout 或最小特征值分析。")