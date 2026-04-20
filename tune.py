import torch
import torch.nn as nn
import multiprocessing as mp
import itertools
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
TRAIN_STEPS = 1000

# ==============================
# 调优配置
# ==============================
DATASET_TYPE = "noisy"  # 可选: "clean" 或 "noisy"
TRAIN_PATH = f"dataset/dataset_{DATASET_TYPE}_train.pt"
TEST_PATH = f"dataset/dataset_{DATASET_TYPE}_test.pt"

# 定义搜索空间 (每个模型 16 组，4 个模型共 64 个训练任务)
PARAM_GRID = {
    'lr': [5e-4],
    'weight_reg': [0.1, 5.0, 10.0, 50.0],
    'weight_skew': [0.1, 1.0, 5.0, 10.0]
}

# ==============================
# 全局 Worker 初始化
# ==============================
# 这部分用于给每个进程分配固定的 GPU ID，并预加载数据到该 GPU 的显存中
local_env = {}

def init_worker(gpu_queue):
    gpu_id = gpu_queue.get()
    device = torch.device(f"cuda:{gpu_id}")
    local_env['device'] = device
    local_env['gpu_id'] = gpu_id
    
    # 每个进程独立读取数据并放入自己掌管的 GPU，避免多进程张量共享报错
    train_data = torch.load(TRAIN_PATH, weights_only=False)
    test_data = torch.load(TEST_PATH, weights_only=False)
    
    local_env['train_data'] = {k: v.to(device) for k, v in train_data.items()}
    local_env['test_data'] = {k: v.to(device) for k, v in test_data.items()}
    
    # 取测试集前 1024 个样本用于验证，节约时间
    local_env['q_val'] = local_env['test_data']["q"][:1024].clone()
    local_env['dq_val'] = local_env['test_data']["dq"][:1024].clone()
    local_env['ddq_val'] = local_env['test_data']["ddq"][:1024].clone()
    local_env['tau_val'] = local_env['test_data']["tau"][:1024].clone()

# ==============================
# 核心训练任务 (在单个 Worker 上执行)
# ==============================
def train_task(args):
    model_name, model_class, params = args
    device = local_env['device']
    train_data = local_env['train_data']
    gpu_id = local_env['gpu_id']
    
    model = model_class(DIM=DIM, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    N_train = train_data["q"].shape[0]
    
    # 训练循环
    for step in range(TRAIN_STEPS):
        model.train()
        idx = torch.randint(0, N_train, (BATCH_SIZE,), device=device)
        
        q = train_data["q"][idx]
        dq = train_data["dq"][idx]
        ddq = train_data["ddq"][idx]
        tau = train_data["tau"][idx]
        
        optimizer.zero_grad()
        
        if "Residual" in model_name:
            residual, M_matrix, _ = model(q, dq, ddq, tau)
            loss_dyn = torch.mean(residual**2)
        else: # Tau model
            tau_pred = model(q, dq, ddq)
            loss_dyn = torch.mean((tau_pred - tau)**2)
            M_matrix = model.M(q)
            
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

    # 验证循环
    model.eval()
    q_val = local_env['q_val']
    dq_val = local_env['dq_val']
    ddq_val = local_env['ddq_val']
    tau_val = local_env['tau_val']
    
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

    # 训练完成，将状态转移至 CPU 并返回
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    
    result = {
        'model_name': model_name,
        'params': params,
        'dyn_mse': val_loss_dyn,
        'condition_number': val_cond,
        'skew_error': val_skew,
        'gpu_id': gpu_id,
        'state_dict': state_dict
    }
    
    return result

# ==============================
# 主调度程序
# ==============================
if __name__ == "__main__":
    # 使用 spawn 启动，确保 CUDA 初始化正常
    mp.set_start_method('spawn', force=True)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    # 构建所有要测试的任务列表
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    models_to_test = {
        "Residual_margin": PINN_Residual,
        "Residual_condition": PINN_Residual,
        "Tau_margin": PINN_Tau,
        "Tau_condition": PINN_Tau
    }
    
    tasks = []
    for model_name, model_class in models_to_test.items():
        for params in param_combinations:
            tasks.append((model_name, model_class, params))
            
    total_tasks = len(tasks)
    print(f"==================================================")
    print(f"🚀 开启 4x GPU 并行调优 | 数据集: {DATASET_TYPE.upper()}")
    print(f"📦 共有 {len(models_to_test)} 个模型架构，每个架构 {len(param_combinations)} 组参数")
    print(f"🔥 总计 {total_tasks} 个训练任务已分配至进程池！")
    print(f"==================================================")
    
    # 设置 4 个 GPU 的队列分配
    num_gpus = 4
    gpu_queue = mp.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)
        
    start_time = time.time()
    all_results = {name: [] for name in models_to_test.keys()}
    
    # 启动 4 进程并行池
    with mp.Pool(processes=num_gpus, initializer=init_worker, initargs=(gpu_queue,)) as pool:
        # 使用 imap_unordered 可以无序、高效地获取返回结果，并更新 tqdm 进度条
        for res in tqdm(pool.imap_unordered(train_task, tasks), total=total_tasks, desc="总体任务进度"):
            all_results[res['model_name']].append(res)
            
            # (可选) 打印单条完成日志，方便随时查看进度
            # print(f"✅ [GPU {res['gpu_id']}] {res['model_name']} | 权重={res['params']['weight_reg']}/{res['params']['weight_skew']} | MSE: {res['dyn_mse']:.5f} | Cond: {res['condition_number']:.2f} | Skew: {res['skew_error']:.6f}")

    time_cost = time.time() - start_time
    print(f"\n🎉 并行调优完成！总耗时: {time_cost/60:.2f} 分钟")

    # ==============================
    # 结果分析与输出
    # ==============================
    print("\n" + "="*80)
    print(f"🏆 每个架构的超参数评估报告 ({DATASET_TYPE.upper()} 数据集)")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n🌟 {model_name} 调优结果 (按 Dyn_MSE 从低到高排序):")
        
        # 按 Dyn_MSE 升序排序
        results_sorted = sorted(results, key=lambda x: x['dyn_mse'])
        
        for r in results_sorted:
            p = r['params']
            print(f"  ➜ reg={p['weight_reg']:<5.1f} | skew={p['weight_skew']:<5.1f} || "
                  f"MSE: {r['dyn_mse']:.6f} | Cond: {r['condition_number']:<8.4f} | Skew: {r['skew_error']:.6f}")
                  
        # 取第一项作为“理论误差最小的最优解”保存
        # （注：你依然可以像我们之前讨论的那样，人工去审视这个列表，找出更具物理意义的帕累托最优解）
        best_run = results_sorted[0]
        torch.save(best_run['state_dict'], f"models/best_{model_name}_{DATASET_TYPE}.pth")
        
        print(f"💾 {model_name} 自动保存的最佳参数 (最低 MSE): reg={best_run['params']['weight_reg']}, skew={best_run['params']['weight_skew']}")
        print("-" * 80)