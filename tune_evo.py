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
DIM = 4

# ==============================
# 最终演化训练配置
# ==============================
DATASET_TYPE = "noisy"
TRAIN_PATH = f"dataset/dataset_{DATASET_TYPE}_train.pt"
TEST_PATH = f"dataset/dataset_{DATASET_TYPE}_test.pt"

BATCH_SIZE = 128      
TRAIN_STEPS = 1500    
DT = 0.005
H_STEPS = 5           

# 🌟 精心调优得出的终极参数字典
FINAL_OPTIMAL_PARAMS = {
    "Residual_margin":    {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 10.0},
    "Residual_condition": {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 10.0},
    "Tau_margin":         {'lr': 0.0005, 'weight_reg': 10.0, 'weight_skew': 10.0, 'weight_evo': 1.0}, 
    "Tau_condition":      {'lr': 0.0005, 'weight_reg': 0.1,  'weight_skew': 10.0, 'weight_evo': 1.0}  
}

# ==============================
# 1. 训练专用可微积分器 (确保 BPTT 梯度不断链)
# ==============================
def get_ddq_diff(model, q, dq, tau):
    M = model.M(q)
    C = model.C(q, dq)
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
# 2. 验证专用游离积分器 (物理空间绝对隔离，0内存泄漏)
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
        # 返回前 detach，使得内部生成的局域计算图全部被 Python 垃圾回收机制销毁
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
# 全局 Worker 初始化
# ==============================
local_env = {}

def init_worker(gpu_queue):
    gpu_id, worker_idx = gpu_queue.get()
    device = torch.device(f"cuda:{gpu_id}")
    local_env['device'] = device
    local_env['gpu_id'] = gpu_id
    local_env['worker_idx'] = worker_idx 
    
    # 设置不同的随机种子，保证 Ensemble 寻优时的差异性
    torch.manual_seed(worker_idx * 1024)
    
    train_data_raw = torch.load(TRAIN_PATH, weights_only=False)
    test_data_raw = torch.load(TEST_PATH, weights_only=False)
    
    STEPS_PER_TRAJ = 2000
    num_train_traj = train_data_raw["q"].shape[0] // STEPS_PER_TRAJ
    num_test_traj = test_data_raw["q"].shape[0] // STEPS_PER_TRAJ
    
    local_env['train_traj'] = {k: v.view(num_train_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in train_data_raw.items()}
    local_env['test_traj'] = {k: v.view(num_test_traj, STEPS_PER_TRAJ, DIM).to(device) for k, v in test_data_raw.items()}

# ==============================
# 核心演化微调任务
# ==============================
def train_task(args):
    try:
        model_name, model_class, seed_idx = args
        device = local_env['device']
        train_traj = local_env['train_traj']
        test_traj = local_env['test_traj']
        gpu_id = local_env['gpu_id']
        worker_idx = local_env['worker_idx']
        
        params = FINAL_OPTIMAL_PARAMS[model_name]
        
        model = model_class(DIM=DIM, device=device).to(device)
        
        # 🛡️ 极其严格的预训练权重加载机制 (智能路径寻找)
        possible_paths = [
            f"models/final_{model_name}_{DATASET_TYPE}.pth",
            f"models/best_{model_name}_{DATASET_TYPE}.pth",
            f"models/final_{model_name}.pth",
            f"models/best_{model_name}.pth"
        ]
        
        loaded_successfully = False
        for path in possible_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                loaded_successfully = True
                break
                
        if not loaded_successfully:
            # 如果没找到，直接抛出异常，绝不盲目瞎跑！
            raise FileNotFoundError(f"找不到预训练权重，检查了以下路径: {possible_paths}")

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        num_traj = train_traj["q"].shape[0]
        STEPS_PER_TRAJ = train_traj["q"].shape[1]
        
        desc_str = f"[GPU {gpu_id} | {model_name}] 实例 {seed_idx+1}/4"
        pbar = tqdm(range(TRAIN_STEPS), desc=desc_str, position=worker_idx, leave=True)
        
        # --- 训练循环 ---
        for step in pbar:
            model.train()
            
            traj_idx = torch.randint(0, num_traj, (BATCH_SIZE,), device=device)
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

        pbar.close()

        # --- 验证循环 ---
        model.eval()
        val_loss_evo_total = 0.0
        
        with torch.no_grad():
            for t_idx in range(5):  # 评估 5 条测试轨迹，向前预测 100 步
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

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        return {
            'model_name': model_name,
            'seed_idx': seed_idx,
            'val_evo_mse': val_loss_evo_total / 5.0,
            'state_dict': state_dict
        }
    except Exception as e:
        import traceback
        print(f"\n[GPU ERROR] 任务 {model_name} 崩溃:\n{traceback.format_exc()}")
        return None

# ==============================
# 主调度程序
# ==============================
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    if not os.path.exists("models_evo_final"):
        os.makedirs("models_evo_final")

    models_to_test = {
        "Residual_margin": PINN_Residual,
        "Residual_condition": PINN_Residual,
        "Tau_margin": PINN_Tau,
        "Tau_condition": PINN_Tau
    }
    
    tasks = []
    # 构造 16 个并发任务
    for model_name, cls in models_to_test.items():
        for seed_idx in range(4):
            tasks.append((model_name, cls, seed_idx))
            
    print("\033[2J\033[H", end="")
    print(f"==========================================================================")
    print(f"🚀 开启 16 进程终极训练 (集成寻优：每种模型训练 4 次，取最强王者！)")
    print(f"==========================================================================\n")
    
    gpu_queue = mp.Queue()
    worker_idx = 0
    # 确保任务完美隔离在 4 张 GPU 上
    for i in range(4):          
        for _ in range(4):      
            gpu_queue.put((i, worker_idx))
            worker_idx += 1
        
    start_time = time.time()
    all_results = {name: [] for name in models_to_test.keys()}
    
    with mp.Pool(processes=16, initializer=init_worker, initargs=(gpu_queue,)) as pool:
        for res in pool.imap_unordered(train_task, tasks):
            if res is not None:
                all_results[res['model_name']].append(res)

    print(f"\n\n\n🎉 终极训练全部完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    print("\n" + "="*80)
    print(f"🏆 终极模型大逃杀结果 (每个模型从 4 个平行实例中选出最佳)")
    print("="*80)
    
    for model_name, results in all_results.items():
        if not results: continue
        print(f"\n🌟 {model_name}")
        results_sorted = sorted(results, key=lambda x: x['val_evo_mse'])
        
        for r in results_sorted:
            medal = "🥇" if r == results_sorted[0] else "  "
            print(f" {medal} 实例 {r['seed_idx']+1}/4 || Evo_MSE: {r['val_evo_mse']:.6e}")
                  
        best_run = results_sorted[0]
        save_path = f"models_evo_final/final_evo_{model_name}_{DATASET_TYPE}.pth"
        torch.save(best_run['state_dict'], save_path)
        print(f"💾 最佳实例已保存至: {save_path}")