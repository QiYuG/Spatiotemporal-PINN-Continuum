import torch
import torch.multiprocessing as mp
import os
from tune_evo import train_task, init_worker, PINN_Residual, DATASET_TYPE

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 仅针对 Residual_condition，指定最佳物理平衡权重 10.0
    task_args = ("Residual_condition", PINN_Residual, {'weight_evo': 10.0})
    
    gpu_queue = mp.Queue()
    gpu_queue.put((0, 0)) # 用 GPU 0 跑这个唯一任务
    
    print("🚀 正在单独纠正 Residual_condition 模型 (强制锁定 evo_weight=10.0)...")
    
    with mp.Pool(processes=1, initializer=init_worker, initargs=(gpu_queue,)) as pool:
        result = pool.map(train_task, [task_args])[0]
        
    if result:
        save_path = f"models_evo/best_evo_Residual_condition_{DATASET_TYPE}.pth"
        torch.save(result['state_dict'], save_path)
        print(f"✅ 纠正完成！已用包含物理结构的完美模型覆盖原文件：{save_path}")
        print(f"该模型的 Evo_MSE (100步): {result['val_evo_mse']:.6e}")