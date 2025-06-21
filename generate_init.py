import os
import torch
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path

import numpy as np

# === CONFIG ===
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial"
task_suite = benchmark_dict[task_suite_name]()
task_id = 9
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] loaded task {task_id}: {task.language}")
NUM_SAMPLES = 20

# === 创建环境 ===
env = OffScreenRenderEnv(bddl_file_name=task_bddl_file)
env.seed(0)

# === 采样状态（flatten后） ===
init_states = []
for i in range(NUM_SAMPLES):
    try:
        env.reset()
        state = env.sim.get_state().flatten()  # 👈 关键：flatten 成 ndarray
        init_states.append(state)
        print(f"[{i+1}/{NUM_SAMPLES}] collected")
    except Exception as e:
        print(f"[{i+1}/{NUM_SAMPLES}] failed: {e}")

env.close()

# === 保存为 torch tensor（no_mat 格式） ===
init_states = np.stack(init_states, axis=0)
save_tensor = torch.tensor(init_states, dtype=torch.float32)
init_path = os.path.join("libero/libero/init_files/libero_spatial", f"{task.name}_mat.pruned_init")
torch.save(save_tensor, init_path)
print(f"[✓] Saved flattened init states to {init_path}")
