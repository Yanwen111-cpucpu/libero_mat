from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import numpy as np
import os
import imageio

# 1. 加载任务
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial"
task_suite = benchmark_dict[task_suite_name]()
task_id = 9
task = task_suite.get_task(task_id)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
test_bddl_file = "tmp/pddl_files/KITCHEN_SCENE1_your_language_1.bddl"
print(f"[info] loaded task {task_id}: {task.language}")

# 2. 设置环境
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id)
env.set_init_state(init_states[0])

# 3. 执行并记录图像帧
replay_images = []

dummy_action = [0.] * 7
for step in range(100):
    obs, reward, done, info = env.step(dummy_action)
    img = obs["agentview_image"]
    replay_images.append(np.flipud(img))

env.close()

# 4. 保存为视频
video_path = f"task_{task_id}_demo.mp4"
with imageio.get_writer(video_path, fps=10) as writer:
    for frame in replay_images:
        writer.append_data(frame)

print(f"[info] video saved to {video_path}")
