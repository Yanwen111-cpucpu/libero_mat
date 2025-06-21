import numpy as np
import matplotlib.pyplot as plt
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


# ✅ 替换为任意你想看的 task
bddl_file = "libero/libero/bddl_files/libero_spatial_mat/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"

# 创建环境，设定初始垫子高度
env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,  # .bddl 文件路径
    camera_names=["frontview","agentview"],
    camera_heights=256,
    camera_widths=256,
)

# 重置环境并渲染图像
obs = env.reset()
for i in range(env.sim.model.nsite):
    name = env.sim.model.site_id2name(i)
    pos = env.sim.data.site_xpos[i]
    print(f"[{i}] {name} -> {pos}")

image = obs["agentview_image"]

import matplotlib.pyplot as plt
plt.imshow(np.flipud(image))
plt.title("Environment with Adjustable Mat")
plt.axis("off")
plt.show()