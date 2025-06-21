import torch
import numpy as np

# 加载两个 init 文件
init_file_1 = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_mat.pruned_init"
init_file_2 = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.pruned_init"

states1 = torch.load(init_file_1)
states2 = torch.load(init_file_2)

print(f'mat:{states1}')
print(f'no_mat:{states2}')

n = min(len(states1), len(states2))

print(f"🔍 Comparing {n} states from each file...\n")

for i in range(n):
    state1 = states1[i]
    state2 = states2[i]

    print(f"\n=== 🔢 Init State {i} ===")
    
    for field in ["qpos", "qvel", "mocap_pos", "mocap_quat", "ctrl"]:
        val1 = getattr(state1, field, None)
        val2 = getattr(state2, field, None)

        if val1 is None or val2 is None:
            print(f"{field}: ❓ Missing in one file")
            continue

        diff = np.linalg.norm(val1 - val2)
        same = np.allclose(val1, val2, atol=1e-6)

        print(f"{field}: {'✅ Same' if same else f'❗Diff (L2={diff:.4e})'}")

        # Optional: uncomment to print actual arrays
        print("  val1:", val1)
        print("  val2:", val2)

print("\n✅ Done comparing.")
