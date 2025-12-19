import os
import sys
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from stable_baselines3 import DQN
import highway_env
from tqdm import trange
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import multiprocessing as mp

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体

# Create the environment
env = gym.make("level2-v0", render_mode=None)
env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.Level1Vehicle"
env.unwrapped.config["training_level"] = 2
env.unwrapped.config["show_trajectories"] = False
env.unwrapped.config["simulation_frequency"] = 15
env.unwrapped.config["policy_frequency"] = 15

sim_freq = env.unwrapped.config.get("simulation_frequency", 15)
steps_per_sec = int(round(sim_freq))

interact_list = []
step_interactions = []

ACC_EXTREME_THRESHOLD = 2.5
STEER_EXTREME_THRESHOLD = 0.25

# Initialize environment first to get spaces
obs, info = env.reset()

# Load model with custom_objects to handle serialization issues

model = DQN.load(
    "dqn_checkpoints/level2_saeg_test/level2_saeg_final_model.zip",
    env= env,
    custom_objects={
        "observation_space": env.observation_space ,
        "action_space": env.action_space,
    },
    device="cuda:0"
)

#计算交互

for episode in trange(100, desc="Episode"):
    done = truncated = False
    obs, info = env.reset()
    step = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        # 计算Ego与其他车辆的交互数
        core_vehicle = env.unwrapped.vehicle.road.vehicles[0]
        # 遍历所有车辆
        interaction = 0
        for vehicle in env.unwrapped.road.vehicles:
            if vehicle != core_vehicle:
                if abs(core_vehicle.position[0] - vehicle.position[0]) < 30 and abs(core_vehicle.lane_index[2] - vehicle.lane_index[2]) <= 1 and abs(core_vehicle.velocity[0] - vehicle.velocity[0]) > 0.3:
                    interaction += 1
        interaction_pair = (step, interaction)
        # print(f"Ego与其他车辆的交互数: {interaction}")
        interact_list.append(interaction)
        step_interactions.append(interaction_pair)
    print(step)

env.close()


interact_mean = np.mean(interact_list) if interact_list else 0.0



print("Ego interaction mean: {:.3f}".format(interact_mean))


# 1) 按 step 分组：step_interactions 需包含 (step, interaction)
bucket = defaultdict(list)
for s, v in step_interactions:   # 确保你在循环里有 step_interactions.append((step, interaction))
    bucket[s].append(v)

# 2) 计算每步平均
steps_sorted = sorted(bucket.keys())
per_step_mean = np.array([np.mean(bucket[s]) for s in steps_sorted], dtype=float)

# 3) 画柱状图
plt.figure(figsize=(10, 4))
plt.bar(steps_sorted, per_step_mean, width=0.9, align='center')
plt.xlabel('step')
plt.ylabel('每步平均交互车辆数')
plt.title('每步平均交互（step_mean）')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('bar_mean_interaction_per_step.png', dpi=200)


