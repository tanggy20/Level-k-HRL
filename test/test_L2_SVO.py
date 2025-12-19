import os
import sys
import time
import math
import numpy as np
from tqdm import trange
import multiprocessing as mp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import gymnasium as gym
import highway_env
from stable_baselines3 import DQN


# ===================== 多进程 worker =====================

def run_svo_batch(episodes_in_batch: int,
                  model_path: str,
                  phi: float,
                  process_id: int,
                  start_global_idx: int,
                  base_seed: int = 0):
    """
    单个进程跑若干个 episode，只统计 SVO 相关变量（时间损失 + 碰撞率）。

    episodes_in_batch : 这个进程要跑多少个 episode
    model_path        : 模型路径
    phi               : SVO 参数（例如 0 或 math.pi/4）
    process_id        : 进程编号，仅用于 tqdm 显示
    start_global_idx  : 这个进程负责的 episode 的全局起始下标 (0-based)
    base_seed         : 全局随机种子偏移

    为了保证“环境相同”，第 k 个全局 episode 使用 seed = base_seed + k。
    如果你对 phi=0 和 phi=pi/4 都用同样的 episodes_per_process / start_global_idx / base_seed，
    那么两次评估的环境是严格对齐的。
    """

    local = {
        "delta_ego": [],
        "delta_others": [],
        "delta_social": [],
    }

    # 每个进程独立创建环境
    env = gym.make("level2-v0", render_mode=None)
    env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.Level1Vehicle"
    env.unwrapped.config["training_level"] = 2
    env.unwrapped.config["show_trajectories"] = False
    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.config["policy_frequency"] = 15


    # 每个进程独立加载模型
    model = DQN.load(
        model_path,
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
        device="cuda:0"   # 或 "cpu"
    )

    # 这个进程负责 episodes_in_batch 个 episode
    for local_ep in trange(episodes_in_batch, desc=f"进程 {process_id}", position=process_id):
        global_ep_idx = start_global_idx + local_ep
        seed = base_seed + global_ep_idx

        obs, info = env.reset(seed=int(seed))
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        # 时间损失：有碰撞的 episode 记 NaN（只用无碰撞的来算平均）
        stats = info.get("time_loss_stats", {})

        local["delta_ego"].append(stats.get("delta_ego", np.nan))
        local["delta_others"].append(stats.get("delta_others", np.nan))
        local["delta_social"].append(stats.get("delta_social", np.nan))


    env.close()
    # 返回这个进程的统计
    return local


# ===================== 单个 phi 的汇总评估函数 =====================

def evaluate_one_phi_multiprocess(model_path: str,
                                  phi: float,
                                  total_episodes: int,
                                  num_processes: int,
                                  base_seed: int = 0,
                                  tag: str = ""):
    """
    多进程评估一个 SVO 参数 phi，对 total_episodes 个 episode 做统计。
    仅统计：
      - 时间损失：delta_ego / delta_others / delta_social
      - 碰撞率：ego_crash_rate / others_crash_rate / any_crash_rate
    """

    # 1) 把总 episode 数平均分配给各个进程（照你原来的写法）
    episodes_per_process = [total_episodes // num_processes] * num_processes
    remainder = total_episodes % num_processes
    for i in range(remainder):
        episodes_per_process[i] += 1

    # 2) 计算每个进程负责的“全局 episode 起始下标”
    start_indices = []
    acc = 0
    for e in episodes_per_process:
        start_indices.append(acc)
        acc += e

    # 3) 组装多进程参数：每个进程知道自己要跑多少个 episode，以及从第几个 global_idx 开始
    args = [
        (episodes_per_process[i], model_path, phi, i, start_indices[i], base_seed)
        for i in range(num_processes)
    ]

    print(f"\n==== 开始评估 {tag} (phi={phi:.3f})，共 {total_episodes} 回合，{num_processes} 进程 ====")

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_svo_batch, args)

    # 4) 汇总所有进程的结果
    delta_ego_all = []
    delta_others_all = []
    delta_social_all = []

    for local in results:
        delta_ego_all.extend(local["delta_ego"])
        delta_others_all.extend(local["delta_others"])
        delta_social_all.extend(local["delta_social"])

    # 5) 计算指标
    delta_ego_mean = float(np.nanmean(delta_ego_all))
    delta_others_mean = float(np.nanmean(delta_others_all))
    delta_social_mean = float(np.nanmean(delta_social_all))

    print(f"\n--- {tag} 结果 (phi={phi:.3f}) ---")
    print(f"总 episode 数          : {total_episodes}")
    print(f"平均时间损失 delta_ego   : {delta_ego_mean:.4f}")
    print(f"平均时间损失 delta_others: {delta_others_mean:.4f}")
    print(f"平均时间损失 delta_social: {delta_social_mean:.4f}")

    return {
        "phi": phi,
        "delta_ego_mean": delta_ego_mean,
        "delta_others_mean": delta_others_mean,
        "delta_social_mean": delta_social_mean,
    }


# ===================== 主程序：分别评估 phi=0 和 phi=pi/4 =====================

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Windows 建议

    # 你自己的两个模型路径
    MODEL_PATH_PHI0  = "dqn_checkpoints/level2_efeg_test/level2_efeg_final_model_0.6.zip"
    MODEL_PATH_PHI45 = "dqn_checkpoints/level2_sapr_test/level2_sapr_final_model_PER_safe.zip"

    TOTAL_EPISODES = 500
    NUM_PROCESSES = max(1, mp.cpu_count() - 20)
    BASE_SEED = 0

    print(f"使用 {NUM_PROCESSES} 个进程进行 SVO 评估，总 episode = {TOTAL_EPISODES}")

    t0 = time.time()

    # 自利型 SVO：phi = 0
    res_phi0 = evaluate_one_phi_multiprocess(
        model_path=MODEL_PATH_PHI0,
        phi=0.0,
        total_episodes=TOTAL_EPISODES,
        num_processes=NUM_PROCESSES,
        base_seed=BASE_SEED,
        tag="phi = 0"
    )

    # 亲社会型 SVO：phi = pi/4
    res_phi45 = evaluate_one_phi_multiprocess(
        model_path=MODEL_PATH_PHI45,
        phi=math.pi / 4.0,
        total_episodes=TOTAL_EPISODES,
        num_processes=NUM_PROCESSES,
        base_seed=BASE_SEED,
        tag="phi = pi/4"
    )

    t1 = time.time()
    print(f"\n总评估耗时: {t1 - t0:.2f} 秒")

