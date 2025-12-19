# filepath: 
import gymnasium as gym
import torch
import numpy as np
import os
import sys
import time
import multiprocessing as mp
from tqdm import trange
import highway_env

# 添加父目录到路径，确保能导入项目模块
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from stable_baselines3 import DQN

# ===================== 配置参数 =====================
MODEL_PATH = "dqn_checkpoints/level2_saeg_test/level2_saeg_final_model_PER_safe.zip"  # 模型路径
TOTAL_EPISODES = 500   # 总测试回合数
N_PROCESSES = 10      # 并行进程数


# ===================== 核心测试函数 =====================
def run_flow_test(episodes_in_batch: int, model_path: str, process_id: int) -> tuple:
    """
    运行测试并统计交通流量 (断面计数法)
    - 使用跨越检测线事件计数
    - 碰撞的 episode 不计入流量统计
    """
    # 1. 创建环境
    env = gym.make("level2-v0", render_mode=None)

    # --- 关键配置：开启 IDM 和换道，模拟真实车流 ---
    env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"
    env.unwrapped.config["training_level"] = 2
    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.config["policy_frequency"] = 15
    env.unwrapped.config["show_trajectories"] = False

    # 2. 加载模型
    model = DQN.load(
        model_path,
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
        device="cuda:0"  # 或 "cpu"
    )

    local_collisions = 0
    flow_samples = []  # 存储每个「无碰撞」回合计算出的流量 (veh/h/lane)

    # 进度条
    iterator = trange(episodes_in_batch, desc=f"进程 {process_id}", position=process_id, leave=False)

    for _ in iterator:
        obs, info = env.reset()
        done = truncated = False

        # --- 获取所有车辆的初始纵向位置 ---
        all_vehicles = env.unwrapped.road.vehicles
        x_positions = [v.position[0] for v in all_vehicles]
        
        x_min = min(x_positions)  # 最后一辆车的位置
        x_max = max(x_positions)  # 最前一辆车的位置
        detection_line_x = (x_min + x_max) / 2  # 中间位置作为截面

        # --- 流量统计变量 ---
        passed_vehicle_ids = set()  # 记录本回合已通过检测线的车辆ID
        last_positions = {}         # 记录上一时刻每辆车的 x 坐标
        step_count = 0

        # 初始化 last_positions
        for v in all_vehicles:
            last_positions[id(v)] = v.position[0]

        last_cross_step = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            # --- 断面计数核心逻辑：检测跨越事件 ---
            for v in env.unwrapped.road.vehicles:
                # 如果想只统计环境车，可以跳过 ego：
                # if v is env.unwrapped.vehicle:
                #     continue

                vid = id(v)
                x_now = v.position[0]

                if vid in last_positions:
                    x_last = last_positions[vid]
                    # 从检测线左侧穿越到右侧，计一次
                    if x_last <= detection_line_x  < x_now:
                        passed_vehicle_ids.add(vid)
                        last_cross_step = step_count

                last_positions[vid] = x_now

        # 检查是否有碰撞发生
        crashed = info.get("crashed", False)

        # --- 回合结束：统计（视是否碰撞而定） ---
        sim_freq = env.unwrapped.config["simulation_frequency"]
        duration_seconds = last_cross_step / sim_freq + 2.0  # 加2秒缓冲
        # print (f"进程 {process_id} 回合结束: 持续时间 {duration_seconds:.2f} 秒, 通过车辆数 {len(passed_vehicle_ids)}, 碰撞: {crashed}")

        # 有碰撞的 episode：只计碰撞，不计流量
        if crashed:
            local_collisions += 1
        else:
            # 只有当仿真持续时间足够长(例如>5秒)，计算才有意义
            if duration_seconds > 5.0:
                duration_hours = duration_seconds / 3600.0

                # 总流量 Q_total = N / T (veh/h)
                q_total = len(passed_vehicle_ids) / duration_hours

                # 换算为每车道流量 (veh/h/lane)
                lanes = env.unwrapped.config["lanes_count"]
                q_lane = q_total / lanes

                flow_samples.append(q_lane)

    env.close()
    return local_collisions, flow_samples


# ===================== 主程序 =====================
if __name__ == '__main__':
    # 设置多进程启动方式
    mp.set_start_method("spawn", force=True)
    
    print(f"开始测试交通流量...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"总回合数: {TOTAL_EPISODES}")
    
    start_time = time.time()

    # 分配任务
    episodes_per_process = TOTAL_EPISODES // N_PROCESSES
    remainder = TOTAL_EPISODES % N_PROCESSES
    args_list = []
    for i in range(N_PROCESSES):
        count = episodes_per_process + (1 if i < remainder else 0)
        if count > 0:
            args_list.append((count, MODEL_PATH, i))

    # 并行执行
    with mp.Pool(processes=N_PROCESSES) as pool:
        results = pool.starmap(run_flow_test, args_list)

    end_time = time.time()

    # --- 汇总结果 ---
    total_collisions = 0
    all_flows = []

    for collisions, flows in results:
        total_collisions += collisions
        all_flows.extend(flows)

    # 计算统计值
    avg_flow = np.mean(all_flows) if all_flows else 0.0
    max_flow = np.max(all_flows) if all_flows else 0.0
    min_flow = np.min(all_flows) if all_flows else 0.0
    collision_rate = total_collisions / TOTAL_EPISODES

    # --- 打印报告 ---
    print(f"\n{'='*30}")
    print(f"      交通流量统计报告      ")
    print(f"{'='*30}")
    print(f"平均流量 : {avg_flow:.2f} veh/h (辆/小时)")
    print(f"最大流量 : {max_flow:.2f} veh/h")
    print(f"最小流量 : {min_flow:.2f} veh/h")
    print(f"{'-'*30}")
    print(f"有效样本 : {len(all_flows)} / {TOTAL_EPISODES}")
    print(f"总耗时   : {end_time - start_time:.2f} 秒")
    print(f"{'='*30}")
