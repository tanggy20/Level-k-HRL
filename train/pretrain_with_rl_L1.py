import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import highway_env
import math
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
import matplotlib.pyplot as plt
import numpy as np


# 性能优化配置
TRAIN = False  # 改为True进行训练
USE_PARALLEL_ENVS = True  # 使用并行环境
N_ENVS =  10  # 减少到4个避免内存过载，同时保持并行效率
USE_GPU = True
RANDOM_SEEDS = [42, 123, 456, 789, 999]
if __name__ == "__main__":
    
    # 优化环境配置函数
    def make_optimized_env(seed = None):
        def _init():
            env = gym.make("level1-v0", render_mode=None)
            # 优化环境配置以提高训练速度
            env.unwrapped.config.update({
                "policy_frequency": 15,
                "simulation_frequency": 15,  # 保持与policy_frequency一致
                ### efficiency_reward:safety_reward:comfort_reward = 4:10:1 ###
                "efficiency_reward": 0.8,
                "safety_reward": 2.0, 
                "comfort_reward": 0.2,
                "svo":0.0,  # 设置SVO值
                "show_trajectories": False,  # 关闭轨迹显示
                "real_time_rendering": False,  # 关闭实时渲染
                "offscreen_rendering": False,  # 关闭离屏渲染
            })
            if seed is not None:
                env.reset(seed=seed)
            return Monitor(env)  # 包装Monitor用于统计
        return _init
    
    # 确定设备
    device = "cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu"

    # Train the model with optimizations
    if TRAIN:
        print(f"\n=== 开始训练 {len(RANDOM_SEEDS)} 个不同seed的模型 ===")
        
        for seed_idx, seed in enumerate(RANDOM_SEEDS):
            print(f"\n🔄 训练模型 {seed_idx + 1}/{len(RANDOM_SEEDS)} (seed={seed})")

            # 设置随机种子
            set_random_seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # 为每个seed创建独立的环境（关键修改1）
            if USE_PARALLEL_ENVS and N_ENVS > 1:
                print(f"创建 {N_ENVS} 个并行环境 (seed={seed})...")
                # 为每个环境设置不同的种子
                env_seeds = [seed + i for i in range(N_ENVS)]
                env = SubprocVecEnv([make_optimized_env(env_seeds[i]) for i in range(N_ENVS)])
            else:
                print(f"创建单个环境 (seed={seed})...")
                env = DummyVecEnv([make_optimized_env(seed)])
            
            # 为每个seed创建独立的保存目录（关键修改2）
            checkpoint_dir = f"./dqn_checkpoints/level1_test_1.0/" 
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 优化的回调设置
            checkpoint_callback = CheckpointCallback(
                save_freq=100000,  # 减少保存频率
                save_path=checkpoint_dir,
                name_prefix=f"level1_test_1.0_{seed}",  # 修改3：包含seed信息
                save_replay_buffer=False,  # 不保存replay buffer以节省时间
            )

            model_params = {
                "policy_kwargs": dict(net_arch=[256, 256]),
                "learning_rate": 1e-4,
                "buffer_size": 50000*N_ENVS,
                "learning_starts": 1000*N_ENVS,
                "batch_size": 32,
                "gamma": 0.99,
                "train_freq": 2,
                "gradient_steps": 1*N_ENVS,  
                "target_update_interval": 500*N_ENVS,
                "seed": seed,  # 修改4：为模型设置种子
            }
            
            # 为每个seed创建新的模型（关键修改5）
            model = DQN(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log=f"dqn_checkpoints/log/L1",  # 修改6：独立的tensorboard日志
                device=device,
                **model_params
            )

            # 开始训练
            model.learn(
                total_timesteps=int(1.0e6), 
                progress_bar=True, 
                callback=checkpoint_callback,
            )
            
            # 保存最终模型（修改7：包含seed信息）
            final_model_path = f"{checkpoint_dir}/level1_final_model_2_{seed}.zip"
            model.save(final_model_path)
            print(f"💾 模型 {seed_idx + 1} 已保存到: {final_model_path}")

            # 清理资源（修改8：每个模型训练后清理）
            del model
            env.close()
            print(f"✅ 模型 {seed_idx + 1}/{len(RANDOM_SEEDS)} (seed={seed}) 训练完成")

        print("\n🎉 所有模型训练完成!")

    
    
###################### 记得修改behavior.py中的Level1Vehicle.act方法
    # 可选：测试阶段（仅在训练完成后运行）
    if not TRAIN:
        print("\n=== 开始模型测试 ===")
        
        # 创建单个测试环境（用于视频录制）
        def make_test_env():
            test_env = gym.make("level1-v0", render_mode="rgb_array")
            test_env.unwrapped.config.update({
                "simulation_frequency": 15,  # 保持与训练一致
                "duration": 20,  # 保持与训练一致
                "policy_frequency":15,
                "efficiency_reward": 0.8,
                "safety_reward": 2.0,
                "comfort_reward": 0.2,
                "svo": math.pi/2,  # 设置SVO值
                "show_trajectories": False,
            })
            test_env.metadata["render_fps"] = 15
            return test_env
        
        # 创建单个环境并包装为向量环境
        single_test_env = make_test_env()
        
        # 方法1：直接加载模型时指定新环境（推荐）
        print("加载模型并设置测试环境...")
        # model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_test_1.0_1200000_steps.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_final_model_2_999.zip", env=single_test_env)
        model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_final_model_2_42.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_with_dl/level1_model", env=single_test_env)
        print("✅ 模型加载成功")
        
        # 创建视频录制目录
        video_dir = "dqn_checkpoints/level1_test_1.0/test_videos"
        # video_dir = "dqn_checkpoints/level1_with_dl/test_videos"
        os.makedirs(video_dir, exist_ok=True)
        

        

        # 包装RecordVideo用于视频录制
        video_env = RecordVideo(
            single_test_env, 
            video_folder=video_dir, 
            episode_trigger=lambda e: True,
            disable_logger=True
        )
        
        
        speed_list = []
        acc_list = []
        steering_list = []
        time_list = []  
        print("开始录制测试视频...")
        for episode in range(5):
            sp_list = []
            ac_list = []
            st_list = []
            ti_list = []
            done = truncated = False
            obs, info = video_env.reset()
            # print(obs.shape)
            print(f"录制第{episode+1}个视频...")
            rewards = []
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                print(action)
                obs, reward, done, truncated, info = video_env.step(action)
                # print(reward)
                rewards.append(reward)

                vehicle = video_env.env.unwrapped.vehicle  # 只获取一个环境的车辆信息
                sp_list.append(vehicle.speed)
                ac_list.append(vehicle.action['acceleration'])
                st_list.append(vehicle.action['steering'])
                ti_list.append(vehicle.timer)
                video_env.render()
            print(sum(rewards), "总奖励")
            speed_list.append(sp_list)
            acc_list.append(ac_list)
            steering_list.append(st_list)
            time_list.append(ti_list)

        video_env.close()
        # 绘制图表
        plt.figure(figsize=(12, 8))
        for i in range(5):
            plt.subplot(3, 1, 1)
            plt.plot(time_list[i], speed_list[i], label=f"Episode {i+1}")
            plt.ylabel("Speed (m/s)")
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(time_list[i], acc_list[i], label=f"Episode {i+1}")
            plt.ylabel("Acceleration (m/s²)")
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(time_list[i], steering_list[i], label=f"Episode {i+1}")
            plt.ylabel("Steering Angle (rad)")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        print("视频录制完成")
        
        # # 初始化统计变量
        # total_collisions = 0  # 总碰撞次数
        # total_episodes = 10000  # 进行的回合数
        # print("开始测试模型...")
        # for episode in range(total_episodes):
        #     done = truncated = False
        #     obs, info = single_test_env.reset()
        #     print(f"第{episode+1}个回合开始...")
        #     rewards = []
            
        #     while not (done or truncated):
        #         action, _states = model.predict(obs, deterministic=True)
        #         obs, reward, done, truncated, info = single_test_env.step(action)
        #         rewards.append(reward)
                
        #         vehicle = single_test_env.env.unwrapped.vehicle  # 获取车辆信息
                
        #         # 检查是否发生碰撞
        #         if vehicle.crashed or not vehicle.on_road:
        #             total_collisions += 1

        #     print(f"第{episode+1}个回合总奖励: {sum(rewards)}")
        #     print(f"第{episode+1}个回合总碰撞次数: {total_collisions}")


        # # 计算碰撞率
        # collision_rate = total_collisions / (total_episodes * 1.0)
        # print(f"\n总碰撞次数: {total_collisions}")
        # print(f"碰撞率: {collision_rate * 100:.2f}%")

        # print("测试完成")