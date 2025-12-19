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
from stable_baselines3.common.utils import get_linear_fn

##修改两个地方：①behavior.py中Level1Vehicle.act方法，取消对原有act方法的注释；②behavior.py中Level1Vehicle类的__init__方法，添加self.GOALS和self.PRIMS属性。

# 性能优化配置
TRAIN = False # 改为True进行训练
USE_PARALLEL_ENVS = True  # 使用并行环境
N_ENVS =  8  # 减少到4个避免内存过载，同时保持并行效率
USE_GPU = True

if __name__ == "__main__":
    
    # 优化环境配置函数
    def make_optimized_env():
        def _init():
            env = gym.make("level1-hdqn-v0", render_mode=None)
            # 优化环境配置以提高训练速度
            env.unwrapped.config.update({
                "policy_frequency": 15,
                "simulation_frequency": 15,  # 保持与policy_frequency一致
                "efficiency_reward": 0.8,
                "safety_reward": 4.0,
                "comfort_reward": 0.2,
                "svo":0.0,  # 设置SVO值
                "show_trajectories": False,  # 关闭轨迹显示
                "real_time_rendering": False,  # 关闭实时渲染
                "offscreen_rendering": False,  # 关闭离屏渲染
            })
            return Monitor(env)  # 包装Monitor用于统计
        return _init
    
    # 创建向量化环境
    if USE_PARALLEL_ENVS and N_ENVS > 1:
        print(f"创建 {N_ENVS} 个并行环境...")
        # 使用SubprocVecEnv提高并行效率
        env = SubprocVecEnv([make_optimized_env() for _ in range(N_ENVS)])
    else:
        print("创建单个环境...")
        env = DummyVecEnv([make_optimized_env()])
        
    # 确定设备
    device = "cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu"

    # checkpoint_dir = "./dqn_checkpoints/level2_test_saal/" # efco efeg efpr efal saco saeg sapr saal
    checkpoint_dir = "./dqn_checkpoints/level1_hdqn/" 
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 优化的回调设置
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 减少保存频率
        save_path=checkpoint_dir,
        name_prefix="level1_hdqn", # efco efeg efpr efal saco saeg sapr saal
        save_replay_buffer=False,  # 不保存replay buffer以节省时间
    )
    

    model_params = {
        "policy_kwargs": dict(net_arch=[256, 256]),
        "learning_rate": 1e-4,
        "buffer_size": 50000*N_ENVS ,
        "learning_starts": 1000*N_ENVS,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 2,
        "gradient_steps": 1*N_ENVS,  
        "target_update_interval": 500*N_ENVS,  #
    }
    
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="dqn_checkpoints/log/L1_HDQN",
        device=device,
        **model_params
    )
    

    # Train the model with optimizations
    if TRAIN:
        
        # 开始训练
        import time
        start_time = time.time()
        
        model.learn(
            total_timesteps=int(2.0e6), 
            progress_bar=True, 
            callback=checkpoint_callback,
        )
        
        end_time = time.time()
        training_time = end_time - start_time
    
        
        model.save(f"{checkpoint_dir}/level1_hdqn_final_model_4.zip")
        print(f"💾 模型已保存到: {checkpoint_dir}/level1_hdqn_final_model_4.zip")

        del model
        print("\n=== 训练阶段完成 ===")
    # 训练完成后清理环境
    env.close()

    
    
    
    # 可选：测试阶段（仅在训练完成后运行）
    if not TRAIN:
        print("\n=== 开始模型测试 ===")
        
        # 创建单个测试环境（用于视频录制）
        def make_test_env():
            test_env = gym.make("level1-hdqn-v0", render_mode="rgb_array")
            test_env.unwrapped.config.update({
                "simulation_frequency": 15,  # 保持与训练一致
                "duration": 20,  # 保持与训练一致
                "policy_frequency":15,
                "efficiency_reward": 0.8,
                "safety_reward": 3.0,
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
        # model = DQN.load("dqn_checkpoints/level2_saeg_test/level2_saeg_final_model.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level2_saco/level2_saco_500000_steps", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_test_1.0_1200000_steps.zip", env=single_test_env)
        model = DQN.load("dqn_checkpoints/level1_hdqn/level1_hdqn_final_model_3", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_with_dl/level1_model", env=single_test_env)
        print("✅ 模型加载成功")
        
        # 创建视频录制目录
        video_dir = "dqn_checkpoints/level1_hdqn/test_videos"
        # video_dir = "dqn_checkpoints/level1_with_dl/test_videos"
        # video_dir = "dqn_checkpoints/level2_saeg_test/test_videos"
        # video_dir = "dqn_checkpoints/level2_saal/test_videos"
        os.makedirs(video_dir, exist_ok=True)
        

        

        # 包装RecordVideo用于视频录制
        video_env = RecordVideo(
            single_test_env, 
            video_folder=video_dir, 
            episode_trigger=lambda e: True,
            disable_logger=True
        )
        
        
        
        print("开始录制测试视频...")
        for episode in range(5):
            done = truncated = False
            obs, info = video_env.reset()
            print(f"录制第{episode+1}个视频...")
            rewards = []
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = video_env.step(action)
                # print(reward)
                rewards.append(reward)
                video_env.render()
            print(sum(rewards), "总奖励")
        video_env.close()
        print("视频录制完成")
        
