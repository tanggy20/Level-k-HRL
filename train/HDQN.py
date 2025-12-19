import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import highway_env
import math
import torch
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from highway_env.envs.common.action import HierarchicalMetaAction
import numpy as np
from collections import deque
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer


# 性能优化配置
TRAIN = True  # 改为True进行训练
USE_PARALLEL_ENVS = True  # 使用并行环境
N_ENVS = 1  # 并行环境数量
USE_GPU = True
logger = configure("dqn_checkpoints/log/L1_HDQN", ["stdout", "tensorboard"])

if __name__ == "__main__":
    class ControlAugmentedEnv(VecEnvWrapper):
        """包装环境以支持将高层目标与观测拼接"""
        def __init__(self, venv, goal_dim=1):
            super().__init__(venv)
            self.goal_dim = goal_dim
            self.current_goals = np.zeros((self.num_envs, goal_dim))  # 存储当前目标
            
            # 扩展观测空间：原始观测 + 目标
            original_obs_space = self.observation_space
            low = np.concatenate([original_obs_space.low, [-np.inf] * goal_dim])
            high = np.concatenate([original_obs_space.high, [np.inf] * goal_dim])
            self.action_space = gym.spaces.Discrete(3)  
            
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                dtype=original_obs_space.dtype
            )
        
        def set_goals(self, goals):
            """设置当前目标，供低层模型使用"""
            self.current_goals = goals
        def reset(self):
            # 重置时目标设为0
            obs, info = super().reset()
            return np.concatenate([obs, self.current_goals], axis=-1), info  # 返回拼接后的观测

        def step_wait(self):
            # 调用包装环境的step
            obs, rewards, dones, infos = super().step_wait()
            # 拼接目标和原始观测
            augmented_obs = np.concatenate([obs, self.current_goals], axis=-1)
            return augmented_obs, rewards, dones, infos
        
    class GoalRefinedEnv(VecEnvWrapper):
        """包装环境以支持将限制高层目标范围大小"""
        def __init__(self, venv):
            super().__init__(venv)
            self.action_space = gym.spaces.Discrete(3)
        
        def reset(self):
            return super().reset()
        
        def step_wait(self):
            obs, rewards, dones, infos = super().step_wait()
            return obs, rewards, dones, infos


    # 环境配置函数
    def make_optimized_env():
        def _init():
            env = gym.make("level1-hdqn-v1", render_mode=None)
            # 优化环境配置
            env.unwrapped.config.update({
                "policy_frequency": 15,
                "simulation_frequency": 15,
                "efficiency_reward": 0.8,
                "safety_reward": 1.0,
                "comfort_reward": 0.2,
                "svo": 0.0,  # 设置SVO值
                "show_trajectories": False,
                "real_time_rendering": False,
                "offscreen_rendering": False,
            })
            return env
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

    checkpoint_dir = "./dqn_checkpoints/level1_hdqn/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 优化的回调设置
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 减少保存频率
        save_path=checkpoint_dir,
        name_prefix="level1_hdqn",
        save_replay_buffer=False,  # 不保存replay buffer以节省时间
    )

    # Model parameters with separation between meta-level and controller-level
    model_params = {
        "learning_rate": 1e-4,
        "buffer_size": 50000 * N_ENVS,
        "learning_starts": 1000 * N_ENVS,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 2,
        "gradient_steps": 1 * N_ENVS,
        "target_update_interval": 500 * N_ENVS,
    }
    meta_env = GoalRefinedEnv(env)
    # 创建高层目标选择模型 (Meta Model)
    meta_model = DQN(
        "MlpPolicy",
        meta_env,
        verbose=0,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        **model_params,
        tensorboard_log="dqn_checkpoints/log/L1_HDQN"
    )

    controller_env = ControlAugmentedEnv(env, goal_dim=1)
    # 创建低层控制器模型 (Controller Model)
    controller_model = DQN(
        "MlpPolicy",
        controller_env,
        verbose=0,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        **model_params,
        tensorboard_log="dqn_checkpoints/log/L1_HDQN"
    )
    meta_model.set_logger(logger)
    controller_model.set_logger(logger)

    # 训练阶段
    if TRAIN:
        import time
        start_time = time.time()
        step = 0

        for episode in range(20000):  # 或根据需要调整训练周期
            print(f"\n=== 训练第 {episode + 1} 个回合 ===")
            state = env.reset()
            print(state.shape)
            done = False
            episode_reward = 0
            meta_state = state.copy()

            while not done:
                step += 1
                
                # 1. 高层离散目标
                goal_idx, _ = meta_model.predict(state)          # int
                # print(goal_idx)
                # 2. 把目标塞进低层包装器（必须是 (N_envs, goal_dim) ）
                goal_idx = np.array([goal_idx])  # 转换为二维数组
                controller_env.set_goals(goal_idx)

                # 3. 低层观测拼接：原始观测(2-D) + 目标(2-D)
                ctrl_state = np.concatenate([state, goal_idx], axis=-1)

                # 4. 低层动作
                prim, _ = controller_model.predict(ctrl_state)
                # print(prim)

                # 5. 组合成 HierarchicalMetaAction 的整数索引
                action = (goal_idx, prim)
                action_idx = HierarchicalMetaAction.PAIRS.index(action)

                # 6. 环境 step
                next_state, reward, done, info = env.step([action_idx])

                # 7. 下一底层状态
                next_ctrl_state = np.concatenate([next_state, goal_idx], axis=-1)

                # 8. 经验回放（全部 numpy）

                meta_model.replay_buffer.add(
                    meta_state, next_state, goal_idx, reward, done, info
                )
                controller_model.replay_buffer.add(
                    ctrl_state, next_ctrl_state, prim, reward, done, info
                )

                state = next_state

                if step % model_params['target_update_interval'] == 0:
                    meta_model.update_target()
                    controller_model.update_target()
                
                print(meta_model.replay_buffer.size(), controller_model.replay_buffer.size(), "replay buffer size")

                # 6. 每回合结束，更新两层网络
            if controller_model.replay_buffer.size() > 1000:
                controller_model.train(gradient_steps=1, batch_size=32)
                print("下层模型更新")
            if meta_model.replay_buffer.size() > 1000:
                meta_model.train(gradient_steps=1, batch_size=32)
                print("上层模型更新")
            # 每200个episode保存一次模型
            if episode % 2000 == 0:
                meta_model.save(f"{checkpoint_dir}/meta_model_{episode}.zip")
                controller_model.save(f"{checkpoint_dir}/controller_model_{episode}.zip")
            print(f"回合 {episode + 1} 完成，奖励: {reward}, 步数: {step}")
            
        end_time = time.time()
        training_time = end_time - start_time
        print(f"训练完成，耗时: {training_time}秒")
        meta_model.save(f"{checkpoint_dir}/final_meta_model.zip")
        controller_model.save(f"{checkpoint_dir}/final_controller_model.zip")

    # 训练完成后清理环境
    env.close()

    # 可选：测试阶段（仅在训练完成后运行）
    if not TRAIN:
        print("\n=== 开始模型测试 ===")

        # 创建单个测试环境（用于视频录制）
        def make_test_env():
            test_env = gym.make("level1-hdqn-v1", render_mode="rgb_array")
            test_env.unwrapped.config.update({
                "simulation_frequency": 15,  # 保持与训练一致
                "duration": 20,  # 保持与训练一致
                "policy_frequency": 15,
                "efficiency_reward": 0.8,
                "safety_reward": 3.0,
                "comfort_reward": 0.2,
                "svo": math.pi / 2,  # 设置SVO值
                "show_trajectories": False,
            })
            test_env.metadata["render_fps"] = 15
            return test_env

        # 加载模型
        # 1. 先建基础环境
        base_env = make_test_env()

        # 2. 用 DummyVecEnv 包一层（训练时是 DummyVecEnv/SubprocVecEnv）
        base_vec = DummyVecEnv([lambda: base_env])

        # 3. 再套训练时用过的高层、低层包装器
        meta_test_env   = GoalRefinedEnv(base_vec)          # 高层只看 3 个目标
        control_test_env = ControlAugmentedEnv(base_vec, goal_dim=1)  # 低层拼接观测

        # 4. 加载模型时把对应包装器传进去
        meta_model = DQN.load(f"{checkpoint_dir}/final_meta_model.zip",
                            env=meta_test_env)

        controller_model = DQN.load(f"{checkpoint_dir}/final_controller_model.zip",
                                    env=control_test_env)
        print("✅ 模型加载成功")

        # 创建视频录制目录
        video_dir = "dqn_checkpoints/level1_hdqn/test_videos"
        os.makedirs(video_dir, exist_ok=True)

        # 包装RecordVideo用于视频录制
        video_env = RecordVideo(
            base_env,
            video_folder=video_dir,
            episode_trigger=lambda e: True,
            disable_logger=True
        )

        print("开始录制测试视频...")
        for episode in range(5):
            done = truncated = False
            obs, info = video_env.reset()
            print(f"录制第{episode + 1}个视频...")
            rewards = []
            while not (done or truncated):
                goal, _ = meta_model.predict(obs)  # 获取高层目标
                goal = np.array([goal])
                ctrl_state = np.concatenate([obs, goal], axis=-1)  # 拼接观测和目标
                prim, _ = controller_model.predict([ctrl_state])  # 根据目标和状态选择动作
                action = (goal, prim)
                action_idx = HierarchicalMetaAction.PAIRS.index(action)
                obs, reward, done, truncated, info = video_env.step(action_idx)
                rewards.append(reward)
                video_env.render()
            print(sum(rewards), "总奖励")
        video_env.close()
        print("视频录制完成")
