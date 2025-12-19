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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from typing import List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch.nn as nn
from stable_baselines3.common.buffers import ReplayBuffer
from types import MethodType
from stable_baselines3.dqn.dqn import DQN
from torch.nn import functional as F
from collections import namedtuple
from collections import deque

# ### 尝试加入注意力机制效果 2025-11-04
# # ===================== AttnObsExtractor：将注意力融入 SB3 DQN =====================
# class AttnObsExtractor(BaseFeaturesExtractor):
#     """
#     观测布局（固定语义槽位）：
#       - L1: 35 = ego(5) + 6槽*5                 （槽位顺序示例：lf, lr, f, r, rf, rr）
#       - L2: 36 = ego(5) + 6槽*5 + v_max(1)      （最大限速/自定义全局量在末尾）
#     每槽5维默认按: [presence, x, y, vx, vy]（presence 用于 attention 的 key_padding_mask）

#     编码流程：
#       1) 还原扁平观测 → ego(5), others(6,5), [v_max]
#       2) 共享线性把每个槽位 5→d（默认 d=64），可选加入槽位类型嵌入（ego/lf/.../rr）
#       3) 以 ego token 为 query，(ego+others) 为 key/value 做 MultiheadAttention（掩码空槽）
#       4) 残差 + LN + FFN（Transformer Encoder 风格的最小块）
#       5) 融合 v_max（若有）后输出 d 维特征给 DQN 的 MLP 头
#     """
#     def __init__(self, observation_space, d_model=64, n_heads=2):
#         super().__init__(observation_space, features_dim=d_model)
#         self.d = d_model
#         self.h = n_heads
#         self.slots = 6
#         self.feats_per = 5
#         self.seq_len = 1 + self.slots  # ego + 6 槽

#         # 共享编码：5 -> d（所有槽位与 ego 复用同一权重，避免“位置=权重”的硬偏置）
#         self.embed = nn.Linear(self.feats_per, self.d)


#         # 注意力（batch_first=True，输入输出均是 (B, S, D)）
#         self.mha = nn.MultiheadAttention(self.d, num_heads=self.h, batch_first=True)

#         # 最小 Transformer block：残差 + LN + FFN
#         self.ln1 = nn.LayerNorm(self.d)
#         self.ffn = nn.Sequential(
#             nn.Linear(self.d, 2*self.d),
#             nn.GELU(),
#             nn.Linear(2*self.d, self.d),
#         )
#         self.ln2 = nn.LayerNorm(self.d)

#         # 融合 v_max（若不存在则以 0 替代），输出维度保持 d
#         self.fuse_vmax = nn.Linear(self.d + 1, self.d)

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         输入：
#           obs: (B, 35) 或 (B, 36) —— SB3 默认批维在前
#         输出：
#           features: (B, d) —— 交给 DQN 的 policy/value MLP
#         """
#         # 保证 batch 维存在
#         if obs.dim() == 1:
#             obs = obs.unsqueeze(0)
#         B, D = obs.shape

#         # 解析是否有 v_max（L2=36 维）
#         has_vmax = (D == 36)

#         # 切片：ego(5), others(6*5), [v_max]
#         ego_raw = obs[:, 0:5]                                   # (B,5)
#         others_raw = obs[:, 5:5 + self.slots*self.feats_per]    # (B,30)
#         others_raw = others_raw.view(B, self.slots, self.feats_per)  # (B,6,5)
#         vmax = obs[:, -1:].clone() if has_vmax else obs.new_zeros(B, 1)  # (B,1)

#         # presence 掩码（True=屏蔽；ego 永远不屏蔽）
#         presence = others_raw[..., 0]                      # (B,6)
#         key_padding_mask = torch.cat(
#             [torch.zeros(B, 1, dtype=torch.bool, device=obs.device),  # ego 不屏蔽
#              (presence < 0.5)], dim=1)                    # (B,7)

#         # 共享编码到 d 维
#         ego_e = self.embed(ego_raw).unsqueeze(1)           # (B,1,d)
#         others_e = self.embed(others_raw)                  # (B,6,d)
#         seq = torch.cat([ego_e, others_e], dim=1) # (B,7,d)

#         # Ego 作为 query，(Ego+Others) 作为 K/V
#         q = seq[:, :1, :]     # (B,1,d)
#         k = v = seq           # (B,7,d)

#         # 注意：need_weights=False 时不返回 head 权重；若后续要可视化，置 True 并存储
#         out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask,
#                           need_weights=False, average_attn_weights=False)  # out: (B,1,d)

#         # 残差 + LN + FFN：与原 ego 表示做残差，再过一层 FFN
#         h = self.ln1(q + out)              # (B,1,d)
#         h2 = self.ffn(h)                   # (B,1,d)
#         g = self.ln2(h + h2).squeeze(1)    # (B,d)

#         # 融合 v_max：把全局 scalar 信息注入表示（若无则相当于拼接 0）
#         g = self.fuse_vmax(torch.cat([g, vmax], dim=-1))  # (B,d)

#         return g
# ================================================================================

# ###课程学习 2025-11-12

# class SoftRetreatCurriculumCallback(BaseCallback):
#     """
# 功能：
#     1) 只改变车辆数的课程学习（通过 env.enable_* / set_* ）；
#     2) 每阶段：探索率“重热(升/降级不同) + 线性回落到该阶段下限”；
#     3) 阈值滞回：升级阈值 up=0.90，降级阈值 down=0.80；
#     4) 烧机回合：阶段切换后的前若干回合不计入统计，避免误判。

# 用法：
#     callback = VehicleCountCurriculumCallback(counts=list(range(5,16)), episodes_per_stage=100, ...)
#     model.learn(..., callback=callback)
#     """
     
#     def __init__(self,
#                  counts: List[int],
#                  episodes_per_stage: int = 70,
#                  up_threshold: float = 0.90,
#                  weak_threshold: float = 0.80,        # 低于此阈值触发“软回退”（加大 prev_ratio）
#                  burn_in_episodes: int = 8,            # 升级后的暖机回合（不计评估）
#                  verbose: int = 0,
#                  use_confidence: bool = True,
#                  confidence_level: float = 0.95,
#                  # 采样占比策略
#                  prev_ratio_init: float = 0.30,        # 升级当下的上一阶段占比
#                  prev_ratio_target: float = 0.15,      # 稳定后的目标占比
#                  prev_ratio_max: float = 0.60,         # 软回退上限
#                  prev_ratio_min: float = 0.10,         # 不低于10%
#                  prev_ratio_step_down: float = 0.05,   # 表现好时每窗口回落步长
#                  prev_ratio_step_up: float = 0.20,     # 表现差时每窗口提升步长
#                  # ε 策略
#                  eps_floor_base: float = 0.06,
#                  eps_floor_slope: float = 0.007,
#                  eps_floor_cap: float = 0.18,
#                  eps_hi_up: float = 0.35,              # 升级后的最高ε（仅暖机期，且不参与评估）
#                  eps_warmup: float = 0.60,
#                  plateau_frac: float = 0.25,
#                  min_ramp_mult: int = 10,
#                  patience_up_windows: int = 2,         # 连续N个窗口通过才晋级
#                  episodes_per_stage_per_env: bool = True):
#         super().__init__(verbose)
#         self.counts = [int(c) for c in counts]
#         self.ep_per_stage = int(episodes_per_stage)
#         self.up_th = float(up_threshold)
#         self.weak_th = float(weak_threshold)
#         self.burn_in_episodes = int(burn_in_episodes)
#         self.use_confidence = bool(use_confidence)
#         self.confidence_level = float(confidence_level)

#         # 采样占比
#         self.prev_ratio = float(prev_ratio_init)
#         self.prev_ratio_target = float(prev_ratio_target)
#         self.prev_ratio_max = float(prev_ratio_max)
#         self.prev_ratio_min = float(prev_ratio_min)
#         self.prev_ratio_step_down = float(prev_ratio_step_down)
#         self.prev_ratio_step_up = float(prev_ratio_step_up)

#         # ε 与窗口
#         self.eps_floor_base = float(eps_floor_base)
#         self.eps_floor_slope = float(eps_floor_slope)
#         self.eps_floor_cap = float(eps_floor_cap)
#         self.eps_hi_up = float(eps_hi_up)
#         self.eps_warmup = float(eps_warmup)
#         self.plateau_frac = float(plateau_frac)
#         self.min_ramp_mult = int(min_ramp_mult)
#         self.episodes_per_stage_per_env = bool(episodes_per_stage_per_env)
#         self._approx_ep_len = 300

#         # 状态
#         self.stage_idx = 0
#         self.stage_ep = 0
#         self.stage_success = 0
#         self._up_streak = 0
#         self.stage_t0_env_steps = 0
#         self._burn_in_env_steps = 0
#         self.stage_ramp_steps = 90000
#         self._eps_floor = 0.10
#         self._eps_hi = 0.35
    
#     # ========== 工具函数 ==========

#     def _eps_floor_for_stage(self, idx: int) -> float:
#         k0 = self.counts[0]
#         k = self.counts[idx]
#         floor = self.eps_floor_base + self.eps_floor_slope * (k - k0)
#         return float(np.clip(floor, self.eps_floor_base, self.eps_floor_cap))

#     def _recompute_ramp_steps(self):
#         durations = self.training_env.env_method("call_unwrapped", "get_config_item", "duration", 20)
#         freqs = self.training_env.env_method("call_unwrapped", "get_config_item", "policy_frequency", 15)
#         dur = int(np.mean(durations)); freq = int(np.mean(freqs))
#         self._approx_ep_len = dur * freq

#         base = int(0.6 * self.ep_per_stage * self._approx_ep_len )
#         LS = int(getattr(self.model, "learning_starts", 10000))
#         self.stage_ramp_steps = max(20000, base, self.min_ramp_mult * LS)
#         self._ep_target = self.ep_per_stage 

#     def _set_eps_schedule(self, eps_hi: float, eps_floor: float, ramp_scale: float = 1.0):
#         self.stage_t0_env_steps = int(getattr(self.model, "num_timesteps", 0))
#         self._eps_floor = float(eps_floor)
#         self._eps_hi = float(max(eps_hi, self._eps_floor + 0.10))
#         curr = float(getattr(self.model, "exploration_rate", 0.0))
#         hi = max(curr, self._eps_hi)

#         ramp_env_steps = int(max(1, self.stage_ramp_steps * ramp_scale))
#         plateau_env_steps = max(
#             int(self.burn_in_episodes * self._approx_ep_len),
#             int(self.plateau_frac * ramp_env_steps)
#         )
#         self._ramp_env_steps = ramp_env_steps
#         self._plateau_env_steps = plateau_env_steps
#         self._burn_in_env_steps = int(self.burn_in_episodes * self._approx_ep_len)

#         def _schedule(_):
#             now = int(getattr(self.model, "num_timesteps", 0))
#             LS = int(getattr(self.model, "learning_starts", 0))
#             if now < LS: return float(self.eps_warmup)
#             s_env = max(0, now - self.stage_t0_env_steps)
#             if s_env < self._plateau_env_steps: return float(hi)
#             span = max(1, self._ramp_env_steps - self._plateau_env_steps)
#             t = min(1.0, (s_env - self._plateau_env_steps) / span)
#             w = 0.5 * (1.0 + math.cos(math.pi * t))
#             return float(self._eps_floor + (hi - self._eps_floor) * w)

#         self.model.exploration_schedule = _schedule
#         self.model.exploration_rate = _schedule(0.0)

#     def _wilson(self, success: int, total: int, conf: float = 0.95):
#         if total <= 0: return 0.0, 1.0
#         from math import sqrt
#         z = 1.96 if abs(conf - 0.95) < 1e-9 else 1.96
#         p = success / total
#         denom = 1 + z**2 / total
#         center = p + z*z / (2*total)
#         margin = z * sqrt((p*(1-p) + z*z/(4*total)) / total)
#         return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)

#     # ------- lifecycle -------
#     def _on_training_start(self) -> None:
#         # 启动课程（先用初始 prev_ratio），并设置阶段0
#         self.training_env.env_method("call_unwrapped", "enable_vehicle_count_curriculum",
#                                      self.counts, self.prev_ratio)
#         self.training_env.env_method("call_unwrapped", "set_curriculum_stage", 0)

#         self._recompute_ramp_steps()
#         eps_floor = self._eps_floor_for_stage(0)
#         self._set_eps_schedule(eps_hi=self.eps_hi_up, eps_floor=eps_floor, ramp_scale=1.0)
#         self.stage_ep = 0; self.stage_success = 0; self._up_streak = 0

#         # 把初始 prev_ratio 同步给环境
        
#         self.training_env.env_method("call_unwrapped", "set_prev_ratio", float(self.prev_ratio))
       

#         self.logger.record("curriculum/stage_idx", self.stage_idx)
#         self.logger.record("curriculum/vehicles_count", self.counts[self.stage_idx])
#         # --- 新增代码结束 ---

#         return True

#     def _adjust_prev_ratio(self, stronger: bool):
#         """stronger=True 表示表现差，增大上一阶段占比；否则向目标值回落。"""
#         if stronger:
#             self.prev_ratio = float(np.clip(self.prev_ratio + self.prev_ratio_step_up,
#                                             self.prev_ratio_min, self.prev_ratio_max))
#         else:
#             # 朝目标值靠拢（回落）
#             if self.prev_ratio > self.prev_ratio_target:
#                 self.prev_ratio = float(max(self.prev_ratio_target, self.prev_ratio - self.prev_ratio_step_down))
#         try:
#             self.training_env.env_method("call_unwrapped", "set_prev_ratio", float(self.prev_ratio))
#         except Exception:
#             pass

#     def _maybe_promote(self, rate: float, lo: float, hi: float):
#         """只晋级，不降级；需要连续 patience_up_windows 个窗口满足条件。"""
#         cond = (lo >= self.up_th) if self.use_confidence else (rate >= self.up_th)
#         if cond and self.stage_idx < len(self.counts) - 1:
#             self._up_streak += 1
#             if self._up_streak >= 2:  # 连续2窗口才升
#                 self.stage_idx += 1
#                 self._up_streak = 0

#                 # 宣布晋级但不禁用旧阶段采样（由 prev_ratio 控制）
#                 self.training_env.env_method("call_unwrapped", "set_curriculum_stage", self.stage_idx)

#                 # 升级：临时抬高 prev_ratio，随后再慢慢降回目标值
#                 self.prev_ratio = max(self.prev_ratio, 0.30)
#                 self._adjust_prev_ratio(stronger=False)

#                 # 重新设定 ε 与窗口
#                 self._recompute_ramp_steps()
#                 eps_floor = self._eps_floor_for_stage(self.stage_idx)
#                 self._set_eps_schedule(eps_hi=self.eps_hi_up, eps_floor=eps_floor, ramp_scale=1.4)

#     def _on_step(self) -> bool:
#         # 应用 ε 调度
#         self.model.exploration_rate = self.model.exploration_schedule(0.0)
#         self.logger.record("train/exploration_rate", float(self.model.exploration_rate))
#         self.logger.record("curriculum/stage_idx", self.stage_idx)
#         self.logger.record("curriculum/vehicles_count", self.counts[self.stage_idx])

#         # 暖机期：不统计
#         now = int(getattr(self.model, "num_timesteps", 0))
#         if now - self.stage_t0_env_steps < self._burn_in_env_steps:
#             return True

#         # 只统计“低ε回合”的成功率（训练可高ε）
#         low_eps = float(self.model.exploration_rate) <= (self._eps_floor + 0.02)

#         dones = self.locals.get("dones", [])
#         infos = self.locals.get("infos", [])
#         for done, info in zip(dones, infos):
#             if not done: continue
#             used_idx = info.get("episode_stage_idx", None)
#             if used_idx is None: continue
#             # 只统计当前“允许的最高阶段”的回合，且低ε
#             if used_idx != self.stage_idx or not low_eps:
#                 continue
#             is_success = bool(info.get("is_success", (not info.get("crashed", False)) and info.get("on_road", True)))
#             self.stage_ep += 1
#             if is_success: self.stage_success += 1

#         # 窗口结束 → 评估并调整 prev_ratio / 决定是否晋级
#         if self.stage_ep >= getattr(self, "_ep_target", self.ep_per_stage):
#             rate = self.stage_success / max(1, self.stage_ep)
#             lo, hi = self._wilson(self.stage_success, self.stage_ep, self.confidence_level) if self.use_confidence else (rate, rate)

#             # 表现差 → 软回退：提高上一阶段采样占比；表现好 → 向目标占比回落
#             self._adjust_prev_ratio(stronger=(hi <= self.weak_th))

#             # 只可能晋级，不会降级
#             self._maybe_promote(rate, lo, hi)


#             # 重置窗口
#             self.stage_ep = 0
#             self.stage_success = 0

#         return True

# class UnwrappedForwarder(gym.Wrapper):
#     """
#     统一中转：让 SubprocVecEnv/DummyVecEnv 都能调用 env.unwrapped 上的方法。
#     用法（回调侧）：
#         env.env_method("call_unwrapped", "method_name", *args, **kwargs)
#     """
#     def call_unwrapped(self, name: str, *args, **kwargs):
#         return getattr(self.unwrapped, name)(*args, **kwargs)

#     def get_config_item(self, key: str, default=None):
#         cfg = getattr(self.unwrapped, "config", {})
#         return cfg.get(key, default)


### Safety Prioritized Experience Replay Buffer 2025-11-12
SafetyReplayBufferSamples = namedtuple(
    "SafetyReplayBufferSamples",
    ["observations", "actions", "next_observations", "dones", "rewards",
     "weights", "indices", "env_indices"]
)


class SafetyPrioritizedReplayBuffer(ReplayBuffer):
    """
    基于 SB3 ReplayBuffer 的简易 PER + Safety 加权实现：
      p_i = (|TD_i| + eps)^alpha * (1 + lam * S_i)
    其中 S_i ∈ [0,1] 是安全强度（碰撞=1，TTC/THW 越危险越接近1）

    兼容性：
      - 支持 vec env
      - 采样时返回 IS 权重 w_i
      - 在训练端调用 model.update_priorities(indices, new_priorities) 回写
    """
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        beta_final: float = 1.0,
        beta_frames: int = 1_000_000,
        eps_priority: float = 1e-3,
        lam_safety: float = 1.0,
    ):
        super().__init__(buffer_size, observation_space, action_space, device,
                         n_envs=n_envs, optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self.alpha = float(alpha)
        self.beta_init = float(beta_init)
        self.beta_final = float(beta_final)
        self.beta_frames = int(beta_frames)
        self.eps_priority = float(eps_priority)
        self.lam_safety = float(lam_safety)

        # Fenwick/segment tree 可更快；此处用简化数组实现（足够好用）
        self.priorities = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # 安全强度缓存（同维度）
        self.safety_scores = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # 用于 β 退火的计数器
        self._num_samples = 0

    def _beta(self):
        # 线性退火
        t = min(1.0, self._num_samples / max(1, self.beta_frames))
        return self.beta_init + t * (self.beta_final - self.beta_init)

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        infos: 来自环境的 info 字典列表（vec-env），这里从中提取安全信号：
          - info['crashed'] → 碰撞
          - info['ttc_min'] / info['thw_min']（可选，如无则默认0）
        你只要确保环境在 info 里提供这些键即可；没有就当0。
        """
        # 先写入父类（会滚动 idx）
        super().add(obs, next_obs, action, reward, done, infos)

        # 计算本次写入位置（最近一次 add() 的结束 idx 是 self.pos）
        # 写入的是 pos-1 的那一批（处理负索引）
        idx = (self.pos - 1) % self.buffer_size

        # 组装安全强度 S ∈ [0,1]
        S_batch = np.zeros((self.n_envs,), dtype=np.float32)
        for e, info in enumerate(infos):
            crashed = float(info.get("crashed", 0.0))
            # 连续指标（如无则为0）
            ttc_min = float(info.get("ttc_min", 10.0))
            thw_min = float(info.get("thw_min", 10.0))
            # 将 TTC/THW 转成危险度（越小越危险），阈值可按需调
            S_ttc = max(0.0, 1.0 - ttc_min / 3.0)   # τ_TTC≈3s
            S_thw = max(0.0, 1.0 - thw_min / 1.5)  # τ_THW≈1.5s
            # 融合，碰撞直接顶到1
            S = max(crashed, 0.5 * S_ttc + 0.5 * S_thw)
            S_batch[e] = np.clip(S, 0.0, 1.0)

        self.safety_scores[idx, :] = S_batch

        # 初始化优先级：用当前最大 p，确保新样本被尽快采到
        max_p = self.priorities.max() if self.priorities.max() > 0 else 1.0
        self.priorities[idx, :] = max_p

    def sample(self, batch_size: int, env: None = None):
        assert self.pos > 0 or self.full, "buffer is empty"
        max_idx = self.buffer_size if self.full else self.pos

        flat_p = self.priorities[:max_idx, :].reshape(-1)
        prob = flat_p / np.clip(flat_p.sum(), 1e-8, None)

        idx_flat = np.random.choice(len(flat_p), size=batch_size, p=prob)
        idx = idx_flat // self.n_envs
        env_idx = idx_flat % self.n_envs

        # IS 权重
        self._num_samples += batch_size
        beta = self._beta()
        w = (len(flat_p) * prob[idx_flat]) ** (-beta)
        w = (w / (w.max() + 1e-8)).astype(np.float32)  # 归一化到 [0,1]

        # ==== 按我们抽到的 (idx, env_idx) 取数据 ====
        obs = self.observations[idx, env_idx, ...]
        next_obs = self.next_observations[idx, env_idx, ...]
        actions = self.actions[idx, env_idx, ...]
        dones = self.dones[idx, env_idx, ...]
        rewards = self.rewards[idx, env_idx, ...]

        return SafetyReplayBufferSamples(
            observations=self.to_torch(obs),
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),
            dones=self.to_torch(dones),
            rewards=self.to_torch(rewards),
            weights=torch.as_tensor(w, device=self.device).unsqueeze(-1),
            indices=idx,
            env_indices=env_idx,
        )

    def update_priorities(self, indices: np.ndarray, env_indices: np.ndarray, td_errors: np.ndarray):
        # td_errors: (B,)
        td_abs = np.abs(td_errors).reshape(-1)
        # 取对应样本的 S
        S = self.safety_scores[indices, env_indices]
        base = (td_abs + self.eps_priority) ** self.alpha
        boost = (1.0 + self.lam_safety * S)
        p_new = base * boost
        self.priorities[indices, env_indices] = np.asarray(p_new, dtype=np.float32)

class SafetyDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 32) -> None:
        self.policy.set_training_mode(True)
        # 若使用 LR schedule，需要在每次调用 train() 时更新一次
        self._update_learning_rate(self.policy.optimizer)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                # Double DQN：在线网选动作、目标网估值
                next_q_values = self.policy.q_net_target(replay_data.next_observations)
                next_actions = self.policy.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q = next_q_values.gather(1, next_actions)

                target_q = replay_data.rewards.reshape(-1, 1) + (1.0 - replay_data.dones.reshape(-1, 1)) * self.gamma * next_q

            current_q = self.policy.q_net(replay_data.observations).gather(
                1, replay_data.actions.long()).reshape(-1, 1)

            # Huber + IS 权重
            per_sample_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")  # (B,1)
            loss = (replay_data.weights * per_sample_loss).mean()

            # # 建议用“权重的加权平均”，而不是简单 mean(weights * loss)，可保持等效学习率规模更稳定
            # w = replay_data.weights.unsqueeze(-1)  # (B,1)
            # loss = (w * per_sample_loss).sum() / (w.sum() + 1e-8)

            self.policy.optimizer.zero_grad()
            loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy.optimizer.step()

            # 软更新（按 target_update_interval）
            # tau = getattr(self, "target_tau", 0.005)  # 可在 __init__ 里设置
            if (self._n_updates + 1) % self.target_update_interval == 0:
                self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

            # 回写优先级（用 TD 误差的绝对值）
            if hasattr(self.replay_buffer, "update_priorities"):
                td = (target_q - current_q).detach().cpu().numpy().squeeze(-1)
                self.replay_buffer.update_priorities(
                    replay_data.indices, replay_data.env_indices, td
                )

            self._n_updates += 1


class CrashRateSlidingCallback(BaseCallback):
    def __init__(self, window_episodes: int = 100, log_every: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.window = int(window_episodes)
        self.log_every = int(log_every)
        self.buf = deque(maxlen=self.window)   # 存最近N个episode是否碰撞（0/1）
        self.sum_crash = 0                     # 滑窗内的碰撞数
        self.total_ep = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            crashed = 1 if bool(info.get("crashed", False)) else 0
            # 若窗口已满，append 前先扣掉将被挤出的最旧值
            if len(self.buf) == self.buf.maxlen:
                self.sum_crash -= self.buf[0]
            self.buf.append(crashed)
            self.sum_crash += crashed
            self.total_ep += 1

            # 记录/打印
            if (self.total_ep % self.log_every == 0) and len(self.buf) > 0:
                rate = self.sum_crash / len(self.buf)
                if self.verbose:
                    print(f"[CrashRate-Sliding] last {len(self.buf)} eps: {rate:.4f} "
                          f"({self.sum_crash}/{len(self.buf)}) @ep={self.total_ep}")
                # 写 TensorBoard
                self.logger.record("safety/crash_rate_sliding", float(rate))
                self.logger.record("safety/episodes_total", int(self.total_ep))
        return True

# 性能优化配置
TRAIN = False # 改为True进行训练
USE_PARALLEL_ENVS = True  # 使用并行环境
N_ENVS =  1  # 减少到4个避免内存过载，同时保持并行效率
USE_GPU = True
# VEH_COUNTS = list(range(3, 16, 3))  # 课程学习车辆数范围：3, 6, 9, 12, 15

if __name__ == "__main__":
    
    # 优化环境配置函数
    def make_optimized_env():
        def _init():
            env = gym.make("level2-v0", render_mode=None)  
            # 优化环境配置以提高训练速度
            env.unwrapped.config.update({
                "policy_frequency": 15,
                "simulation_frequency": 15,  # 保持与policy_frequency一致
                ### efficiency_reward:safety_reward:comfort_reward = 6:4:1 ###
                "efficiency_reward": 0.8,
                "safety_reward": 1.6,
                "comfort_reward": 0.4,
                "svo": math.pi/4,  # 设置SVO值
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
    checkpoint_dir = "./dqn_checkpoints/level2_sapr_test/" 
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 优化的回调设置
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # 减少保存频率
        save_path=checkpoint_dir,
        name_prefix="level2_sapr_test", # efco efeg efpr efal saco saeg sapr saal
        save_replay_buffer=False,  # 不保存replay buffer以节省时间
    )

    # # 4) 课程回调：每阶段 100 回合，>90% 升级，<90% 降级，且≈10% 回放旧阶段
    # callback = SoftRetreatCurriculumCallback(
    # counts=VEH_COUNTS,
    # episodes_per_stage=70,
    # up_threshold=0.90,
    # weak_threshold=0.80,
    # burn_in_episodes=8,
    # use_confidence=True,
    # prev_ratio_init=0.30,
    # prev_ratio_target=0.15,
    # prev_ratio_max=0.60,
    # prev_ratio_min=0.10,
    # )

    crash_cb = CrashRateSlidingCallback(window_episodes=1000, log_every=100, verbose=1)
    callback = CallbackList([checkpoint_callback, crash_cb])

    model_params = {
        # "policy_kwargs": dict(net_arch=[256, 256],
        #                       features_extractor_class=AttnObsExtractor,
        #                       features_extractor_kwargs=dict(d_model=64, n_heads=2)
        #                      ),
        "policy_kwargs": dict(net_arch=[256, 256],),
        "learning_rate": 1e-4,
        "buffer_size": 50000*N_ENVS ,
        "learning_starts": 1000*N_ENVS,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 2,
        "gradient_steps": 1*N_ENVS,  
        "target_update_interval": 500*N_ENVS,  #
    }
    
    # ===== Safety-PER 接入参数 =====
    replay_kwargs = dict(
    alpha=0.6,          # PER 强度
    beta_init=0.4,      # IS 初值
    beta_final=1.0,     # 退火到 1
    beta_frames=int(1e6),
    eps_priority=1e-3,
    lam_safety=1.0,     # 安全放大系数 λ（0.5~2.0 试）
    )


    model = SafetyDQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="dqn_checkpoints/log/L2",
        device=device,
        replay_buffer_class=SafetyPrioritizedReplayBuffer,
        replay_buffer_kwargs=replay_kwargs,
        **model_params
    )
    

    # Train the model with optimizations
    if TRAIN:
        
        # 开始训练
        import time
        start_time = time.time()
        
        model.learn(
            total_timesteps=int(1.0e6), 
            progress_bar=True, 
            callback=callback,
        )
        
        end_time = time.time()
        training_time = end_time - start_time


        model.save(f"{checkpoint_dir}/level2_sapr_final_model_PER_NORMAL.zip")
        print(f"💾 模型已保存到: {checkpoint_dir}/level2_sapr_final_model_PER_NORMAL.zip")

        del model
        print("\n=== 训练阶段完成 ===")
    # 训练完成后清理环境
    env.close()

    
    
    
    # 可选：测试阶段（仅在训练完成后运行）
    if not TRAIN:
        print("\n=== 开始模型测试 ===")
        
        # 创建单个测试环境（用于视频录制）
        def make_test_env():
            test_env = gym.make("level2-v0", render_mode="rgb_array")
            test_env.unwrapped.config.update({
                "simulation_frequency": 15,  # 保持与训练一致
                "duration": 20,  # 保持与训练一致
                "policy_frequency":15,
                "efficiency_reward": 0.8,
                "safety_reward": 1.4,
                "comfort_reward": 0.2,
                "svo": 0.0,  # 设置SVO值
                "show_trajectories": False,
            })
            test_env.metadata["render_fps"] = 15
            return test_env
        
        # 创建单个环境并包装为向量环境
        single_test_env = make_test_env()
        
 
        print("加载模型并设置测试环境...")
        model = DQN.load("dqn_checkpoints/level2_saeg_test/level2_saeg_final_model.zip", env=single_test_env)
        # model = SafetyDQN.load("dqn_checkpoints/level2_sapr_test/level2_sapr_final_model_PER_radical.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level2_efco_test/level2_efco_test_160000_steps.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level2_saco/level2_saco_500000_steps", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_test_1.0_1200000_steps.zip", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_test_1.0/level1_final_model", env=single_test_env)
        # model = DQN.load("dqn_checkpoints/level1_with_dl/level1_model", env=single_test_env)
        print("✅ 模型加载成功")
        
        # 创建视频录制目录
        # video_dir = "dqn_checkpoints/level1_test_1.0/test_videos"
        # video_dir = "dqn_checkpoints/level1_with_dl/test_videos"
        # video_dir = "dqn_checkpoints/level2_saeg_test/test_videos"
        video_dir = "dqn_checkpoints/level2_saeg_test/test_videos"
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
                print(f"执行动作: {action}")
                obs, reward, done, truncated, info = video_env.step(action)
                # print(reward)
                rewards.append(reward)
                video_env.render()
            print(sum(rewards), "总奖励")
        video_env.close()
        print("视频录制完成")
        
