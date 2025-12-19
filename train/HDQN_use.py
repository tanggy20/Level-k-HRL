import os
import sys
import random
from collections import deque
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter

import highway_env
from highway_env.envs.common.action import HierarchicalMetaAction


# ================== 全局配置 ==================
checkpoint_dir = "./dqn_checkpoints/level1_hdqn/"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

STATE_DIM  = 35
GOAL_DIM   = 3   # 0=左换道, 1=保持, 2=右换道
ACTION_DIM = 3   # 0=减速,  1=保持, 2=加速

# ================== 车道数固定为 3（全局常量） ==================
LANES_COUNT = 3

# 并行配置
N_ENVS_TRAIN = 1
N_ENVS_TEST  = 1
USE_SUBPROC  = True  # N_ENVS_TRAIN=1 也能用，但 DummyVecEnv 更省事


# ======= 超参数 =======
NET_HIDDEN = 256
LEARNING_RATE = 1e-4

BUFFER_SIZE_PER_ENV = 50_000
LEARNING_STARTS_PER_ENV = 1_000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.005

TRAIN_FREQ = 2
GRADIENT_STEPS_PER_ENV = 1
TARGET_UPDATE_INTERVAL_PER_ENV = 500

# epsilon（底层）
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS   = 0.05
EXPLORATION_SCHEDULE_STEPS = 100_000

# ================== 上层：单独 learning_starts & 最短持续步数 ==================
META_LEARNING_STARTS = 500
META_MIN_STEPS = 10
META_MAX_STEPS = 20

# 上层 epsilon：按“决策次数”退火
META_EXPLORATION_SCHEDULE_DECISIONS = 15_000


# ================== (goal, prim) -> action_id 映射（避免每步 .index()） ==================
PAIR2IDX = {pair: i for i, pair in enumerate(HierarchicalMetaAction.PAIRS)}
DEFAULT_ACTION_ID = PAIR2IDX.get((1, 1), 1)

def pair_to_action(goal: int, prim: int) -> int:
    return PAIR2IDX.get((int(goal), int(prim)), DEFAULT_ACTION_ID)


# ================== info 解析工具 ==================
def get_terminal_obs(info: dict, fallback_obs: np.ndarray) -> np.ndarray:
    if not info:
        return np.asarray(fallback_obs, dtype=np.float32)
    obs = info.get("terminal_observation", None)
    if obs is None:
        return np.asarray(fallback_obs, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def goal_delta(goal: int) -> int:
    # 0=左, 1=保持, 2=右
    if goal == 0:
        return -1
    if goal == 2:
        return +1
    return 0


def build_goal_mask(lane_id, lanes_count, goal_dim=GOAL_DIM):
    """
    边界车道禁止无效换道：
      - lane_id==0 禁左
      - lane_id==lanes_count-1 禁右
    """
    valid = list(range(goal_dim))
    if lane_id is None or lanes_count is None:
        return valid
    if lanes_count <= 1:
        return [1]
    if lane_id <= 0 and 0 in valid:
        valid.remove(0)
    if lane_id >= lanes_count - 1 and 2 in valid:
        valid.remove(2)
    return valid


# ================== Safety-PER ReplayBuffer（底层） ==================
class SafetyPrioritizedReplayBufferHDQN:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_init: float = 0.4,
        beta_final: float = 1.0,
        beta_frames: int = 1_000_000,
        eps_priority: float = 1e-3,
        lam_safety: float = 1.0,
    ):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_init = float(beta_init)
        self.beta_final = float(beta_final)
        self.beta_frames = int(beta_frames)
        self.eps_priority = float(eps_priority)
        self.lam_safety = float(lam_safety)

        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.safety_scores = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0
        self.n_samples = 0

    def __len__(self):
        return len(self.buffer)

    def _beta(self) -> float:
        t = min(1.0, self.n_samples / max(1, self.beta_frames))
        return self.beta_init + t * (self.beta_final - self.beta_init)

    def add(self, state, goal, prim, reward, next_state, done, safety_score: float):
        data = (state, int(goal), int(prim), float(reward), next_state, float(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        idx = self.pos
        self.safety_scores[idx] = float(np.clip(safety_score, 0.0, 1.0))

        max_p = self.priorities.max() if self.priorities.max() > 0 else 1.0
        self.priorities[idx] = max_p

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        size = len(self.buffer)
        assert size > 0, "Replay buffer is empty"

        scaled_p = self.priorities[:size] ** self.alpha
        sum_p = scaled_p.sum()
        prob = (scaled_p / sum_p) if sum_p > 0 else (np.ones(size, dtype=np.float32) / float(size))

        indices = np.random.choice(size, size=batch_size, p=prob)
        self.n_samples += batch_size

        beta = self._beta()
        weights = (size * prob[indices]) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        weights = weights.astype(np.float32)

        batch = [self.buffer[i] for i in indices]
        states, goals, prims, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.array(goals, dtype=np.int64),
            np.array(prims, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        td_abs = np.abs(td_errors).reshape(-1)
        S = self.safety_scores[indices]
        p_new = (td_abs + self.eps_priority) * (1.0 + self.lam_safety * S)
        self.priorities[indices] = np.asarray(p_new, dtype=np.float32)


# ================== 线性 ε 调度 ==================
class LinearSchedule:
    def __init__(self, schedule_timesteps, final_val, init_val=1.0):
        self.schedule_timesteps = int(schedule_timesteps)
        self.final_val = float(final_val)
        self.init_val = float(init_val)

    def step(self, t):
        fraction = min(float(t) / max(1, self.schedule_timesteps), 1.0)
        return self.init_val + (self.final_val - self.init_val) * fraction


# ================== 网络定义 ==================
class MetaModel(nn.Module):
    def __init__(self, input_shape, goal_size):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, NET_HIDDEN)
        self.fc2 = nn.Linear(NET_HIDDEN, NET_HIDDEN)
        self.fc3 = nn.Linear(NET_HIDDEN, goal_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ControllerModel(nn.Module):
    def __init__(self, goal_shape, input_shape, action_size):
        super().__init__()
        self.input_dim = goal_shape + input_shape
        self.fc1 = nn.Linear(self.input_dim, NET_HIDDEN)
        self.fc2 = nn.Linear(NET_HIDDEN, NET_HIDDEN)
        self.fc3 = nn.Linear(NET_HIDDEN, action_size)

    def forward(self, goal_onehot, state):
        x = torch.cat([goal_onehot, state], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ================== HDQN Agent ==================
class HDQNAgent:
    def __init__(self, state_dim, action_dim, goal_dim, train: bool, n_envs_train: int = 1):
        self.state_dim  = state_dim
        self.goal_dim   = goal_dim
        self.action_dim = action_dim
        self.n_envs_train = max(1, n_envs_train)

        self.buffer_size = BUFFER_SIZE_PER_ENV * self.n_envs_train
        self.train_start = LEARNING_STARTS_PER_ENV * self.n_envs_train
        self.batch_size = BATCH_SIZE
        self.meta_batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.lr = LEARNING_RATE

        self.train_freq = TRAIN_FREQ
        self.gradient_steps = GRADIENT_STEPS_PER_ENV * self.n_envs_train
        self.target_update_interval = TARGET_UPDATE_INTERVAL_PER_ENV * self.n_envs_train

        self.meta_min_steps = META_MIN_STEPS
        self.meta_max_steps = META_MAX_STEPS

        self.memory = SafetyPrioritizedReplayBufferHDQN(capacity=self.buffer_size)
        self.meta_memory = deque(maxlen=self.buffer_size)

        self.meta_model = MetaModel(self.state_dim, self.goal_dim).to(device)
        self.controller_model = ControllerModel(self.goal_dim, self.state_dim, self.action_dim).to(device)
        self.target_meta_model = MetaModel(self.state_dim, self.goal_dim).to(device)
        self.target_controller_model = ControllerModel(self.goal_dim, self.state_dim, self.action_dim).to(device)
        self.update_targets(hard=True)

        self.meta_optimizer = optim.Adam(self.meta_model.parameters(), lr=self.lr)
        self.controller_optimizer = optim.Adam(self.controller_model.parameters(), lr=self.lr)

        self.rewards_r, self.steps_r, self.episodes_r = [], [], []
        self.average_r, self.average_steps_r = [], []
        self.total_steps = 0
        self.num_timesteps = 0
        self.last_train_step = 0
        self._n_calls = 0
        self._n_updates = 0

        # 分离探索
        self.epsilon_low = EXPLORATION_INITIAL_EPS
        self.epsilon_low_schedule = LinearSchedule(EXPLORATION_SCHEDULE_STEPS, EXPLORATION_FINAL_EPS, EXPLORATION_INITIAL_EPS)

        self.epsilon_meta = EXPLORATION_INITIAL_EPS
        self.meta_decision_count = 0
        self.epsilon_meta_schedule = LinearSchedule(META_EXPLORATION_SCHEDULE_DECISIONS, EXPLORATION_FINAL_EPS, EXPLORATION_INITIAL_EPS)

        self.is_train = train
        if train:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = os.path.join("dqn_checkpoints/log/L1_HDQN", f"logs_{current_time}")
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard日志: {self.log_dir}")
        else:
            self.writer = None

        # 上层状态
        self.current_goals = None
        self.need_new_goal = None
        self.meta_start_states = None
        self.meta_return = None
        self.meta_len = None
        self.meta_target_lanes = None

        # 缓存当前车道
        self.last_lane_ids = None

        # ====== Phase control ======
        self.phase = "joint"          # "low_pretrain" / "meta_train" / "joint"
        self.train_controller = True
        self.train_meta = True

    def set_phase(self, phase: str):
        assert phase in ("low_pretrain", "meta_train", "joint")
        self.phase = phase

        if phase == "low_pretrain":
            # 只训下层
            self.train_controller = True
            self.train_meta = False

            for p in self.controller_model.parameters():
                p.requires_grad_(True)
            for p in self.meta_model.parameters():
                p.requires_grad_(False)

            # 上层 goal 强制随机（不退火到贪婪）
            self.epsilon_meta = 1.0

        elif phase == "meta_train":
            # 只训上层
            self.train_controller = False
            self.train_meta = True

            for p in self.controller_model.parameters():
                p.requires_grad_(False)
            for p in self.meta_model.parameters():
                p.requires_grad_(True)

            # 下层固定贪婪执行（给上层稳定的“动力学”）
            self.epsilon_low = 0.0

            # 重新初始化上层 option 状态 & 只保留新一轮 meta 轨迹
            self.meta_memory.clear()
            self.meta_decision_count = 0
            self.current_goals = None
            self.need_new_goal = None
            self.meta_start_states = None
            self.meta_return = None
            self.meta_len = None
            self.meta_target_lanes = None
            self.last_lane_ids = None

            self.last_train_step = 0

        else:
            # 联合微调（可选）
            self.train_controller = True
            self.train_meta = True
            for p in self.controller_model.parameters():
                p.requires_grad_(True)
            for p in self.meta_model.parameters():
                p.requires_grad_(True)

    def update_targets(self, hard: bool = True, tau: float = 1.0):
        if hard:
            self.target_meta_model.load_state_dict(self.meta_model.state_dict())
            self.target_controller_model.load_state_dict(self.controller_model.state_dict())
        else:
            for tp, lp in zip(self.target_meta_model.parameters(), self.meta_model.parameters()):
                tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)
            for tp, lp in zip(self.target_controller_model.parameters(), self.controller_model.parameters()):
                tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    def reset_env_goal_state(self, env_idx: int):
        if self.need_new_goal is not None and 0 <= env_idx < len(self.need_new_goal):
            self.need_new_goal[env_idx] = True
        if self.meta_target_lanes is not None and 0 <= env_idx < len(self.meta_target_lanes):
            self.meta_target_lanes[env_idx] = None

    # ================== 上层：SMDP + 至少持续 10 步 + 固定车道数 3 ==================
    def select_goal_batch(self, states_np: np.ndarray, infos, prev_dones=None) -> np.ndarray:
        n = states_np.shape[0]

        # -------- init buffers --------
        if (getattr(self, "current_goals", None) is None or
            getattr(self, "need_new_goal", None) is None or
            getattr(self, "meta_start_states", None) is None or
            getattr(self, "meta_return", None) is None or
            getattr(self, "meta_len", None) is None or
            len(self.current_goals) != n):

            self.current_goals = np.zeros(n, dtype=np.int64)
            self.need_new_goal = np.ones(n, dtype=bool)
            self.meta_start_states = np.array(states_np, copy=True)
            self.meta_return = np.zeros(n, dtype=np.float32)
            self.meta_len = np.zeros(n, dtype=np.int32)

            self.meta_target_lanes = np.array([None] * n, dtype=object)
            self.last_lane_ids = np.array([None] * n, dtype=object)

            if not hasattr(self, "meta_decision_count"):
                self.meta_decision_count = 0

        if prev_dones is None:
            prev_dones = np.zeros(n, dtype=bool)
        else:
            prev_dones = np.array(prev_dones, dtype=bool)

        # ===== (1) update running option using previous infos/dones =====
        if infos is not None:
            for i in range(n):
                info_i = infos[i] or {}

                lane_id = info_i.get("lane_index", None)          # 你这里是 int
                tgt_lane = self.meta_target_lanes[i]              # 可能 None
                crashed = bool(info_i.get("crashed", False))

                if lane_id is not None:
                    self.last_lane_ids[i] = int(lane_id)

                episode_done = bool(prev_dones[i]) or crashed
                episode_terminal = episode_done

                # 关键：没有激活 option，就不要累计 return/len
                if self.need_new_goal[i]:
                    continue

                g = int(self.current_goals[i])

                # accumulate discounted option return
                t = int(self.meta_len[i])
                h_r_step = float(info_i.get("h_reward", 0.0))
                self.meta_return[i] += (self.gamma ** t) * h_r_step
                self.meta_len[i] = t + 1
                k = int(self.meta_len[i])  # 更新后的长度

                # termination
                if episode_done:
                    meta_done = True
                else:
                    if g == 1:  # KEEP：固定驻留（建议你单独设 meta_keep_steps）
                        meta_done = (k >= self.meta_min_steps)
                    else:       # LC：到达目标 or 超时（并且至少执行 meta_min_steps 防抖）
                        reached = (lane_id is not None and tgt_lane is not None and int(lane_id) == int(tgt_lane))
                        meta_done = reached or (k >= self.meta_max_steps)

                if meta_done:
                    # print(f"[Meta结束] Env {i}: steps={k}, return={self.meta_return[i]:.3f}, goal={g}, "
                    #     f"h_safety={info_i.get('h_safety_reward', 0.0):.3f}, "
                    #     f"eff={info_i.get('efficiency_reward', 0.0):.3f}, "
                    #     f"comfort={info_i.get('comfort_reward', 0.0):.3f}")

                    if self.is_train:
                        s0 = self.meta_start_states[i]
                        s1 = states_np[i]
                        if episode_terminal:
                            s1 = get_terminal_obs(info_i, s1)

                        self.remember_meta(s0, g, float(self.meta_return[i]), s1, bool(episode_terminal), k)

                    # reset option state
                    self.need_new_goal[i] = True
                    self.meta_return[i] = 0.0
                    self.meta_len[i] = 0
                    self.meta_target_lanes[i] = None

        # ===== (2) sample new goals for envs that need it =====
        need_indices = np.where(self.need_new_goal)[0]
        if len(need_indices) > 0:
            with torch.no_grad():
                s = torch.as_tensor(states_np[need_indices], dtype=torch.float32, device=device)
                q = self.meta_model(s).cpu().numpy()

            if self.is_train:
                if self.phase == "low_pretrain":
                    self.epsilon_meta = 1.0
                else:
                    self.epsilon_meta = self.epsilon_meta_schedule.step(self.meta_decision_count)

            new_goals = np.zeros(len(need_indices), dtype=np.int64)
            for j, idx in enumerate(need_indices):
                lane_id = self.last_lane_ids[idx]

                # lane 未知：最稳是只允许 KEEP
                if lane_id is None:
                    valid_goals = [1]
                else:
                    valid_goals = build_goal_mask(int(lane_id), LANES_COUNT, goal_dim=self.goal_dim)

                q_row = q[j].copy()
                for gg in range(self.goal_dim):
                    if gg not in valid_goals:
                        q_row[gg] = -1e9
                q_str = ", ".join([f"{val:.3f}" for val in q_row])
                # print(f"[Meta Q值] Env {idx} | Lane: {lane_id} | Q: [{q_str}] | Valid: {valid_goals}")
                greedy_goal = int(np.argmax(q_row))
                if self.phase == "low_pretrain":
                    if (1 in valid_goals) and (np.random.rand() < 0.8):
                        new_goals[j] = 1
                    else:
                        cand = [g for g in valid_goals if g != 1]
                        new_goals[j] = int(np.random.choice(cand if len(cand) > 0 else [1]))
                else:
                    if self.is_train and (np.random.rand() <= self.epsilon_meta):
                        new_goals[j] = int(np.random.choice(valid_goals))
                    else:
                        new_goals[j] = greedy_goal

            for idx, g in zip(need_indices, new_goals):
                self.current_goals[idx] = int(g)
                self.meta_start_states[idx] = np.array(states_np[idx], copy=True)
                self.need_new_goal[idx] = False

                lane_id = self.last_lane_ids[idx]
                if lane_id is not None:
                    desired = int(np.clip(int(lane_id) + goal_delta(int(g)), 0, LANES_COUNT - 1))
                    self.meta_target_lanes[idx] = desired
                else:
                    self.meta_target_lanes[idx] = None

            self.meta_decision_count += len(need_indices)

        return np.array(self.current_goals, dtype=np.int64)


    # ================== 底层动作 ==================
    def act_batch(self, goals: np.ndarray, states_np: np.ndarray) -> np.ndarray:
        n = states_np.shape[0]
        with torch.no_grad():
            g_onehot = torch.nn.functional.one_hot(
                torch.as_tensor(goals, dtype=torch.long, device=device),
                num_classes=self.goal_dim
            ).float()
            s = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            q = self.controller_model(g_onehot, s)
            greedy = torch.argmax(q, dim=1).cpu().numpy()

        if self.is_train:
            if self.phase == "meta_train":
                prims = greedy
            else:
                self.epsilon_low = self.epsilon_low_schedule.step(self.num_timesteps)
                mask = (np.random.rand(n) <= self.epsilon_low)
                rand_actions = np.random.randint(0, self.action_dim, size=n)
                prims = np.where(mask, rand_actions, greedy)
        else:
            prims = greedy
        return prims.astype(np.int64)

    # ================== 底层记忆 ==================
    def remember(self, env_idx: int, state, goal, prim, l_reward, next_state, done, info: dict):

        env_done = bool(done)
        env_terminal = env_done 

        if env_done:
            next_state = get_terminal_obs(info, next_state)

        if self.meta_len is not None and 0 <= env_idx < len(self.meta_len):
            executed_len = int(self.meta_len[env_idx]) + 1
        else:
            executed_len = 1

        lane_id = info.get("lane_index")
        tgt_lane = None
        if self.meta_target_lanes is not None and 0 <= env_idx < len(self.meta_target_lanes):
            tgt_lane = self.meta_target_lanes[env_idx]

        goal_completed = False
        if (tgt_lane is not None) and (lane_id is not None):
            goal_completed = (int(lane_id) == int(tgt_lane))

        if goal == 1:  # KEEP
            meta_should_end = (executed_len >= self.meta_min_steps) and env_done
        else:
            meta_should_end = goal_completed or (executed_len >= self.meta_max_steps)

        # done_low = bool(env_terminal or meta_should_end)
        done_low = bool(env_terminal)

        crashed = float(info.get("crashed", 0.0))
        ttc_min = float(info.get("ttc_min", 10.0))
        S_ttc = max(0.0, 1.0 - ttc_min / 5.0)
        S = crashed + (1.0 - crashed) * S_ttc
        S = float(np.clip(S, 0.0, 1.0))

        self.memory.add(state, goal, prim, l_reward, next_state, done_low, S)

    # ================== 上层记忆 ==================
    def remember_meta(self, state, goal, h_reward, next_state, done, k_steps: int):
        if not self.train_meta:
            return
        self.meta_memory.append((state, goal, h_reward, next_state, done, int(k_steps)))

    def replay(self):
        if not self.is_train:
            return
        import random as _rnd

        # ---------- 底层更新 ----------
        if self.train_controller and len(self.memory) >= self.train_start:
            (
                states_np, goals_np, prims_np, l_rewards_np,
                next_states_np, dones_np, weights_np, indices
            ) = self.memory.sample(self.batch_size)

            states      = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            goals_t     = torch.as_tensor(goals_np, dtype=torch.int64, device=device)
            prims_t     = torch.as_tensor(prims_np, dtype=torch.int64, device=device)
            l_rewards_t = torch.as_tensor(l_rewards_np, dtype=torch.float32, device=device)
            next_states = torch.as_tensor(next_states_np, dtype=torch.float32, device=device)
            dones_t     = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
            weights     = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

            g_onehot = torch.nn.functional.one_hot(goals_t, num_classes=self.goal_dim).float()

            self.controller_optimizer.zero_grad()
            q_all = self.controller_model(g_onehot, states)
            q_pred = q_all.gather(1, prims_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_online = self.controller_model(g_onehot, next_states)
                next_prim_online = q_next_online.argmax(dim=1)

                q_next_target_all = self.target_controller_model(g_onehot, next_states)
                q_next = q_next_target_all.gather(1, next_prim_online.unsqueeze(1)).squeeze(1)

                q_tgt = l_rewards_t + self.gamma * q_next * (1 - dones_t)

            per_sample_loss = F.smooth_l1_loss(q_pred, q_tgt, reduction="none")
            loss = (weights * per_sample_loss).mean()
            loss.backward()
            self.controller_optimizer.step()

            td_errors = (q_tgt - q_pred).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        # ---------- 上层更新（SMDP gamma^k） ----------
        if self.train_meta and len(self.meta_memory) >= META_LEARNING_STARTS:
            minibatch = _rnd.sample(self.meta_memory, min(len(self.meta_memory), self.meta_batch_size))
            m_states, m_goals, h_rewards, m_next_states, m_dones, k_steps = zip(*minibatch)

            m_states      = torch.as_tensor(np.array(m_states, dtype=np.float32), device=device)
            m_goals_t     = torch.as_tensor(np.array(m_goals, dtype=np.int64), device=device)
            h_rewards_t   = torch.as_tensor(np.array(h_rewards, dtype=np.float32), device=device)
            m_next_states = torch.as_tensor(np.array(m_next_states, dtype=np.float32), device=device)
            m_dones_t     = torch.as_tensor(np.array(m_dones, dtype=np.float32), device=device)
            k_steps_t     = torch.as_tensor(np.array(k_steps, dtype=np.float32), device=device)

            self.meta_optimizer.zero_grad()
            q_all = self.meta_model(m_states)
            q_pred = q_all.gather(1, m_goals_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next_online = self.meta_model(m_next_states)
                next_goal_online = q_next_online.argmax(dim=1)

                q_next_target_all = self.target_meta_model(m_next_states)
                q_next = q_next_target_all.gather(1, next_goal_online.unsqueeze(1)).squeeze(1)

                discount = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=device), k_steps_t)
                q_tgt = h_rewards_t + discount * q_next * (1 - m_dones_t)

            meta_loss = F.mse_loss(q_pred, q_tgt)
            meta_loss.backward()
            self.meta_optimizer.step()

        self._n_updates += 1

    def save(self, episode):
        if episode % 1000 == 0 and episode > 0:
            torch.save(self.meta_model.state_dict(), os.path.join(checkpoint_dir, f"meta_model_{episode}.pth"))
            torch.save(self.controller_model.state_dict(), os.path.join(checkpoint_dir, f"controller_model_{episode}.pth"))
            print(f"[保存] 第 {episode} 回合权重已保存")

    def load(self, episode):
        self.meta_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"meta_model_{episode}.pth"), map_location=device))
        self.controller_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"controller_model_{episode}.pth"), map_location=device))
        self.update_targets(hard=True)
        print(f"[加载] 权重已加载：episode={episode}")

    def record_metrics(self, episode):
        if (not self.is_train) or (len(self.rewards_r) == 0):
            return
        self.writer.add_scalar("reward/episode_reward", self.rewards_r[-1], episode)
        self.writer.add_scalar("reward/episode_steps", self.steps_r[-1], episode)
        if len(self.average_r) > 0:
            self.writer.add_scalar("reward/avg_reward_100ep", self.average_r[-1], episode)
        if len(self.average_steps_r) > 0:
            self.writer.add_scalar("reward/avg_steps_100ep", self.average_steps_r[-1], episode)

        self.writer.add_scalar("train/epsilon_low", self.epsilon_low, episode)
        self.writer.add_scalar("train/epsilon_meta", self.epsilon_meta, episode)
        self.writer.add_scalar("train/meta_decision_count", self.meta_decision_count, episode)
        self.writer.add_scalar("train/num_timesteps", self.num_timesteps, episode)
        self.writer.add_scalar("train/num_updates", self._n_updates, episode)

    def close_writer(self):
        if self.is_train and self.writer is not None:
            self.writer.close()
            print("训练结束，TensorBoard日志已关闭")


# ================== VecEnv 工厂 ==================
def make_env_factory(env_id: str, render_mode, cfg: dict, seed_base: int, rank: int,
                     record: bool = False, video_dir: str = None, record_first_k: int = 0):
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if cfg:
            env.unwrapped.config.update(cfg)
        if record:
            assert render_mode == "rgb_array"
            assert video_dir is not None
            sub_dir = os.path.join(video_dir, f"env{rank}")
            os.makedirs(sub_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=sub_dir,
                episode_trigger=lambda ep_id: ep_id < record_first_k
            )
        env.reset(seed=seed_base + rank)
        return env
    return _init


# ================== 训练 ==================
def train_parallel(agent: HDQNAgent, episodes_target: int, seed: int = 42):
    train_cfg = {
        "policy_frequency": 10, "simulation_frequency": 10,
        "efficiency_reward": 0.8, "safety_reward": 2.0, "comfort_reward": 0.4,
        "svo": 0.0, "show_trajectories": False, "real_time_rendering": False,
        "offscreen_rendering": False,
    }

    seed_base = seed
    set_random_seed(seed_base)
    random.seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)

    env_fns = [make_env_factory("level1-hdqn-v1", 'human', train_cfg, seed_base, i)
               for i in range(N_ENVS_TRAIN)]
    vec_env: VecEnv = (SubprocVecEnv(env_fns) if USE_SUBPROC else DummyVecEnv(env_fns))

    n_envs = N_ENVS_TRAIN
    states = vec_env.reset()
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    ep_steps   = np.zeros(n_envs, dtype=np.int32)
    episodes_done = 0

    infos = None
    prev_dones = np.zeros(n_envs, dtype=bool)
    print(f"=== 训练开始：n_envs={n_envs}, 目标回合={episodes_target} ===")

    while episodes_done < episodes_target:
        # goals = agent.select_goal_batch(states, infos, prev_dones)
        goals = [1] * n_envs  # 训练初期强制 KEEP
        prims = agent.act_batch(goals, states)
        actions = np.array([pair_to_action(g, p) for g, p in zip(goals, prims)], dtype=np.int32)
        next_states, rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            info_i = infos[i] or {}
            l_r = float(info_i.get("l_reward", rewards[i]))
            if agent.train_controller:
                agent.remember(i, states[i], goals[i], prims[i], l_r, next_states[i], bool(dones[i]), info_i)

        ep_rewards += rewards
        ep_steps += 1

        agent.num_timesteps += n_envs
        agent.total_steps += n_envs

        if agent.is_train:
            ready = False
            if agent.train_controller and agent.num_timesteps > agent.train_start:
                ready = True
            if agent.train_meta and len(agent.meta_memory) >= META_LEARNING_STARTS:
                ready = True
            if ready and (agent.num_timesteps - agent.last_train_step) >= agent.train_freq:
                for _ in range(agent.gradient_steps):
                    agent.replay()
                agent.last_train_step = agent.num_timesteps

        agent._n_calls += 1
        step_interval = max(agent.target_update_interval // agent.n_envs_train, 1)
        if agent._n_calls % step_interval == 0:
            agent.update_targets(hard=False, tau=TAU)

        for i in range(n_envs):
            if dones[i]:
                episodes_done += 1
                agent.reset_env_goal_state(i)

                agent.rewards_r.append(float(ep_rewards[i]))
                agent.steps_r.append(int(ep_steps[i]))
                agent.episodes_r.append(episodes_done)

                win = min(100, len(agent.rewards_r))
                agent.average_r.append(float(np.mean(agent.rewards_r[-win:])))
                agent.average_steps_r.append(float(np.mean(agent.steps_r[-win:])))

                agent.record_metrics(episodes_done)
                agent.save(episodes_done)

                ep_rewards[i] = 0.0
                ep_steps[i] = 0
                print(f"[训练] 回合 {episodes_done}/{episodes_target} 完成，奖励={agent.rewards_r[-1]:.2f}, 步数={agent.steps_r[-1]}")

        prev_dones = dones.copy()
        states = next_states

    vec_env.close()
    agent.close_writer()
    print("=== 训练完成 ===")


# ================== 测试（录制视频） ==================
def test_single(test_agent: HDQNAgent, record_episodes: int = 5, seed: int = 2025):
    test_agent.is_train = False
    test_agent.epsilon_low = 0.0
    test_agent.epsilon_meta = 0.0

    test_cfg = {
        "policy_frequency": 10, "simulation_frequency": 10,
        "efficiency_reward": 0.8, "safety_reward": 1.0, "comfort_reward": 0.2,
        "svo": 0.0, "show_trajectories": False, "real_time_rendering": False,
        "offscreen_rendering": False,
        "duration": 30,
        "lanes_count": LANES_COUNT,
    }

    env = gym.make("level1-hdqn-v1", render_mode="rgb_array")
    env.unwrapped.config.update(test_cfg)
    env.metadata["render_fps"] = 10

    video_dir = "./dqn_checkpoints/level1_hdqn/test_videos/"
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda e: True,
        disable_logger=True
    )


    print(f"=== 测试开始（录制 {record_episodes} 回合）===")

    for episode in range(record_episodes):
        state, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_steps = 0

        # 清空上层状态
        test_agent.current_goals = None
        test_agent.need_new_goal = None
        test_agent.meta_start_states = None
        test_agent.meta_return = None
        test_agent.meta_len = None
        test_agent.meta_target_lanes = None
        test_agent.last_lane_ids = None

        prev_info = None
        prev_done = False

        print(f"[测试] 开始第 {episode + 1}/{record_episodes} 回合")
        while not done:
            states_batch = state.reshape(1, -1)
            infos_batch = [prev_info] if prev_info is not None else None
            dones_batch = np.array([prev_done], dtype=bool)

            goals_batch = test_agent.select_goal_batch(states_batch, infos_batch, dones_batch)
            prims_batch = test_agent.act_batch(goals_batch, states_batch)

            goal = int(goals_batch[0])
            prim = int(prims_batch[0])

            action = pair_to_action(1, prim)
            print(f"[测试] Step {episode_steps}: 选择目标={goal}, 低层动作={prim}, 最终动作={action}")
            
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f"[测试] 底层奖励 {info.get('l_reward', 0.0):.3f}, 安全奖励 {info.get('l_safety_reward', 0.0):.3f}, 效率奖励 {info.get('efficiency_reward', 0.0):.3f}, 舒适奖励 {info.get('l_comfort_reward', 0.0):.3f}")
            done = terminated or truncated

            episode_reward += float(reward)
            episode_steps += 1

            prev_info = info
            prev_done = done
            state = next_state

            # 你的打印
            # print(f"[测试] Step {episode_steps}: 期望车道={info['target_lane_index']}, 实际车道={info['lane_index']}")

        print(f"[测试] 回合完成，奖励={episode_reward:.2f}, 步数={episode_steps}")

    env.close()
    print("=== 测试结束 ===")


# ================== 主入口 ==================
if __name__ == "__main__":
    # EPISODES = 10000
    EP_LOW = 1000
    EP_HIGH = 1000
    EP_BOTH = 1000
    TRAIN = True # True=训练；False=测试

    if TRAIN:
        agent = HDQNAgent(STATE_DIM, ACTION_DIM, GOAL_DIM, train=True, n_envs_train=N_ENVS_TRAIN)
        # train_parallel(agent, EPISODES, seed=42)
         # ===== 阶段A：pretrain controller =====
        agent.set_phase("low_pretrain")
        train_parallel(agent, EP_LOW, seed=42)

        # ===== 阶段B：train meta with frozen controller =====
        agent.set_phase("meta_train")
        train_parallel(agent, EP_HIGH, seed=43)
    else:
        test_agent = HDQNAgent(STATE_DIM, ACTION_DIM, GOAL_DIM, train=False, n_envs_train=N_ENVS_TRAIN)
        test_agent.load(episode=1000)
        test_single(test_agent, record_episodes=5, seed=2025)
