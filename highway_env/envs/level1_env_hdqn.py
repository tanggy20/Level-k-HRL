from typing import Dict, Text
import numpy as np
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.behavior import Level1Vehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.action import HierarchicalMetaAction

Observation = np.ndarray

#########运行的时候记得修改behavior

# ========= TTC 评分函数 =========
def _ttc_score(d: float,
               dv: float,
               tau: float,
               d_safe: float = 10.0,
               beta: float = 0.8) -> float:
    """
    分段线性 TTC 安全评分。

    Parameters
    ----------
    d : 前/后车与自车的纵向距离  (m, 必须≥0)
    dv: 速度差 = ego_speed - other_speed (m/s)
        - 前车：dv_front = ego - front
        - 后车：dv_rear  = rear - ego   (注意符号与前车相反)
    tau: TTC 阈值 (s) -- 前车 3.0, 后车 2.0 为常用选择
    d_safe: 距离阈值，如果车距小于此值直接判极危险
    beta: 安全区最高得分 (0<b≤1)，建议 0.7~0.9

    Returns
    -------
    score ∈ [-1, beta]
        -1    : 极危险 (TTC<1s 或 距离<d_safe)
         0    : 临界安全 (TTC≈tau)
         beta : 非常安全 (TTC≥2*tau 或 后车更慢且距离充足)
    """
    # ---- 1. 极端危险：距离过近 ----
    if d < d_safe:
        return -1.0                      # 直接扣满分

    # ---- 2. 处理 dv≤0: 对向或后车更慢 ----
    #   即便 dv≤0，也不能无限安全；仍按 dv=0.1 近似
    dv_eff = max(dv, 0.1)
    ttc = d / dv_eff                     # s

    # ---- 3. 分段线性映射 ----
    if ttc < 1.0:                        # <1 s
        return -1.0
    elif ttc < tau:                      # 危险区 (-1 → 0)
        return (ttc - 1.0) / (tau - 1.0) - 1.0
    elif ttc < 2 * tau:                  # 过渡区 (0 → beta)
        return beta * (ttc - tau) / tau
    else:                                # 安全区
        return beta


class Level1HDQNEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    HIGH_LEVEL_ACTIONS: Dict[int, str] = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
    }
    HIGH_LEVEL_ACTIONS_INDEXES = {v: k for k, v in HIGH_LEVEL_ACTIONS.items()}

    LOW_LEVEL_ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER',
    }

    LOW_LEVEL_ACTIONS_INDEXES = {v: k for k, v in LOW_LEVEL_ACTIONS.items()}

    # ========= 1. 单对车辆 TTC 计算 =========
    @staticmethod
    def _pair_ttc(ego: Vehicle, other: Vehicle, front: bool = True) -> float:
        """
        计算 ego 与 other 的一维 TTC（秒）:
          - front=True  : other 在 ego 前方，车距 d = x_other - x_ego，dv = v_ego - v_other
          - front=False : other 在 ego 后方，车距 d = x_ego - x_other，dv = v_other - v_ego
        只有在 “d>0 且 dv>0（正在逼近）” 时 TTC 有意义，否则返回 +inf。
        """
        if other is None:
            return float("inf")

        if front:
            d = other.position[0] - ego.position[0]
            dv = ego.speed - other.speed
        else:
            d = ego.position[0] - other.position[0]
            dv = other.speed - ego.speed

        if d <= 0.0 or dv <= 0.0:
            return float("inf")

        return float(d / dv)  # 单位：秒

    # ========= 2. 场景级最小 TTC =========
    def _compute_ttc_min(self, ego: Vehicle) -> float:
        """
        计算本步场景中 “最危险”的 TTC：
          - 当前车道前车
          - 若有目标车道，目标车道前车 & 后车
        返回：ttc_min（秒），若完全安全则给个较大值（例如 10s）
        """
        ttcs = []

        # 当前车道：前车
        front_cur, _ = ego.road.neighbour_vehicles(ego, ego.lane_index)
        ttc_front_cur = self._pair_ttc(ego, front_cur, front=True)
        ttcs.append(ttc_front_cur)

        # 目标车道：只有在存在不同 target_lane_index 时才考虑
        try:
            target_lane_index = ego.target_lane_index
        except AttributeError:
            target_lane_index = ego.lane_index

        if target_lane_index is not None and target_lane_index != ego.lane_index:
            front_t, rear_t = ego.road.neighbour_vehicles(ego, target_lane_index)

            # 目标车道前车：变道后追尾风险
            ttc_front_t = self._pair_ttc(ego, front_t, front=True)
            ttcs.append(ttc_front_t)

            # 目标车道后车：切入被追尾风险
            ttc_rear_t = self._pair_ttc(ego, rear_t, front=False)
            ttcs.append(ttc_rear_t)

        # 去掉 inf，只保留真正有逼近关系的
        ttcs = [t for t in ttcs if np.isfinite(t)]
        if len(ttcs) == 0:
            return 10.0  # 场景完全安全时给一个较大的 TTC

        return float(min(ttcs))



    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "observation_1": {"type": "Kinematics"},
                "action": {
                    "type": "HierarchicalMetaAction",
                },
                "lanes_count": 3,
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 30,  # [s]
                "ego_spacing": 1.0,
                "vehicles_density": 1.5,
                "efficiency_reward": 0.8, # safety
                "safety_reward": 1.0,
                "comfort_reward": 0.2,
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "training_level": 1,
            }
        )
        return config


    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        for v in self.controlled_vehicles:
            v.last_action = 4

        # 缓存 reward 分量
        self._last_reward_components = None

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=33
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        def sample_driver_profile():
            typ = self.np_random.choice(['aggressive', 'normal', 'cautious'], p=[0.2, 0.6, 0.2])
            base = 33.0
            
            # typ = self.np_random.choice(['aggressive', 'normal', 'cautious'], p=[0.3, 0.4, 0.3])
            if typ =="aggressive":
                v_cap = float(self.np_random.uniform(base-3, base))  # 最大期望速度 30-33 m/s       
            elif typ =="normal":
                v_cap = float(self.np_random.uniform(base-5, base-3))  # 最大期望速度 28-30 m/s     
            else:
                v_cap = float(self.np_random.uniform(base-7, base-5))  # 最大期望速度 26-28 m/s
            return v_cap

        veh_n = self.config["vehicles_count"] + self.config["controlled_vehicles"]
        numbers = np.arange(veh_n)
        ego_num = np.random.choice(np.arange(0, 6), replace=False)
        for num in numbers:
            if num == ego_num:
                vehicle = Level1Vehicle.create_random(
                    self.road,
                    speed=28,
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"],
                )
                vehicle.color = (0, 0, 255)
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.target_speed = sample_driver_profile()
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        if ego_num != 0:
            self.road.vehicles[0], self.road.vehicles[ego_num] = self.road.vehicles[ego_num], self.road.vehicles[0]

    def _reward(self, action: Action) -> float:
        """
        The reward includes safety, efficiency, comfort and collision-avoiding.
        :param action: the last action performed
        :return: the corresponding reward
        """

        if self.vehicle.crashed or not self.vehicle.on_road:
            r = -100 if self.vehicle.crashed else -50
            self._last_reward_components = {
                "safety_reward": 0.0,
                "h_safety_reward": 0.0,
                "l_safety_reward": 0.0,
                "efficiency_reward": 0.0,
                "comfort_reward": 0.0,
                "h_comfort_reward": 0.0,
                "l_comfort_reward": 0.0,
            }
            return r
        
        rewards = self._rewards(action)
        self._last_reward_components = rewards

        r = (self.config["efficiency_reward"] * rewards["efficiency_reward"] +
             self.config["safety_reward"] * rewards["safety_reward"] +
             self.config["comfort_reward"] * rewards["comfort_reward"])
       

        if self.config["normalize_reward"]:
            max_r = sum(self.config[k] for k in ["efficiency_reward",
                                                "safety_reward",
                                                "comfort_reward"])
            r = utils.lmap(r, [0, max_r], [0, 1])
        return r
    
    def _rewards(self, action: Action) -> Dict[Text, float]:
        """
        返回若干奖励分量：
            - safety_reward      : 综合安全（高层+低层），给环境总体 reward 用
            - h_safety_reward    : 高层安全（变道相关），给上层 DDQN 用
            - l_safety_reward    : 低层安全（跟驰相关），给下层 DDQN 用
            - efficiency_reward  : 速度效率
            - comfort_reward     : 动作/纵向舒适
        """
        high_action, low_action = HierarchicalMetaAction.PAIRS[action]
        lane_penalty = -0.2

        # ===== 1. 效率：前向速度映射到 [0,1] =====
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        efficiency_reward = np.clip(scaled_speed, 0, 1)

        # ===== 2. 舒适：动作平滑 + 可选纵向舒适 =====
        # A. 纵向舒适 (Longitudinal): 避免急加急减 (Bang-Bang Control)
        # low_action 定义: 0=SLOWER, 1=IDLE, 2=FASTER
        # diff=0 (保持动作): 最舒适 (+1.0)
        # diff=1 (平滑切换, 如 加速->保持): 可接受 (+0.5)
        # diff=2 (剧烈切换, 如 加速->减速): 顿挫感强 (+0.0)
        high_prev_action, low_prev_action = HierarchicalMetaAction.PAIRS[self.vehicle.last_action]
        diff = abs(low_action - low_prev_action)
        if diff == 0:
            comfort_long = 1.0
        elif diff == 1:
            comfort_long = 0.5
        else:
            comfort_long = 0.0
        # B. 横向舒适 (Lateral): 变道本身会带来侧向加速度，保持车道最舒适
        if high_action == 1:  # IDLE
            comfort_lat = 1.0
        else:
            comfort_lat = 0.5
        comfort_reward = 0.7 * comfort_long + 0.3 * comfort_lat
        self.vehicle.last_action = action

        # ===== 3. 低层安全：本车道跟驰 =====
        front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        if front is not None:
            d_front = front.position[0] - self.vehicle.position[0]
            dv_front = self.vehicle.speed - front.speed
            l_safety_reward = _ttc_score(d_front, dv_front, tau=3.0)
        else:
            l_safety_reward = 1.0    # 前方空旷

        # ===== 4. 高层安全：与变道相关 =====
        # high_action: 0=左变道, 1=不变道, 2=右变道

        if high_action == 0 or high_action == 2:
        
            front, rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
            if front is not None:
                d_front = front.position[0] - self.vehicle.position[0]
                dv_front = self.vehicle.speed - front.speed
                front_score = _ttc_score(d_front, dv_front, tau=3.0)
            else:                                  # 目标车道前方空旷
                front_score = 1.0
            
            if rear is not None:
                d_rear = self.vehicle.position[0] - rear.position[0]
                dv_rear = rear.speed - self.vehicle.speed
                rear_score = _ttc_score(d_rear, dv_rear, tau=2.0)
            else:                                  # 无后车
                rear_score = 1.0

            h_safety_reward = 0.5 * front_score + 0.5 * rear_score + lane_penalty
        else:
            h_safety_reward = l_safety_reward   
        
        

        if high_action == 0 or high_action == 2:
            safety_reward = h_safety_reward
        else:
            safety_reward = l_safety_reward

        ego_speed = self.vehicle.speed
        ego_target_speed = self.vehicle.target_speed
        v1 = self.road.vehicles[1]
        v2 = self.road.vehicles[2]
        # print(f"[实时速度] Ego: {ego_speed:.1f} m/s, 车1: {v1.speed:.1f} m/s, 车2: {v2.speed:.1f} m/s")     
        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward,
            "h_safety_reward": h_safety_reward,
            "h_comfort_reward": comfort_lat,
            "l_comfort_reward": comfort_long,
            "l_safety_reward": l_safety_reward,
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        # print(self.time)
        return self.time >= self.config["duration"]
    

    # ========= 重写 step：把上下层奖励放进 info =========
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        rewards = getattr(self, '_last_reward_components', None)
        if self.vehicle.crashed or not self.vehicle.on_road:
            h_reward = l_reward = reward
            info['ttc_min'] = 0.0
        else:
            h_reward = rewards["h_safety_reward"] * self.config["safety_reward"] + \
                    rewards["efficiency_reward"] * self.config["efficiency_reward"] + \
                    rewards["h_comfort_reward"] * self.config["comfort_reward"]
            l_reward = rewards["l_safety_reward"] * self.config["safety_reward"] + \
                    rewards["efficiency_reward"] * self.config["efficiency_reward"] + \
                    rewards["l_comfort_reward"] * self.config["comfort_reward"]
            max_r = self.config["safety_reward"] + self.config["efficiency_reward"] + self.config["comfort_reward"]
            if self.config["normalize_reward"]:
                h_reward = utils.lmap(h_reward, [0, max_r], [0, 1])
                l_reward = utils.lmap(l_reward, [0, max_r], [0, 1])
            info['ttc_min'] = self._compute_ttc_min(self.vehicle)
        info["h_reward"] = h_reward
        info["l_reward"] = l_reward
        info["efficiency_reward"] = rewards["efficiency_reward"]
        info["h_safety_reward"] = rewards["h_safety_reward"]
        info["l_safety_reward"] = rewards["l_safety_reward"]
        info['h_comfort_reward'] = rewards["h_comfort_reward"]
        info['l_comfort_reward'] = rewards["l_comfort_reward"]
        info["comfort_reward"] = rewards["comfort_reward"]


        return obs, reward, terminated, truncated, info