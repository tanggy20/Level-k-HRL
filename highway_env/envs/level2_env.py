from typing import Dict, Text
import math
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import Level2Vehicle
from highway_env.vehicle.kinematics import Vehicle
from typing import List, Optional

Observation = np.ndarray

###  self.np_random  gymnasium/stable_baselines3的标准做法生成随机数


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
        - 前车:dv_front = ego - front
        - 后车:dv_rear  = rear - ego   (注意符号与前车相反)
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
    

class Level2Env(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    ACTIONS: Dict[int, str] = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    # # ========= 课程学习：环境侧状态 =========
    # _curriculum_enabled: bool = False
    # _curriculum_counts: List[int] | None = None
    # _curriculum_stage_idx: int = 0
    # _curriculum_replay_prev_ratio: float = 0.0

    # # 软回退相关
    # _curriculum_prev_ratio: float = 0.10   # “上一阶段”采样占比（可动态调整）
    # _curriculum_older_ratio: float = 0.10  # （可选）更早阶段的占比做少量巩固

    # # 当前 episode 实际使用的阶段（考虑了旧阶段回放抽样）
    # _episode_stage_idx_used: Optional[int] = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "KinematicSpeedObservation",},
                "observation1": {"type": "Kinematics",}, ###修改观测type 2025-10-31/ AbstractEnv增加observation1
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 3,
                "vehicles_count": 15,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 1,
                "vehicles_density": 1.5,
                "efficiency_reward": 1.0, # efficiency-prioritized(1.0,0.8,0.2) safety-prioritized(0.8,1.0,0.2)
                "safety_reward": 0.8,
                "comfort_reward": 0.2,
                "svo": -math.pi/4, #competitive:-pi/6 egoistic:0 prosocial:pi/4 altruistic:pi/2
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.Level1Vehicle",
                "training_level": 2,
                # 他人奖励参数
                "tau_current": 3.0,
                "tau_target": 2.0,
                "acc_k": 3.0,
                "w_safety": 0.7,
                "w_curr": 0.7,
                "w_targ": 0.3,
            }
        )
        return config
    


    
    # ### curriculum learning 课程学习训练方式 2025-11-11

    # # ====== 1) 启用“仅车辆数”的课程学习 ====== 
    # def enable_vehicle_count_curriculum(self, counts, replay_prev_ratio: float = 0.10,older_ratio: float = 0.10):
    #     """
    #     counts: 课程每个阶段的车辆数列表，例如 [6, 9, 12, 15]
    #     replay_prev_ratio: 在当前阶段中，以该概率抽取“之前任意阶段”的车辆数进行训练（≈10%）
    #     """
    #     if not counts or any(c < 0 for c in counts):
    #         raise ValueError("counts 必须为非负整数列表")
    #     self._curriculum_enabled = True
    #     self._curriculum_counts = [int(c) for c in counts]
    #     self._curriculum_stage_idx = 0
    #     self._curriculum_replay_prev_ratio = float(np.clip(replay_prev_ratio, 0.0, 1.0))
    #     self._curriculum_prev_ratio = float(np.clip(replay_prev_ratio, 0.0, 1.0))
    #     self._curriculum_older_ratio = float(np.clip(older_ratio, 0.0, 1.0))

    # def set_curriculum_stage(self, stage_idx: int):
    #     """由外部回调设置当前阶段索引（0..len(counts)-1），下次 reset 生效"""
    #     if not self._curriculum_enabled:
    #         raise RuntimeError("课程学习未启用")
    #     if stage_idx < 0 or stage_idx >= len(self._curriculum_counts):
    #         raise ValueError("stage_idx 超出范围")
    #     self._curriculum_stage_idx = int(stage_idx)

    # def set_prev_ratio(self, r: float):
    #     """软回退核心：动态调高/调低上一阶段采样占比。"""
    #     if not self._curriculum_enabled:
    #         raise RuntimeError("请先调用 enable_vehicle_count_curriculum()")
    #     self._curriculum_prev_ratio = float(np.clip(r, 0.0, 1.0))

    # # ====== 2) 在 reset 时，为“本回合”选择车辆数（含10%回放早期阶段） ======
    # def _pick_vehicles_count_for_episode(self) -> int:
    #     # 没启用课程就按 config 走
    #     if not self._curriculum_enabled or self._curriculum_counts is None:
    #         return self.config["vehicles_count"]
    #     # 启用课程学习


    #     # 10% 概率抽取“之前任一阶段”（均匀挑一个），否则用当前阶段
    #     if self._curriculum_stage_idx > 0 and self.np_random.random() < self._curriculum_replay_prev_ratio:
    #         chosen_idx = int(self.np_random.integers(0, self._curriculum_stage_idx))
    #     else:
    #         chosen_idx = self._curriculum_stage_idx

    #     self._episode_stage_idx_used = int(chosen_idx)  # 记录本回合实际阶段索引
    #     return int(self._curriculum_counts[chosen_idx])

    # def get_config_item(self, key: str, default=None):
    #     return self.config.get(key, default)

    def _reset(self) -> None:
        # self.config["vehicles_count"] = self._pick_vehicles_count_for_episode()
        self._create_road()
        self._create_vehicles()
        for v in self.controlled_vehicles:
            v.last_action = 1
        
        ### 统计对比SVO
        self._episode_t: Dict[Text, float] = {}        # 车辆累计时间
        self._veh_dist: Dict[Text, float] = {}      # 车辆累计行驶距离
        self._veh_v_des: Dict[Text, float] = {}     # 车辆理想速度（只记录一次）

    # # ====== 3) 暴露 info，便于统计成功率与“本回合阶段” ======
    # def _info(self, obs, action) -> dict:
    #     info = {
    #         "speed": self.vehicle.speed,
    #         "crashed": self.vehicle.crashed,
    #         "action": action,
    #         "lane_index": self.vehicle.lane_index[2],
    #         "target_lane_index": self.vehicle.target_lane_index[2],
    #         "is_success":(not self.vehicle.crashed) and self.vehicle.on_road,
    #         "episode_stage_idx": self._episode_stage_idx_used
    #     }
    #     try:
    #         info["rewards"] = self._rewards(action)
    #     except NotImplementedError:
    #         pass
    #     return info
    

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=35
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
    ### 设置训练场景（随机选择+固定交互场景）

    def _create_vehicles(self) -> None:
        """Create vehicles with random or interaction scenarios."""
        # 随机选择场景类型 (70% 随机场景, 30% 交互场景)
        scene_type = self.np_random.choice(['random', 'cut_in', 'cut_out', 'cut_in_front'], p=[0.85, 0.05, 0.05, 0.05])
        # scene_type = self.np_random.choice(['random', 'cut_in', 'cut_out', 'cut_in_front'], p=[1.0, 0.0, 0.0, 0.0])
        if scene_type == 'random':
            self._create_random_vehicles()
        elif scene_type == 'cut_in':
            self._create_interaction_vehicles_cut_in()
        elif scene_type == 'cut_out':
            self._create_interaction_vehicles_cut_out()
        elif scene_type == 'cut_in_front':
            self._create_interaction_vehicles_cut_in_front()
        
            

    def _create_random_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        self.controlled_vehicles = []

        veh_n = self.config["vehicles_count"] + self.config["controlled_vehicles"]
        numbers = np.arange(veh_n)
        ego_num = veh_n // 2  # Ego车辆编号设在中间

        base = 33
        def sample_driver_profile():
            typ = self.np_random.choice(['aggressive', 'normal', 'cautious'], p=[0.2, 0.6, 0.2])
            
            # typ = self.np_random.choice(['aggressive', 'normal', 'cautious'], p=[0.3, 0.4, 0.3])
            if typ =="aggressive":
                v_cap = float(self.np_random.uniform(base-3, base))  # 最大期望速度 30-33 m/s       
            elif typ =="normal":
                v_cap = float(self.np_random.uniform(base-5, base-3))  # 最大期望速度 28-30 m/s     
            else:
                v_cap = float(self.np_random.uniform(base-7, base-5))  # 最大期望速度 26-28 m/s
            return v_cap
        ### 2 2 2 FOR BEST  3 3 3 FOR TEST 3 2 2 FOR TRAINING 
        for num in numbers:
            if num == ego_num:
                vehicle = Level2Vehicle.create_random(
                    self.road,
                    speed=27,
                    # speed = self.np_random.uniform(22, 24),
                    # lane_id=self.config["initial_lane_id"],
                    lane_id=0,
                    spacing=self.config["ego_spacing"],
                )
                vehicle.color = (0, 0, 255)
                vehicle.max_speed_limit = self.np_random.uniform(30,32)
                # vehicle.max_speed_limit = 35
                self.controlled_vehicles.append(vehicle)
                self.road.vehicles.append(vehicle)
            else:
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.enable_lane_change = True
                vehicle.randomize_behavior()
                vehicle.max_speed_limit = sample_driver_profile()
                # vehicle.max_speed_limit = 33
                self.road.vehicles.append(vehicle)
        self.road.vehicles[0], self.road.vehicles[ego_num] = self.road.vehicles[ego_num], self.road.vehicles[0]


    ### 创建交互场景(切入) 2025-10-30
    def _create_interaction_vehicles_cut_in(self) -> None:
        """Create vehicles for cut-in interaction scenarios."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        # 1) Ego 车道随机
        ego_lane_id = int(self.np_random.integers(0, self.config["lanes_count"]))

        # 2) 随机选一个相邻车道作为切入起始车道
        if ego_lane_id == 0:
            adj_lane_id = 1
        elif ego_lane_id == self.config["lanes_count"] - 1:
            adj_lane_id = ego_lane_id - 1
        else:
            if self.np_random.random() < 0.5:
                adj_lane_id = ego_lane_id - 1
            else:
                adj_lane_id = ego_lane_id + 1


        # 3) 创建Ego车辆 (Level2Vehicle)
        ego_vehicle = Level2Vehicle.create_random(
            self.road,
            speed=self.np_random.uniform(24, 26),
            # speed = self.np_random.uniform(22, 24),
            lane_id=self.config["initial_lane_id"] or ego_lane_id,  
            spacing=self.config["ego_spacing"],
        )
        ego_vehicle.color = (0, 0, 255)        # 蓝色标识
        ego_vehicle.max_speed_limit = self.np_random.uniform(31,33)
        # ego_vehicle.max_speed_limit = 35
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)


        # 4) Ego 车道前车(Level1Vehicle)
        front_distance_1 = self.np_random.uniform(40, 45)  # 随机距离
        front_x_1 = ego_vehicle.position[0] + front_distance_1

        front_speed_1 = self.np_random.uniform(25, 28)  # 随机速度25-28m/s
        front_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            front_x_1,
            speed=front_speed_1,
            lane_id=ego_vehicle.lane_index[2],
        )
        front_vehicle_1.max_speed_limit = self.np_random.uniform(32,33)
        # 5) 切入车道前车(Level1Vehicle)
        cut_in_distance_1 = self.np_random.uniform(20, 24)  # 随机距离
        cut_in_x_1 = ego_vehicle.position[0] + cut_in_distance_1
        cut_in_speed_1 = self.np_random.uniform(28, 30)  # 随机速度25-30m/s
        
        cut_in_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            cut_in_x_1,
            speed=cut_in_speed_1,
            lane_id=adj_lane_id,
        )

        cut_in_vehicle_1.lane_change_forced = True  # 允许变道
        cut_in_vehicle_1.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], ego_lane_id)  # 切入
        cut_in_vehicle_1.max_speed_limit = self.np_random.uniform(31, 32)  # 目标速度提高
        
        front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(front_vehicle_1)

        cut_in_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(cut_in_vehicle_1)



    ### 创建交互场景(切出) 2025-10-30
    def _create_interaction_vehicles_cut_out(self) -> None:
        """Create vehicles for cut-out interaction scenarios."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        # 1) Ego 车道随机
        ego_lane_id = int(self.np_random.integers(0, self.config["lanes_count"]))
        # ego_lane_id = 0

        # 2) 随机选一个相邻车道作为切出期望车道
        if ego_lane_id == 0:
            target_lane_id = 1
        elif ego_lane_id == self.config["lanes_count"] - 1:
            target_lane_id = ego_lane_id - 1
        else:
            if self.np_random.random() < 0.5:
                target_lane_id = ego_lane_id - 1
            else:
                target_lane_id = ego_lane_id + 1


        # 3) 创建Ego车辆 (Level2Vehicle)
        ego_vehicle = Level2Vehicle.create_random(
            self.road,
            speed=self.np_random.uniform(25, 27),                 
            # speed = self.np_random.uniform(22, 24),                           
            lane_id=self.config["initial_lane_id"] or ego_lane_id,  
            spacing=self.config["ego_spacing"],
        )
        ego_vehicle.color = (0, 0, 255)        # 蓝色标识
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        # ego_vehicle.max_speed_limit = 35 ### 新增

        # 4) Ego 车道前车(Level1Vehicle)
        front_distance_1 = self.np_random.uniform(38, 45)  # 随机距离
        front_x_1 = ego_vehicle.position[0] + front_distance_1

        front_speed_1 = self.np_random.uniform(25, 28)  # 随机速度25-28m/s
        front_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            front_x_1,
            speed=front_speed_1,
            lane_id=ego_vehicle.lane_index[2],
        )
        front_vehicle_1.max_speed_limit = 30
        # 5) 切出车道前车(Level1Vehicle)
        cut_out_distance_1 = self.np_random.uniform(20, 24)  # 随机距离
        cut_out_x_1 = ego_vehicle.position[0] + cut_out_distance_1
        cut_out_speed_1 = self.np_random.uniform(28, 30)  # 随机速度25-30m/s

        cut_out_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            cut_out_x_1,
            speed=cut_out_speed_1,
            lane_id=ego_vehicle.lane_index[2],
        )

        cut_out_vehicle_1.lane_change_forced = True  # 允许变道
        cut_out_vehicle_1.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], target_lane_id)  # 切出
        cut_out_vehicle_1.max_speed_limit = self.np_random.uniform(28,31)  # 目标速度降低
        front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(front_vehicle_1)

        cut_out_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(cut_out_vehicle_1)


    ### 创建交互场景(切入有相邻车) 2025-10-31
    def _create_interaction_vehicles_cut_in_front(self) -> None:
        """Create vehicles for cut-in-front interaction scenarios."""

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        # 1) Ego 车道随机
        ego_lane_id = int(self.np_random.integers(0, self.config["lanes_count"]))
        # ego_lane_id = 0

        # 2) 随机选一个相邻车道作为切入起始车道
        if ego_lane_id == 0:
            adj_lane_id = 1
        elif ego_lane_id == self.config["lanes_count"] - 1:
            adj_lane_id = ego_lane_id - 1
        else:
            if self.np_random.random() < 0.5:
                adj_lane_id = ego_lane_id - 1
            else:
                adj_lane_id = ego_lane_id + 1


        # 3) 创建Ego车辆 (Level2Vehicle)
        ego_vehicle = Level2Vehicle.create_random(
            self.road,
            speed=self.np_random.uniform(23, 26),    
            # speed = self.np_random.uniform(22, 24),                       
            lane_id=self.config["initial_lane_id"] or ego_lane_id,  
            spacing=self.config["ego_spacing"],
        )
        ego_vehicle.color = (0, 0, 255)        # 蓝色标识
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        # ego_vehicle.max_speed_limit = 35

        # 4) Ego 车道前车(Level1Vehicle)
        front_distance_1 = self.np_random.uniform(38, 45)  # 随机距离
        front_x_1 = ego_vehicle.position[0] + front_distance_1

        front_speed_1 = self.np_random.uniform(25, 28)  # 随机速度25-28m/s
        front_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            front_x_1,
            speed=front_speed_1,
            lane_id=ego_vehicle.lane_index[2],
        )
        # 5) 切入车道前车(Level1Vehicle)
        cut_in_distance_1 = self.np_random.uniform(20, 24)  # 随机距离
        cut_in_x_1 = ego_vehicle.position[0] + cut_in_distance_1
        cut_in_speed_1 = self.np_random.uniform(28, 30)  # 随机速度25-30m/s
        
        cut_in_vehicle_1 = other_vehicles_type.create_specific(
            self.road,
            cut_in_x_1,
            speed=cut_in_speed_1,
            lane_id=adj_lane_id,
        )




        cut_in_vehicle_1.lane_change_forced = True  # 允许变道
        cut_in_vehicle_1.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], ego_lane_id)  # 切入
        cut_in_vehicle_1.max_speed_limit = self.np_random.uniform(31, 32)  # 目标速度提高
        front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(front_vehicle_1)

        cut_in_vehicle_1.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(cut_in_vehicle_1)
        # 6) 切入车道后车(Level1Vehicle)
        cut_in_distance_2 = self.np_random.uniform(10, 15)  # 随机距离
        cut_in_x_2 = ego_vehicle.position[0] + cut_in_distance_2
        cut_in_speed_2 = self.np_random.uniform(22, 25)  # 随机速度22-25m/s
        cut_in_vehicle_2 = other_vehicles_type.create_specific(
            self.road,
            cut_in_x_2,
            speed=cut_in_speed_2,
            lane_id=adj_lane_id,
        )
        cut_in_vehicle_2.max_speed_limit = self.np_random.uniform(31,33)
        cut_in_vehicle_2.lane_change_forced = True  # 允许变道
        cut_in_vehicle_2.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], adj_lane_id)  
        cut_in_vehicle_2.randomize_behavior()     # 随机化驾驶行为
        self.road.vehicles.append(cut_in_vehicle_2)
        
        if ego_lane_id == 1:
            # 7) 另一侧车道切入车辆(Level1Vehicle)
            cut_in_distance_3 = self.np_random.uniform(10, 15)  # 随机距离
            cut_in_x_3 = ego_vehicle.position[0] + cut_in_distance_3
            cut_in_speed_3 = self.np_random.uniform(22, 25)  # 随机速度22-25m/s
            cut_in_vehicle_3 = other_vehicles_type.create_specific(
                self.road,
                cut_in_x_3,
                speed=cut_in_speed_3,
                lane_id=3 - adj_lane_id - ego_lane_id,
            )
            cut_in_vehicle_3.max_speed_limit = self.np_random.uniform(31,33)
            cut_in_vehicle_3.lane_change_forced = True  # 允许变道
            cut_in_vehicle_3.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], 3 - adj_lane_id - ego_lane_id)
            cut_in_vehicle_3.randomize_behavior()     # 随机化驾驶行为
            self.road.vehicles.append(cut_in_vehicle_3)


   

     # ---------- 车辆加速度模型 ----------
    @staticmethod
    def _idm_acc(follower, leader=None):
        v_max = getattr(follower, 'max_speed_limit', 33)
        if v_max is None:  # 处理 max_speed_limit 为 None 的情况
            v_max = 33
        free_acc = 3 * (1 - (follower.speed / v_max) ** 4)
        if leader is None:
            return free_acc
        desired_gap = 10 + follower.speed + follower.speed * (follower.speed - leader.speed) / (2 * math.sqrt(15))
        d = leader.position[0] - follower.position[0]
        return free_acc - 3 * (desired_gap / utils.not_zero(d)) ** 2
    
    # ---------- 单车道评分 ----------
    def _lane_score(self, rear, front, tau):
        """评估 ego 对同车道 *rear* 车辆的综合影响 (0~1).
        - rear : 后车 (None 表示无后车, 直接返回 1)
        - front: rear 原本跟随的前车 (可能为 None)
        - tau  : TTC 阈值 (秒)
        """
        if rear is None:
            return 1.0  # 该车道后方无人 → 不产生扰动

        # ---- 1) 安全分 (TTC) → [0,1] ----
        d  = abs(rear.position[0] - self.vehicle.position[0])           # 与 ego 距离
        dv = rear.speed - self.vehicle.speed                            # 速度差 (rear − ego)
        beta = 0.8                                                      # 原 _ttc_score 上限
        raw_ttc = _ttc_score(d, dv, tau=tau, beta=beta)                # ∈ [-1,beta]
        s = np.clip(raw_ttc, 0, beta) / beta                           # 归一化到 [0,1]

        # ---- 2) 舒适分 (Δa) → (0,1] ----
        a_nom = self._idm_acc(rear, front)                              # rear 跟随 "原前车" 或自由行驶
        a_act = self._idm_acc(rear, self.vehicle)                       # rear 实际跟随 ego
        k = self.config["acc_k"]
        c = math.exp(-abs(a_act - a_nom) / k)                           # 0< c ≤1

        # ---- 3) 综合 ----
        w_s = self.config["w_safety"]
        return w_s * s + (1 - w_s) * c
    
    # ---------- 他车奖励 ----------
    def _others_reward(self):
        curr_front, curr_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        targ_front, targ_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)

        r_curr = self._lane_score(curr_rear, curr_front, self.config["tau_current"])
        r_targ = self._lane_score(targ_rear, targ_front, self.config["tau_target"])

        return self.config["w_curr"] * r_curr + self.config["w_targ"] * r_targ
    
    def _reward(self, action: Action) -> float:
        """
        The reward includes safety, efficiency, comfort and env-effect.
        :param action: the last action performed
        :return: the corresponding reward
        """
        self_rewards = self._rewards(action)
        self_reward = sum(
            self.config.get(name, 0) * reward for name, reward in self_rewards.items()
        )
        if self.config["normalize_reward"]:
            max_r = sum(self.config[k] for k in ["efficiency_reward",
                                                "safety_reward",
                                                "comfort_reward"])
            self_reward = utils.lmap(
                self_reward,
                [0, max_r],
                [0, 1],
            )
        # current_front, current_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        # target_front, target_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        # if not current_rear:
        #     current_reward = 0
        # elif not current_front:
        #     current_reward = self.acceleration(current_rear, self.vehicle) - 3 * (1 - np.power(current_rear.speed / 33, 4))
        # else:
        #     current_rear_a = self.acceleration(current_rear, self.vehicle)
        #     current_rear_pred_a = self.acceleration(current_rear, current_front)
        #     current_reward = current_rear_pred_a - current_rear_a
        # if not target_rear:
        #     target_reward = 0
        # elif not target_front:
        #     target_reward = self.acceleration(target_rear, self.vehicle) - 3 * (1 - np.power(target_rear.speed / 33, 4))
        # else:
        #     target_rear_a = self.acceleration(target_rear, self.vehicle)
        #     target_rear_pred_a = self.acceleration(target_rear, target_front)
        #     target_reward = target_rear_pred_a - target_rear_a

        # others_reward = np.clip(current_reward, -3, 3) + np.clip(target_reward, -3, 3)
        # if self.config["normalize_reward"]:
        #     others_reward = utils.lmap(
        #         others_reward,
        #         [-6, 6],
        #         [0, 1],
        #     )
        others_reward = self._others_reward()
        reward = max(math.cos(self.config["svo"]),0.1) * self_reward + math.sin(self.config["svo"]) * others_reward
        if not self.vehicle.on_road:
            reward = -100
        if self.vehicle.crashed:
            reward = -200

        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        """Aggregate different reward components."""
        lane_penalty = -0.2
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        efficiency_reward = np.clip(scaled_speed, 0, 1)

        if self.vehicle.last_action == self.vehicle.action:
            comfort_reward = 1
        else:
            comfort_reward = 0
        self.vehicle.last_action = self.vehicle.action
        

        if action == 0 or action == 2:
        #     min_rear_distance = 999
        #     min_front_distance = 999
        #     closest_front, closest_rear = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)
        #     if closest_front:
        #         min_front_distance = closest_front.position[0] - self.vehicle.position[0]
        #     if closest_rear:
        #         min_rear_distance = self.vehicle.position[0] - closest_rear.position[0]
        #     if closest_rear and (closest_rear.speed - self.vehicle.speed)>0:
        #         rear_reward = min_rear_distance / (closest_rear.speed - self.vehicle.speed + 0.01) / 3
        #         rear_reward = np.clip(rear_reward, 0, 1)
        #     else:
        #         rear_reward = 1
        #     if closest_front and (self.vehicle.speed - closest_front.speed)>0:
        #         front_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
        #         front_reward = np.clip(front_reward, 0, 1)
        #     else:
        #         front_reward = 1
        #     safety_reward = 0.5 * rear_reward + 0.5 * front_reward
        #     if min_rear_distance < 5 or min_front_distance < 5:
        #         safety_reward = 0
        # else:
        #     min_front_distance = 999
        #     closest_front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
        #     if closest_front:
        #         min_front_distance = closest_front.position[0] - self.vehicle.position[0]
        #     if closest_front and (self.vehicle.speed - closest_front.speed)>0:
        #         safety_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 3
        #         safety_reward = np.clip(safety_reward, 0, 1)
        #     elif closest_front and min_front_distance < 50:
        #         safety_reward = min_front_distance / 50.0
        #         safety_reward = np.clip(safety_reward, 0, 1)
        #     else:
        #         safety_reward = 1
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

            safety_reward = 0.5 * front_score + 0.5 * rear_score + lane_penalty
        
        else:
            front, _ = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.lane_index)
            if front is not None:
                d_front = front.position[0] - self.vehicle.position[0]
                dv_front = self.vehicle.speed - front.speed
                safety_reward = _ttc_score(d_front, dv_front, tau=3.0)
            else:
                safety_reward = 1.0    # 前方空旷
        ego_speed = self.vehicle.speed
        ego_target_speed = self.vehicle.target_speed
        # v1 = self.road.vehicles[1]
        # v2 = self.road.vehicles[2]
        # print(f"[实时速度] Ego: {ego_speed:.1f} m/s, 车1: {v1.speed:.1f} m/s, 车2: {v2.speed:.1f} m/s") 
        # print(f"[实时速度] Ego: {ego_speed:.1f} m/s, 目标速度: {ego_target_speed:.1f} m/s, 最大理想速度: {self.vehicle.max_speed_limit:.1f} m/s")
        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward
        }
        
    def _compute_time_loss(self, ego):
        """
        根据 self._episode_t, self._veh_dist, self._veh_v_des
        计算自车/他车/整体的相对时间损失，并返回一个 dict
        """

        ego_vid = id(ego)

        deltas = {}  # vid -> delta_i

        for vid, L in self._veh_dist.items():
            v_des = self._veh_v_des.get(vid)
            T_free = L / v_des  # 理想时间
            # 相对时间损失
            T_true = self._episode_t.get(vid)  # 实际时间
            delta = (T_true - T_free) / T_free
            deltas[vid] = delta

        # 自车
        delta_ego = deltas.get(ego_vid)

        # 其他车
        others = [d for vid, d in deltas.items() if vid != ego_vid]
        delta_others = float(np.mean(others)) if len(others) > 0 else 0.0

        # 整体
        all_d = list(deltas.values())
        delta_social = float(np.mean(all_d)) if len(all_d) > 0 else 0.0

        return {
            "delta_ego": float(delta_ego),
            "delta_others": float(delta_others),
            "delta_social": float(delta_social),
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
        return self.time >= self.config["duration"]
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # ==== 时间损失统计：每一步更新 ====
        # 用 dt：仿真一步的时间长度，你可以用 self.time_step 或由 config 计算
        dt = 1.0 / self.config["simulation_frequency"]

    

        # 遍历当前路上的所有车
        for v in self.road.vehicles:
            vid = id(v)
            # 记录理想速度（只记录一次）
            if vid not in self._veh_v_des:
                # 自己按需求选：target_velocity 或 车道限速
                v_des = v.max_speed_limit if v.max_speed_limit is not None else 33
                self._veh_v_des[vid] = v_des

            if getattr(v, "crashed", False):
                continue  # 碰撞的车不计入时间损失统计

            # 累积行驶距离
            self._veh_dist[vid] = self._veh_dist.get(vid, 0.0) + v.speed * dt
            self._episode_t[vid] = self._episode_t.get(vid, 0.0) + dt

            

        # ==== 在一个 episode 结束时，计算时间损失指标，塞进 info ====
        if terminated or truncated:
            info["time_loss_stats"] = self._compute_time_loss(self.vehicle)

        return obs, reward, terminated, truncated, info

