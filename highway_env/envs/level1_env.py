from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.behavior import Level1Vehicle, StackelbergLateralVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray

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


class Level1Env(AbstractEnv):
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

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction", 
                },
                "lanes_count": 3,
                "vehicles_count": 20,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 20,  # [s]
                "ego_spacing": 1.0,
                "vehicles_density": 1.5,
                "efficiency_reward": 0.8, # safety
                "safety_reward": 1.0,
                "comfort_reward": 0.2,
                "reward_speed_range": [23, 33],
                "normalize_reward": True,
                "offroad_terminal": True,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle", #Level0Vehicle可以设置为IDMVehicle
                "training_level": 1,
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        for v in self.controlled_vehicles:
            v.last_action = 1

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=33
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    # ### 原始正常训练
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []

        veh_n = self.config["vehicles_count"] + self.config["controlled_vehicles"]
        numbers = np.arange(veh_n)
        ego_num = self.np_random.choice(np.arange(0, 6), replace=False)
        for num in numbers:
            if num == ego_num:
                vehicle = Level1Vehicle.create_random(
                    self.road,
                    speed=30,
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
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
        if ego_num != 0:
            self.road.vehicles[0], self.road.vehicles[ego_num] = self.road.vehicles[ego_num], self.road.vehicles[0]

    # ### 交互场景测试（纵向） 2025.10.24
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""

    #     self.controlled_vehicles = []

    #     # 1. 创建Ego车辆 (Level1Vehicle)
    #     ego_vehicle = Level1Vehicle.create_random(
    #         self.road,
    #         speed=30,                           # 固定速度30m/s
    #         lane_id=self.config["initial_lane_id"] or 0,  # 默认中间车道
    #         spacing=self.config["ego_spacing"],
    #     )
    #     ego_vehicle.color = (0, 0, 255)        # 蓝色标识
    #     self.controlled_vehicles.append(ego_vehicle)
    #     self.road.vehicles.append(ego_vehicle)

    #     # 2. 创建2辆慢速前车 (IDMVehicle)
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

    #     front_distance_1 = self.np_random.uniform(20, 25)  # 随机距离
    #     front_x_1 = ego_vehicle.position[0] + front_distance_1

    #     front_speed_1 = self.np_random.uniform(20, 25)  # 随机速度30-33m/s
    #     front_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         front_x_1,
    #         speed=front_speed_1,
    #         lane_id=ego_vehicle.lane_index[2],
    #     )

    #     front_distance_2 = self.np_random.uniform(0, 1)  # 随机距离
    #     front_x_2 = ego_vehicle.position[0] + front_distance_2
    #     front_speed_2 = self.np_random.uniform(25, 30)  # 随机速度25-30m/s
    #     front_vehicle_2 = other_vehicles_type.create_specific(
    #         self.road,
    #         front_x_2,
    #         speed=front_speed_2,
    #         lane_id=ego_vehicle.lane_index[2] + 1,
    #     )

        
    #     front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(front_vehicle_1)

    #     front_vehicle_2.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(front_vehicle_2)


    # ### 交互场景测试（横向切入） 2025.10.25
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""

    #     self.controlled_vehicles = []

    #     # 1. 创建Ego车辆 (Level1Vehicle)
    #     ego_vehicle = Level1Vehicle.create_random(
    #         self.road,
    #         speed=25,                           # 固定速度25m/s
    #         lane_id=self.config["initial_lane_id"] or 0,  # 默认中间车道
    #         spacing=self.config["ego_spacing"],
    #     )
    #     ego_vehicle.color = (0, 0, 255)        # 蓝色标识
    #     self.controlled_vehicles.append(ego_vehicle)
    #     self.road.vehicles.append(ego_vehicle)

    #     # 2. 创建2辆慢速前车 (IDMVehicle)
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     front_distance_1 = self.np_random.uniform(25, 30)  # 随机距离
    #     front_x_1 = ego_vehicle.position[0] + front_distance_1

    #     front_speed_1 = self.np_random.uniform(30, 33)  # 随机速度30-33m/s
    #     front_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         front_x_1,
    #         speed=front_speed_1,
    #         lane_id=ego_vehicle.lane_index[2],
    #     )

    #     cut_in_distance_1 = self.np_random.uniform(15, 20)  # 随机距离
    #     cut_in_x_1 = ego_vehicle.position[0] + cut_in_distance_1
    #     cut_in_speed_1 = self.np_random.uniform(25, 30)  # 随机速度25-30m/s
    #     cut_in_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         cut_in_x_1,
    #         speed=cut_in_speed_1,
    #         lane_id=ego_vehicle.lane_index[2] + 1,
    #     )

    #     # cut_in_vehicle_1.enable_lane_change = True  # 允许变道
    #     cut_in_vehicle_1.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], 0)  # 目标变到Ego车道
    #     cut_in_vehicle_1.target_speed = front_speed_1  # 目标速度不变

    #     behind_distance_1 = self.np_random.uniform(2, 5)
    #     behind_x_1 = ego_vehicle.position[0] + behind_distance_1
    #     behind_speed_1 = self.np_random.uniform(25, 30)  # 随机速度25-30m/s
    #     behind_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         behind_x_1,
    #         speed=behind_speed_1,
    #         lane_id=ego_vehicle.lane_index[2] + 1,
    #     )
    #     behind_vehicle_1.target_speed = 32

    #     front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(front_vehicle_1)

    #     cut_in_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(cut_in_vehicle_1)

    #     behind_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(behind_vehicle_1)

    ### 交互场景测试（横向切出） 2025.10.27
    # def _create_vehicles(self) -> None:
    #     """Create some new random vehicles of a given type, and add them on the road."""

    #     self.controlled_vehicles = []

    #     # 1. 创建Ego车辆 (Level1Vehicle)
    #     ego_vehicle = Level1Vehicle.create_random(
    #         self.road,
    #         speed=20,                           # 固定速度20m/s
    #         lane_id=self.config["initial_lane_id"] or 0,  # 默认中间车道
    #         spacing=self.config["ego_spacing"],
    #     )
    #     ego_vehicle.color = (0, 0, 255)        # 蓝色标识
    #     self.controlled_vehicles.append(ego_vehicle)
    #     self.road.vehicles.append(ego_vehicle)

    #     # 2. 创建2辆慢速前车 (IDMVehicle)
    #     other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     front_distance_1 = self.np_random.uniform(49, 50)  # 随机距离
    #     front_x_1 = ego_vehicle.position[0] + front_distance_1

    #     front_speed_1 = self.np_random.uniform(22, 25)  # 随机速度25-28m/s
    #     front_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         front_x_1,
    #         speed=front_speed_1,
    #         lane_id=ego_vehicle.lane_index[2],
    #     )

    #     cut_out_distance_1 = self.np_random.uniform(25, 26)  # 随机距离
    #     cut_out_x_1 = ego_vehicle.position[0] + cut_out_distance_1
    #     cut_out_speed_1 = self.np_random.uniform(22, 25)  # 随机速度22-25m/s
    #     cut_out_vehicle_1 = other_vehicles_type.create_specific(
    #         self.road,
    #         cut_out_x_1,
    #         speed=cut_out_speed_1,
    #         lane_id=ego_vehicle.lane_index[2],
    #     )

    #     # cut_in_vehicle_1.enable_lane_change = True  # 允许变道
    #     cut_out_vehicle_1.target_lane_index = (ego_vehicle.lane_index[0], ego_vehicle.lane_index[1], 1)  # 切出
    #     cut_out_vehicle_1.target_speed = front_speed_1 + 1.0  # 目标速度不变

    #     # behind_distance_1 = self.np_random.uniform(2, 5)
    #     # behind_x_1 = ego_vehicle.position[0] + behind_distance_1
    #     # behind_speed_1 = self.np_random.uniform(25, 30)  # 随机速度25-30m/s
    #     # behind_vehicle_1 = other_vehicles_type.create_specific(
    #     #     self.road,
    #     #     behind_x_1,
    #     #     speed=behind_speed_1,
    #     #     lane_id=ego_vehicle.lane_index[2] + 1,
    #     # )
    #     # behind_vehicle_1.target_speed = 32

    #     front_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(front_vehicle_1)

    #     cut_out_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     self.road.vehicles.append(cut_out_vehicle_1)

    #     # behind_vehicle_1.randomize_behavior()     # 随机化驾驶行为
    #     # self.road.vehicles.append(behind_vehicle_1)


    def _reward(self, action: Action) -> float:
        """
        The reward includes safety, efficiency, comfort and collision-avoiding.
        :param action: the last action performed
        :return: the corresponding reward
        """
        if self.vehicle.crashed or not self.vehicle.on_road:
            return -100 if self.vehicle.crashed else -50
        r = sum(self.config[k] * v for k, v in self._rewards(action).items())
        if self.config["normalize_reward"]:
            max_r = sum(self.config[k] for k in ["efficiency_reward",
                                                "safety_reward",
                                                "comfort_reward"])
            r = utils.lmap(r, [0, max_r], [0, 1])
        return r
    

    def _rewards(self, action: Action) -> Dict[Text, float]:
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
        #         rear_reward = min_rear_distance / (closest_rear.speed - self.vehicle.speed + 0.01) / 10
        #         rear_reward = np.clip(rear_reward, 0, 1)
        #     else:
        #         rear_reward = 1
        #     if closest_front and (self.vehicle.speed - closest_front.speed)>0:
        #         front_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 10
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
        #         safety_reward = min_front_distance / (self.vehicle.speed - closest_front.speed + 0.01) / 10
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
            
        # 🖨️ 实时打印速度
        ego_speed = self.vehicle.speed
        v1 = self.road.vehicles[1]
        v2 = self.road.vehicles[2]
        # print(f"[实时速度] Ego: {ego_speed:.1f} m/s, 车1: {v1.speed:.1f} m/s, 车2: {v2.speed:.1f} m/s") 
        
        if self.road.vehicles[2].lane_index[2] == 0:
            self.road.vehicles[2].enable_lane_change = False  # 禁止再次变道
        return {
            "safety_reward": safety_reward,
            "efficiency_reward": efficiency_reward,
            "comfort_reward": comfort_reward,
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
