import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import highway_env  
from tqdm import trange
import matplotlib.pyplot as plt
import multiprocessing as mp

# ===================== 全局常量配置 =====================
EPS = 1e-3
TTC_CLIP_MAX = 100.0  # TTC 截断上限（秒）
THW_CLIP_MAX = 5.0    # THW 截断上限（秒）
YMAX = 0.10           # 直方图 y 轴上限（概率）
TTC_BINS = 60
THW_BINS = 50

# --- 换道判断阈值（可按需要微调） ---
LANE_CHANGE_VY_THRESHOLD = 0.3   # 横向速度阈值 (m/s)
LANE_CHANGE_LAT_THRESHOLD = 0.5  # 横向偏移阈值 (m)

# --- 是否在本脚本里画图 ---
PLOT_FIGURES = False  # 默认不画图，只保存 npy；需要图的话改成 True

# ========= 三种 L2 风格的模型路径 & 权重（按你自己的模型改） =========
STYLE_CONFIG = [
    # (风格名, 模型路径, 在 Level-2 人群中的比例权重)
    ("aggressive",
     "dqn_checkpoints/level2_efeg_test/level2_efeg_final_model_PER_RADICAL.zip",
     0.3),
    ("normal",
     "dqn_checkpoints/level2_efeg_test/level2_efeg_final_model_PER_NORMAL.zip",
     0.4),
    ("conservative",
     "dqn_checkpoints/level2_saeg_test/level2_saeg_final_model_PER_safe.zip",
     0.3),
]

# ========= HighD 预先统计好的 TTC/THW（只看跟驰）路径 =========
HIGH_D_TTC_CF_PATH = "highd_stats/highd_level2_ttc_cf.npy"
HIGH_D_THW_CF_PATH = "highd_stats/highd_level2_thw_cf.npy"

# ========= 仿真统计结果的保存目录（本脚本新增） =========
SIM_SAVE_DIR = "sim_stats"  # 里面会自动存 npy
os.makedirs(SIM_SAVE_DIR, exist_ok=True)


# ===================== 驾驶状态判定：跟驰 / 换道 =====================
def get_driving_state(vehicle) -> str:
    """
    判断车辆当前的驾驶状态：跟驰 或 换道
    返回："car_following" 或 "lane_changing"
    """
    vy = abs(vehicle.velocity[1])  # 横向速度
    # print(f"横向速度: {vy}")
    if vehicle.lane is not None:
        _, lat_offset = vehicle.lane.local_coordinates(vehicle.position)
        lat_offset = abs(lat_offset)
        # print(f"横向偏移: {lat_offset}")
    else:
        lat_offset = 0.0

    if vy > LANE_CHANGE_VY_THRESHOLD or lat_offset > LANE_CHANGE_LAT_THRESHOLD:
        return "lane_changing"
    else:
        return "car_following"


# ===================== 计算前车 & TTC/THW =====================
def _leader_same_lane(road, ego):
    """
    返回与自车同车道、且纵向在前方的最近车辆 (leader) 及其 Δx。
    若找不到则返回 (None, np.inf)。
    """
    leader, min_dx = None, np.inf
    ex = float(ego.position[0])

    try:
        ego_lane = ego.lane_index[2]
    except Exception:
        ego_lane = None

    for v in road.vehicles:
        if v is ego:
            continue
        try:
            lane = v.lane_index[2]
        except Exception:
            continue
        if ego_lane is not None and lane != ego_lane:
            continue
        dx = float(v.position[0] - ex)
        if dx > 0.0 and dx < min_dx:
            min_dx = dx
            leader = v
    return leader, min_dx


def compute_ttc_thw(road, ego):
    """
    计算同车道前车的 TTC 与 THW：
      - TTC = Δx / Δv，仅当自车对前车闭合（Δv > 0）才为有限，否则记为 ∞
      - THW = Δx / v_ego，仅当 v_ego > 0 才为有限，否则记为 ∞
    若无前车则两者均为 ∞
    """
    leader, dx = _leader_same_lane(road, ego)
    if leader is None:
        return np.inf, np.inf

    rel_v = float(ego.velocity[0] - leader.velocity[0])  # 自车对前车相对速度
    ttc = (dx / rel_v) if (rel_v > EPS) else np.inf

    v_ego = float(ego.velocity[0])
    thw = (dx / v_ego) if (v_ego > EPS) else np.inf

    return ttc, thw


# ===================== 单进程：某一风格跑若干回合 =====================
def run_episodes_for_style(episodes_in_batch: int, model_path: str, process_id: int):
    """
    由单个进程运行若干回合，使用指定的 L2 风格模型。
    返回：(碰撞次数, 该进程的统计字典)
    """
    local_collisions = 0
    local = {
        "ttc_all": [],
        "thw_all": [],
        "ttc_cf": [],   # 跟驰 car-following
        "thw_cf": [],
        "ttc_lc": [],   # 换道 lane-changing
        "thw_lc": [],
        # 新增：速度
        "v_all": [],
        "v_cf": [],
        "v_lc": [],
    }

    # 创建环境
    env = gym.make("level2-v0", render_mode=None)
    env.unwrapped.config["other_vehicles_type"] = "highway_env.vehicle.behavior.Level1Vehicle"
    env.unwrapped.config["training_level"] = 2
    env.unwrapped.config["show_trajectories"] = False
    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.config["policy_frequency"] = 15

    # 加载模型（多进程建议用 CPU）
    model = DQN.load(
        model_path,
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        },
        device="cpu",
    )

    for _ in trange(episodes_in_batch, desc=f"进程 {process_id}", position=process_id):
        obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            ego = env.unwrapped.road.vehicles[0]

            # 当前状态：跟驰 / 换道
            state = get_driving_state(ego)

            # TTC / THW
            ttc, thw = compute_ttc_thw(env.unwrapped.road, ego)

            # 纵向速度 v（m/s）
            v_long = float(ego.velocity[0])
            local["v_all"].append(v_long)
            if state == "car_following":
                local["v_cf"].append(v_long)
            else:
                local["v_lc"].append(v_long)

            if np.isfinite(ttc):
                ttc = min(ttc, TTC_CLIP_MAX)
                local["ttc_all"].append(ttc)
                if state == "car_following":
                    local["ttc_cf"].append(ttc)
                else:
                    local["ttc_lc"].append(ttc)

            if np.isfinite(thw):
                thw = min(thw, THW_CLIP_MAX)
                local["thw_all"].append(thw)
                if state == "car_following":
                    local["thw_cf"].append(thw)
                else:
                    local["thw_lc"].append(thw)

        if info.get("crashed", False):
            local_collisions += 1

    env.close()
    return local_collisions, local


# ===================== 合并多进程统计 =====================
def merge_local_stats(sample_dicts):
    merged = {
        "ttc_all": [],
        "thw_all": [],
        "ttc_cf": [],
        "thw_cf": [],
        "ttc_lc": [],
        "thw_lc": [],
        "v_all": [],
        "v_cf": [],
        "v_lc": [],
    }
    for d in sample_dicts:
        for k in merged.keys():
            merged[k].extend(d.get(k, []))
    return merged


# ===================== 统计描述输出 =====================
def describe_samples(name, data):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        print(f"{name}: empty")
        return
    mean = np.mean(data)
    p5, p50, p95 = np.percentile(data, [5, 50, 95])
    print(f"{name}: N={len(data)}, mean={mean:.2f}, P5={p5:.2f}, P50={p50:.2f}, P95={p95:.2f}")


# ===================== 画直方图（可选） =====================
def plot_hist_density(
    data,
    filename="hist.png",
    xmax=100.0,
    ymax=0.12,
    bins=60,
    xlabel="Value"
):
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    data = data[data < xmax]
    if len(data) == 0:
        print(f"[WARN] {filename}: empty, skip plot.")
        return

    bin_edges = np.linspace(0, xmax, bins + 1)
    weights = np.ones_like(data) / len(data)

    plt.figure(figsize=(8, 5))
    plt.hist(
        data,
        bins=bin_edges,
        weights=weights,
        density=False,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_hist_density_compare(
    data_sim,
    data_highd,
    filename,
    xmax,
    ymax,
    bins,
    label_sim="Sim Level-2",
    label_highd="HighD",
    xlabel="Value"
):
    data_sim = np.asarray(data_sim, float)
    data_sim = data_sim[np.isfinite(data_sim)]
    data_sim = data_sim[data_sim < xmax]

    data_highd = np.asarray(data_highd, float)
    data_highd = data_highd[np.isfinite(data_highd)]
    data_highd = data_highd[data_highd < xmax]

    if len(data_sim) == 0 or len(data_highd) == 0:
        print(f"[WARN] {filename}: sim or highd empty, skip plot.")
        return

    bin_edges = np.linspace(0, xmax, bins + 1)
    w_sim = np.ones_like(data_sim) / len(data_sim)
    w_highd = np.ones_like(data_highd) / len(data_highd)

    plt.figure(figsize=(8, 5))
    plt.hist(
        data_highd,
        bins=bin_edges,
        weights=w_highd,
        density=False,
        alpha=0.5,
        label=label_highd,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.hist(
        data_sim,
        bins=bin_edges,
        weights=w_sim,
        density=False,
        alpha=0.5,
        label=label_sim,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_hist_density_multi_styles(
    style_data_dict,
    filename,
    xmax,
    ymax,
    bins,
    xlabel="Value"
):
    plt.figure(figsize=(8, 5))
    bin_edges = np.linspace(0, xmax, bins + 1)

    for style_name, data in style_data_dict.items():
        data = np.asarray(data, float)
        data = data[np.isfinite(data)]
        data = data[data < xmax]
        if len(data) == 0:
            continue
        weights = np.ones_like(data) / len(data)
        plt.hist(
            data,
            bins=bin_edges,
            weights=weights,
            density=False,
            alpha=0.35,
            label=style_name,
            linewidth=0.3,
            edgecolor="black",
        )

    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ===================== 主程序 =====================
if __name__ == "__main__":
    # Windows 下多进程需要
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 总 Level-2 回合数（会按 0.3/0.4/0.3 分给三个风格）
    TOTAL_EPISODES = 600
    # 并行进程数，不想并行就改成 1
    NUM_PROCESSES = max(1, mp.cpu_count() - 22)

    print(f"使用 {NUM_PROCESSES} 个进程进行测试。")

    style_results = {}
    t0 = time.time()

    # ---------- 三种风格分别仿真 ----------
    for style_idx, (style_name, model_path, weight) in enumerate(STYLE_CONFIG):
        episodes_style = int(round(TOTAL_EPISODES * weight))
        if episodes_style <= 0:
            continue

        print(f"\n===== 风格 {style_name}：模型 {model_path}, 回合数 {episodes_style} =====")

        # 按进程平均分配回合数
        episodes_per_process = [episodes_style // NUM_PROCESSES] * NUM_PROCESSES
        remainder = episodes_style % NUM_PROCESSES
        for i in range(remainder):
            episodes_per_process[i] += 1

        args = [(ep, model_path, i) for i, ep in enumerate(episodes_per_process)]

        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.starmap(run_episodes_for_style, args)

        collision_counts, sample_dicts = zip(*results)
        total_collisions_style = sum(collision_counts)
        merged_samples = merge_local_stats(sample_dicts)

        style_results[style_name] = {
            "episodes": episodes_style,
            "collisions": total_collisions_style,
            "samples": merged_samples,
        }

        # 输出该风格在“跟驰”场景下的 TTC/THW/速度 描述统计
        describe_samples(f"[{style_name}] TTC(cf)", merged_samples["ttc_cf"])
        describe_samples(f"[{style_name}] THW(cf)", merged_samples["thw_cf"])
        describe_samples(f"[{style_name}] v(cf)", merged_samples["v_cf"])
        describe_samples(f"[{style_name}] v(lc)", merged_samples["v_lc"])

        # 保存该风格的 npy
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_ttc_cf.npy"),
                np.asarray(merged_samples["ttc_cf"], dtype=np.float32))
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_thw_cf.npy"),
                np.asarray(merged_samples["thw_cf"], dtype=np.float32))
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_ttc_lc.npy"),
                np.asarray(merged_samples["ttc_lc"], dtype=np.float32))
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_thw_lc.npy"),
                np.asarray(merged_samples["thw_lc"], dtype=np.float32))
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_v_cf.npy"),
                np.asarray(merged_samples["v_cf"], dtype=np.float32))
        np.save(os.path.join(SIM_SAVE_DIR, f"sim_L2_{style_name}_v_lc.npy"),
                np.asarray(merged_samples["v_lc"], dtype=np.float32))

    t1 = time.time()
    print(f"\n全部风格仿真完成，总耗时 {t1 - t0:.1f} s")

    # ---------- ① Level-2 整体（混合 0.3/0.4/0.3） ----------
    ttc_cf_all, thw_cf_all = [], []
    ttc_lc_all, thw_lc_all = [], []
    v_cf_all, v_lc_all = [], []

    total_episodes = 0
    total_collisions = 0

    for style_name, res in style_results.items():
        total_episodes += res["episodes"]
        total_collisions += res["collisions"]
        s = res["samples"]
        ttc_cf_all.extend(s["ttc_cf"])
        thw_cf_all.extend(s["thw_cf"])
        ttc_lc_all.extend(s["ttc_lc"])
        thw_lc_all.extend(s["thw_lc"])
        v_cf_all.extend(s["v_cf"])
        v_lc_all.extend(s["v_lc"])

    collision_rate = total_collisions / max(total_episodes, 1)
    print(f"\n===== Level-2 整体结果（三风格混合） =====")
    print(f"总回合数: {total_episodes}")
    print(f"总碰撞次数: {total_collisions}")
    print(f"碰撞率: {collision_rate:.2%}")
    describe_samples("[Level-2 overall] TTC(cf)", ttc_cf_all)
    describe_samples("[Level-2 overall] THW(cf)", thw_cf_all)
    describe_samples("[Level-2 overall] v(cf)", v_cf_all)
    describe_samples("[Level-2 overall] v(lc)", v_lc_all)

    # 保存整体 npy
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_ttc_cf.npy"),
            np.asarray(ttc_cf_all, dtype=np.float32))
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_thw_cf.npy"),
            np.asarray(thw_cf_all, dtype=np.float32))
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_ttc_lc.npy"),
            np.asarray(ttc_lc_all, dtype=np.float32))
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_thw_lc.npy"),
            np.asarray(thw_lc_all, dtype=np.float32))
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_v_cf.npy"),
            np.asarray(v_cf_all, dtype=np.float32))
    np.save(os.path.join(SIM_SAVE_DIR, "sim_L2_overall_v_lc.npy"),
            np.asarray(v_lc_all, dtype=np.float32))

