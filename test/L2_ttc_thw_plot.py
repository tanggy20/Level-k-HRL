import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ===================== 全局常量配置 =====================
TTC_CLIP_MAX = 100.0  # TTC 横轴截断上限（秒）
THW_CLIP_MAX = 5.0    # THW 横轴截断上限（秒）
YMAX_THW = 0.06       # THW 直方图 y 轴上限
YMAX_TTC = 0.10       # TTC 直方图 y 轴上限
YMAX_SPEED = 0.20     # 速度直方图 y 轴上限
TTC_BINS = 40
THW_BINS = 50
SPEED_BINS = 50
SPEED_XMAX = 34.0     # 速度横轴上限（按你仿真可能的最大值改）

# ====== 路径配置（按你的工程调整） ======
# 仿真统计结果目录（要和你仿真脚本里的 SIM_SAVE_DIR 一致）
SIM_SAVE_DIR = "sim_stats"
os.makedirs(SIM_SAVE_DIR, exist_ok=True)

# 这就是你上传的 highD 统计文件路径（你本地放哪就写哪）
HIGH_D_NPZ_PATH = "sim_stats/highd_stats_subset_1.npz"  # 如果你放在 highd_stats/ 目录下就写 "highd_stats/highd_stats_subset.npz"

# ===================== JS 散度计算函数 =====================
def compute_js_divergence(data1, data2, bins, xmin=0, xmax=100):
    """
    计算两个数据集之间的 Jensen-Shannon 散度
    
    :param data1: 第一个数据集 (numpy array)
    :param data2: 第二个数据集 (numpy array)
    :param bins: 直方图的 bin 数量
    :param xmin: 直方图的最小值
    :param xmax: 直方图的最大值
    :return: JS 散度值 (0 到 1 之间，0 表示完全相同)
    """
    # 过滤无效值
    data1 = np.asarray(data1, float)
    data1 = data1[np.isfinite(data1)]
    data1 = data1[(data1 >= xmin) & (data1 < xmax)]
    
    data2 = np.asarray(data2, float)
    data2 = data2[np.isfinite(data2)]
    data2 = data2[(data2 >= xmin) & (data2 < xmax)]
    
    if len(data1) == 0 or len(data2) == 0:
        return np.nan
    
    # 使用相同的 bin 边界计算直方图
    bin_edges = np.linspace(xmin, xmax, bins + 1)
    
    hist1, _ = np.histogram(data1, bins=bin_edges, density=False)
    hist2, _ = np.histogram(data2, bins=bin_edges, density=False)
    
    # 归一化为概率分布（确保和为 1）
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)
    
    # 添加小常数避免除零
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # 重新归一化
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # 计算 JS 散度（scipy 返回的是 JS 距离，即 sqrt(JS divergence)）
    js_distance = jensenshannon(hist1, hist2)
    js_divergence = js_distance ** 2  # JS 散度 = JS 距离的平方
    
    return js_divergence

# ===================== 通用画图函数 =====================

def plot_hist_density_compare(
    data_sim,
    data_highd,
    filename,
    xmin,
    xmax,
    ymax,
    bins,
    label_sim="Sim",
    label_highd="HighD",
    xlabel="Value"
):
    """仿真 vs HighD 对比直方图"""

    data_sim = np.asarray(data_sim, float)
    data_sim = data_sim[np.isfinite(data_sim)]
    data_sim = data_sim[(data_sim >= xmin) & (data_sim < xmax)]

    data_highd = np.asarray(data_highd, float)
    data_highd = data_highd[np.isfinite(data_highd)]
    data_highd = data_highd[(data_highd >= xmin) & (data_highd < xmax)]

    if len(data_sim) == 0 or len(data_highd) == 0:
        print(f"[WARN] {filename}: sim or highd empty, skip plot.")
        return
    
    # 计算 JS 散度
    js_div = compute_js_divergence(data_sim, data_highd, bins, 0, xmax)


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
        linewidth=1.0,
    )
    plt.hist(
        data_sim,
        bins=bin_edges,
        weights=w_sim,
        density=False,
        alpha=0.5,
        label=label_sim,
        edgecolor="black",
        linewidth=1.0,
    )

    # 在图上标注 JS 散度
    plt.text(
        0.95, 0.95,
        f"JS Divergence = {js_div:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, ymax+0.05)
    plt.savefig(filename, dpi=200)
    plt.close()


# ===================== 主流程 =====================
if __name__ == "__main__":
    SAVE_DIR = SIM_SAVE_DIR + "/pictures"
    # ---------- 1. 读取 HighD npz ----------
    if not os.path.exists(HIGH_D_NPZ_PATH):
        raise FileNotFoundError(f"找不到 HighD npz 文件：{HIGH_D_NPZ_PATH}")

    hd = np.load(HIGH_D_NPZ_PATH, allow_pickle=True)
    # 里面的 key 在我这边看到是：
    # ['cf_dhw', 'cf_thw', 'cf_speed', 'lc_ttc', 'lc_speed']
    cf_thw_highd = hd["cf_thw"]      # 跟驰 THW
    cf_speed_highd = hd["cf_speed"]  # 跟驰速度
    lc_ttc_highd = hd["lc_ttc"]      # 换道 TTC
    lc_speed_highd = hd["lc_speed"]  # 换道速度

    # ---------- 2. 读取仿真 overall 统计 ----------
    # 确保你之前的仿真脚本已经在 SIM_SAVE_DIR 下保存了这些文件：
    # - sim_L2_overall_thw_cf.npy
    # - sim_L2_overall_v_cf.npy
    # - sim_L2_overall_ttc_lc.npy
    # - sim_L2_overall_v_lc.npy
    sim_thw_cf_path = os.path.join(SIM_SAVE_DIR, "sim_L2_overall_thw_cf.npy")
    sim_v_cf_path   = os.path.join(SIM_SAVE_DIR, "sim_L2_overall_v_cf.npy")
    sim_ttc_lc_path = os.path.join(SIM_SAVE_DIR, "sim_L2_overall_ttc_lc.npy")
    sim_v_lc_path   = os.path.join(SIM_SAVE_DIR, "sim_L2_overall_v_lc.npy")

    for p in [sim_thw_cf_path, sim_v_cf_path, sim_ttc_lc_path, sim_v_lc_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到仿真统计结果文件：{p}，请先运行仿真统计脚本。")

    thw_cf_sim = np.load(sim_thw_cf_path)
    v_cf_sim   = np.load(sim_v_cf_path)
    ttc_lc_sim = np.load(sim_ttc_lc_path)
    v_lc_sim   = np.load(sim_v_lc_path)

    # ---------- 3. 画图：跟驰 THW（HighD vs Sim） ----------
    out_thw_cf = os.path.join(SAVE_DIR, "cf_thw_sim_vs_highd.png")
    plot_hist_density_compare(
        data_sim=thw_cf_sim,
        data_highd=cf_thw_highd,
        filename=out_thw_cf,
        xmin=0,
        xmax=THW_CLIP_MAX,
        ymax=YMAX_THW,
        bins=THW_BINS,
        label_sim="Sim L2 (car-following)",
        label_highd="HighD (car-following)",
        xlabel="THW (s)",
    )
    print(f"已生成跟驰 THW 对比图: {out_thw_cf}")

    # ---------- 5. 画图：换道 TTC（HighD vs Sim） ----------
    out_ttc_lc = os.path.join(SAVE_DIR, "lc_ttc_sim_vs_highd.png")
    plot_hist_density_compare(
        data_sim=ttc_lc_sim,
        data_highd=lc_ttc_highd,
        filename=out_ttc_lc,
        xmin=0,
        xmax=TTC_CLIP_MAX,
        ymax=YMAX_TTC,
        bins=TTC_BINS,
        label_sim="Sim L2 (lane-change)",
        label_highd="HighD (lane-change)",
        xlabel="TTC (s)",
    )
    print(f"已生成换道 TTC 对比图: {out_ttc_lc}")


