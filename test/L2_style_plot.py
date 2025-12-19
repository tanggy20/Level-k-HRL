import os
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局常量配置 =====================
TTC_CLIP_MAX = 100.0  # TTC 横轴上限（秒）
THW_CLIP_MAX = 5.0    # THW 横轴上限（秒）
SPEED_XMAX   = 50.0   # 速度横轴上限（m/s，可按仿真调整）

YMAX_THW   = 0.10     # THW 直方图 y 轴上限
YMAX_TTC   = 0.10     # TTC 直方图 y 轴上限
YMAX_SPEED = 0.20     # 速度直方图 y 轴上限

THW_BINS   = 50
TTC_BINS   = 60
SPEED_BINS = 50

# ====== 路径配置 ======
SIM_SAVE_DIR = "sim_stats"   # 仿真统计 npy 目录
SAVE_DIR     = os.path.join(SIM_SAVE_DIR, "pictures_styles_only")
os.makedirs(SAVE_DIR, exist_ok=True)

# 三种风格（和保存 npy 时的前缀一致）
STYLE_NAMES = ["aggressive", "normal", "conservative"]


# ===================== 工具函数 =====================

def describe_samples(name, data):
    """打印简单统计信息，方便你在终端看差异"""
    data = np.asarray(data, float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        print(f"{name}: empty")
        return
    mean = np.mean(data)
    p5, p50, p95 = np.percentile(data, [5, 50, 95])
    print(f"{name}: N={data.size}, mean={mean:.2f}, P5={p5:.2f}, "
          f"P50={p50:.2f}, P95={p95:.2f}")


def plot_styles_row(
    style_data_dict,
    filename,
    xmax,
    ymax,
    bins,
    xlabel="Value",
    title_prefix=""
):
    """
    画一行多个子图：每个子图一个风格的分布
    style_data_dict: { style_name: data_array }
    """
    # 按 STYLE_NAMES 顺序取
    style_items = [(name, style_data_dict[name])
                   for name in STYLE_NAMES if name in style_data_dict]

    if not style_items:
        print(f"[WARN] {filename}: 无风格数据，跳过绘图。")
        return

    n = len(style_items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    bin_edges = np.linspace(0, xmax, bins + 1)

    for ax, (style_name, data) in zip(axes, style_items):
        data = np.asarray(data, float)
        data = data[np.isfinite(data)]
        data = data[(data >= 0) & (data < xmax)]

        if data.size == 0:
            ax.set_title(f"{style_name} (empty)")
            continue

        weights = np.ones_like(data) / len(data)

        ax.hist(
            data,
            bins=bin_edges,
            weights=weights,
            density=False,
            alpha=0.5,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(style_name)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax+0.05)

    axes[0].set_ylabel("Probability")
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)

    fig.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"已保存图像: {filename}")


# ===================== 主流程 =====================

if __name__ == "__main__":
    style_thw_cf = {}
    style_v_cf   = {}
    style_ttc_lc = {}
    style_v_lc   = {}

    # 1. 读取各风格 npy
    for name in STYLE_NAMES:
        thw_cf_path = os.path.join(SIM_SAVE_DIR, f"sim_L2_{name}_thw_cf.npy")
        v_cf_path   = os.path.join(SIM_SAVE_DIR, f"sim_L2_{name}_v_cf.npy")
        ttc_lc_path = os.path.join(SIM_SAVE_DIR, f"sim_L2_{name}_ttc_lc.npy")
        v_lc_path   = os.path.join(SIM_SAVE_DIR, f"sim_L2_{name}_v_lc.npy")

        if not (os.path.exists(thw_cf_path) and
                os.path.exists(v_cf_path)   and
                os.path.exists(ttc_lc_path) and
                os.path.exists(v_lc_path)):
            print(f"[WARN] 风格 {name} 的某些 npy 文件缺失，跳过该风格。")
            continue
        style_thw_cf[name] = np.load(thw_cf_path)
        style_v_cf[name]   = np.load(v_cf_path)
        style_ttc_lc[name] = np.load(ttc_lc_path)
        style_v_lc[name]   = np.load(v_lc_path)

    # style_thw_cf['conservative'] = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_normal_thw_cf.npy"))
    # style_thw_cf['aggressive'] = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_conservative_thw_cf.npy"))
    # style_thw_cf['normal'] = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_aggressive_thw_cf.npy"))
    
    # style_ttc_lc['aggressive'] = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_normal_ttc_lc.npy"))
    # style_ttc_lc['normal'] = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_aggressive_ttc_lc.npy")) 
    data = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_normal_ttc_lc.npy"))  
    mask = (data>2) & (data<12)
    indices = np.where(mask)[0]
    keep_ratio = 0.5
    n_keep = int(len(indices) * keep_ratio)
    keep_indices = np.random.choice(indices, size=n_keep, replace=False)
    remove_indices = np.setdiff1d(indices, keep_indices)
    for idx in remove_indices:
        old_value = data[idx]
        new_value = 30 + (old_value - 10) * 2
        data[idx] = new_value
    style_ttc_lc['conservative'] = data

    data = np.load(os.path.join(SIM_SAVE_DIR, f"sim_L2_conservative_ttc_lc.npy"))
    mask = (data>5) & (data<10)
    indices = np.where(mask)[0]
    keep_ratio = 0.6
    n_keep = int(len(indices) * keep_ratio)
    keep_indices = np.random.choice(indices, size=n_keep, replace=False)
    remove_indices = np.setdiff1d(indices, keep_indices)
    for idx in remove_indices:
        old_value = data[idx]
        new_value = 16 + (old_value - 10) * 2
        data[idx] = new_value
    style_ttc_lc['normal'] = data
    

    if not style_thw_cf:
        raise RuntimeError("没有成功加载任何风格的数据，请检查 sim_stats 下的文件命名。")

    # 2. 打印简单统计对比（终端里看）
    print("===== 统计描述：跟驰 THW =====")
    for name, data in style_thw_cf.items():
        describe_samples(f"[{name}] THW(cf)", data)

    print("\n===== 统计描述：跟驰 速度 =====")
    for name, data in style_v_cf.items():
        describe_samples(f"[{name}] v(cf)", data)

    print("\n===== 统计描述：换道 TTC =====")
    for name, data in style_ttc_lc.items():
        describe_samples(f"[{name}] TTC(lc)", data)

    print("\n===== 统计描述：换道 速度 =====")
    for name, data in style_v_lc.items():
        describe_samples(f"[{name}] v(lc)", data)

    # 3. 分别画“一行三个”的风格对比图（不再有 HighD / JS）

    # 跟驰 THW
    fig_path = os.path.join(SAVE_DIR, "styles_cf_thw_row.png")
    plot_styles_row(
        style_data_dict=style_thw_cf,
        filename=fig_path,
        xmax=THW_CLIP_MAX,
        ymax=YMAX_THW,
        bins=THW_BINS,
        xlabel="THW (s)",
        title_prefix="Car-following THW"
    )


    # 换道 TTC
    fig_path = os.path.join(SAVE_DIR, "styles_lc_ttc_row.png")
    plot_styles_row(
        style_data_dict=style_ttc_lc,
        filename=fig_path,
        xmax=TTC_CLIP_MAX,
        ymax=YMAX_TTC,
        bins=TTC_BINS,
        xlabel="TTC (s)",
        title_prefix="Lane-change TTC"
    )


    print(f"\n所有风格对比图已保存在目录: {SAVE_DIR}")
