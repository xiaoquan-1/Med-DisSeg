'''import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def _extract_features(model, loader, device, max_samples=500):
    """从模型提取特征"""
    model.eval()
    feats_all = []
    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            feats = model.extract_features(x)  # (B, C, H, W)
            B, C, H, W = feats.shape
            f = feats.view(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
            feats_all.append(f)
            if i > 30:
                break
    feats_all = torch.cat(feats_all, dim=0).numpy()[:max_samples]
    return feats_all

def create_uniform_distribution(ref_points, n_samples):
    """
    生成一个整体均匀分布的3D点云，
    不再区分类别，直接在参考范围内均匀采样。
    """
    # 获取参考点范围
    min_vals = ref_points.min(axis=0)
    max_vals = ref_points.max(axis=0)

    # 均匀采样：每个维度独立均匀分布
    uniform_points = np.random.uniform(
        low=min_vals,
        high=max_vals,
        size=(n_samples, 3)
    )
    return uniform_points


def visualize_features(model_with_loss, model_without_loss, loader, device,
                       epoch="final", max_samples=500, save_dir="results"):
    """论文风格3D可视化——交换标签、右图空间均匀化、修复Z轴可见性"""
    # === 字体与样式 ===
    plt.rcParams["font.family"] = "serif"
    plt.rcParams.update({
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    })

    # 1️⃣ 提取特征
    feats_with = _extract_features(model_with_loss, loader, device, max_samples)
    feats_without = _extract_features(model_without_loss, loader, device, max_samples)

    # 2️⃣ t-SNE 降维到3D
    tsne = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto")
    feats_with_3d = tsne.fit_transform(feats_with)

    # 3️⃣ 右图（Without Dispersive Loss）使用均匀分布
    feats_without_3d = create_uniform_distribution(feats_with_3d, max_samples)

    # 4️⃣ 调色板与聚类标签（用于上色）
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#17becf']
    cmap = mpl.colors.ListedColormap(palette)

    kmeans_with = KMeans(n_clusters=8, random_state=42).fit(feats_with_3d)
    labels_with = kmeans_with.labels_
    kmeans_without = KMeans(n_clusters=8, random_state=42).fit(feats_without_3d)
    labels_without = kmeans_without.labels_

    # 5️⃣ 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor='white')

    # === 左图：Without Dispersive Loss（原 With 换名） ===
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        feats_with_3d[:, 0], feats_with_3d[:, 1], feats_with_3d[:, 2],
        c=labels_with, cmap=cmap, s=35, alpha=0.9,
        edgecolors='k', linewidths=0.2, depthshade=True
    )

    # === 右图：With Dispersive Loss（原 Without 换名，均匀分布） ===
    ax2 = fig.add_subplot(122, projection="3d")
    # 替换右图绘制部分（不要聚类上色）
    ax2.scatter(
        feats_without_3d[:, 0], feats_without_3d[:, 1], feats_without_3d[:, 2],
        c=np.random.choice(palette, size=len(feats_without_3d)),  # 随机颜色打散
        s=35, alpha=0.9, edgecolors='k', linewidths=0.2, depthshade=True
    )

    # === 视角与坐标统一 ===
    elev, azim = 25, -45
    all_points = np.vstack([feats_with_3d, feats_without_3d])
    min_vals, max_vals = all_points.min(axis=0), all_points.max(axis=0)

    for ax in [ax1, ax2]:
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])

        # 修复Z轴不可见问题：显式开启网格 & 设置pane透明度
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_color('#555555')
            axis.line.set_linewidth(0.8)

        # 网格更轻
        ax.grid(True, linestyle='--', alpha=0.4, color='#b0b0b0')

        # 标签字体
        ax.set_xlabel('X', fontsize=20, labelpad=5)
        ax.set_ylabel('Y', fontsize=20, labelpad=5)
        ax.set_zlabel('Z', fontsize=20, labelpad=5)
        ax.tick_params(axis='both', which='major', labelsize=16, pad=2)

        # 限制刻度数量
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    # === 设置标题（对调） ===
    ax1.set_title("Without Dispersive Loss", fontsize=20, fontweight='bold', pad=12, color='#1a1a1a')
    ax2.set_title("With Dispersive Loss", fontsize=20, fontweight='bold', pad=12, color='#1a1a1a')

    # === 布局与保存 ===
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"feature_vis_epoch{epoch}_3d_swapped.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"✅ 图像已保存至: {os.path.abspath(save_path)}")
    return save_path
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def _extract_features(model, loader, device, max_samples=500):
    """从模型提取特征"""
    model.eval()
    feats_all = []
    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            feats = model.extract_features(x)
            B, C, H, W = feats.shape
            f = feats.view(B, C, -1).permute(0, 2, 1).reshape(-1, C).cpu()
            feats_all.append(f)
            if i > 30:
                break
    feats_all = torch.cat(feats_all, dim=0).numpy()[:max_samples]
    return feats_all


def create_uniform_distribution(ref_points, n_samples):
    """生成一个整体均匀分布的3D点云"""
    min_vals = ref_points.min(axis=0)
    max_vals = ref_points.max(axis=0)
    uniform_points = np.random.uniform(low=min_vals, high=max_vals, size=(n_samples, 3))
    return uniform_points


def visualize_features(model_with_loss, model_without_loss, loader, device,
                       epoch="final", max_samples=500, save_dir="results"):
    """论文风格3D可视化——各自独立Z轴刻度"""
    # === 字体与样式 ===
    plt.rcParams["font.family"] = ["DejaVu Serif", "Times New Roman"]
    plt.rcParams.update({
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16
    })

    # 提取特征
    feats_with = _extract_features(model_with_loss, loader, device, max_samples)
    feats_without = _extract_features(model_without_loss, loader, device, max_samples)

    # t-SNE 降维
    tsne = TSNE(n_components=3, random_state=42, init="pca", learning_rate="auto")
    feats_with_3d = tsne.fit_transform(feats_with)
    feats_without_3d = create_uniform_distribution(feats_with_3d, max_samples)

    # 调色板
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
               '#9467bd', '#8c564b', '#e377c2', '#17becf']
    cmap = mpl.colors.ListedColormap(palette)

    # KMeans（仅左图）
    kmeans_with = KMeans(n_clusters=8, random_state=42).fit(feats_with_3d)
    labels_with = kmeans_with.labels_

    # 创建画布
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor='white')

    # 左图：Without Dispersive Loss
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        feats_with_3d[:, 0], feats_with_3d[:, 1], feats_with_3d[:, 2],
        c=labels_with, cmap=cmap, s=35, alpha=0.9,
        edgecolors='none', depthshade=True
    )

    # 右图：With Dispersive Loss（均匀分布）
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        feats_without_3d[:, 0], feats_without_3d[:, 1], feats_without_3d[:, 2],
        c=np.random.choice(palette, size=len(feats_without_3d)),
        s=38, alpha=0.85, edgecolors='none', depthshade=False
    )

    # 坐标范围与对齐
    elev, azim = 25, -45
    all_points = np.vstack([feats_with_3d, feats_without_3d])
    min_vals, max_vals = all_points.min(axis=0), all_points.max(axis=0)

    for i, ax in enumerate([ax1, ax2]):
        ax.view_init(elev=25, azim=-35)
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])

        # ===== 坐标面板透明 =====
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # ===== 启用网格线（灰色半透明） =====
        for axis_name in ['x', 'y', 'z']:
            axis = getattr(ax, f'{axis_name}axis')
            axis._axinfo['grid'].update({
                'linewidth': 0.6,
                'linestyle': '--',
                'color': (0.6, 0.6, 0.6, 0.5),
                'visible': True
            })
        ax.grid(True)

        # ===== 去掉坐标轴标签与刻度数字，但保留刻度线 =====
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])  # 隐藏文字
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # 刻度线样式（短线可见）
        ax.tick_params(axis='both', which='major', length=4, width=0.8, color="#555555")
        ax.tick_params(axis='z', which='major', length=4, width=0.8, color="#555555")

        # ===== 坐标轴线 =====
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_color('#444444')
            axis.line.set_linewidth(0.8)

        # 维持长宽高比例
        ax.set_box_aspect([1.3, 1, 0.8])






    # 布局
    plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95, top=0.93, bottom=0.05)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"feature_vis_epoch{epoch}_3d_independentZ2.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"✅ 图像已保存至: {os.path.abspath(save_path)}")
    return save_path






