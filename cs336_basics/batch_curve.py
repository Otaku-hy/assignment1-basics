import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cs336_basics.training_infra import *
from cs336_basics.generate_text import load_model

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

def grad_sq_norm(model: torch.nn.Module, data, B: int, context_length: int, device: str) -> float:
    x, y = load_data(data, B, context_length, device)
    model.zero_grad()
    cross_entropy(model(x), y).backward()
    return sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None).item()

def noise_scale(model, data, context_length, device, Bs=(8, 16, 32, 64, 128), repeats=8):
    xs = [1 / B for B in Bs]
    ys = [np.mean([grad_sq_norm(model, data, B, context_length, device) for _ in range(repeats)]) for B in Bs]
    slope, intercept = np.polyfit(xs, ys, 1)   # E|g_B|² = |G|² + tr(Σ)/B

    # 拟合的直线：截距 = 信号 |G|²，斜率/B = 噪声；noise/signal = 1 的那个 B 就是 B_noise
    print(f"{'B':>5} {'1/B':>8} {'E|g_B|^2':>10} {'fit':>10} {'noise':>10} {'noise/signal':>13}")
    for B, x, y in zip(Bs, xs, ys):
        noise = slope * x
        print(f"{B:>5} {x:>8.4f} {y:>10.4g} {intercept + noise:>10.4g} {noise:>10.4g} {noise / intercept:>13.3f}")
    print(f"signal |G|^2 = intercept = {intercept:.4g}, tr(Sigma) = slope = {slope:.4g}")
    return slope / intercept, xs, ys, slope, intercept   # B_noise（单位是序列）和画图用的数据

def plot_curves(results: list[dict], out_path: str):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(3.4 * (len(results) + 1), 3.6),
                             facecolor=SURFACE, layout="constrained")
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(color=AXIS, labelcolor=MUTED, labelsize=8)

    # 每个 ckpt 一张：点是实测，直线是拟合；截距那条横线是信号，
    # 橙点是 noise = signal 的位置（直线高度正好是截距的 2 倍），它的横坐标就是 1/B_noise
    for ax, r in zip(axes, results):
        x_max = max(r["xs"]) * 1.05
        x_line = np.linspace(0, x_max, 100)
        ax.plot(x_line, r["intercept"] + r["slope"] * x_line, color=BLUE, linewidth=1.5, label="linear fit")
        ax.plot(r["xs"], r["ys"], "o", color=BLUE, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="measured")
        ax.axhline(r["intercept"], color=INK_2, linewidth=0.8)
        ax.text(x_max, r["intercept"], "signal |G|²", ha="right", va="bottom", color=INK_2, fontsize=8)

        x_star = 1 / r["b_noise"]
        if 0 < x_star <= x_max:
            ax.plot([x_star], [2 * r["intercept"]], "o", color=ORANGE, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate("noise = signal", (x_star, 2 * r["intercept"]), xytext=(6, -12),
                        textcoords="offset points", color=INK_2, fontsize=8)

        ticks = [0] + [x for x in r["xs"] if 1 / x <= 32]
        ax.set_xticks(ticks, ["0"] + [f"1/{round(1 / x)}" for x in ticks[1:]])
        ax.set_ylim(bottom=0)
        ax.set_title(f"step {r['step']}: B_noise ≈ {r['b_noise']:.0f}", color=INK, fontsize=10)
        ax.set_xlabel("1/B", color=INK_2, fontsize=9)
    axes[0].set_ylabel("E|g_B|²", color=INK_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    # 最后一张：B_noise 随训练的变化
    ax = axes[-1]
    steps = [r["step"] for r in results]
    b_noise = [r["b_noise"] for r in results]
    ax.plot(steps, b_noise, color=BLUE, linewidth=1.5, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(f"{b_noise[-1]:.0f}", (steps[-1], b_noise[-1]), xytext=(0, 8),
                textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title("B_noise over training", color=INK, fontsize=10)
    ax.set_xlabel("step", color=INK_2, fontsize=9)
    ax.set_ylabel("B_noise (sequences)", color=INK_2, fontsize=9)

    fig.suptitle("E|g_B|² = |G|² + tr(Σ)/B,   B_noise = tr(Σ) / |G|²  (the B where noise = signal)",
                 color=INK, fontsize=11)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"plot saved to {out_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = "runs/bchs_coarse/bch_64"
    results = []
    for ckpt_name in ["ckpt_5000.pt", "ckpt_10000.pt", "ckpt_15000.pt", "ckpt_20000.pt"]:
        model, args = load_model(run_dir, ckpt_name, device)
        train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
        np.random.seed(0)   # 每个 ckpt 用同一组 batch，方便比较
        print(f"== {ckpt_name} ==")
        b_noise, xs, ys, slope, intercept = noise_scale(model, train_data, args.context_length, device)
        print(f"B_noise = slope / intercept = {b_noise:.1f} seqs = {b_noise * args.context_length:.0f} tokens\n")
        step = int(ckpt_name.removeprefix("ckpt_").removesuffix(".pt"))
        results.append({"step": step, "b_noise": b_noise, "xs": xs, "ys": ys, "slope": slope, "intercept": intercept})
    plot_curves(results, os.path.join(run_dir, "noise_scale.png"))
