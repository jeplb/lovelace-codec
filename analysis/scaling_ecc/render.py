"""draw lovelace-only panels from current computational source rows."""
import hashlib
import os
from pathlib import Path
import shutil
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/lovelace-scaling-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/lovelace-scaling-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import beta
HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE
plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial"], "font.size":7, "axes.labelsize":7, "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7, "pdf.fonttype":42, "svg.fonttype":"none", "svg.hashsalt":"lovelace-scaling-ecc", "savefig.dpi":600, "axes.spines.top":False, "axes.spines.right":False})
WIDTH_MM = 183
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / (name + ".pdf"), metadata={"CreationDate":None, "ModDate":None})
    fig.savefig(FIGURE_DIR / (name + ".svg"), metadata={"Date":None})
    fig.savefig(FIGURE_DIR / (name + ".png"), dpi=600)
    plt.close(fig)


def scaling():
    native = pd.read_csv(HERE / "scaling/native_format_accounting.csv").iloc[0]
    capacities = pd.read_csv(HERE / "scaling/length_limited_capacity.csv")
    files = pd.read_csv(HERE / "scaling/file_size_scaling.csv")
    measured = pd.read_csv(HERE / "benchmark/runtime_memory_measurements.csv")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM/25.4, WIDTH_MM/1.618/25.4))
    parts = ["useful_payload_region_nt", "primer_nt", "address_nt", "marker_nt", "header_nt", "digest_nt", "pilots_nt"]
    labels = ["Payload", "Primers", "Address", "Markers", "Header", "Digest", "Pilots"]
    start = 0
    for index, (part, label) in enumerate(zip(parts, labels)):
        value = native[part] / native.designed_nt
        axes[0,0].barh(0, value, left=start, color=COLORS[index % len(COLORS)], label=label)
        start += value
    axes[0,0].set(yticks=[0], yticklabels=["Lovelace"], xlabel="Complete strand allocation [fraction]", xlim=(0,1))
    axes[0,0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5,-.32))
    for color, limit in zip(COLORS, (200,250,300)):
        rows = capacities[(capacities["format"] == "proposed representation allocation") & (capacities.strand_limit_nt == limit) & (capacities.address_length_bits <= 32)]
        axes[0,1].plot(rows.address_length_bits, rows.payload_bits_per_strand, marker="o", ms=2, color=color, label=f"{limit} nt")
    axes[0,1].set(xlabel="Address length [bits]", ylabel="Payload [bits per strand]")
    axes[0,1].legend(frameon=False)
    rows = files[(files.strand_limit_nt == 300) & (files.parity_packets_per_stripe == 8) & (files.scenario == "1 KiB")]
    rows = rows[rows.supported_file_size_after_metadata_ecc_bytes > 0]
    bounds = rows[rows.theoretical_lower_file_capacity_bytes > 0]
    axes[1,0].fill_between(bounds.address_length_bits.to_numpy(float), bounds.theoretical_lower_file_capacity_bytes.to_numpy(float), bounds.theoretical_upper_file_capacity_bytes.to_numpy(float), color=COLORS[2], alpha=.1)
    point = rows[rows.address_length_bits == 16]
    axes[1,0].scatter(point.address_length_bits, point.verified_native_book_file_capacity_bytes, marker="^", facecolors="white", edgecolors=COLORS[2], s=12)
    axes[1,0].plot(rows.address_length_bits, rows.supported_file_size_after_metadata_ecc_bytes, marker="o", ms=2, color=COLORS[2])
    axes[1,0].set(xlabel="Address length [bits]", ylabel="Supported payload [bytes]", yscale="log")
    rows = measured[measured.stage == "lovelace_cell_likelihoods"]
    grouped = rows.groupby("problem_size").elapsed_seconds.agg(["median", "min", "max"])
    axes[1,1].plot(grouped.index, grouped["median"], color=COLORS[2], marker="o", ms=2)
    axes[1,1].fill_between(grouped.index.to_numpy(float), grouped["min"].to_numpy(float), grouped["max"].to_numpy(float), color=COLORS[2], alpha=.15)
    axes[1,1].set(xlabel="Input reads [count]", ylabel="Elapsed time [s]", xscale="log", yscale="log")
    for letter, ax in zip("abcd", axes.flat):
        ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontsize=7, fontweight="bold")
    save(fig, "lovelace_scaling")


def interval(successes, trials):
    low = 0 if successes == 0 else beta.ppf(.025, successes, trials - successes + 1)
    high = 1 if successes == trials else beta.ppf(.975, successes + 1, trials - successes)
    return low, high


def ecc():
    summary = pd.read_csv(HERE / "sequence/recovery_summary.csv")
    unequal = pd.read_csv(HERE / "unequal/summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_MM/25.4, WIDTH_MM/1.618/25.4))
    rows = summary[(summary.payload == "random") & (summary.condition.isin(["dropout", "baseline"])) & (summary.budget == "per_strand")]
    methods = (("unprotected",0), ("RS",8), ("RS",24), ("repetition",24))
    markers = ("o", "s", "^", "D")
    for color, (arm, parity) in zip(COLORS, methods):
        selected = rows[(rows.arm == arm) & (rows.r == parity)].sort_values("level")
        axes[0,0].plot(selected.level, selected.recovery, marker="o", ms=2, color=color, label=f"{arm} {parity}")
        axes[0,0].fill_between(selected.level.to_numpy(float), selected.ci_low.to_numpy(float), selected.ci_high.to_numpy(float), color=color, alpha=.1)
    axes[0,0].set(xlabel="Template dropout [fraction]", ylabel="Exact object recovery [fraction]", ylim=(-.04,1.04))
    axes[0,0].legend(frameon=False, ncol=1, handlelength=1.0)
    rows = summary[(summary.payload != "random") & (summary.condition == "dropout") & (summary.level == .2)]
    for color, image in zip(COLORS, ("image_A", "image_B")):
        for budget in ("per_strand", "total"):
            chosen = rows[(rows.payload == image) & (rows.budget == budget)]
            base = chosen[chosen.arm == "unprotected"].iloc[0]
            for marker, (arm, parity) in zip(markers, methods):
                row = chosen[(chosen.arm == arm) & (chosen.r == parity)].iloc[0]
                for ax, cost in zip((axes[0,1], axes[0,2]), ("designed_nt", "sequenced_nt")):
                    x, y = row[cost] / base[cost], row.recovery
                    ax.errorbar(x, y, yerr=[[y-row.ci_low], [row.ci_high-y]], fmt="none", ecolor=color, lw=.45, alpha=.5, capsize=1)
                    ax.scatter([x], [y], marker=marker, s=12, facecolors=color if budget == "per_strand" else "white", edgecolors=color, linewidths=.6)
    axes[0,1].set(xlabel="Designed nt [relative]", ylabel="Exact object recovery [fraction]", ylim=(-.04,1.04))
    axes[0,2].set(xlabel="Sequenced nt [relative]", ylabel="Exact object recovery [fraction]", ylim=(-.04,1.04))
    for color, image in zip(COLORS, ("image_A", "image_B")):
        for marker, arm in (("o", "uniform"), ("s", "unequal")):
            row = unequal[(unequal.image == image) & (unequal.arm == arm) & (unequal.condition == "unknown erroneous packet") & (unequal.probability == .05)].iloc[0]
            x = (0 if image == "image_A" else 2) + (arm == "unequal")
            successes, trials = int(row.exact_quantized_stream), int(row.realizations)
            y = successes / trials
            low, high = interval(successes, trials)
            axes[1,0].errorbar(x, y, yerr=[[y-low], [high-y]], fmt="none", color=color, lw=.6, capsize=1.5)
            axes[1,0].scatter([x], [y], color=color, marker=marker, s=14)
            axes[1,1].scatter([x], [row.component_error_fraction_mean], color=color, marker=marker, s=14)
    for ax in (axes[1,0], axes[1,1]):
        ax.set_xticks([0,1,2,3], ["A uniform", "A unequal", "B uniform", "B unequal"], rotation=35, ha="right")
    axes[1,0].set_ylabel("Exact coefficient recovery [fraction]")
    axes[1,0].set_ylim(-.04,1.04)
    axes[1,1].set_ylabel("Wrong or missing components [fraction]")
    axes[1,2].axis("off")
    axes[1,2].text(0, .95, "Computational extension", fontsize=7, va="top")
    axes[1,2].text(0, .8, "32 data packets + r parity packets\nComplete addressed strands\nCell likelihoods\nOuter correction + SHA-256", fontsize=7, va="top", linespacing=1.8)
    axes[1,2].text(0, .38, "Blue image A (orange image B)\nFilled 16 reads per strand\nOpen fixed total reads", fontsize=7, va="top", linespacing=1.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=marker, ls="none", color="black", ms=3, label=f"{arm} {parity}") for marker, (arm, parity) in zip(markers, methods)]
    axes[1,2].legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -.06), ncol=2, frameon=False, handlelength=.7, columnspacing=.5, borderaxespad=0)
    for letter, ax in zip("abcdef", axes.flat):
        ax.text(-.16, 1.08, letter, transform=ax.transAxes, fontsize=7, fontweight="bold")
    save(fig, "lovelace_ecc")


if __name__ == "__main__":
    scaling()
    ecc()
    root = HERE.parents[1]
    for stem, name in (("lovelace_scaling", "lovelace_scaling_ecc.pdf"),
                       ("lovelace_ecc", "lovelace_ecc.pdf")):
        shutil.copyfile(HERE / (stem + ".pdf"), root / name)
