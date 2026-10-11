"""render the fixed-model read-support argument from measured source tables."""

import csv
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


os.environ.setdefault('MPLCONFIGDIR', '/tmp/lovelace-si-inference-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
ARMS = ('control', 'position', 'position_weighted')
NAMES = ('Control', 'Position-dependent', 'Position-weighted')
COLORS = ('#0072B2', '#D55E00', '#009E73')
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'figure.titlesize': 7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'lovelace-si-inference',
    'savefig.dpi': 600, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .5, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def read_table(name):
    with (FIGURE_DIR / name).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def panel_letter(ax, letter):
    ax.text(-.17, 1.07, letter, transform=ax.transAxes, weight='bold',
            va='top', ha='left', fontsize=7)


def curve(ax, rows, metric, image, arm, color, offset=0.):
    points = sorted([row for row in rows if row['method'] == arm and row['image'] == image
        and row['metric'] == metric], key=lambda row: float(row['fraction']))
    x = np.array([float(row['fraction']) for row in points]) * 100
    center = np.array([float(row['mean']) for row in points]) + offset
    lower = np.array([float(row['q05']) for row in points]) + offset
    upper = np.array([float(row['q95']) for row in points]) + offset
    style = '-' if image == 'A' else '--'
    marker = 'o' if image == 'A' else 's'
    ax.fill_between(x, lower, upper, color=color, alpha=.09, linewidth=0)
    ax.plot(x, center, color=color, ls=style, marker=marker, ms=2.2, lw=.9,
            markerfacecolor=color if image == 'A' else 'white', markeredgewidth=.6)


def main():
    support = read_table('support_resolution.tsv')
    ablation = read_table('partner_ablation.tsv')
    sampling = read_table('read_sampling_summary.tsv')
    width = 183 / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(width, width / GOLDEN_RATIO))
    fig.subplots_adjust(left=.14, right=.985, bottom=.14, top=.92, wspace=.40, hspace=.62)
    ax = axes[0, 0]
    columns = [(image, arm) for image in 'AB' for arm in ARMS]
    statuses = [(support_class, resolved) for support_class in ('complete', 'partial', 'absent')
                for resolved in ('True', 'False')]
    counts = np.array([[int(next(row['scalar_count'] for row in support
        if row['method'] == arm and row['image'] == image and row['direct_support'] == support_class
        and row['resolved'] == resolved)) for image, arm in columns] for support_class, resolved in statuses])
    ax.imshow(np.log1p(counts), cmap='cividis', aspect='auto', vmin=0, vmax=np.log1p(512))
    for row in range(6):
        for column in range(6):
            ax.text(column, row, str(counts[row, column]), ha='center', va='center',
                    color='black' if counts[row, column] > 100 else 'white')
    ax.set_xticks(range(6), ['C A', 'P A', 'W A', 'C B', 'P B', 'W B'])
    ax.set_yticks(range(6), ['Complete resolved', 'Complete unresolved', 'Partial resolved',
                            'Partial unresolved', 'Absent resolved', 'Absent unresolved'])
    ax.set_xlabel('Method and image')
    ax.tick_params(length=0)
    ax.axvline(2.5, color='white', lw=.7)
    panel_letter(ax, 'a')
    ax = axes[0, 1]
    deltas = []
    labels = []
    for image in 'AB':
        for arm, name, color in zip(ARMS, NAMES, COLORS):
            before = next(row for row in ablation if row['method'] == arm and row['image'] == image and row['partner_information'] == 'True')
            after = next(row for row in ablation if row['method'] == arm and row['image'] == image and row['partner_information'] == 'False')
            position = len(labels)
            labels.append(f'{arm[0].upper() if arm != "position_weighted" else "W"} {image}')
            for shift, field, filled in ((-.16, 'exact_scalar_matches', True), (.16, 'resolved_labels', False)):
                delta = int(after[field]) - int(before[field])
                ax.barh(position + shift, delta, height=.27, facecolor=color if filled else 'none',
                    edgecolor=color, linewidth=.75)
                deltas.append(dict(method=arm, image=image, metric=field, paired_change=delta))
    ax.set_yticks(range(6), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=.5)
    ax.set_xlabel('Change after disabling partners [labels]')
    ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor='0.4', edgecolor='0.4', linewidth=.75),
               plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='0.4', linewidth=.75)],
              ['Exact matches', 'Resolved'], loc='lower left', bbox_to_anchor=(-.01, 1.02),
              ncol=2, frameon=False, borderaxespad=0, handlelength=1.2, columnspacing=1.)
    panel_letter(ax, 'b')
    recovery_spec = axes[1, 0].get_subplotspec().subgridspec(2, 1, hspace=.14)
    axes[1, 0].remove()
    for subrow, (metric, label) in enumerate((('exact_scalar_matches', 'Exact labels'),
                                             ('resolved_labels', 'Resolved labels'))):
        ax = fig.add_subplot(recovery_spec[subrow, 0])
        for arm, color in zip(ARMS, COLORS):
            for image in 'AB':
                curve(ax, sampling, metric, image, arm, color)
        ax.set_ylabel(label)
        ax.set_xticks([10, 25, 50, 75, 100])
        ax.set_xlim(7, 103)
        ax.set_ylim(120, 515)
        ax.set_yticks([200, 350, 500])
        if subrow == 0:
            ax.tick_params(labelbottom=False)
            panel_letter(ax, 'c')
        else:
            ax.set_xlabel('Original parent reads sampled [%]')
    ax = axes[1, 1]
    for arm, color in zip(ARMS, COLORS):
        for image in 'AB':
            curve(ax, sampling, 'image_rmse_normalized_designed', image, arm, color)
    ax.set_xlabel('Original parent reads sampled [%]')
    ax.set_ylabel('Normalized image RMSE')
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xlim(7, 103)
    panel_letter(ax, 'd')
    methods = [Line2D([], [], color=color, lw=1.3, label=name) for color, name in zip(COLORS, NAMES)]
    images = [Line2D([], [], color='0.25', lw=.9, ls=style, marker=marker, ms=3,
        label=f'Image {image}', markerfacecolor='0.25' if image == 'A' else 'white')
        for image, style, marker in (('A', '-', 'o'), ('B', '--', 's'))]
    fig.legend(methods + images, NAMES + ('Image A', 'Image B'), loc='lower center',
        bbox_to_anchor=(.57, .002), ncol=5, frameon=False, columnspacing=1.1,
        handlelength=1.7, handletextpad=.5)
    fig.savefig(FIGURE_DIR / 'SuppFigure_Lovelace_Inference.svg', metadata={'Date': None})
    fig.savefig(FIGURE_DIR / 'SuppFigure_Lovelace_Inference.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGURE_DIR / 'SuppFigure_Lovelace_Inference.png', dpi=600)
    plt.close(fig)
    with (FIGURE_DIR / 'panel_b_source_data.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, deltas[0], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(deltas)


if __name__ == '__main__':
    main()
