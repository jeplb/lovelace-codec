"""render position measurements from the saved source tables."""

import csv
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


os.environ.setdefault('MPLCONFIGDIR', '/tmp/lovelace-si-position-mpl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE
NAME = 'SuppFigure_Lovelace_Position'
WIDTH_MM = 183
HEIGHT_MM = WIDTH_MM / ((1 + np.sqrt(5)) / 2)
FONT_SIZE = 7
PALETTE = {3: '#0072B2', 7: '#D55E00', 10: '#009E73'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'lovelace-si-position', 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .5,
    'xtick.major.width': .5, 'ytick.major.width': .5,
})


def read(name):
    with (HERE / name).open() as handle:
        return list(csv.DictReader(handle))


def query(rows, **fields):
    selected = [row for row in rows if all(row[key] == str(value) for key, value in fields.items())]
    return sorted(selected, key=lambda row: int(row['cell_position']))


def values(rows, key):
    return np.array([float(row[key]) for row in rows])


def style_axis(ax, last=False):
    ax.set_xlim(.65, 8.35)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
    ax.tick_params(length=2, pad=2)
    if last:
        ax.set_xlabel('Payload cell position', labelpad=2)
    else:
        ax.tick_params(labelbottom=False)


def count_range(rows, key):
    counts = [int(row[key]) for row in rows]
    return str(min(counts)) if min(counts) == max(counts) else f'{min(counts)}–{max(counts)}'


def main():
    runs = read('run_length_source_data.csv')
    calls = read('cell_agreement_source_data.csv')
    standardized = read('uniform_state_agreement.csv')
    diagnostics = read('registration_diagnostic_source_data.csv')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    outer = fig.add_gridspec(1, 3, left=.071, right=.984, top=.895, bottom=.17,
                            width_ratios=[1.03, 1.06, 1.16], wspace=.43)
    left = outer[0].subgridspec(3, 1, hspace=.25)
    middle = outer[1].subgridspec(4, 1, hspace=.65)
    right = outer[2].subgridspec(3, 1, height_ratios=[1, 1, 1.13], hspace=.51)
    axes_a = [fig.add_subplot(left[index]) for index in range(3)]
    axes_b = [fig.add_subplot(middle[index]) for index in range(4)]
    axes_c = [fig.add_subplot(right[index]) for index in range(3)]
    x = np.arange(1, 9)
    for base, ax in zip('ACT', axes_a):
        lengths = sorted({int(row['designed_length']) for row in runs if row['population'] == 'native_data' and row['base'] == base})
        for length in lengths:
            for orientation, line, offset in [('forward', '-', -.06), ('reverse_complement', '--', .06)]:
                rows = query(runs, population='native_data', base=base, designed_length=length, orientation=orientation)
                ax.plot(x + offset, values(rows, 'mean'), color=PALETTE[length], linestyle=line, linewidth=.9)
                ax.fill_between(x + offset, values(rows, 'q25'), values(rows, 'q75'),
                                color=PALETTE[length], alpha=.09)
            ax.axhline(length, color=PALETTE[length], linewidth=.4, alpha=.25)
        ax.set_ylim(1.7, 11.2)
        ax.set_yticks([3, 7, 10])
        ax.set_ylabel(f'{base} run length [nt]', labelpad=2)
        style_axis(ax, ax == axes_a[-1])
    for state, ax in enumerate(axes_b):
        for orientation, line, marker in [('forward', '-', 'o'), ('reverse_complement', '--', 's')]:
            rows = query(calls, population='native_data', method='control', orientation=orientation, designed_state=state)
            for outcome, color in [('correct', '#0072B2'), ('incorrect', '#D55E00'), ('uncalled', '#009E73')]:
                ax.plot(x, values(rows, outcome + '_fraction'), color=color, linestyle=line,
                        marker=marker, markersize=1.8, linewidth=.8)
        fwd = query(calls, population='native_data', method='control', orientation='forward', designed_state=state)
        rev = query(calls, population='native_data', method='control', orientation='reverse_complement', designed_state=state)
        support_text = (f'n F {count_range(fwd, "observations")}  R {count_range(rev, "observations")}\n'
                        f'd F {count_range(fwd, "designs")}  R {count_range(rev, "designs")}')
        ax.text(.01, 1.04, support_text,
                transform=ax.transAxes, ha='left', va='bottom')
        ax.set_ylim(-.04, 1.035)
        ax.set_yticks([0, .5, 1])
        ax.set_ylabel(f'{state:02b}\nFraction', labelpad=2)
        style_axis(ax, state == 3)
    population_styles = [
        ('native_data', '#777777', '-', 'Native'),
        ('matched_independent_data', '#0072B2', '-', 'Matched'),
        ('independent_data_all_complete_ACT8', '#CC79A7', '--', 'Independent all'),
    ]
    for orientation, ax in zip(('forward', 'reverse_complement'), axes_c[:2]):
        for population, color, line, label in population_styles:
            rows = query(standardized, population=population, method='control', orientation=orientation)
            ax.plot(x, values(rows, 'correct_fraction'), color=color, linestyle=line,
                    linewidth=.9, marker='o' if population == 'matched_independent_data' else None,
                    markersize=2)
        ax.set_ylim(.5, 1.02)
        ax.set_yticks([.5, .75, 1])
        ax.set_ylabel('Equal-state\nagreement', labelpad=2)
        support_rows = [query(calls, population=population, method='control', orientation=orientation, designed_state='all')[0] for population, color, line, label in population_styles]
        label = 'F' if orientation == 'forward' else 'R'
        support_text = (label + '   n ' + ' / '.join(row['observations'] for row in support_rows)
                        + '\nd ' + ' / '.join(row['designs'] for row in support_rows))
        ax.text(.04, .04, support_text,
                transform=ax.transAxes, ha='left', va='bottom')
        style_axis(ax, False)
    ax = axes_c[-1]
    for population, color, label in [
            ('expected_assisted_full_posterior', '#999999', 'Full-cell filter'),
            ('expected_assisted_held_out_cell', '#D55E00', 'Held-out cell')]:
        for orientation, line in [('forward', '-'), ('reverse_complement', '--')]:
            rows = query(diagnostics, population=population, orientation=orientation)
            ax.plot(x, values(rows, 'run_exact_error'), color=color, linestyle=line, linewidth=.9)
    ax.set_ylim(.08, .37)
    ax.set_yticks([.1, .2, .3])
    ax.set_ylabel('Run error\nfraction', labelpad=2)
    style_axis(ax, True)
    ax.text(.0, 1.07, 'Expected-assisted diagnostics', transform=ax.transAxes, ha='left', va='bottom')
    for letter, xpos in zip('abc', [.033, .36, .666]):
        fig.text(xpos, .958, letter, fontsize=FONT_SIZE, fontweight='bold', va='top')
    fig.legend([Line2D([], [], color=PALETTE[length], linewidth=1) for length in (3, 7, 10)],
               ['Designed 3', 'Designed 7', 'Designed 10'],
               loc='upper left', bbox_to_anchor=(.066, .969), ncol=1,
               frameon=False, handlelength=1.3, labelspacing=.13, borderaxespad=0)
    fig.legend([Line2D([], [], color=color, linewidth=1) for color in ('#0072B2', '#D55E00', '#009E73')],
               ['Correct', 'Incorrect', 'Uncalled'],
               loc='upper left', bbox_to_anchor=(.393, .969), ncol=3,
               frameon=False, handlelength=.8, handletextpad=.25,
               columnspacing=.5, labelspacing=.13, borderaxespad=0)
    fig.legend([Line2D([], [], color=color, linestyle=line, linewidth=1) for population, color, line, label in population_styles],
               [label for population, color, line, label in population_styles],
               loc='upper left', bbox_to_anchor=(.699, .969), ncol=1,
               frameon=False, handlelength=1.3, labelspacing=.13, borderaxespad=0)
    fig.legend([Line2D([], [], color='#555555', linestyle=line, linewidth=1) for line in ('-', '--')],
               ['Forward', 'Reverse'], loc='lower left', bbox_to_anchor=(.07, .008), ncol=2,
               frameon=False, handlelength=1.8, columnspacing=1.3, borderaxespad=0)
    fig.text(.374, .049, 'n observations   d designs\nF forward   R reverse', ha='left', va='bottom')
    fig.legend([Line2D([], [], color=color, linewidth=1) for color in ('#999999', '#D55E00')],
               ['Full-cell filter', 'Held-out cell'], loc='lower left', bbox_to_anchor=(.69, .008),
               ncol=1, frameon=False, handlelength=1.3, labelspacing=.13, borderaxespad=0)
    fig.savefig(FIGURE_DIR / f'{NAME}.svg', metadata={'Date': None})
    fig.savefig(FIGURE_DIR / f'{NAME}.pdf', metadata={'Creator': 'matplotlib', 'CreationDate': None, 'ModDate': None})
    fig.savefig(FIGURE_DIR / f'{NAME}.png', dpi=600)
    text_sizes = sorted({text.get_fontsize() for text in fig.findobj(matplotlib.text.Text) if text.get_text()})
    assert text_sizes == [FONT_SIZE]
    plt.close(fig)


if __name__ == '__main__':
    main()
