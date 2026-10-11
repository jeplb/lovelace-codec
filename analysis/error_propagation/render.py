"""render the error propagation figure from the completed source tables."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tsv_records import read_output

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lovelace-si-error-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/lovelace-si-error-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE
WIDTH_MM = 183
HEIGHT_MM = WIDTH_MM / ((1 + np.sqrt(5)) / 2)
ARMS = ('control', 'position', 'position_weighted')
COLORS = dict(control='#0072B2', position='#D55E00', position_weighted='#009E73')
NAMES = dict(control='Control', position='Position-dependent', position_weighted='Position-weighted')
FONT_SIZE = 7


def table(name):
    with (HERE / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 7,
        'axes.labelsize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'lovelace-si-errors',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2,
        'ytick.major.size': 2, 'lines.linewidth': .8, 'savefig.dpi': 600,
    })
    assert 'Arial' in font_manager.FontProperties(fname=font_manager.findfont('Arial', fallback_to_default=False)).get_name()
    findings = read_output(HERE / 'numerical_findings.tsv')
    summary = {(r['image'], r['method']): r for r in findings['current_streams']}
    arrays = np.load(HERE / 'error_arrays.npz')
    digit = table('digit_error_summary.csv')
    diag = table('coefficient_metadata_diagnostics.csv')
    concentration = table('fourier_group_concentration.csv')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor='white')

    def axes(x, top, width, height):
        return fig.add_axes([x / WIDTH_MM, 1 - (top + height) / HEIGHT_MM,
                             width / WIDTH_MM, height / HEIGHT_MM])

    def panel(letter, x, top):
        fig.text(x / WIDTH_MM, 1 - top / HEIGHT_MM, letter, fontweight='bold', va='top', fontsize=FONT_SIZE)

    handles = [Line2D([], [], color=COLORS[arm], lw=1.1, label=NAMES[arm]) for arm in ARMS]
    handles += [Line2D([], [], marker='o', color='0.25', linestyle='-', ms=3, label='Image A'),
                Line2D([], [], marker='s', color='0.25', linestyle='--', ms=3, label='Image B')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, 1.005), ncol=5,
               handlelength=1.5, columnspacing=1.2, handletextpad=.5, frameon=False)

    panel('a', 3, 9)
    ax = axes(15, 15, 42, 36)
    stage_source = []
    for image, marker, offset in [('A', 'o', -.10), ('B', 's', .10)]:
        representation = next(r for r in findings['representation_stages'] if r['image'] == image)
        values = [representation['crop_rmse'], representation['quantized_to_original_rmse']]
        ax.plot(np.arange(2) + offset, values, color='0.4', marker=marker, ms=3,
                linestyle='-' if image == 'A' else '--')
        for x, value, stage in zip(range(2), values, ('crop', 'quantized')):
            stage_source.append(dict(panel='a', image=image, method='representation', stage=stage,
                x=x + offset, value=value, metric='pixel RMSE to original', domain='physical_intensity'))
        for j, arm in enumerate(ARMS, 2):
            value = summary[(image, arm)]['physical_pixel_rmse_to_original']
            ax.plot(j + offset, value, color=COLORS[arm], marker=marker, ms=3.5, linestyle='none')
            stage_source.append(dict(panel='a', image=image, method=arm, stage='read-derived',
                x=j + offset, value=value, metric='pixel RMSE to original', domain='physical_intensity'))
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Crop', 'Quantized', 'Control', 'P', 'W'], rotation=55, ha='right')
    ax.set_ylabel('RMSE to original [intensity]')
    ax.set_ylim(0, .185)
    ax.set_yticks([0, .05, .10, .15])
    ax.set_xlim(-.5, 4.5)

    panel('b', 64, 9)
    ax = axes(77, 15, 42, 36)
    digit_source = []
    for image, marker in [('A', 'o'), ('B', 's')]:
        for arm in ARMS:
            selected = [r for r in digit if r['image'] == image and r['method'] == arm]
            x = [int(r['incorrect_digit_count']) for r in selected]
            y = [float(r['mean_absolute_normalized_error_if_wrong']) for r in selected]
            ax.scatter(x, y, marker=marker, color=COLORS[arm], s=13,
                       facecolors='none' if image == 'B' else COLORS[arm], linewidths=.75)
            digit_source.extend(dict(panel='b', image=image, method=arm,
                digit_position_msb_first=int(r['digit_position_msb_first']),
                incorrect_digits=int(r['incorrect_digit_count']),
                mean_absolute_isolated_digit_error=float(r['mean_absolute_normalized_error_if_wrong']),
                domain='normalized') for r in selected)
    for label, y in [('High', .65), ('Middle', .26), ('Low', .075)]:
        ax.text(.8, y, label, va='center', fontsize=FONT_SIZE, color='.3')
    ax.set_xlabel('Incorrect digits [count]')
    ax.set_ylabel('Mean digit error [normalized]')
    ax.set_xlim(0, 27)
    ax.set_ylim(0, .73)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([0, .2, .4, .6])

    panel('d', 125, 9)
    ax = axes(146, 15, 32, 36)
    combos = [('designed', 'designed'), ('estimated', 'designed'), ('designed', 'decoded'), ('estimated', 'decoded')]
    row_keys = [(image, arm) for image in 'AB' for arm in ARMS]
    matrix = np.asarray([[float(next(r for r in diag if r['image'] == image and r['method'] == arm and
        r['coefficient_source'] == coeff and r['metadata_source'] == meta)['pixel_rmse_to_designed'])
        for coeff, meta in combos] for image, arm in row_keys])
    ax.imshow(matrix, cmap='cividis', vmin=0, vmax=.18, aspect='auto', interpolation='nearest')
    for r in range(6):
        for c in range(4):
            value = matrix[r, c]
            label = '0' if value < .0005 else f'{value:.3f}'
            ax.text(c, r, label, ha='center', va='center', color='white' if value < .09 else 'black', fontsize=FONT_SIZE)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'{image} ' + dict(control='Control', position='P', position_weighted='W')[arm]
                        for image, arm in row_keys])
    for tick, (_, arm) in zip(ax.get_yticklabels(), row_keys):
        tick.set_color(COLORS[arm])
    ax.set_xticks(range(4))
    ax.set_xticklabels(['D D', 'E D', 'D R', 'E R'])
    ax.set_xlabel('Coefficients and metadata')
    ax.tick_params(length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    diagnostic_source = [dict(panel='d', image=image, method=arm, coefficient_source=coeff,
        metadata_source=meta, value=matrix[r, c], metric='pixel RMSE to designed', domain='physical_intensity')
        for r, (image, arm) in enumerate(row_keys) for c, (coeff, meta) in enumerate(combos)]

    panel('c', 3, 65)
    signed = LinearSegmentedColormap.from_list('signed_wong', ['#0072B2', '#FFFFFF', '#D55E00'])
    maps = [arrays[f'group_location_change_{image}_position_weighted'] for image in 'AB']
    bound = max(float(np.abs(values).max()) for values in maps)
    map_source = []
    for i, (image, values) in enumerate(zip('AB', maps)):
        ax = axes(15 + 40 * i, 70, 32, 32)
        artist = ax.imshow(values, cmap=signed, vmin=-bound, vmax=bound, origin='lower',
                           interpolation='nearest', extent=(-16.5, 15.5, -16.5, 15.5), aspect='equal')
        ax.add_patch(Rectangle((-8.5, -8.5), 16, 16, fill=False, edgecolor='.3', lw=.6))
        ax.text(-14.5, 12.5, image, fontsize=FONT_SIZE, va='center')
        ax.set_xticks([-16, 0, 15])
        ax.set_yticks([-16, 0, 15])
        ax.set_xlabel('Column frequency [index]')
        if i == 0:
            ax.set_ylabel('Row frequency [index]')
        else:
            ax.set_yticklabels([])
        for r in range(32):
            for c in range(32):
                map_source.append(dict(panel='c', image=image, comparison='Position-weighted minus Control',
                    row_frequency=r - 16, column_frequency=c - 16, value=float(values[r, c]),
                    metric='group pixel MSE change equally divided across conjugate grid sites', domain='normalized'))
    color_axis = axes(91, 73, 2, 26)
    colorbar = fig.colorbar(artist, cax=color_axis)
    colorbar.set_ticks([-bound, 0, bound])
    colorbar.set_ticklabels([f'{-bound * 1e4:.1f}', '0', f'{bound * 1e4:.1f}'])
    colorbar.set_label('MSE change [0.0001 normalized]', fontsize=FONT_SIZE, labelpad=3)
    colorbar.outline.set_linewidth(.5)
    colorbar.ax.tick_params(labelsize=FONT_SIZE, length=2)

    ax = axes(123, 70, 55, 32)
    curve_source = []
    for image, linestyle, marker in [('A', '-', 'o'), ('B', '--', 's')]:
        for arm in ARMS:
            selected = sorted([r for r in concentration if r['image'] == image and r['method'] == arm],
                              key=lambda r: int(r['contribution_rank']))
            ranks = [int(r['contribution_rank']) for r in selected]
            assert min(ranks) > 0, 'strictly positive ranks required for logarithmic axis'
            fractions = [float(r['cumulative_mse_fraction']) for r in selected]
            ax.plot(ranks, fractions, color=COLORS[arm], linestyle=linestyle, linewidth=.85)
            ax.plot(5, fractions[4], marker=marker, color=COLORS[arm], ms=3,
                    markerfacecolor='none' if image == 'B' else COLORS[arm], linestyle='none')
            curve_source.extend(dict(panel='c', image=image, method=arm, contribution_rank=x,
                cumulative_mse_fraction=y, metric='cumulative normalized readout pixel MSE fraction')
                for x, y in zip(ranks, fractions))
    ax.set_xscale('log')
    ax.set_xlim(1, 144)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([1, 5, 20, 144])
    ax.set_xticklabels(['1', '5', '20', '144'])
    ax.minorticks_off()
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel('Ranked conjugate groups [count]')
    ax.set_ylabel('Cumulative pixel MSE [fraction]')

    source_rows = stage_source + digit_source + diagnostic_source + map_source + curve_source
    fields = list(dict.fromkeys(k for r in source_rows for k in r))
    with (HERE / 'SuppFigure_Lovelace_Error_Propagation_source_data.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(source_rows)
    stem = HERE / 'SuppFigure_Lovelace_Error_Propagation'
    fig.savefig(stem.with_suffix('.svg'), metadata={'Date': None}, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None}, facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=600, facecolor='white')
    fig.savefig(HERE / 'SuppFigure_Lovelace_Error_Propagation_print_preview.png', dpi=300, facecolor='white')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    bounds = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
            outside.append(text.get_text())
    assert not outside, outside
    plt.close(fig)
    print(json.dumps(dict(svg=True, pdf=True, png=True, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)))


if __name__ == '__main__':
    main()
