"""reconstruct and score saved predictions, then assemble the Lovelace figure."""

import argparse
import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import struct
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lovelace-integration-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/lovelace-integration-cache')
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tsv_records import read_output, write_output
INPUTS = HERE / 'evidence'
ORDINARY = INPUTS / 'ordinary'
CURRENT = ORDINARY / '13_observation_model'
ARMS = ('control', 'position', 'position_weighted')
NAMES = dict(control='Control', position='Position', position_weighted='Weighted')
STATES = np.array([[3, 7, 10], [3, 10, 7], [7, 3, 10], [7, 10, 3]])
WIDTH_MM = 183
HEIGHT_MM = WIDTH_MM / ((1 + np.sqrt(5)) / 2)
FONT_SIZE = 7
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 7, 'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
    'figure.titlesize': FONT_SIZE, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'lovelace-integration', 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .5,
    'xtick.major.width': .5, 'ytick.major.width': .5,
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_record(path):
    return read_output(path)


def write_record(path, value):
    write_output(path, value)


def read_table(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def inverse(components, mean=0., scale=1.):
    from reconstruction import from_components
    return from_components(components, mean, scale)


def unpack_stream(path):
    from reconstruction import reconstruct_stream
    from fourier_codec import dequantize
    stream = Path(path).read_bytes()
    decoded = reconstruct_stream(stream)
    labels = decoded['codes']
    physical = dequantize(labels, decoded['scale'], 6)
    return dict(labels=labels, normalized=-1 + 2 * labels.astype(np.float64) / 63,
                physical=np.stack((physical.real, physical.imag), axis=-1),
                mean=decoded['mean'], scale=decoded['scale'], checksum=decoded['digest_matches'],
                pixels=decoded['image'])


def address_pattern(identity):
    from stream_format import address_pattern as encode_address
    return encode_address(identity)


def validate_sources():
    for image in 'AB':
        assert (ORDINARY / f'v2_primary_{image}.bin').read_bytes() == (
            CURRENT / f'predictions/control_{image}.bin').read_bytes()
    import fourier_codec as codec
    from stream_format import pack, make_header
    from reconstruction import ssim
    codec.pack = pack
    codec.make_header = make_header
    codec.ssim = ssim
    originals = np.load(INPUTS / 'codec/source/images_float64.npy')
    np.testing.assert_array_equal(np.roll(originals[0], (3, 5), axis=(0, 1)), originals[1])
    np.testing.assert_array_equal(originals, np.load(INPUTS / 'standalone_codec/source_images/images.npy'))
    import encoder
    assert encoder.encode_images(originals) == (ORDINARY / 'frozen_inputs/Lovelace_S12_480.fasta').read_bytes()
    reference = np.load(ORDINARY / 'frozen_inputs/design_reference.npz')
    design_states = reference['expected_cell_states']
    fasta_lines = (ORDINARY / 'frozen_inputs/Lovelace_S12_480.fasta').read_text().splitlines()
    sequences = [line for line in fasta_lines if not line.startswith('>')]
    assert len(sequences) == 480
    for index in range(2):
        saved = np.load(INPUTS / f'codec/source/S12/image_{index}.npz')
        mean, scale, full, block = codec.forward(originals[index])
        labels = codec.quantize(block, scale, 6)
        np.testing.assert_array_equal(labels, saved['codes'])
        np.testing.assert_array_equal(labels, reference['quantized_codes'][index])
        payload = codec.pack(labels, 6)
        header = codec.make_header(payload, mean, scale, 6)
        assert payload == (INPUTS / f'codec/source/S12/image_{index}_payload.bin').read_bytes()
        assert header == (INPUTS / f'codec/source/S12/image_{index}_header.bin').read_bytes()
        digits = ((np.frombuffer(header + payload, np.uint8)[:, None] >> [6, 4, 2, 0]) & 3).reshape(224, 8)
        np.testing.assert_array_equal(digits, design_states[index * 224:(index + 1) * 224])
        decoded = codec.dequantize(labels, scale, 6)
        pixels, imaginary_rms = codec.inverse(decoded, mean)
        np.testing.assert_allclose(pixels, saved['reconstruction'], atol=5e-15, rtol=0)
    for identity, (states, sequence) in enumerate(zip(design_states, sequences)):
        from stream_format import strand
        rebuilt = strand(identity, states)
        assert len(rebuilt) == 298 and rebuilt == sequence
    example = dict(identifier='lovelace_0032', address=32, selection='lowest image A payload address',
                   address_pattern=address_pattern(32), states=design_states[32],
                   lengths=STATES[design_states[32]], sequence=sequences[32],
                   scalar_index=0, component='real', coefficient_frequency=[-8, -8],
                   quantized_label=int(reference['quantized_codes'][0, 0, 0, 0]),
                   normalized_value=float(-1 + 2 * reference['quantized_codes'][0, 0, 0, 0] / 63))
    write_record(HERE / 'molecular_example.tsv', example)
    return originals, reference, codec, example


def metrics(prediction, target, codec):
    from reconstruction import pixel_metrics
    values = pixel_metrics(prediction, target)
    return dict(pixel_count=np.asarray(prediction).size, pixel_rmse=values['rmse'],
                pixel_mae=values['mae'], pixel_mse=values['mse'],
                psnr_db=values['psnr_db'], ssim=values['ssim'], psnr_data_range=1.,
                ssim_definition='frozen codec 7 by 7 uniform window, sample covariance, 26 by 26 interior',
                ssim_evaluated_centers=26 * 26)


def encoding_stages(originals, codec):
    arrays, rows = {}, []
    for index, image in enumerate('AB'):
        mean, scale, full, block = codec.forward(originals[index])
        crop, crop_imaginary_rms = codec.inverse(block, mean)
        labels = codec.quantize(block, scale, 6)
        rounded, rounded_imaginary_rms = codec.inverse(codec.dequantize(labels, scale, 6), mean)
        saved = np.load(INPUTS / f'codec/source/S12/image_{index}.npz')
        np.testing.assert_array_equal(labels, saved['codes'])
        np.testing.assert_allclose(rounded, saved['reconstruction'], atol=5e-15, rtol=0)
        retained = np.zeros_like(full)
        retained[8:24, 8:24] = block
        arrays.update({f'original_{image}': originals[index],
                       f'mean_removed_{image}': originals[index] - mean,
                       f'full_spectrum_{image}': full, f'retained_spectrum_{image}': retained,
                       f'crop_{image}': crop, f'rounded_{image}': rounded,
                       f'crop_difference_{image}': crop - originals[index],
                       f'quantization_difference_{image}': rounded - crop})
        for stage, prediction, target_name, target in [
                ('crop', crop, 'original', originals[index]),
                ('rounding_increment', rounded, 'crop', crop),
                ('rounded', rounded, 'original', originals[index])]:
            rows.append(dict(image=image, stage=stage, target=target_name,
                             domain='physical_intensity', **metrics(prediction, target, codec),
                             mean=mean, coefficient_scale=scale,
                             crop_inverse_imaginary_rms=crop_imaginary_rms,
                             rounded_inverse_imaginary_rms=rounded_imaginary_rms,
                             metric_clipping=False, alignment=False))
    mean_limit = max(float(np.abs(arrays[f'mean_removed_{image}']).max()) for image in 'AB')
    magnitude_limit = max(float(np.abs(arrays[f'full_spectrum_{image}']).max()) for image in 'AB')
    difference_limit = max(float(np.abs(arrays[f'{stage}_difference_{image}']).max())
                           for stage in ['crop', 'quantization'] for image in 'AB')
    display = dict(image='A', stage_metrics_cover_images=['A', 'B'],
                   grayscale_range=[0, 1], mean_removed_range=[-mean_limit, mean_limit],
                   magnitude_range=[0, magnitude_limit], magnitude_transform='absolute value, linear scale',
                   signed_difference_range=[-difference_limit, difference_limit],
                   difference_definitions=['crop minus original', 'rounded minus crop'],
                   shared_difference_scale_across_both_images_and_stages=True,
                   crop_indices=[8, 24], crop_outline_pixel_edges=[7.5, 23.5],
                   retained_spectrum_display='full 32 by 32 grid, zero outside the retained block',
                   signed_palette=['#0072B2', '#FFFFFF', '#D55E00'], magnitude_palette='cividis',
                   quantized_coordinates='signed real and imaginary components, not magnitude',
                   rounded_replay_absolute_tolerance=5e-15, inverse_uses_real_part=True,
                   interpolation='nearest', metric_clipping=False)
    np.savez_compressed(HERE / 'encoding_arrays.npz', **arrays)
    write_table(HERE / 'encoding_metrics.csv', rows)
    return arrays, display


def assemble():
    originals, reference, codec, example = validate_sources()
    encoding = encoding_stages(originals, codec)
    label_rows = read_table(CURRENT / 'predictions/labels.tsv')
    partitions = read_table(CURRENT / 'evaluation/partitions.tsv')
    comparisons, components, coverage, arrays = [], [], [], {}
    for index, image in enumerate('AB'):
        source = np.load(INPUTS / f'codec/source/S12/image_{index}.npz')
        true_labels = reference['quantized_codes'][index]
        true_norm = -1 + 2 * true_labels.astype(float) / 63
        design_norm = inverse(true_norm)
        original_norm = (originals[index] - float(source['mean'])) / float(source['scale'])
        arrays[f'original_{image}'] = originals[index]
        arrays[f'designed_{image}'] = source['reconstruction']
        arrays[f'zero_{image}'] = np.full((32, 32), float(source['mean']))
        zero_mse = float(np.mean(design_norm**2))
        for name in ['original', 'designed', 'zero']:
            values = dict(original=original_norm, designed=design_norm, zero=np.zeros((32, 32)))[name]
            for target_name, target in [('designed', design_norm), ('original', original_norm)]:
                comparisons.append(dict(cohort='reference', image=image, method=name,
                    domain='normalized', target=target_name, **metrics(values, target, codec),
                    mse_relative_to_zero=float(np.mean((values - target)**2) / np.mean(target**2))))
            for target_name, target in [('designed', source['reconstruction']), ('original', originals[index])]:
                comparisons.append(dict(cohort='reference', image=image, method=name,
                    domain='physical_intensity', target=target_name,
                    **metrics(arrays[f'{name}_{image}'], target, codec),
                    header_provenance='designed mean for the zero reference'))
        methods = []
        for arm in ARMS:
            pred = unpack_stream(CURRENT / f'predictions/{arm}_{image}.bin')
            saved = np.load(CURRENT / 'evaluation/rendered_arrays.npz')[f'{arm}_{image}']
            np.testing.assert_array_equal(pred['pixels'], saved)
            arrays[f'{arm}_{image}'] = pred['pixels']
            rows = sorted([row for row in label_rows if row['image'] == image and row['arm'] == arm
                           and row['kind'] == 'payload'], key=lambda row: int(row['scalar_index']))
            assert len(rows) == 512
            np.testing.assert_array_equal([int(row['label']) for row in rows], pred['labels'].ravel())
            mask = np.array([row['support_class'] == 'complete' for row in rows]).reshape(16, 16, 2)
            resolved = np.array([row['resolved'] == 'True' for row in rows]).reshape(16, 16, 2)
            correct = pred['labels'] == true_labels
            counts = dict(expected_scalars=512, expected_complex_coefficients=256,
                emitted_estimates=512, directly_supported_scalars=int(mask.sum()),
                partial_support_scalars=sum(row['support_class'] == 'partial' for row in rows),
                no_direct_support_scalars=sum(row['support_class'] == 'none' for row in rows),
                resolved_correct=int((resolved & correct).sum()), resolved_wrong=int((resolved & ~correct).sum()),
                unresolved=int((~resolved).sum()), map_correct=int(correct.sum()),
                map_label_errors=int((~correct).sum()), map_complex_errors=int((~correct.all(-1)).sum()),
                missing_emitted_estimates=0, coefficient_rmse_available=float(np.sqrt(np.mean((pred['normalized'][mask] - true_norm[mask])**2))),
                coefficient_error_evaluated_components=int(mask.sum()),
                coefficient_rmse_all=float(np.sqrt(np.mean((pred['normalized'] - true_norm)**2))),
                header_mean=pred['mean'], header_scale=pred['scale'], checksum_pass=pred['checksum'])
            old = next(r for r in partitions if r['arm'] == arm and r['image'] == image and r['region'] == 'payload_scalars')
            for key in ['resolved_correct', 'resolved_wrong', 'unresolved', 'map_correct']:
                assert counts[key] == int(old[key]), (arm, image, key)
            coverage.append(dict(image=image, method=arm, **counts))
            for scalar, row in enumerate(rows):
                components.append(dict(image=image, method=arm, scalar_index=scalar,
                    expected_label=int(true_labels.ravel()[scalar]), estimated_label=int(pred['labels'].ravel()[scalar]),
                    expected_normalized=true_norm.ravel()[scalar], estimated_normalized=pred['normalized'].ravel()[scalar],
                    resolved=bool(resolved.ravel()[scalar]), direct_support=row['support_class'],
                    provenance=row['provenance'], missing_estimate=False))
            methods.append((arm, pred['normalized'], mask, 'unresolved MAP labels retained, no truth fill'))
            for target_name, target in [('designed', source['reconstruction']), ('original', originals[index])]:
                comparisons.append(dict(cohort='NEW_ABC_ONLY', image=image, method=arm,
                    domain='physical_intensity', target=target_name, **metrics(pred['pixels'], target, codec),
                    **counts, header_provenance='own decoded mean and scale',
                    missing_policy='unresolved MAP labels retained, no truth fill'))
        for name, values, mask, missing_policy in methods:
            pixels = inverse(np.where(np.isfinite(values), values, 0.))
            for target_name, target in [('designed', design_norm), ('original', original_norm)]:
                comparisons.append(dict(cohort='NEW_ABC_ONLY', image=image, method=name,
                    domain='normalized', target=target_name, **metrics(pixels, target, codec),
                    expected_scalars=512, expected_complex_coefficients=256,
                    emitted_estimates=int(np.isfinite(values).sum()),
                    directly_supported_scalars=int(mask.sum()) if name in ARMS else None,
                    coefficient_error_evaluated_components=int(mask.sum()),
                    missing_emitted_estimates=int((~np.isfinite(values)).sum()),
                    coefficient_rmse_available=float(np.sqrt(np.mean((values[mask] - true_norm[mask])**2))),
                    coefficient_rmse_all=float(np.sqrt(np.mean((np.nan_to_num(values) - true_norm)**2))),
                    mse_relative_to_zero=float(np.mean((pixels - target)**2) / np.mean(target**2)),
                    header_provenance='unit scale and zero mean by definition, no fitted affine transform',
                    missing_policy=missing_policy))
    write_table(HERE / 'reconstruction_comparison.csv', comparisons)
    write_table(HERE / 'coefficient_recovery.csv', coverage)
    write_table(HERE / 'coefficient_source_data.csv', components)
    np.savez_compressed(HERE / 'reconstruction_arrays.npz', **arrays)
    return comparisons, coverage, arrays, example, encoding


def draw(comparisons, coverage, arrays, example, encoding):
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor='white')

    def text(x, y, value, **kwargs):
        return fig.text(x / WIDTH_MM, 1 - y / HEIGHT_MM, value, va='top', **kwargs)

    def axes(x, y, width, height):
        return fig.add_axes([x / WIDTH_MM, 1 - (y + height) / HEIGHT_MM,
                             width / WIDTH_MM, height / HEIGHT_MM])

    def arrow(x, y, length=7):
        ax = axes(x, y - 1, length, 2)
        ax.annotate('', xy=(1, .5), xytext=(0, .5),
                    arrowprops=dict(arrowstyle='->', lw=.6, color='#009E73'))
        ax.set_axis_off()

    def down_arrow(x, y, length=3):
        ax = axes(x - 1, y, 2, length)
        ax.annotate('', xy=(.5, 0), xytext=(.5, 1),
                    arrowprops=dict(arrowstyle='->', lw=.6, color='#009E73'))
        ax.set_axis_off()

    stage_arrays, display = encoding
    signed = LinearSegmentedColormap.from_list('signed_intensity', display['signed_palette'])

    def image_tile(x, y, size, value, cmap='gray', limits=(0, 1), outline=False):
        ax = axes(x, y, size, size)
        ax.imshow(value, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
        if outline:
            ax.add_patch(Rectangle((7.5, 7.5), 16, 16, fill=False, edgecolor='white', lw=.6))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#009E73')
            spine.set_linewidth(.6)
        return ax

    def tint(color, fraction):
        return tuple(1 - fraction * (1 - v) for v in to_rgb(color))

    def card(x, y, width, height, color, fraction=.06, edge=None):
        patch = FancyBboxPatch((x / WIDTH_MM, 1 - (y + height) / HEIGHT_MM),
                               width / WIDTH_MM, height / HEIGHT_MM,
                               boxstyle='round,pad=0,rounding_size=0.004',
                               transform=fig.transFigure, facecolor=tint(color, fraction),
                               edgecolor=edge or 'none', linewidth=.5, zorder=-1)
        fig.add_artist(patch)

    def line(points, color='#444444', style='-', width=.5):
        xs, ys = zip(*points)
        item = Line2D(np.array(xs) / WIDTH_MM, 1 - np.array(ys) / HEIGHT_MM,
                      transform=fig.transFigure, color=color, linestyle=style, linewidth=width)
        fig.add_artist(item)

    scalar_labels = np.load(INPUTS / 'codec/source/S12/image_0.npz')['codes']
    capacity = dict(complex_coefficients=scalar_labels[..., 0].size,
                    scalar_labels=scalar_labels.size, bits_per_label=6,
                    coefficient_payload_bits=8 * (INPUTS / 'codec/source/S12/image_0_payload.bin').stat().st_size,
                    cells_per_strand=len(example['states']), bits_per_cell=2,
                    strand_payload_bits=2 * len(example['states']),
                    payload_nt=int(example['lengths'].sum()), strand_nt=len(example['sequence']),
                    forward_primer_nt=25, address_nt=80, marker_nt=4, markers_per_strand=2,
                    downstream_primer_nt=25)
    assert capacity['scalar_labels'] * capacity['bits_per_label'] == capacity['coefficient_payload_bits']
    assert sum(capacity[k] for k in ['forward_primer_nt', 'address_nt', 'payload_nt', 'downstream_primer_nt']) + 2 * capacity['marker_nt'] == capacity['strand_nt']

    text(2, 2, 'a', weight='bold')
    for x, width, color, heading in [
            (7, 82, '#0072B2', 'Image processing and coefficient selection'),
            (92, 43, '#009E73', 'Coefficient quantization\nand payload formatting'),
            (138, 43, '#CC79A7', 'Two-bit cells and\noligo design')]:
        card(x, 1.5, width, 7, color, .16)
        text(x + width / 2, 3.3 if x == 7 else 2.3, heading, ha='center', weight='bold')
    card(92, 10, 43, 51, '#009E73', .045)
    card(138, 10, 43, 31, '#CC79A7', .055)

    positions = [7, 28, 49, 70]
    titles = ['Source image A', 'Mean removed\n(signed intensity)', 'Full Fourier\nmagnitude', 'Retained Fourier\nmagnitude']
    values = [stage_arrays['original_A'], stage_arrays['mean_removed_A'],
              np.abs(stage_arrays['full_spectrum_A']), np.abs(stage_arrays['retained_spectrum_A'])]
    palettes = ['gray', signed, 'cividis', 'cividis']
    limits = [(0, 1), display['mean_removed_range'], display['magnitude_range'], display['magnitude_range']]
    for index, (x, title, value, cmap, span) in enumerate(zip(positions, titles, values, palettes, limits)):
        card(x, 10, 19, 30, '#0072B2', .045)
        text(x + 9.5, 11, title, ha='center')
        image_tile(x + 2.5, 19, 14, value, cmap, span, outline=index >= 2)
        if index < 3:
            arrow(x + 17, 26, 6)
    text(16.5, 34.2, '32 × 32 pixels', ha='center')
    text(58.5, 34.2, '32 × 32 grid', ha='center')
    text(79.5, 34.2, '16 × 16 retained', ha='center', color='#009E73')
    text(79.5, 37, '(256 coefficients)', ha='center')
    mean_legend = axes(31, 34.2, 13, 1.3)
    mean_legend.imshow(np.linspace(-1, 1, 256)[None, :], cmap=signed, vmin=-1, vmax=1,
                       interpolation='nearest', aspect='auto')
    mean_legend.set_axis_off()
    for x, value in [(31, '−'), (37.5, '0'), (44, '+')]:
        text(x, 35.8, value, ha='center')
    arrow(89.3, 26, 2.4)

    # inverse views form the lower row of the first group
    for x, title, name in [(9.5, 'Original', 'original'), (37.5, 'After crop', 'crop'),
                           (65.5, 'After rounding', 'rounded')]:
        text(x + 5.5, 42, title, ha='center')
        image_tile(x, 46, 11, stage_arrays[f'{name}_A'])
    for x, name, label in [(24, 'crop', 'Crop − source'), (52, 'quantization', 'Rounded − crop')]:
        arrow(x - 3, 48, 16)
        image_tile(x, 51, 7, stage_arrays[f'{name}_difference_A'], signed,
                   display['signed_difference_range'])
        text(x + 3.5, 58.3, label, ha='center')
    text(7, 62, 'Shared difference scale', color='#444444')
    legend = axes(42, 62, 25, 1.4)
    legend.imshow(np.linspace(-1, 1, 256)[None, :], cmap=signed, vmin=-1, vmax=1,
                  interpolation='nearest', aspect='auto')
    legend.set_axis_off()
    for x, value in [(40.5, '−'), (54.5, '0'), (68.5, '+')]:
        text(x, 61.8, value, ha='center')

    text(113.5, 11.5, f"{capacity['complex_coefficients']} complex coefficients", ha='center')
    line([(113.5, 14.6), (103.5, 17)])
    line([(113.5, 14.6), (123.5, 17)])
    for x, name, component, color in [(93.5, 'Real parts', 0, '#009E73'),
                                       (114.5, 'Imaginary parts', 1, '#0072B2')]:
        card(x, 18, 19.5, 8, color, .12)
        text(x + 9.75, 18.8, name + '\n256 values', ha='center')
        down_arrow(x + 9.75, 26.3, 1.8)
        text(x + 9.75, 28.2, 'Six-bit\nquantization\n0 to 63', ha='center')
        down_arrow(x + 9.75, 36.7, 1.8)
        card(x, 39, 19.5, 8, color, .02, edge=color)
        label = int(scalar_labels[0, 0, component])
        bits = f'{label:06b}'
        digits = '  '.join(bits[j:j + 2] for j in range(0, 6, 2))
        text(x + 9.75, 40, f'{label} → {bits}\n{digits}', ha='center')
    line([(103.25, 47.4), (103.25, 49.5), (123.75, 49.5), (123.75, 47.4)])
    down_arrow(113.5, 49.5, 2)
    card(97, 52, 33, 8.5, '#009E73', .01, edge='#009E73')
    text(113.5, 53, f"{capacity['scalar_labels']} six-bit labels\n{capacity['coefficient_payload_bits']:,} payload bits", ha='center')
    arrow(135.3, 28.5, 2.4)

    card(139.5, 14.6, 40, 25.4, '#CC79A7', 0)
    text(159.5, 11.3, 'Six-bit label → three cells', ha='center')
    bits = f"{example['quantized_label']:06b}"
    text(159.5, 15.8, bits + ' → ' + '  '.join(bits[j:j + 2] for j in range(0, 6, 2)), ha='center')
    cell_colors = ['#D55E00', '#009E73', '#0072B2']
    for index, (x, color) in enumerate(zip([140.3, 153.6, 166.9], cell_colors)):
        card(x, 20, 12.3, 13.5, color, .09, edge=color)
        a, c, t = example['lengths'][index]
        text(x + 6.15, 20.6, f'Cell {index + 1}\n({bits[2 * index:2 * index + 2]})\nA{a} C{c}\nT{t}', ha='center')
    text(159.5, 35, 'Each cell totals 20 nt\nThree A/C/T runs', ha='center')
    card(138, 42.2, 43, 6.5, '#CC79A7', .09)
    text(159.5, 42.8, example['identifier'] + f"\n{capacity['cells_per_strand']} cells carry {capacity['strand_payload_bits']} payload bits", ha='center')

    # segment widths use nucleotide counts, including both markers
    bar_x, bar_y, bar_width, bar_height = 139.5, 52, 40, 2.5
    lengths = [25, 80, 4, 160, 4, 25]
    segment_colors = ['#777777', '#E69F00', '#CC79A7', '#0072B2', '#CC79A7', '#777777']
    cursor, centers = bar_x, []
    for count, color in zip(lengths, segment_colors):
        width = bar_width * count / capacity['strand_nt']
        fig.add_artist(Rectangle((cursor / WIDTH_MM, 1 - (bar_y + bar_height) / HEIGHT_MM),
                                 width / WIDTH_MM, bar_height / HEIGHT_MM,
                                 transform=fig.transFigure, facecolor=color, edgecolor='white', linewidth=.4))
        centers.append(cursor + width / 2)
        cursor += width
    text(159.5, 49, 'GGGG', ha='center', color='#777777')
    line([(157, 51.4), (centers[2], 51.8)], color='#777777')
    line([(162, 51.4), (centers[4], 51.8)], color='#777777')
    text(139, 55.3, 'F primer')
    text(180, 55.3, 'R primer', ha='right')
    text(143, 58.2, '16 × 5 nt address')
    text(181, 58.2, '8 cells (160 nt)', ha='right')
    line([(centers[1], 54.7), (centers[1], 57.9)], color='#777777')
    line([(centers[3], 54.7), (centers[3], 57.9)], color='#777777')
    ruler = axes(139.5, 60.6, 40, 2)
    ruler.annotate('', xy=(1, .5), xytext=(0, .5),
                   arrowprops=dict(arrowstyle='|-|', lw=.5, color='#444444'))
    ruler.set_axis_off()
    text(159.5, 60.8, f"{capacity['strand_nt']} nt", ha='center', bbox=dict(facecolor='white', edgecolor='none', pad=.5))

    text(2, 65, 'b', weight='bold')
    colors = ['#0072B2', '#D55E00', '#D9D9D9']
    for x, label, color in zip([9, 29, 47], ['Correct', 'Wrong', 'Unresolved'], colors):
        ax = axes(x, 65.3, 2.5, 1.8)
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none'))
        ax.set_axis_off()
        text(x + 3.5, 65, label)
    ax = axes(23, 69, 42, 16)
    ordered = [next(row for row in coverage if row['image'] == image and row['method'] == arm)
               for image in 'AB' for arm in ARMS]
    for y, row in enumerate(ordered):
        left = 0
        for key, color in zip(['resolved_correct', 'resolved_wrong', 'unresolved'], colors):
            ax.barh(y, row[key], left=left, height=.74, color=color, edgecolor='none')
            left += row[key]
    ax.set_yticks(range(6), [row['image'] + ' ' + NAMES[row['method']] for row in ordered])
    ax.set_xlim(0, 512)
    ax.set_xticks([0, 256, 512])
    ax.set_ylim(5.6, -.6)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', length=2, pad=1)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Scalar labels [count]', labelpad=1)

    text(2, 93, 'd', weight='bold')
    text(9, 93, '○  A   □  B', color='#444444')
    text(65, 93, '0 / 6 checksums', ha='right')
    ax = axes(23, 97, 42, 10)
    for index, image in enumerate('AB'):
        selected = [next(row for row in comparisons if row['image'] == image and row['method'] == arm
                    and row['domain'] == 'normalized' and row['target'] == 'designed') for arm in ARMS]
        ax.scatter([row['mse_relative_to_zero'] for row in selected],
                   np.arange(3) + (index - .5) * .18, marker=['o', 's'][index],
                   s=17, facecolors=['#0072B2', '#009E73'][index], edgecolors='white', linewidths=.4)
    ax.axvline(1, color='#777777', ls=(0, (2, 2)), lw=.7)
    ax.set_yticks(range(3), [NAMES[arm] for arm in ARMS])
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    ax.set_ylim(2.5, -.5)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', length=2, pad=1)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('MSE / zero-reference MSE', labelpad=1)

    text(70, 65, 'c', weight='bold')
    names = ['original', 'designed', *ARMS, 'zero']
    labels = ['Original', 'Designed', 'Control', 'Position', 'Weighted', 'Zero']
    centers = [82, 100, 118, 136, 154, 172]
    for center, label in zip(centers, labels):
        text(center, 65, label, ha='center')
    for image, top in [('A', 70), ('B', 94)]:
        text(71, top + 5, image, weight='bold')
        for center, name in zip(centers, names):
            ax = axes(center - 7.5, top, 15, 15)
            ax.imshow(arrays[f'{name}_{image}'], cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest', origin='upper')
            ax.set_axis_off()
            row = next(r for r in comparisons if r['image'] == image and r['method'] == name
                       and r['domain'] == 'physical_intensity' and r['target'] == 'original')
            text(center, top + 15.3, f"{row['pixel_rmse']:.3f}", ha='center')
    text(71, 89, 'RMSE to source', color='#444444')
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        assert item.get_fontsize() == FONT_SIZE, item.get_text()
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > width + .5 or box.y1 > height + .5:
            outside.append(item.get_text())
    assert not outside, outside
    for suffix in ['svg', 'pdf']:
        metadata = {'Date': None} if suffix == 'svg' else {'CreationDate': None, 'ModDate': None}
        fig.savefig(FIGURE_DIR / f'Figure_Lovelace.{suffix}', facecolor='white', metadata=metadata)
    fig.savefig(FIGURE_DIR / 'Figure_Lovelace_600dpi.png', dpi=600, facecolor='white')
    fig.savefig(FIGURE_DIR / 'Figure_Lovelace.png', dpi=600, facecolor='white')
    fig.savefig(FIGURE_DIR / 'Figure_Lovelace_print_preview.png', dpi=300, facecolor='white')
    plt.close(fig)


def main():
    comparisons, coverage, arrays, example, encoding = assemble()
    draw(comparisons, coverage, arrays, example, encoding)


if __name__ == '__main__':
    main()
