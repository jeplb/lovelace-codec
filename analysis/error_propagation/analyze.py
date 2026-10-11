"""replay saved coefficient streams and decompose their image errors."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import struct
import sys

sys.dont_write_bytecode = True
import numpy as np
from tsv_records import write_output

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE
INTEGRATION = HERE.parent / 'main_figure'
ARMS = ('control', 'position', 'position_weighted')
N = 32 * 32
HEADER = struct.Struct('>4sBBBBBBHddI32s')
FIELDS = (
    ('magic', 0, 4), ('version', 4, 5), ('height', 5, 6), ('width', 6, 7),
    ('retained_height', 7, 8), ('retained_width', 8, 9), ('component_bits', 9, 10),
    ('flags', 10, 12), ('mean', 12, 20), ('scale', 20, 28), ('payload_bytes', 28, 32),
)


def read_csv(path, delimiter=','):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_record(name, value):
    write_output(HERE / name, value)


def load_codec(integration):
    import fourier_codec
    from checksum import inspect
    from stream_format import unpack
    from reconstruction import ssim
    fourier_codec.inspect = inspect
    fourier_codec.unpack = unpack
    fourier_codec.ssim = ssim
    return fourier_codec


def parse_stream(path, codec):
    content = Path(path).read_bytes()
    valid, checksum, fields, payload = codec.inspect(content, 6)
    assert valid, path
    return dict(content=content, labels=codec.unpack(payload, 6), mean=fields[8],
                scale=fields[9], valid=valid, checksum=checksum)


def normalized_block(labels):
    components = -1 + 2 * np.asarray(labels, np.float64) / 63
    return components[..., 0] + 1j * components[..., 1]


def complex_inverse(full):
    return np.fft.ifft2(np.fft.ifftshift(full), norm='ortho')


def embed(block):
    full = np.zeros((32, 32), np.complex128)
    full[8:24, 8:24] = np.asarray(block).reshape(16, 16)
    return full


def spectrum_groups():
    remaining = {(r, c) for r in range(8, 24) for c in range(8, 24)}
    groups = []
    while remaining:
        point = min(remaining)
        partner = ((32 - point[0]) % 32, (32 - point[1]) % 32)
        retained = sorted({point, partner} & remaining)
        remaining.difference_update(retained)
        locations = sorted({point, partner})
        groups.append(dict(
            group_id=len(groups), locations=locations, retained=retained,
            kind='self' if point == partner else 'paired' if len(retained) == 2 else 'crop_boundary',
            representative=locations[0],
        ))
    return groups


def group_decomposition(delta, groups):
    full = embed(delta)
    pixels = complex_inverse(full).real
    rows, parts, location_map = [], [], np.zeros((32, 32), np.float64)
    for group in groups:
        restricted = np.zeros_like(full)
        for point in group['retained']:
            restricted[point] = full[point]
        z = complex_inverse(restricted)
        contribution = float(np.mean(z.real ** 2))
        for point in group['locations']:
            location_map[point] = contribution / len(group['locations'])
        parts.append(z.real)
        point = group['retained'][0]
        partner = ((32 - point[0]) % 32, (32 - point[1]) % 32)
        if group['kind'] == 'self':
            analytic = float(full[point].real ** 2 / N)
        else:
            analytic = float(abs(full[point] + np.conj(full[partner])) ** 2 / (2 * N))
        np.testing.assert_allclose(contribution, analytic, rtol=1e-12, atol=1e-17)
        rows.append(dict(group_id=group['group_id'], group_kind=group['kind'],
            retained_grid_coordinates=json.dumps(group['retained']),
            full_conjugate_coordinates=json.dumps(group['locations']),
            retained_frequencies=json.dumps([[r - 16, c - 16] for r, c in group['retained']]),
            representative_k_row=group['representative'][0] - 16,
            representative_k_column=group['representative'][1] - 16,
            retained_count=len(group['retained']), normalized_pixel_mse=contribution,
            normalized_imaginary_mse=float(np.mean(z.imag ** 2)), analytic_pixel_mse=analytic))
    parts = np.stack(parts)
    np.testing.assert_allclose(parts.sum(axis=0), pixels, rtol=1e-12, atol=2e-16)
    np.testing.assert_allclose(sum(r['normalized_pixel_mse'] for r in rows),
                               np.mean(pixels ** 2), rtol=1e-12, atol=1e-16)
    gram = parts.reshape(len(groups), -1) @ parts.reshape(len(groups), -1).T / N
    off_diagonal = gram - np.diag(np.diag(gram))
    return rows, parts, location_map, float(np.max(np.abs(off_diagonal)))


def mse(values):
    return float(np.mean(np.asarray(values) ** 2))


def checksum_record(stream, truth, codec, unresolved, image, arm):
    content, reference = stream['content'], truth
    payload_xor = np.frombuffer(content[64:], np.uint8) ^ np.frombuffer(reference[64:], np.uint8)
    prefix_xor = np.frombuffer(content[:32], np.uint8) ^ np.frombuffer(reference[:32], np.uint8)
    digest_xor = np.frombuffer(content[32:64], np.uint8) ^ np.frombuffer(reference[32:64], np.uint8)
    incorrect = [name for name, start, end in FIELDS if content[start:end] != reference[start:end]]
    carried = content[32:64]
    calculated = hashlib.sha256(content[:32] + content[64:]).digest()
    designed_digest = reference[32:64]
    assert calculated == hashlib.sha256(content[:32] + content[64:448]).digest()
    assert stream['checksum'] == (carried == calculated)
    return dict(image=image, method=arm, payload_bits_wrong=int(np.unpackbits(payload_xor).sum()),
        payload_bytes_wrong=int(np.count_nonzero(payload_xor)),
        metadata_fields_wrong=json.dumps(incorrect), metadata_fields_wrong_count=len(incorrect),
        metadata_prefix_bits_wrong=int(np.unpackbits(prefix_xor).sum()),
        metadata_prefix_bytes_wrong=int(np.count_nonzero(prefix_xor)),
        unresolved_scalar_labels=unresolved, checksum_algorithm='SHA-256',
        covered_byte_ranges_zero_based_half_open='[0,32) and [64,448)',
        excluded_digest_byte_range='[32,64)', covered_bytes=416,
        covered_content_agrees_with_designed=content[:32] + content[64:] == reference[:32] + reference[64:],
        recovered_digest_agrees_with_designed=carried == designed_digest,
        recovered_digest_bits_wrong=int(np.unpackbits(digest_xor).sum()),
        recovered_digest_bytes_wrong=int(np.count_nonzero(digest_xor)),
        recovered_digest_matches_recomputed=carried == calculated,
        designed_digest_matches_recomputed=designed_digest == calculated,
        header_valid=stream['valid'], overall_checksum_pass=stream['checksum'],
        stream_exact=content == reference, recovered_digest_hex=carried.hex(),
        recomputed_digest_hex=calculated.hex(), designed_digest_hex=designed_digest.hex())


def main(integration=INTEGRATION):
    integration = Path(integration)
    codec = load_codec(integration)
    encoding = np.load(integration / 'encoding_arrays.npz')
    saved_pixels = np.load(integration / 'reconstruction_arrays.npz')
    support_rows = read_csv(integration / 'coefficient_source_data.csv')
    support = {(r['image'], r['method'], int(r['scalar_index'])): r for r in support_rows}
    evidence_path = integration / 'evidence/ordinary/13_observation_model/predictions/labels.tsv'
    evidence = {(r['image'], r['arm'], int(r['scalar_index'])): r
                for r in read_csv(evidence_path, '\t') if r['kind'] == 'payload'}
    groups = spectrum_groups()
    arrays, scalar_rows, digit_rows, bit_rows = {}, [], [], []
    stage_rows, interaction_rows, diagnostic_rows, checksum_rows = [], [], [], []
    group_rows, concentration_rows, changes = [], [], []
    stream_summary, stage_summary, concentration_summary = [], [], []
    by_image_arm, validation = {}, dict(group_count=len(groups),
        paired_groups=sum(g['kind'] == 'paired' for g in groups),
        crop_boundary_groups=sum(g['kind'] == 'crop_boundary' for g in groups),
        self_groups=sum(g['kind'] == 'self' for g in groups),
        full_error_max_absolute_residual=0., group_pixel_sum_max_absolute_residual=0.,
        group_energy_max_absolute_residual=0., group_orthogonality_max_absolute_inner_product=0.,
        decoded_metadata_group_pixel_max_absolute_residual=0., decoded_metadata_group_mse_max_absolute_residual=0.,
        pixel_metrics_clipped=False, affine_correction=False)
    for index, image in enumerate('AB'):
        source = integration / f'evidence/codec/source/S12/image_{index}.npz'
        header_path = integration / f'evidence/codec/source/S12/image_{index}_header.bin'
        payload_path = integration / f'evidence/codec/source/S12/image_{index}_payload.bin'
        stored = np.load(source)
        truth = header_path.read_bytes() + payload_path.read_bytes()
        true_labels = stored['codes']
        true_mean, true_scale = float(stored['mean']), float(stored['scale'])
        original, crop, designed = [encoding[f'{key}_{image}'] for key in ('original', 'crop', 'rounded')]
        np.testing.assert_allclose(codec.inverse(codec.dequantize(true_labels, true_scale, 6), true_mean)[0],
                                   designed, atol=5e-15, rtol=0)
        crop_error, quant_error = crop - original, designed - crop
        np.testing.assert_array_equal(crop_error, encoding[f'crop_difference_{image}'])
        np.testing.assert_array_equal(quant_error, encoding[f'quantization_difference_{image}'])
        true_norm = normalized_block(true_labels)
        design_norm = codec.inverse(true_norm, 0.)[0]
        arrays.update({f'original_{image}': original, f'crop_{image}': crop,
            f'designed_{image}': designed, f'crop_error_{image}': crop_error,
            f'quantization_error_{image}': quant_error, f'design_normalized_{image}': design_norm})
        for name, error, target in [('crop', crop_error, 'original'),
                                   ('quantization_increment', quant_error, 'cropped'),
                                   ('quantized_to_original', designed - original, 'original')]:
            stage_rows.append(dict(image=image, method='representation', stage=name, target=target,
                domain='physical_intensity', pixel_mse=mse(error), pixel_rmse=np.sqrt(mse(error))))
        stage_summary.append(dict(image=image, mean=true_mean, scale=true_scale,
            crop_rmse=np.sqrt(mse(crop_error)), quantization_increment_rmse=np.sqrt(mse(quant_error)),
            quantized_to_original_rmse=np.sqrt(mse(designed - original)),
            crop_quantization_interaction=2 * float(np.mean(crop_error * quant_error))))
        for arm in ARMS:
            stream_path = integration / f'evidence/ordinary/13_observation_model/predictions/{arm}_{image}.bin'
            stream = parse_stream(stream_path, codec)
            estimated = stream['labels']
            estimated_norm = normalized_block(estimated)
            pixels = codec.inverse(codec.dequantize(estimated, stream['scale'], 6), stream['mean'])[0]
            np.testing.assert_array_equal(pixels, saved_pixels[f'{arm}_{image}'])
            error = pixels - designed
            total_error = pixels - original
            coefficient_only = codec.inverse(codec.dequantize(estimated, true_scale, 6), true_mean)[0]
            metadata_only = codec.inverse(codec.dequantize(true_labels, stream['scale'], 6), stream['mean'])[0]
            coeff_error = coefficient_only - designed
            metadata_error = metadata_only - designed
            mixed_interaction = pixels - coefficient_only - metadata_only + designed
            normalized_error = codec.inverse(estimated_norm - true_norm, 0.)[0]
            components_expected = np.stack((true_norm.real, true_norm.imag), axis=-1)
            components_estimated = np.stack((estimated_norm.real, estimated_norm.imag), axis=-1)
            delta_labels = estimated.astype(np.int16) - true_labels.astype(np.int16)
            assert np.count_nonzero(delta_labels) == 512 - sum(
                support[(image, arm, scalar)]['expected_label'] == support[(image, arm, scalar)]['estimated_label']
                for scalar in range(512))
            stage_sum = crop_error + quant_error + error
            residual = float(np.max(np.abs(stage_sum - total_error)))
            validation['full_error_max_absolute_residual'] = max(validation['full_error_max_absolute_residual'], residual)
            np.testing.assert_allclose(stage_sum, total_error, atol=2e-16, rtol=1e-12)
            diag_sum = coeff_error + metadata_error + mixed_interaction
            np.testing.assert_allclose(diag_sum, error, atol=2e-16, rtol=1e-12)
            stages = [('crop', crop_error), ('quantization', quant_error), ('readout', error)]
            for name, value in stages + [('total', total_error)]:
                stage_rows.append(dict(image=image, method=arm, stage=name, target='original' if name == 'total' else
                    'original' if name == 'crop' else 'cropped' if name == 'quantization' else 'designed',
                    domain='physical_intensity', pixel_mse=mse(value), pixel_rmse=np.sqrt(mse(value))))
            cross_terms = []
            for i, (left, a) in enumerate(stages):
                for right, b in stages[i + 1:]:
                    cross = 2 * float(np.mean(a * b))
                    interaction_rows.append(dict(image=image, method=arm, left_stage=left, right_stage=right,
                        domain='physical_intensity', twice_mean_product=cross))
                    cross_terms.append(cross)
            reconstructed_mse = sum(mse(v) for _, v in stages) + sum(cross_terms)
            np.testing.assert_allclose(reconstructed_mse, mse(total_error), rtol=1e-12, atol=1e-16)
            for coeff_source, metadata_source, diagnostic in [
                    ('designed', 'designed', designed), ('estimated', 'designed', coefficient_only),
                    ('designed', 'decoded', metadata_only), ('estimated', 'decoded', pixels)]:
                diagnostic_rows.append(dict(image=image, method=arm, coefficient_source=coeff_source,
                    metadata_source=metadata_source, diagnostic_mixed=coeff_source != ('estimated' if metadata_source == 'decoded' else 'designed'),
                    pixel_rmse_to_designed=np.sqrt(mse(diagnostic - designed)),
                    pixel_mse_to_designed=mse(diagnostic - designed),
                    pixel_rmse_to_original=np.sqrt(mse(diagnostic - original)),
                    pixel_mse_to_original=mse(diagnostic - original), domain='physical_intensity'))
                arrays[f'diagnostic_{image}_{arm}_{coeff_source}_{metadata_source}'] = diagnostic
            decomp, parts, location_map, orthogonality = group_decomposition(estimated_norm - true_norm, groups)
            physical_delta = codec.dequantize(estimated, stream['scale'], 6) - codec.dequantize(true_labels, true_scale, 6)
            physical_groups, physical_parts, _, _ = group_decomposition(physical_delta, groups)
            mean_error = np.full((32, 32), stream['mean'] - true_mean)
            physical_group_sum = physical_parts.sum(0)
            physical_residual = float(np.max(np.abs(physical_group_sum + mean_error - error)))
            physical_mean_cross = 2 * float(np.mean(physical_group_sum * mean_error))
            physical_group_mse = sum(r['normalized_pixel_mse'] for r in physical_groups) + mse(mean_error) + physical_mean_cross
            np.testing.assert_allclose(physical_group_sum + mean_error, error, rtol=1e-12, atol=4e-16)
            np.testing.assert_allclose(physical_group_mse, mse(error), rtol=1e-12, atol=1e-16)
            validation['decoded_metadata_group_pixel_max_absolute_residual'] = max(
                validation['decoded_metadata_group_pixel_max_absolute_residual'], physical_residual)
            validation['decoded_metadata_group_mse_max_absolute_residual'] = max(
                validation['decoded_metadata_group_mse_max_absolute_residual'], abs(physical_group_mse - mse(error)))
            physical_by_id = {r['group_id']: r for r in physical_groups}
            for row in decomp:
                row['physical_readout_mse_with_decoded_scale'] = physical_by_id[row['group_id']]['normalized_pixel_mse']
            norm_mse = mse(normalized_error)
            validation['group_orthogonality_max_absolute_inner_product'] = max(
                validation['group_orthogonality_max_absolute_inner_product'], orthogonality)
            validation['group_pixel_sum_max_absolute_residual'] = max(validation['group_pixel_sum_max_absolute_residual'],
                float(np.max(np.abs(parts.sum(0) - normalized_error))))
            validation['group_energy_max_absolute_residual'] = max(validation['group_energy_max_absolute_residual'],
                abs(sum(r['normalized_pixel_mse'] for r in decomp) - norm_mse))
            ranked = sorted(decomp, key=lambda r: (-r['normalized_pixel_mse'], r['group_id']))
            cumulative = 0.
            for rank, row in enumerate(ranked, 1):
                cumulative += row['normalized_pixel_mse']
                row.update(image=image, method=arm, physical_coefficient_only_pixel_mse=true_scale ** 2 * row['normalized_pixel_mse'],
                    contribution_rank=rank, normalized_mse_fraction=row['normalized_pixel_mse'] / norm_mse if norm_mse else 0.,
                    cumulative_mse_fraction=cumulative / norm_mse if norm_mse else 0.)
                concentration_rows.append(dict(image=image, method=arm, contribution_rank=rank,
                    cumulative_mse_fraction=row['cumulative_mse_fraction'], group_id=row['group_id']))
            group_rows.extend(ranked)
            concentration_summary.append(dict(image=image, method=arm, normalized_pixel_mse=norm_mse,
                nonzero_error_groups=sum(r['normalized_pixel_mse'] > 1e-25 for r in decomp),
                top_group=ranked[0], top_three_fraction=sum(r['normalized_mse_fraction'] for r in ranked[:3]),
                top_five_fraction=sum(r['normalized_mse_fraction'] for r in ranked[:5])))
            for scalar in range(512):
                coefficient, part = divmod(scalar, 2)
                row, col = divmod(coefficient, 16)
                expected_label = int(true_labels.ravel()[scalar])
                estimate = int(estimated.ravel()[scalar])
                wrong_bits = [bit + 1 for bit in range(6) if ((expected_label ^ estimate) >> (5 - bit)) & 1]
                sr, ev = support[(image, arm, scalar)], evidence[(image, arm, scalar)]
                digits_expected = [(expected_label >> shift) & 3 for shift in (4, 2, 0)]
                digits_estimated = [(estimate >> shift) & 3 for shift in (4, 2, 0)]
                digit_label_delta = [(a - b) * weight for a, b, weight in zip(digits_estimated, digits_expected, (16, 4, 1))]
                assert sum(digit_label_delta) == estimate - expected_label
                scalar_rows.append(dict(image=image, method=arm, scalar_index=scalar, coefficient_index=coefficient,
                    component='real' if part == 0 else 'imaginary', crop_row=row, crop_column=col,
                    grid_row=row + 8, grid_column=col + 8, k_row=row - 8, k_column=col - 8,
                    expected_label=expected_label, estimated_label=estimate, label_difference=estimate - expected_label,
                    incorrect_bits_count=len(wrong_bits), incorrect_bit_positions_msb_first=json.dumps(wrong_bits),
                    encoded_digits=json.dumps(digits_expected), estimated_digits=json.dumps(digits_estimated),
                    digit_label_error_contributions=json.dumps(digit_label_delta),
                    normalized_component_error=float(components_estimated.ravel()[scalar] - components_expected.ravel()[scalar]),
                    physical_component_error=float(2 * true_scale / 63 * (estimate - expected_label)),
                    physical_component_error_with_decoded_scale=float(
                        (-stream['scale'] + 2 * stream['scale'] * estimate / 63) -
                        (-true_scale + 2 * true_scale * expected_label / 63)),
                    resolved=sr['resolved'], direct_support=sr['direct_support'], provenance=sr['provenance'],
                    saved_cell_observation_counts=ev['depth_cells'], used_conjugate_partner=ev['coupled'],
                    cell_locations=json.dumps([[index * 224 + (256 + scalar * 3 + d) // 8,
                                                (256 + scalar * 3 + d) % 8] for d in range(3)])))
            expected_digits = np.stack([(true_labels >> shift) & 3 for shift in (4, 2, 0)], axis=-1).reshape(512, 3)
            estimated_digits = np.stack([(estimated >> shift) & 3 for shift in (4, 2, 0)], axis=-1).reshape(512, 3)
            isolated = (estimated_digits.astype(np.int16) - expected_digits.astype(np.int16)) * [16, 4, 1] * 2 / 63
            np.testing.assert_allclose(isolated.sum(1), (components_estimated - components_expected).ravel(), atol=3e-16, rtol=1e-12)
            for digit in range(3):
                wrong = estimated_digits[:, digit] != expected_digits[:, digit]
                magnitude = np.abs(isolated[:, digit])
                digit_rows.append(dict(image=image, method=arm, digit_position_msb_first=digit + 1,
                    binary_positions=json.dumps([2 * digit + 1, 2 * digit + 2]), label_place_weight=(16, 4, 1)[digit],
                    expected_scalar_count=512, incorrect_digit_count=int(wrong.sum()),
                    mean_absolute_normalized_error_all=float(magnitude.mean()),
                    mean_absolute_normalized_error_if_wrong=float(magnitude[wrong].mean()) if wrong.any() else 0.,
                    normalized_scalar_error_sum_squares=float(np.sum(isolated[:, digit] ** 2)),
                    physical_scalar_error_sum_squares=float(np.sum((true_scale * isolated[:, digit]) ** 2)),
                    physical_mean_absolute_error_if_wrong=float(true_scale * magnitude[wrong].mean()) if wrong.any() else 0.))
            for bit in range(6):
                wrong = ((estimated ^ true_labels) >> (5 - bit)) & 1
                bit_rows.append(dict(image=image, method=arm, bit_position_msb_first=bit + 1,
                    label_place_weight=2 ** (5 - bit), incorrect_bit_count=int(wrong.sum())))
            digit_squared = np.sum(isolated ** 2)
            digit_cross = float(np.sum(isolated.sum(1) ** 2) - digit_squared)
            resolved = sum(support[(image, arm, scalar)]['resolved'] == 'True' for scalar in range(512))
            checksums = checksum_record(stream, truth, codec, 512 - resolved, image, arm)
            checksum_rows.append(checksums)
            stream_summary.append(dict(image=image, method=arm, matching_scalar_labels=int((estimated == true_labels).sum()),
                resolved_correct_labels=sum(support[(image, arm, scalar)]['resolved'] == 'True' and
                    support[(image, arm, scalar)]['estimated_label'] == support[(image, arm, scalar)]['expected_label'] for scalar in range(512)),
                normalized_pixel_rmse_to_designed=np.sqrt(norm_mse), physical_pixel_rmse_to_designed=np.sqrt(mse(error)),
                physical_pixel_rmse_to_original=np.sqrt(mse(total_error)),
                physical_coefficient_only_rmse=np.sqrt(mse(coeff_error)), physical_metadata_only_rmse=np.sqrt(mse(metadata_error)),
                decoded_mean_difference=stream['mean'] - true_mean, decoded_scale_difference=stream['scale'] - true_scale,
                stage_mse_sum=sum(mse(v) for _, v in stages), stage_interaction_sum=sum(cross_terms),
                total_mse=mse(total_error), reconstructed_total_mse=reconstructed_mse,
                normalized_scalar_sum_squared_error=float(np.sum((components_estimated - components_expected) ** 2)),
                digit_isolated_normalized_error_sum_squares=float(digit_squared), digit_normalized_cross_term_sum=digit_cross,
                mixed_metadata_coefficient_interaction_rmse=np.sqrt(mse(mixed_interaction)), checksum_pass=stream['checksum']))
            stream_summary[-1].update(physical_group_mse_sum=sum(r['normalized_pixel_mse'] for r in physical_groups),
                mean_error_mse=mse(mean_error), mean_group_interaction=physical_mean_cross,
                physical_readout_mse_reconstructed_from_groups_and_mean=physical_group_mse)
            arrays.update({f'pixels_{image}_{arm}': pixels, f'readout_error_{image}_{arm}': error,
                f'total_error_{image}_{arm}': total_error, f'coefficient_only_error_{image}_{arm}': coeff_error,
                f'metadata_only_error_{image}_{arm}': metadata_error, f'mixed_interaction_{image}_{arm}': mixed_interaction,
                f'normalized_readout_error_{image}_{arm}': normalized_error,
                f'normalized_coefficient_error_{image}_{arm}': estimated_norm - true_norm,
                f'group_pixel_errors_{image}_{arm}': parts, f'group_location_mse_map_{image}_{arm}': location_map,
                f'physical_group_pixel_errors_{image}_{arm}': physical_parts,
                f'mean_error_{image}_{arm}': mean_error})
            by_image_arm[(image, arm)] = {r['group_id']: r for r in decomp}
        for arm in ('position', 'position_weighted'):
            for group in groups:
                base = by_image_arm[(image, 'control')][group['group_id']]
                revised = by_image_arm[(image, arm)][group['group_id']]
                members = [2 * ((r - 8) * 16 + c - 8) + part for r, c in group['retained'] for part in (0, 1)]
                details = []
                for scalar in members:
                    rows = [r for r in scalar_rows if r['image'] == image and r['scalar_index'] == scalar]
                    b = next(r for r in rows if r['method'] == 'control')
                    a = next(r for r in rows if r['method'] == arm)
                    if b['estimated_label'] != a['estimated_label']:
                        details.append(dict(scalar_index=scalar, component=a['component'], k_row=a['k_row'], k_column=a['k_column'],
                            expected_label=a['expected_label'], control_label=b['estimated_label'], revised_label=a['estimated_label'],
                            control_label_error=b['label_difference'], revised_label_error=a['label_difference']))
                changes.append(dict(image=image, comparison_method=arm, group_id=group['group_id'],
                    group_kind=group['kind'], retained_frequencies=revised['retained_frequencies'],
                    control_normalized_pixel_mse=base['normalized_pixel_mse'], revised_normalized_pixel_mse=revised['normalized_pixel_mse'],
                    normalized_pixel_mse_change=revised['normalized_pixel_mse'] - base['normalized_pixel_mse'],
                    physical_coefficient_mse_change=true_scale ** 2 * (revised['normalized_pixel_mse'] - base['normalized_pixel_mse']),
                    changed_scalar_labels=json.dumps(details)))
            delta = arrays[f'group_location_mse_map_{image}_{arm}'] - arrays[f'group_location_mse_map_{image}_control']
            arrays[f'group_location_change_{image}_{arm}'] = delta
    for image in 'AB':
        for arm in ('position', 'position_weighted'):
            rows = [r for r in changes if r['image'] == image and r['comparison_method'] == arm]
            old = next(r for r in stream_summary if r['image'] == image and r['method'] == 'control')
            new = next(r for r in stream_summary if r['image'] == image and r['method'] == arm)
            np.testing.assert_allclose(sum(r['normalized_pixel_mse_change'] for r in rows),
                new['normalized_pixel_rmse_to_designed'] ** 2 - old['normalized_pixel_rmse_to_designed'] ** 2,
                rtol=1e-12, atol=1e-17)
    write_csv('scalar_error_ledger.csv', scalar_rows)
    write_csv('digit_error_summary.csv', digit_rows)
    write_csv('bit_error_summary.csv', bit_rows)
    write_csv('stage_error_metrics.csv', stage_rows)
    write_csv('stage_interaction_terms.csv', interaction_rows)
    write_csv('coefficient_metadata_diagnostics.csv', diagnostic_rows)
    write_csv('checksum_failure_table.csv', checksum_rows)
    write_csv('fourier_group_contributions.csv', group_rows)
    write_csv('fourier_group_concentration.csv', concentration_rows)
    write_csv('fourier_group_changes.csv', changes)
    write_csv('stream_summary.csv', stream_summary)
    np.savez_compressed(HERE / 'error_arrays.npz', **arrays)
    findings = dict(representation_stages=stage_summary, current_streams=stream_summary,
        fourier_groups=concentration_summary,
        weighted_vs_control={image: dict(
            group_mse_change=sum(r['normalized_pixel_mse_change'] for r in changes
                                 if r['image'] == image and r['comparison_method'] == 'position_weighted'),
            greatest_losses=sorted([r for r in changes if r['image'] == image and r['comparison_method'] == 'position_weighted'],
                                   key=lambda r: r['normalized_pixel_mse_change'], reverse=True)[:5],
            greatest_gains=sorted([r for r in changes if r['image'] == image and r['comparison_method'] == 'position_weighted'],
                                  key=lambda r: r['normalized_pixel_mse_change'])[:5]) for image in 'AB'},
        checksum_table=checksum_rows, validation=validation)
    write_record('numerical_findings.tsv', findings)
    print(json.dumps(dict(streams=len(stream_summary), checksum_passes=sum(r['checksum_pass'] for r in stream_summary),
        full_error_max_residual=validation['full_error_max_absolute_residual'],
        group_energy_max_residual=validation['group_energy_max_absolute_residual'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', type=Path, default=INTEGRATION)
    main(parser.parse_args().integration)
