"""trace read support and rerun only frozen aggregation and inference."""

import csv
from collections import Counter
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import struct
import sys
import time

import numpy as np
import yaml
from tsv_records import append_table_rows, read_output, write_output


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRENT = ROOT / 'data/analysis_inputs/13_observation_model'
FROZEN = CURRENT.parent / 'frozen_inputs'
V1 = CURRENT.parent / '12_format_aware_decode'
V2 = V1 / 'v2'
INTEGRATION = HERE.parent / 'main_figure'
ARMS = ('control', 'position', 'position_weighted')
FRACTIONS = (.10, .25, .50, .75, 1.)
N_DRAWS = 50
SEED = 1308102026
sys.dont_write_bytecode = True
for path in (V1, V2, CURRENT):
    sys.path.insert(0, str(path))

from inference import decode_arm
from format_constraints import pairing_rows, scalar_rows
from cell_model import aggregate
from suffix_weighting import build_references, score_weights


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tsv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_record(path, value):
    write_output(path, value)


def inverse(values, mean=0.):
    from reconstruction import from_components
    return from_components(values, mean)


def ablation_decoder():
    # disable partners. keep the dc constraint and scale gate
    def decode_without_partners(scores, supported, config, arm_config, proven_bits=()):
        return decode_arm(scores, supported, config, arm_config, proven_bits, partner_enabled=False)
    return decode_without_partners, dict(partner_enabled=False,
        implementation_sha256=sha(ROOT / 'inference.py'), dc_constraint_retained=True,
        header_rules_retained=True, scale_gate_retained=True)


def decode(scores, identities, weights, config, partner_enabled=True):
    summed, depth = aggregate(scores, identities, weights, n_addresses=480)
    function = decode_arm if partner_enabled else ablation_decoder()[0]
    result = function(summed, depth > 0, config, dict(header_mask='v2', coupling=True),
                      config['header_mask']['proven_bits'])
    return result, summed, depth


def metrics(result, truth, designed_physical, originals):
    rows = []
    label_arrays, resolved_arrays = [], []
    for image_index, image in enumerate('AB'):
        content = result['streams'][image]
        payload_bits = np.unpackbits(np.frombuffer(content[64:], dtype=np.uint8), bitorder='big')
        labels = payload_bits.reshape(512, 6) @ (2 ** np.arange(5, -1, -1))
        scalar_rows_image = sorted([row for row in result['label_rows']
            if row['image'] == image and row['kind'] == 'payload'], key=lambda row: row['scalar_index'])
        resolution = np.array([row['resolved'] for row in scalar_rows_image], dtype=bool)
        np.testing.assert_array_equal(labels, [row['label'] for row in scalar_rows_image])
        true_labels = truth[image_index].reshape(512)
        correct = labels == true_labels
        error = 2 * (labels.astype(float) - true_labels) / 63
        normalized = inverse(-1 + 2 * labels.astype(float) / 63)
        designed = inverse(-1 + 2 * true_labels.astype(float) / 63)
        fields = struct.unpack('>4sBBBBBBHddI32s', content[:64])
        mean, scale = fields[8:10]
        checksum_match = content[32:64] == hashlib.sha256(content[:32] + content[64:]).digest()
        header_valid = bool(fields[:8] == (b'LVIM', 1, 32, 32, 16, 16, 6, 1)
            and np.isfinite(mean) and 0 <= mean <= 1 and np.isfinite(scale)
            and 0 < scale <= 32 and fields[10] == 384)
        physical = None
        if np.isfinite(mean) and np.isfinite(scale) and scale > 0:
            with np.errstate(over='ignore', invalid='ignore'):
                pixels = inverse(-scale + 2 * scale * labels.astype(float) / 63, mean)
                physical_error = float(np.sqrt(np.mean((pixels - designed_physical[image_index]) ** 2)))
                original_error = float(np.sqrt(np.mean((pixels - originals[image_index]) ** 2)))
                if np.isfinite(physical_error) and np.isfinite(original_error):
                    physical = (physical_error, original_error)
        rows.append(dict(image=image, scalar_count=512, resolved_labels=int(resolution.sum()),
            resolved_correct=int((resolution & correct).sum()), resolved_wrong=int((resolution & ~correct).sum()),
            unresolved_labels=int((~resolution).sum()), exact_scalar_matches=int(correct.sum()),
            coefficient_rmse_normalized=float(np.sqrt(np.mean(error ** 2))),
            image_rmse_normalized_designed=float(np.sqrt(np.mean((normalized - designed) ** 2))),
            image_rmse_physical_designed=None if physical is None else physical[0],
            image_rmse_physical_original=None if physical is None else physical[1],
            physical_image_finite=physical is not None, header_valid=header_valid,
            carried_checksum_matches=checksum_match, checksum_pass=header_valid and checksum_match,
            stream_exact=content == true_stream(truth, image_index),
            stream_sha256=hashlib.sha256(content).hexdigest(),
            conjugate_enabled_scalars=sum(row['coupled'] for row in scalar_rows_image)))
        label_arrays.append(labels)
        resolved_arrays.append(resolution)
    return rows, np.array(label_arrays), np.array(resolved_arrays)


def true_stream(truth, image_index):
    # the scoring reference comes from saved encoded cells only.
    with np.load(FROZEN / 'design_reference.npz', allow_pickle=False) as saved:
        states = saved['expected_cell_states'][224 * image_index:224 * (image_index + 1)]
    return (states.reshape(-1, 4) @ np.array([64, 16, 4, 1])).astype(np.uint8).tobytes()


def ledger(full_results, units, truth):
    layout = {(row['image'], row['scalar_index']): row for row in scalar_rows()}
    pairs = {row['coefficient_index']: row for row in pairing_rows()}
    parents = units['parents']
    identities = units['identities']
    ids = units['unit_ids']
    address_units = {address: np.flatnonzero(identities == address) for address in range(448)}
    output, dependencies, support_cells = [], [], []
    for address, indices in address_units.items():
        for index in indices:
            for cell in range(8):
                support_cells.append(dict(address=address, cell=cell, raw_parent_id=parents[index],
                    unit_id=ids[index], frozen_unit_index=int(index)))
    for arm, result in full_results.items():
        for row in result['label_rows']:
            if row['kind'] != 'payload':
                continue
            image, scalar = row['image'], row['scalar_index']
            location = layout[image, scalar]
            pairing = pairs[row['coefficient_index']]
            values = dict(method=arm, image=image, scalar_index=scalar,
                coefficient_index=row['coefficient_index'], freq_u=pairing['freq_u'], freq_v=pairing['freq_v'],
                fourier_row=pairing['freq_u'] + 16, fourier_column=pairing['freq_v'] + 16,
                part=row['part'], direct_support='absent' if row['support_class'] == 'none' else row['support_class'],
                direct_supported_cells=row['supported_cell_count'], conjugate_information_used=row['coupled'],
                dc_format_information_used=row['dc_fixed'], resolution_flag=row['resolved'],
                provenance=row['provenance'], emitted_six_bit_label=row['label'],
                encoded_six_bit_label=int(truth['AB'.index(image)].reshape(-1)[scalar]),
                exact_agreement=row['label'] == int(truth['AB'.index(image)].reshape(-1)[scalar]),
                normalized_component_error=2 * (row['label'] - int(truth['AB'.index(image)].reshape(-1)[scalar])) / 63,
                partner_scalar_index=row['partner_scalar_index'], scalar_crosses_strand=location['global_address'] != location['end_address'])
            for digit in range(3):
                address, cell = location[f'address_{digit}'], location[f'cell_{digit}']
                indices = address_units[address]
                values.update({f'cell_{digit}_address': address, f'cell_{digit}_position': cell,
                    f'cell_{digit}_contributing_parents': len(set(parents[indices])),
                    f'cell_{digit}_parent_ids': json.dumps(sorted(set(parents[indices].tolist()))),
                    f'cell_{digit}_unit_ids': json.dumps(ids[indices].tolist())})
            output.append(values)
            if row['coupled']:
                partner = layout[image, int(row['partner_scalar_index'])]
                for digit in range(3):
                    address, cell = partner[f'address_{digit}'], partner[f'cell_{digit}']
                    indices = address_units[address]
                    dependencies.append(dict(method=arm, image=image, destination_scalar=scalar,
                        source_partner_scalar=int(row['partner_scalar_index']), digit_position=digit,
                        source_address=address, source_cell=cell, contributing_parent_count=len(set(parents[indices])),
                        source_parent_ids=json.dumps(sorted(set(parents[indices].tolist()))),
                        source_unit_ids=json.dumps(ids[indices].tolist()), independent_new_observations=False))
    cross = [dict(method=arm, image=image, direct_support=support, resolved=resolved,
        scalar_count=sum(row['method'] == arm and row['image'] == image
            and row['direct_support'] == support and row['resolution_flag'] == resolved for row in output))
        for arm in ARMS for image in 'AB' for support in ('complete', 'partial', 'absent') for resolved in (True, False)]
    return output, dependencies, support_cells, cross


def main():
    started = time.time()
    HERE.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((CURRENT / 'config.yaml').read_text())
    selection = read_output(CURRENT / 'fitting/selection.tsv')
    with np.load(CURRENT / 'predictions/weight_inputs.npz', allow_pickle=False) as saved:
        units = {key: saved[key].copy() for key in ('identities', 'parents', 'unit_ids',
            'unit_log_scores', 'global_unit_log_scores', 'weights')}
    all_parents = np.array(sorted({row['raw_parent_id'] for row in read_tsv(FROZEN / 'read_audit.tsv')}))
    assert set(units['parents']).issubset(set(all_parents))
    assert len(set(zip(units['parents'], units['identities']))) == len(units['parents'])
    full_reference, full_other_counts = build_references(units['unit_log_scores'], units['identities'], units['parents'])
    full_weighting = score_weights(units['unit_log_scores'], full_reference, full_other_counts,
        selection['selected_candidate'], config)
    np.testing.assert_array_equal(full_weighting['weights'], units['weights'])
    with np.load(FROZEN / 'design_reference.npz', allow_pickle=False) as saved:
        truth = saved['quantized_codes'].copy()
    designed_physical = np.array([np.load(INTEGRATION / f'evidence/codec/source/S12/image_{i}.npz')['reconstruction'] for i in range(2)])
    originals = np.load(INTEGRATION / 'evidence/codec/source/images_float64.npy')
    full_results, full_metrics, full_labels, full_resolution, ablated_metrics = {}, [], [], [], []
    for arm in ARMS:
        scores = units['global_unit_log_scores'] if arm == 'control' else units['unit_log_scores']
        weights = units['weights'] if arm == 'position_weighted' else np.ones(scores.shape[:2])
        result, summed, depth = decode(scores, units['identities'], weights, config)
        full_results[arm] = result
        for image in 'AB':
            assert result['streams'][image] == (CURRENT / f'predictions/{arm}_{image}.bin').read_bytes()
            assert result['streams'][image] == (INTEGRATION / f'evidence/ordinary/13_observation_model/predictions/{arm}_{image}.bin').read_bytes()
        rows, labels, resolution = metrics(result, truth, designed_physical, originals)
        full_metrics.extend(dict(method=arm, partner_information=True, **row) for row in rows)
        full_labels.append(labels)
        full_resolution.append(resolution)
        ablated, ablated_sum, ablated_depth = decode(scores, units['identities'], weights, config, False)
        np.testing.assert_array_equal(summed, ablated_sum)
        np.testing.assert_array_equal(depth, ablated_depth)
        assert [row['scale_hex'] for row in result['gates']] == [row['scale_hex'] for row in ablated['gates']]
        for image in 'AB':
            assert result['streams'][image][:64] == ablated['streams'][image][:64]
            (HERE / f'no_partner_{arm}_{image}.bin').write_bytes(ablated['streams'][image])
        rows, _, _ = metrics(ablated, truth, designed_physical, originals)
        ablated_metrics.extend(dict(method=arm, partner_information=False, **row) for row in rows)
    ledgers, dependencies, support_cells, cross = ledger(full_results, units, truth)
    for name, rows in [('scalar_evidence_ledger.tsv', ledgers), ('conjugate_dependencies.tsv', dependencies),
                       ('direct_cell_observations.tsv', support_cells), ('support_resolution.tsv', cross),
                       ('partner_ablation.tsv', full_metrics + ablated_metrics)]:
        write_tsv(HERE / name, rows)
    admitted_parent_set = set(units['parents'])
    admitted_parent_counts = Counter(units['parents'])
    write_tsv(HERE / 'parent_universe.tsv', [dict(raw_parent_id=parent,
        admitted_for_current_prediction=parent in admitted_parent_set,
        admitted_unit_count=admitted_parent_counts[parent]) for parent in all_parents])
    write_tsv(HERE / 'parent_observation_mapping.tsv', [dict(raw_parent_id=parent, unit_id=unit,
        address=int(address), frozen_unit_index=index) for index, (parent, unit, address)
        in enumerate(zip(units['parents'], units['unit_ids'], units['identities']))])
    rng = np.random.default_rng(SEED)
    samples, draw_metrics, draw_labels, draw_resolved, streams = [], [], [], [], []
    parents_index = {parent: index for index, parent in enumerate(all_parents)}
    unit_parent_indices = np.array([parents_index[parent] for parent in units['parents']])
    sample_target = HERE / 'sampled_parent_ids.tsv.gz'
    sample_fields = ['fraction', 'draw', 'parent_count', 'parent_set_sha256', 'seed', 'raw_parent_ids']
    with gzip.open(sample_target, 'wt', newline='') as handle:
        append_table_rows(handle, [], sample_fields, header=True)
        for fraction in FRACTIONS:
            sample_count = int(np.floor(fraction * len(all_parents)))
            for draw in range(N_DRAWS):
                selected = np.sort(rng.choice(len(all_parents), sample_count, replace=False)) if fraction < 1 else np.arange(len(all_parents))
                sample_ids = all_parents[selected]
                set_digest = hashlib.sha256(('\n'.join(sample_ids) + '\n').encode()).hexdigest()
                record = dict(fraction=fraction, draw=draw, parent_count=sample_count,
                              parent_set_sha256=set_digest, seed=SEED, raw_parent_ids=sample_ids.tolist())
                append_table_rows(handle, [record], sample_fields, header=False)
                included = np.isin(unit_parent_indices, selected)
                ids, parents = units['identities'][included], units['parents'][included]
                sample = dict(fraction=fraction, draw=draw, original_parent_count=len(all_parents),
                    sampled_parent_count=sample_count, admitted_unit_count=int(included.sum()),
                    admitted_parent_count=len(set(parents)), parent_set_sha256=set_digest,
                    paired_across_methods=True, all_parent_units_retained=True)
                samples.append(sample)
                labels_per_draw, resolved_per_draw, streams_per_draw = [], [], []
                if fraction < 1:
                    position_scores = units['unit_log_scores'][included]
                    references, other_counts = build_references(position_scores, ids, parents)
                    weighted = score_weights(position_scores, references, other_counts, selection['selected_candidate'], config)
                for arm_index, arm in enumerate(ARMS):
                    if fraction == 1:
                        result = full_results[arm]
                        rows = [{key: value for key, value in row.items() if key not in ('method', 'partner_information')}
                            for row in full_metrics if row['method'] == arm]
                        labels, resolution = full_labels[arm_index], full_resolution[arm_index]
                    else:
                        scores = units['global_unit_log_scores'][included] if arm == 'control' else position_scores
                        weights = weighted['weights'] if arm == 'position_weighted' else np.ones(scores.shape[:2])
                        result, _, _ = decode(scores, ids, weights, config)
                        rows, labels, resolution = metrics(result, truth, designed_physical, originals)
                    draw_metrics.extend(dict(fraction=fraction, draw=draw, method=arm, **row) for row in rows)
                    labels_per_draw.append(labels)
                    resolved_per_draw.append(resolution)
                    streams_per_draw.append([np.frombuffer(result['streams'][image], dtype=np.uint8) for image in 'AB'])
                draw_labels.append(labels_per_draw)
                draw_resolved.append(resolved_per_draw)
                streams.append(streams_per_draw)
            print(f'completed fraction {fraction} with {N_DRAWS} paired draws', flush=True)
    aggregates = []
    metric_fields = ('resolved_labels', 'resolved_correct', 'exact_scalar_matches',
                     'coefficient_rmse_normalized', 'image_rmse_normalized_designed')
    for fraction in FRACTIONS:
        for arm in ARMS:
            for image in 'AB':
                selected = [row for row in draw_metrics if row['fraction'] == fraction
                            and row['method'] == arm and row['image'] == image]
                for field in metric_fields:
                    values = np.array([row[field] for row in selected])
                    aggregates.append(dict(fraction=fraction, method=arm, image=image, metric=field,
                        mean=float(values.mean()), q05=float(np.quantile(values, .05)),
                        q95=float(np.quantile(values, .95)), draw_count=len(values)))
    write_tsv(HERE / 'sample_summary.tsv', samples)
    write_tsv(HERE / 'read_sampling_draws.tsv', draw_metrics)
    write_tsv(HERE / 'read_sampling_summary.tsv', aggregates)
    np.savez_compressed(HERE / 'sampling_predictions.npz', labels=np.array(draw_labels),
        resolved=np.array(draw_resolved), streams=np.array(streams),
        fractions=np.repeat(FRACTIONS, N_DRAWS), draw_ids=np.tile(np.arange(N_DRAWS), len(FRACTIONS)), methods=np.array(ARMS))
    _, ablation_record = ablation_decoder()
    write_record(HERE / 'findings.tsv', dict(full_data=full_metrics, partner_disabled=ablated_metrics,
        support_resolution=cross, read_sampling=aggregates, original_parent_count=len(all_parents),
        admitted_parent_count=len(admitted_parent_set), admitted_units=len(units['parents']),
        scalar_crosses_strand_count_per_image=sum(row['scalar_crosses_strand'] for row in ledgers
            if row['method'] == 'control' and row['image'] == 'A'),
        all_full_data_streams_equal=True, all_full_data_checksum_outcomes_preserved=True))


if __name__ == '__main__':
    main()
