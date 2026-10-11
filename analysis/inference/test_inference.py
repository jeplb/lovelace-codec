"""verify parent pairing, packed-cell evidence and inference isolation."""

from collections import Counter
import gzip
import hashlib
import json

import numpy as np

from .analyze import ARMS, CURRENT, FRACTIONS, FROZEN, HERE, N_DRAWS, SEED, ablation_decoder, read_tsv
from format_constraints import scalar_rows
from suffix_weighting import build_references
from tsv_records import iter_table_rows


def test_ledger_uses_actual_packing_and_distinct_parent_support():
    layout = {(row['image'], row['scalar_index']): row for row in scalar_rows()}
    ledger = read_tsv(HERE / 'scalar_evidence_ledger.tsv')
    assert len(ledger) == len(layout) * len(ARMS)
    for row in ledger:
        cells = layout[row['image'], int(row['scalar_index'])]
        observed = []
        for digit in range(3):
            assert int(row[f'cell_{digit}_address']) == cells[f'address_{digit}']
            assert int(row[f'cell_{digit}_position']) == cells[f'cell_{digit}']
            parents = json.loads(row[f'cell_{digit}_parent_ids'])
            assert len(set(parents)) == len(parents) == int(row[f'cell_{digit}_contributing_parents'])
            observed.append(bool(parents))
        expected_support = 'complete' if all(observed) else 'partial' if any(observed) else 'absent'
        assert row['direct_support'] == expected_support
        assert int(row['direct_supported_cells']) == sum(observed)
        encoded, emitted = int(row['encoded_six_bit_label']), int(row['emitted_six_bit_label'])
        assert float(row['normalized_component_error']) == (
            float(row['estimated_normalized_component']) - float(row['encoded_normalized_component']))
        assert float(row['label_spacing_component_error']) == 2 * (emitted - encoded) / 63
        assert float(row['dequantized_component_error_designed_scale']) == (
            float(row['estimated_dequantized_component_designed_scale']) - float(row['encoded_dequantized_component']))
    cross = read_tsv(HERE / 'support_resolution.tsv')
    for arm in ARMS:
        for image in 'AB':
            assert sum(int(row['scalar_count']) for row in cross if row['method'] == arm and row['image'] == image) == 512


def test_conjugate_dependency_edges_trace_original_observations():
    layout = {(row['image'], row['scalar_index']): row for row in scalar_rows()}
    for row in read_tsv(HERE / 'conjugate_dependencies.tsv'):
        source = layout[row['image'], int(row['source_partner_scalar'])]
        digit = int(row['digit_position'])
        assert int(row['source_address']) == source[f'address_{digit}']
        assert int(row['source_cell']) == source[f'cell_{digit}']
        assert row['independent_new_observations'] == 'False'
        assert len(json.loads(row['source_parent_ids'])) == int(row['contributing_parent_count'])


def test_sampling_precedes_eligibility_and_preserves_parent_pairing():
    universe = {row['raw_parent_id'] for row in read_tsv(FROZEN / 'read_audit.tsv')}
    mapping = read_tsv(HERE / 'parent_observation_mapping.tsv')
    admitted_parents = [row['raw_parent_id'] for row in mapping]
    summary = {(float(row['fraction']), int(row['draw'])): row for row in read_tsv(HERE / 'sample_summary.tsv')}
    frequencies = Counter()
    sets_by_fraction = {fraction: set() for fraction in FRACTIONS}
    rng = np.random.default_rng(SEED)
    sorted_universe = sorted(universe)
    with gzip.open(HERE / 'sampled_parent_ids.tsv.gz', 'rt', newline='') as handle:
        for sample in iter_table_rows(handle):
            fraction, draw = sample['fraction'], sample['draw']
            frequencies[fraction] += 1
            parents = sample['raw_parent_ids']
            chosen = set(parents)
            assert chosen.issubset(universe)
            assert len(chosen) == len(parents) == int(np.floor(len(universe) * fraction))
            assert sample['seed'] == SEED
            if fraction < 1:
                expected_indices = np.sort(rng.choice(len(universe), len(parents), replace=False))
                assert parents == [sorted_universe[index] for index in expected_indices]
            sets_by_fraction[fraction].add(sample['parent_set_sha256'])
            assert hashlib.sha256(('\n'.join(parents) + '\n').encode()).hexdigest() == sample['parent_set_sha256']
            row = summary[fraction, draw]
            retained = [parent for parent in admitted_parents if parent in chosen]
            assert len(retained) == int(row['admitted_unit_count'])
            assert len(set(retained)) == int(row['admitted_parent_count'])
            assert row['parent_set_sha256'] == sample['parent_set_sha256']
            assert row['all_parent_units_retained'] == row['paired_across_methods'] == 'True'
    assert frequencies == {fraction: N_DRAWS for fraction in FRACTIONS}
    assert all(len(digests) == N_DRAWS if fraction < 1 else len(digests) == 1
               for fraction, digests in sets_by_fraction.items())
    draws = read_tsv(HERE / 'read_sampling_draws.tsv')
    for key, row in summary.items():
        selected = [item for item in draws if (float(item['fraction']), int(item['draw'])) == key]
        assert len(selected) == len(ARMS) * 2
        assert {item['parent_set_sha256'] for item in selected} == {row['parent_set_sha256']}


def test_every_full_data_sample_reproduces_current_streams_and_counts():
    saved = np.load(HERE / 'sampling_predictions.npz', allow_pickle=False)
    subset = saved['fractions'] == 1
    assert int(subset.sum()) == N_DRAWS
    metrics = read_tsv(HERE / 'partner_ablation.tsv')
    for arm_index, arm in enumerate(ARMS):
        for image_index, image in enumerate('AB'):
            expected = np.frombuffer((CURRENT / f'predictions/{arm}_{image}.bin').read_bytes(), dtype=np.uint8)
            np.testing.assert_array_equal(saved['streams'][subset, arm_index, image_index], np.tile(expected, (N_DRAWS, 1)))
            row = next(item for item in metrics if item['method'] == arm and item['image'] == image and item['partner_information'] == 'True')
            assert np.all(saved['resolved'][subset, arm_index, image_index].sum(axis=-1) == int(row['resolved_labels']))


def test_partner_ablation_preserves_headers_dc_and_aggregation():
    _, record = ablation_decoder()
    assert record['dc_constraint_retained'] and record['scale_gate_retained'] and record['header_rules_retained']
    for arm in ARMS:
        for image in 'AB':
            original = (CURRENT / f'predictions/{arm}_{image}.bin').read_bytes()
            ablated = (HERE / f'no_partner_{arm}_{image}.bin').read_bytes()
            assert original[:64] == ablated[:64]
            original_labels = np.unpackbits(np.frombuffer(original[64:], dtype=np.uint8)).reshape(512, 6) @ (2 ** np.arange(5, -1, -1))
            ablated_labels = np.unpackbits(np.frombuffer(ablated[64:], dtype=np.uint8)).reshape(512, 6) @ (2 ** np.arange(5, -1, -1))
            np.testing.assert_array_equal(original_labels[272:274], ablated_labels[272:274])
    metrics = read_tsv(HERE / 'partner_ablation.tsv')
    assert all(int(row['conjugate_enabled_scalars']) == 0 for row in metrics if row['partner_information'] == 'False')


def test_weighting_reference_excludes_all_views_of_scored_parent():
    scores = np.arange(4 * 8 * 4).reshape(4, 8, 4).astype(float)
    identities = np.array([4, 4, 4, 5])
    parents = np.array(['same', 'same', 'other', 'same'])
    reference, count = build_references(scores, identities, parents)
    expected = np.exp(scores[2] - scores[2].max(axis=-1, keepdims=True))
    expected /= expected.sum(axis=-1, keepdims=True)
    np.testing.assert_array_equal(reference[0], expected)
    np.testing.assert_array_equal(reference[1], expected)
    assert np.all(count[:2] == 1)
    assert np.all(count[3] == 0)
    np.testing.assert_array_equal(reference[3], np.full((8, 4), .25))
