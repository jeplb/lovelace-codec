"""check saved evidence accounting and extraction comparisons."""

import csv
import gzip
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tsv_records import read_output
import sys

import numpy as np

from . import analyze


HERE = Path(__file__).resolve().parent


def read(name):
    opener = gzip.open if name.endswith('.gz') else open
    with opener(HERE / name, 'rt') as handle:
        return list(csv.DictReader(handle))


def test_ties_and_out_of_alphabet_remain_uncalled():
    assert analyze.maximum_calls([[0., 0., -1., -1.]]).tolist() == [-1]
    model = np.full((3, 3, 20), .05)
    counts = np.full((1, 8, 3), 3)
    counts[0, 0, 0] = 21
    original = counts.copy()
    assert analyze.likelihood_calls(counts, model).tolist() == [[-1] * 8]
    np.testing.assert_array_equal(counts, original)


def test_frozen_resolution_rule_matches_engine():
    from scalar_inference import normalize, resolved
    with np.load(analyze.CURRENT / 'predictions/weight_inputs.npz', allow_pickle=False) as saved:
        scores = saved['global_unit_log_scores'][:24]
        counts = saved['counts'][:24]
    with np.load(analyze.CURRENT / 'fitting/model.npz', allow_pickle=False) as saved:
        probabilities = saved['global_probabilities']
    config = {'inference': {'resolved_probability': .95, 'tie_tolerance': 0.}}
    expected = []
    for score in scores.reshape(-1, 4):
        posterior = normalize(score)
        selected = int(score.argmax())
        expected.append(selected if resolved(posterior, selected, config) else -1)
    actual = analyze.likelihood_calls(counts, probabilities, .95)
    np.testing.assert_array_equal(actual, np.array(expected).reshape(-1, 8))


def test_every_native_cell_and_total_is_retained():
    ledger = read('cell_observations.csv.gz')
    data = [row for row in ledger if row['role'] == 'data']
    findings = read_output(HERE / 'numerical_findings.tsv')
    assert len(data) == findings['native_data_cells']
    assert len({(row['unit_id'], row['cell_position']) for row in ledger}) == len(ledger)
    assert all(int(row['observed_total']) == sum(int(row['observed_' + base]) for base in 'ACT') for row in ledger)
    assert sum(int(row['observed_total']) != 20 for row in data) == findings['native_data_cells_total_not_twenty']
    assert {row['orientation'] for row in data} == {'forward', 'reverse_complement'}
    assert len({row['parent_read_id'] for row in data}) == findings['native_data_parent_reads']


def test_agreement_categories_and_designed_state_counts_partition():
    rows = read('cell_agreement_source_data.csv')
    for row in rows:
        assert sum(int(row[key]) for key in ('correct', 'incorrect', 'uncalled')) == int(row['observations'])
        assert abs(sum(float(row[key + '_fraction']) for key in ('correct', 'incorrect', 'uncalled')) - 1) < 1e-12
        assert int(row['parent_reads']) <= int(row['observations'])
    lookup = {(r['population'], r['method'], r['orientation'], r['cell_position'], r['designed_state']): r for r in rows}
    for row in rows:
        if row['designed_state'] != 'all':
            continue
        keys = (row['population'], row['method'], row['orientation'], row['cell_position'])
        assert sum(int(lookup[(*keys, str(state))]['observations']) for state in range(4)) == int(row['observations'])


def test_matching_uses_read_design_orientation_and_retains_all_pairs():
    rows = read('matched_cell_observations.csv.gz')
    populations = read('extraction_population_ledger.csv')
    matched = {r['unit_id']: r for r in populations if r['matched'] == 'True' and r['role'] == 'data'}
    assert len(rows) == len(matched) * 8
    assert {row['unit_id'] for row in rows} == set(matched)
    for row in rows:
        origin = matched[row['unit_id']]
        assert row['parent_read_id'] == origin['parent_read_id']
        assert row['identity'] == origin['independent_identity']
        assert row['orientation'] == origin['independent_orientation']
    assert all(row['native_' + base] == row['independent_' + base] for row in rows for base in 'ACT')


def test_independent_outside_alphabet_not_dropped():
    rows = read('independent_cell_observations.csv.gz')
    outside = [r for r in rows if r['in_fixed_alphabet'] == 'False']
    findings = read_output(HERE / 'numerical_findings.tsv')
    assert len(outside) == findings['independent_data_cells_outside_fixed_alphabet']
    assert all(row['control_call'] == row['position_call'] == '-1' for row in outside)


def test_expected_assisted_diagnostics_are_separate_and_reproduced():
    rows = read('registration_diagnostic_source_data.csv')
    assert all((row['expected_assisted'] == 'True') == row['population'].startswith('expected_assisted') for row in rows)
    full = [r for r in rows if r['population'] == 'expected_assisted_full_posterior' and r['orientation'] == 'all']
    held = [r for r in rows if r['population'] == 'expected_assisted_held_out_cell' and r['orientation'] == 'all']
    assert len({row['observations'] for row in full}) == 1
    assert len({row['observations'] for row in held}) > 1
