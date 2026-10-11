"""summarize saved cell observations without changing any decoder output."""

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sys

import numpy as np
import yaml
from tsv_records import read_table as read_tsv_records_table, write_output


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSES = ROOT / 'data/analysis_inputs'
FROZEN = ANALYSES / 'frozen_inputs'
REGISTRATION = ANALYSES / '11_registration_check'
CURRENT = ANALYSES / '13_observation_model'
STATES = np.array(((3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)))
ORIENTATIONS = ('all', 'forward', 'reverse_complement')
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(path, value):
    write_output(path, value)


def write_table(name, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(HERE / name, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_table(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def read_records_table(path):
    return read_tsv_records_table(path)


def load_design():
    sequences = {}
    for line in (FROZEN / 'Lovelace_S12_480.fasta').read_text().splitlines():
        if line.startswith('>'):
            identity = int(line.rsplit('_', 1)[1])
            sequences[identity] = ''
        else:
            sequences[identity] += line.strip()
    design = []
    for identity in range(480):
        sequence = sequences[identity]
        assert len(sequence) == 298
        assert sequence[105:109] == sequence[269:273] == 'GGGG'
        runs = [(base, len(list(group))) for base, group in itertools.groupby(sequence[109:269])]
        assert ''.join(base for base, length in runs) == 'ACT' * 8
        triplets = np.array([length for base, length in runs]).reshape(8, 3)
        matched = (triplets[:, None, :] == STATES).all(-1)
        assert (matched.sum(-1) == 1).all()
        design.append(matched.argmax(-1))
    design = np.array(design)
    with np.load(FROZEN / 'design_reference.npz', allow_pickle=False) as reference:
        np.testing.assert_array_equal(design, reference['expected_cell_states'])
        assert reference['quantized_codes'].shape == (2, 16, 16, 2)
    return design


def maximum_calls(scores, valid=None):
    """do not convert ties into observed symbol calls."""
    scores = np.asarray(scores)
    calls = scores.argmax(-1)
    unique = (scores == scores.max(-1, keepdims=True)).sum(-1) == 1
    if valid is not None:
        unique &= valid
    return np.where(unique, calls, -1)


def likelihood_calls(counts, probabilities, resolution_threshold=None):
    """retain out-of-alphabet cells as uncalled without clipping their runs."""
    counts = np.asarray(counts)
    valid = ((counts >= 1) & (counts <= 20)).all(-1)
    from cell_model import cell_log_scores
    # invalid diagnostic cells stay uncalled after placeholder scoring
    scores = cell_log_scores(np.clip(counts, 1, 20), probabilities)
    calls = maximum_calls(scores, valid)
    if resolution_threshold is not None:
        posterior = np.exp(scores - scores.max(-1, keepdims=True))
        posterior /= posterior.sum(-1, keepdims=True)
        # apply the frozen resolution probability and positive posterior margin.
        calls = np.where(posterior.max(-1) >= resolution_threshold, calls, -1)
    return calls


def nearest_calls(counts):
    distance = np.abs(np.asarray(counts)[:, :, None, :] - STATES).sum(-1)
    return maximum_calls(-distance)


def population(name, records, counts, design, global_model, position_model, resolution_threshold=.95):
    identities = np.array([row['identity'] for row in records])
    return dict(name=name, records=records, counts=np.asarray(counts),
                states=design[identities], identities=identities,
                parents=np.array([row['raw_parent_id'] for row in records]),
                orientations=np.array([row['orientation'] for row in records]),
                calls=dict(control=likelihood_calls(counts, global_model),
                           position=likelihood_calls(counts, position_model),
                           control_resolved_095=likelihood_calls(counts, global_model, resolution_threshold),
                           position_resolved_095=likelihood_calls(counts, position_model, resolution_threshold),
                           nearest_absolute_distance=nearest_calls(counts)))


def subset_population(pop, name, take):
    return dict(name=name, records=[row for row, selected in zip(pop['records'], take) if selected],
                **{key: pop[key][take] for key in ('counts', 'states', 'identities', 'parents', 'orientations')},
                calls={key: calls[take] for key, calls in pop['calls'].items()})


def support(pop, take):
    return dict(observations=int(take.sum()),
                parent_reads=len(set(pop['parents'][take])),
                designs=len(set(pop['identities'][take])))


def summarize_calls(pop):
    rows = []
    for orientation in ORIENTATIONS:
        orientation_take = np.ones(len(pop['records']), bool) if orientation == 'all' else pop['orientations'] == orientation
        for method, calls in pop['calls'].items():
            for position in range(8):
                for state in ('all', 0, 1, 2, 3):
                    take = orientation_take.copy()
                    if state != 'all':
                        take &= pop['states'][:, position] == state
                    count = int(take.sum())
                    correct = int(((calls[:, position] == pop['states'][:, position]) & take).sum())
                    uncalled = int(((calls[:, position] == -1) & take).sum())
                    incorrect = count - correct - uncalled
                    rows.append(dict(population=pop['name'], method=method, orientation=orientation,
                                     cell_position=position + 1, designed_state=state,
                                     **support(pop, take), correct=correct, incorrect=incorrect, uncalled=uncalled,
                                     correct_fraction=correct / count if count else 0,
                                     incorrect_fraction=incorrect / count if count else 0,
                                     uncalled_fraction=uncalled / count if count else 0))
    return rows


def run_distributions(pop):
    rows, histograms, totals = [], [], []
    for orientation in ORIENTATIONS:
        orientation_take = np.ones(len(pop['records']), bool) if orientation == 'all' else pop['orientations'] == orientation
        for position in range(8):
            cell_totals = pop['counts'][:, position].sum(-1)
            for total in np.unique(cell_totals[orientation_take]):
                take = orientation_take & (cell_totals == total)
                totals.append(dict(population=pop['name'], orientation=orientation,
                                   cell_position=position + 1, observed_total=int(total),
                                   designed_total=20, **support(pop, take)))
            for base, letter in enumerate('ACT'):
                desired = STATES[pop['states'][:, position], base]
                for length in sorted(set(STATES[:, base])):
                    take = orientation_take & (desired == length)
                    observed = pop['counts'][take, position, base]
                    if not len(observed):
                        continue
                    q05, q25, median, q75, q95 = np.quantile(observed, [.05, .25, .5, .75, .95])
                    rows.append(dict(population=pop['name'], orientation=orientation,
                                     cell_position=position + 1, base=letter, designed_length=int(length),
                                     **support(pop, take), mean=float(observed.mean()),
                                     standard_deviation=float(observed.std(ddof=1)), q05=q05, q25=q25,
                                     median=median, q75=q75, q95=q95,
                                     exact_fraction=float((observed == length).mean()),
                                     total_not_twenty=int((cell_totals[take] != 20).sum())))
                    values, frequencies = np.unique(observed, return_counts=True)
                    for value, frequency in zip(values, frequencies):
                        histograms.append(dict(population=pop['name'], orientation=orientation,
                                               cell_position=position + 1, base=letter, designed_length=int(length),
                                               observed_length=int(value), count=int(frequency)))
    return rows, histograms, totals


def extraction_ledger(native_records, parser_records):
    parser_by_record = {row['record_index']: row for row in parser_records}
    pairs, native_indices, independent_records = [], [], []
    for index, row in enumerate(native_records):
        record_index = int(row['unit_id'].split(':', 1)[0].rsplit('_', 1)[1])
        independent = parser_by_record[record_index]
        matched = (independent.get('complete_ACT8', False)
                   and independent.get('identity') == row['identity']
                   and independent.get('orientation') == row['orientation'])
        equal_bounds = (independent.get('payload_start') == row['oriented_payload_start']
                        and independent.get('payload_end') == row['oriented_payload_end'])
        equal_runs = independent.get('payload_run_lengths') == list(np.array(row['observed_direct']).reshape(-1))
        pairs.append(dict(unit_id=row['unit_id'], parent_read_id=row['raw_parent_id'],
                          record_index=record_index, identity=row['identity'], role=row['role'],
                          orientation=row['orientation'], independent_status=independent['reason'],
                          independent_identity=independent.get('identity', ''),
                          independent_orientation=independent.get('orientation', ''),
                          independent_complete_ACT8=independent['complete_ACT8'],
                          matched=matched, equal_bounds=equal_bounds, equal_runs=equal_runs))
        if matched:
            native_indices.append(index)
            independent_records.append(independent)
    return pairs, np.array(native_indices), independent_records


def paired_summary(native, independent):
    rows, details = [], []
    assert len(native['records']) == len(independent['records'])
    for i, (left, right) in enumerate(zip(native['records'], independent['records'])):
        assert left['raw_parent_id'] == right['raw_parent_id']
        assert left['identity'] == right['identity'] and left['orientation'] == right['orientation']
        for pos in range(8):
            details.append(dict(unit_id=left['unit_id'], parent_read_id=left['raw_parent_id'],
                                record_index=right['record_index'], identity=left['identity'],
                                orientation=left['orientation'], role=left['role'], cell_position=pos + 1,
                                designed_state=int(native['states'][i, pos]),
                                native_A=int(native['counts'][i, pos, 0]), native_C=int(native['counts'][i, pos, 1]),
                                native_T=int(native['counts'][i, pos, 2]),
                                independent_A=int(independent['counts'][i, pos, 0]), independent_C=int(independent['counts'][i, pos, 1]),
                                independent_T=int(independent['counts'][i, pos, 2]),
                                native_control_call=int(native['calls']['control'][i, pos]),
                                independent_control_call=int(independent['calls']['control'][i, pos])))
    for orientation in ORIENTATIONS:
        orient_take = np.ones(len(native['records']), bool) if orientation == 'all' else native['orientations'] == orientation
        for pos in range(8):
            for state in ('all', 0, 1, 2, 3):
                take = orient_take.copy()
                if state != 'all':
                    take &= native['states'][:, pos] == state
                truth = native['states'][take, pos]
                a, b = [p['calls']['control'][take, pos] for p in (native, independent)]
                rows.append(dict(orientation=orientation, cell_position=pos + 1, designed_state=state,
                                 **support(native, take),
                                 identical_triplets=int((native['counts'][take, pos] == independent['counts'][take, pos]).all(-1).sum()),
                                 identical_calls=int((a == b).sum()),
                                 native_correct=int((a == truth).sum()), independent_correct=int((b == truth).sum()),
                                 native_uncalled=int((a == -1).sum()), independent_uncalled=int((b == -1).sum()),
                                 independent_only_correct=int(((b == truth) & (a != truth)).sum()),
                                 native_only_correct=int(((a == truth) & (b != truth)).sum())))
    return rows, details


def diagnostics(native, masks):
    rows = []
    for name, (mask, assisted) in masks.items():
        for orientation in ORIENTATIONS:
            orientation_take = np.ones(len(native['records']), bool) if orientation == 'all' else native['orientations'] == orientation
            for pos in range(8):
                take = mask[:, pos] & orientation_take
                truth = STATES[native['states'][take, pos]]
                observed = native['counts'][take, pos]
                rows.append(dict(population=name, expected_assisted=assisted,
                                 orientation=orientation, cell_position=pos + 1,
                                 **support(native, take),
                                 run_observations=int(len(observed) * 3),
                                 run_error_count=int((observed != truth).sum()),
                                 run_exact_error=float((observed != truth).mean()),
                                 cell_exact_error=float((observed != truth).any(-1).mean()),
                                 control_correct_fraction=float((native['calls']['control'][take, pos] == native['states'][take, pos]).mean())))
    return rows


def main():
    design = load_design()
    decoder_config = yaml.safe_load((CURRENT / 'config.yaml').read_text())
    resolution_threshold = decoder_config['inference']['resolved_probability']
    assert resolution_threshold == .95 and decoder_config['inference']['tie_tolerance'] == 0.
    frozen_records = read_records_table(FROZEN / 'F_complete_new_units.tsv')
    frozen_counts = np.array([row['observed_direct'] for row in frozen_records])
    with np.load(CURRENT / 'fitting/model.npz', allow_pickle=False) as model:
        global_model = model['global_probabilities']
        position_model = model['selected_probabilities']
    native_all = population('native_all_roles', frozen_records, frozen_counts, design, global_model, position_model, resolution_threshold)
    native_data_take = np.array([row['role'] == 'data' for row in frozen_records])
    native = subset_population(native_all, 'native_data', native_data_take)
    with np.load(CURRENT / 'predictions/weight_inputs.npz', allow_pickle=False) as saved:
        index = {row['unit_id']: i for i, row in enumerate(frozen_records)}
        ordered = np.array([index[value] for value in saved['unit_ids']])
        np.testing.assert_array_equal(frozen_counts[ordered], saved['counts'])
        np.testing.assert_array_equal(native_all['calls']['control'][ordered], maximum_calls(saved['global_unit_log_scores']))
        np.testing.assert_array_equal(native_all['calls']['position'][ordered], maximum_calls(saved['unit_log_scores']))
    with gzip.open(REGISTRATION / 'strict_cell_calls.tsv.gz', 'rt') as handle:
        audit = list(csv.DictReader(handle, delimiter='\t'))
    nearest_saved = np.array([int(row['called_state']) for row in audit]).reshape(-1, 8)
    np.testing.assert_array_equal(native_all['calls']['nearest_absolute_distance'], nearest_saved)
    parsers = read_records_table(REGISTRATION / 'parser_records.tsv.gz')
    parser_inventory = [dict(record_index=row['record_index'], parent_read_id=row['raw_parent_id'],
                             read_id=row['read_id'], extraction_status=row['reason'],
                             orientation=row.get('orientation', ''), identity=row.get('identity', ''),
                             orientation_supported=row['orientation_supported'], address_supported=row['address_supported'],
                             payload_resolved=row['payload_resolved'], complete_ACT8=row['complete_ACT8'],
                             pure_ACT=row.get('pure_ACT', False), cell_count=row.get('cell_count', ''),
                             ordinary_position_population=bool(row['complete_ACT8'] and row['identity'] < 448)) for row in parsers]
    extraction_rows, matched_indices, independent_records = extraction_ledger(frozen_records, parsers)
    matched_native_all = subset_population(native_all, 'matched_native_all_roles', np.isin(np.arange(len(frozen_records)), matched_indices))
    assert [row['unit_id'] for row in matched_native_all['records']] == [frozen_records[i]['unit_id'] for i in matched_indices]
    independent_counts = np.array([row['payload_run_lengths'] for row in independent_records]).reshape(-1, 8, 3)
    independent_matched_all = population('matched_independent_all_roles', independent_records, independent_counts, design, global_model, position_model)
    matched_data = matched_native_all['identities'] < 448
    matched_native = subset_population(matched_native_all, 'matched_native_data', matched_data)
    independent_matched = subset_population(independent_matched_all, 'matched_independent_data', matched_data)
    independent_data_records = [row for row in parsers if row['complete_ACT8'] and row['identity'] < 448]
    independent_data_counts = np.array([row['payload_run_lengths'] for row in independent_data_records]).reshape(-1, 8, 3)
    independent_data = population('independent_data_all_complete_ACT8', independent_data_records, independent_data_counts, design, global_model, position_model)
    agreement_lookup = {row['unit_id']: row for row in extraction_rows}
    agree = np.array([agreement_lookup[row['unit_id']]['matched'] and agreement_lookup[row['unit_id']]['equal_bounds'] and agreement_lookup[row['unit_id']]['equal_runs'] for row in frozen_records])
    agreement_pop = subset_population(native_all, 'native_data_independent_agreement', agree & native_data_take)
    populations = [native, matched_native, independent_matched, independent_data, agreement_pop]
    calls_rows, run_rows, histogram_rows, total_rows = [], [], [], []
    for pop in populations:
        calls_rows += summarize_calls(pop)
        runs, histograms, totals = run_distributions(pop)
        run_rows += runs
        histogram_rows += histograms
        total_rows += totals
    with np.load(REGISTRATION / 'chimera_arrays.npz', allow_pickle=False) as saved:
        assert list(saved['unit_id']) == [row['unit_id'] for row in frozen_records]
        posterior = saved['posterior']
        hidden_posterior = saved['leave_cell_out_posterior']
    masks = dict(native_data=(np.ones((len(native['records']), 8), bool), False),
                 independent_agreement=(np.repeat(agree[native_data_take, None], 8, axis=1), False),
                 expected_assisted_full_posterior=(np.repeat((posterior[native_data_take] < .5)[:, None], 8, axis=1), True),
                 expected_assisted_held_out_cell=(hidden_posterior[native_data_take] < .5, True))
    diagnostic_rows = diagnostics(native, masks)
    saved_gradients = read_table(REGISTRATION / 'gradient_before_after.tsv')
    mapping = dict(native_data='before', independent_agreement='independent_parser_agreement',
                   expected_assisted_full_posterior='posterior_below_0.5',
                   expected_assisted_held_out_cell='leave_cell_out_posterior_below_0.5')
    max_gradient_difference = 0
    for row in diagnostic_rows:
        ref = next(value for value in saved_gradients if value['subset'] == mapping[row['population']] and value['orientation'] == row['orientation'] and int(value['cell_position']) + 1 == row['cell_position'] and value['metric'] == 'run_exact_error')
        assert row['observations'] == int(ref['units'])
        difference = abs(row['run_exact_error'] - float(ref['value']))
        assert difference < 1e-14
        max_gradient_difference = max(max_gradient_difference, difference)
    paired_rows, paired_detail = paired_summary(matched_native, independent_matched)
    observation_ledger = []
    for i, row in enumerate(frozen_records):
        for pos in range(8):
            triplet = frozen_counts[i, pos]
            observation_ledger.append(dict(extraction='native', unit_id=row['unit_id'], parent_read_id=row['raw_parent_id'],
                                           identity=row['identity'], role=row['role'], orientation=row['orientation'],
                                           cell_position=pos + 1, extraction_status='direct_ACT8_in_fixed_alphabet',
                                           intended_state=int(native_all['states'][i, pos]),
                                           designed_A=int(STATES[native_all['states'][i, pos], 0]),
                                           designed_C=int(STATES[native_all['states'][i, pos], 1]),
                                           designed_T=int(STATES[native_all['states'][i, pos], 2]),
                                           observed_A=int(triplet[0]), observed_C=int(triplet[1]), observed_T=int(triplet[2]),
                                           observed_total=int(triplet.sum()), control_call=int(native_all['calls']['control'][i, pos]),
                                           position_call=int(native_all['calls']['position'][i, pos]),
                                           control_resolved_095_call=int(native_all['calls']['control_resolved_095'][i, pos]),
                                           position_resolved_095_call=int(native_all['calls']['position_resolved_095'][i, pos]),
                                           audit_nearest_call=int(native_all['calls']['nearest_absolute_distance'][i, pos]),
                                           expected_assisted_full_posterior=float(posterior[i]),
                                           expected_assisted_held_out_cell_posterior=float(hidden_posterior[i, pos])))
    independent_ledger = []
    for i, row in enumerate(independent_data_records):
        for pos in range(8):
            triplet = independent_data_counts[i, pos]
            independent_ledger.append(dict(extraction='independent', record_index=row['record_index'], parent_read_id=row['raw_parent_id'],
                                           identity=row['identity'], orientation=row['orientation'], cell_position=pos + 1,
                                           extraction_status=row['reason'], in_fixed_alphabet=bool(((triplet >= 1) & (triplet <= 20)).all()),
                                           intended_state=int(independent_data['states'][i, pos]),
                                           observed_A=int(triplet[0]), observed_C=int(triplet[1]), observed_T=int(triplet[2]),
                                           observed_total=int(triplet.sum()), control_call=int(independent_data['calls']['control'][i, pos]),
                                           position_call=int(independent_data['calls']['position'][i, pos]),
                                           control_resolved_095_call=int(independent_data['calls']['control_resolved_095'][i, pos]),
                                           position_resolved_095_call=int(independent_data['calls']['position_resolved_095'][i, pos])))
    standardized = []
    for pop in populations:
        for method in pop['calls']:
            for orientation in ORIENTATIONS:
                for pos in range(1, 9):
                    selected = [row for row in calls_rows if row['population'] == pop['name'] and row['method'] == method and row['orientation'] == orientation and row['cell_position'] == pos and row['designed_state'] != 'all']
                    assert len(selected) == 4 and all(row['observations'] for row in selected)
                    standardized.append(dict(population=pop['name'], method=method, orientation=orientation,
                                             cell_position=pos, state_prior='uniform_four_states',
                                             correct_fraction=float(np.mean([row['correct_fraction'] for row in selected])),
                                             incorrect_fraction=float(np.mean([row['incorrect_fraction'] for row in selected])),
                                             uncalled_fraction=float(np.mean([row['uncalled_fraction'] for row in selected]))))
    endpoints = []
    for pop in populations:
        for orientation in ORIENTATIONS:
            selected = [row for row in calls_rows if row['population'] == pop['name'] and row['method'] == 'control' and row['orientation'] == orientation and row['designed_state'] == 'all' and row['cell_position'] in (1, 8)]
            endpoints.append(dict(population=pop['name'], orientation=orientation, first=selected[0], last=selected[1]))
    findings = dict(cohort='NEW_ABC_ONLY', confidence='high for description and unknown for physical cause',
                    native_data_units=len(native['records']), native_data_parent_reads=len(set(native['parents'])),
                    native_data_designs=len(set(native['identities'])), matched_data_units=len(matched_native['records']),
                    identical_extraction_data_units=len(agreement_pop['records']),
                    independent_complete_ACT8_data_records=len(independent_data['records']),
                    native_data_cells=int(len(native['records']) * 8),
                    native_data_cells_total_not_twenty=int((native['counts'].sum(-1) != 20).sum()),
                    independent_data_cells_outside_fixed_alphabet=int((~((independent_data['counts'] >= 1) & (independent_data['counts'] <= 20)).all(-1)).sum()),
                    control_endpoints=endpoints, expected_assisted_diagnostic_endpoints=[row for row in diagnostic_rows if row['cell_position'] in (1, 8)],
                    uniform_state_endpoints=[row for row in standardized if row['cell_position'] in (1, 8)],
                    run_length_endpoints=[row for row in run_rows if row['population'] == 'native_data' and row['cell_position'] in (1, 8)],
                    fixed_resolution_endpoints=[row for row in calls_rows if row['method'] == 'control_resolved_095' and row['designed_state'] == 'all' and row['cell_position'] in (1, 8)],
                    maximum_saved_gradient_difference=max_gradient_difference,
                    matched_triplet_differences=sum(row['observations'] - row['identical_triplets'] for row in paired_rows if row['orientation'] == 'all' and row['designed_state'] == 'all'),
                    missing_comparisons=['no unique eight-position alignment exists for incomplete or interrupted independent payloads',
                                         'a new cell likelihood is not defined beyond observed lengths 1 through 20',
                                         'the held-out-cell expected-assisted masks select different populations at each position',
                                         'all same-record same-design same-orientation complete pairs have identical runs so discordant paired extraction effects cannot be estimated'],
                    causal_interpretation='position refers to the encoded payload and does not establish sequencing-entry distance or physical cause')
    write_table('cell_observations.csv.gz', observation_ledger, True)
    write_table('independent_cell_observations.csv.gz', independent_ledger, True)
    write_table('extraction_population_ledger.csv', extraction_rows)
    write_table('independent_extraction_population_ledger.csv', parser_inventory)
    write_table('run_length_source_data.csv', run_rows)
    write_table('run_length_histograms.csv', histogram_rows)
    write_table('cell_total_source_data.csv', total_rows)
    write_table('cell_agreement_source_data.csv', calls_rows)
    write_table('uniform_state_agreement.csv', standardized)
    write_table('matched_extraction_source_data.csv', paired_rows)
    write_table('matched_cell_observations.csv.gz', paired_detail, True)
    write_table('registration_diagnostic_source_data.csv', diagnostic_rows)
    write_record(HERE / 'numerical_findings.tsv', findings)
    print(json.dumps({key: findings[key] for key in ('native_data_units', 'matched_data_units', 'identical_extraction_data_units', 'native_data_cells_total_not_twenty', 'matched_triplet_differences')}))


if __name__ == '__main__':
    main()
