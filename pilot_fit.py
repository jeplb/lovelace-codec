"""pure pilot identity grouped selection of position shrinkage."""

import math

import numpy as np
from tsv_records import read_output, write_output

from cell_model import (BASES, LENGTHS, cell_log_scores, discriminability, global_counts,
                         global_probabilities, position_counts, position_model,
                         support_flags, validate_counts, validate_states)


def grouped_folds(identities, parents):
    identities = np.asarray(identities)
    parents = np.asarray(parents)
    if identities.ndim != 1 or not np.issubdtype(identities.dtype, np.integer) or parents.shape != identities.shape:
        raise ValueError("identity and parent arrays must describe the same units")
    if ((identities < 448) | (identities >= 472)).any():
        raise ValueError("fitting accepts only fit pilot identities 448 through 471")
    adjacency = {int(identity): set() for identity in identities}
    for parent in set(parents.tolist()):
        shared = set(identities[parents == parent].tolist())
        for identity in shared:
            adjacency[identity].update(shared)
    remaining = set(adjacency)
    components = []
    while remaining:
        frontier = {min(remaining)}
        component = set()
        while frontier:
            component.update(frontier)
            frontier = set().union(*(adjacency[identity] for identity in frontier)) - component
        remaining -= component
        components.append(sorted(component))
    if len(components) < 2:
        raise ValueError("grouped cross validation requires two independent identity components")
    folds = []
    for fold, heldout in enumerate(components):
        test = np.isin(identities, heldout)
        train_indices, test_indices = np.flatnonzero(~test), np.flatnonzero(test)
        if set(parents[train_indices].tolist()) & set(parents[test_indices].tolist()):
            raise ValueError("a raw parent crosses the validation boundary")
        folds.append(dict(fold=fold, heldout_identities=heldout,
                          train_indices=train_indices, test_indices=test_indices))
    return folds


def _candidate_priority(value):
    return (1, 0.0) if value == "infinity" else (0, float(value))


def select_lambda(counts, states, identities, parents, config):
    """select from held out read cell log likelihoods without data truth."""
    counts = validate_counts(counts)
    states = validate_states(states, len(counts))
    identities, parents = np.asarray(identities), np.asarray(parents)
    if identities.shape != (len(counts),) or parents.shape != (len(counts),):
        raise ValueError("pilot metadata must match the observations")
    config = config.get("position_model", config)
    candidates = list(config["lambda_grid"])
    if not candidates or len(set(candidates)) != len(candidates) or "infinity" not in candidates:
        raise ValueError("lambda grid must be unique and include exact infinity")
    for value in candidates:
        if value != "infinity" and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError("finite lambda candidates must be positive and finite")
    support = config["support"]
    folds = grouped_folds(identities, parents)
    totals = {value: 0.0 for value in candidates}
    heldout_rows, fold_models = [], []
    inventory_count = 0
    for fold in folds:
        train, test = fold["train_indices"], fold["test_indices"]
        global_histogram = global_counts(counts[train], states[train])
        global_w = global_probabilities(global_histogram, config.get("global_pseudocount_per_bin", 0.5))
        histogram, support_rows = position_counts(counts[train], states[train], identities[train], parents[train])
        observed_cells = len(test) * 8
        inventory_count += observed_cells
        for value in candidates:
            model = position_model(histogram, global_w, value, support_rows, support)
            scores = cell_log_scores(counts[test], model)
            selected_scores = np.take_along_axis(scores, states[test, :, None], axis=-1)[..., 0]
            total = float(selected_scores.sum(dtype=np.float64))
            totals[value] += total
            heldout_rows.append(dict(
                fold=fold["fold"], heldout_identities=fold["heldout_identities"], lambda_value=value,
                heldout_units=len(test), heldout_read_cells=observed_cells,
                total_log_likelihood=total, mean_log_likelihood_per_read_cell=total / observed_cells))
        fold_models.append(dict(**fold, global_counts=global_histogram,
                                position_counts=histogram, support_rows=support_rows,
                                global_probabilities=global_w))
    means = {value: totals[value] / inventory_count for value in candidates}
    selected = max(candidates, key=lambda value: (means[value], _candidate_priority(value)))
    for fold in fold_models:
        model = position_model(fold["position_counts"], fold["global_probabilities"], selected,
                               fold["support_rows"], support)
        fold["selected_lambda"] = selected
        fold["selected_probabilities"] = model
        test = fold["test_indices"]
        fold["test_log_scores"] = cell_log_scores(counts[test], model)
        fold["test_identities"] = identities[test]
        fold["test_parents"] = parents[test]
        fold["test_states"] = states[test]
        fold["fold_id"] = fold["fold"]
        fold["unit_log_scores"] = fold["test_log_scores"]
        fold["identities"] = identities[test]
        fold["parents"] = parents[test]
        fold["inventory_identities"] = fold["heldout_identities"]
        state_inventory = {}
        for identity in fold["heldout_identities"]:
            designated = states[identities == identity]
            if not np.all(designated == designated[0]):
                raise ValueError("pilot identity has inconsistent designed states")
            state_inventory[int(identity)] = designated[0].copy()
        fold["states"] = state_inventory
    full_global_counts = global_counts(counts, states)
    full_global_w = global_probabilities(full_global_counts, config.get("global_pseudocount_per_bin", 0.5))
    histogram, support_rows = position_counts(counts, states, identities, parents)
    model = position_model(histogram, full_global_w, selected, support_rows, support)
    flags = support_flags(support_rows, support)
    for row in support_rows:
        row["supported"] = bool(flags[row["position"], BASES.index(row["base"]), LENGTHS.index(row["designed_length"])])
        row["fallback_to_global"] = not row["supported"] or selected == "infinity"
    global_information = discriminability(full_global_w)
    discriminability_rows = []
    for position in range(8):
        information = discriminability(model[position])
        discriminability_rows.append(dict(position=position,
                                          global_mutual_information_nats=global_information,
                                          position_mutual_information_nats=information,
                                          change_nats=information - global_information))
    return dict(
        selected_lambda=selected,
        candidate_scores=[dict(lambda_value=value, heldout_read_cells=inventory_count,
                               total_log_likelihood=totals[value],
                               mean_log_likelihood_per_read_cell=means[value], selected=value == selected)
                          for value in candidates],
        heldout_rows=heldout_rows, fold_models=fold_models,
        full_global_counts=full_global_counts, full_global_probabilities=full_global_w,
        full_counts=histogram, full_support_rows=support_rows, selected_model=model,
        discriminability_rows=discriminability_rows)


def main():
    import argparse
    import json
    from pathlib import Path

    from cell_model import global_from_document
    from decoder import write_tsv
    from suffix_weighting import select_weighting

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='reproduce pilot model and weighting selection')
    parser.add_argument('--training', type=Path, default=root / 'data/prepared/training_inputs.npz')
    parser.add_argument('--control', type=Path, default=root / 'data/prepared/clean_channel.tsv')
    parser.add_argument('--config', type=Path, default=root / 'codebook/model_config.tsv')
    parser.add_argument('--output', type=Path, default=root / 'analysis/results/pilot_fit')
    args = parser.parse_args()
    config = read_output(args.config)
    with np.load(args.training, allow_pickle=False) as saved:
        counts = saved['counts'].copy()
        states = saved['states'].copy()
        identities = saved['identities'].copy()
        parents = saved['parents'].copy()
    position = select_lambda(counts, states, identities, parents, config)
    control = global_from_document(read_output(args.control))
    np.testing.assert_array_equal(position['full_global_probabilities'], control)
    weighting = select_weighting(position['fold_models'], config)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'model.npz',
                        global_counts=position['full_global_counts'],
                        global_probabilities=control,
                        position_counts=position['full_counts'],
                        selected_probabilities=position['selected_model'],
                        support_flags=support_flags(position['full_support_rows'], config['position_model']['support']))
    selection = dict(selected_lambda=position['selected_lambda'],
                     selected_candidate=weighting['selected_candidate'],
                     selected_name=weighting['winner'],
                     position_candidate_scores=position['candidate_scores'],
                     weighting_candidate_metrics=weighting['candidate_metrics'])
    write_output(args.output / 'selection.tsv', selection)
    for name, value in (('position_cv.tsv', position['heldout_rows']),
                        ('position_cv_summary.tsv', position['candidate_scores']),
                        ('weighting_cv.tsv', weighting['fold_metrics']),
                        ('weighting_cv_summary.tsv', weighting['candidate_metrics']),
                        ('support.tsv', position['full_support_rows']),
                        ('discriminability.tsv', position['discriminability_rows'])):
        write_tsv(args.output / name, value)
    print(json.dumps(dict(selected_lambda=position['selected_lambda'],
                          selected_weighting=weighting['winner']), sort_keys=True))


if __name__ == '__main__':
    main()
