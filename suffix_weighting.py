"""parent-excluded references and leave-cell-out suffix weights."""

import numpy as np

from cell_model import aggregate, softmax


MODEL_NAMES = ('no_switch',) + tuple('boundary_' + str(i) for i in range(1, 8)) + ('outlier',)


def build_references(unit_log_scores, identities, parents):
    """exclude every view of the scored raw parent before normalization."""
    scores = np.asarray(unit_log_scores, dtype=np.float64)
    identities = np.asarray(identities)
    parents = np.asarray(parents)
    assert scores.shape == (len(identities), 8, 4)
    assert parents.shape == identities.shape
    assert np.isfinite(scores).all()
    groups = {}
    for index, (identity, parent) in enumerate(zip(identities, parents)):
        groups.setdefault(int(identity), {}).setdefault(parent, []).append(index)
    reference = np.empty_like(scores)
    parent_count = np.zeros(scores.shape[:2], dtype=np.int64)
    for parent_groups in groups.values():
        indices = list(parent_groups.values())
        sums = np.array([scores[rows].sum(axis=0) for rows in indices])
        prefix = np.zeros((len(indices) + 1, 8, 4), dtype=np.float64)
        suffix = np.zeros_like(prefix)
        for index in range(len(indices)):
            prefix[index + 1] = prefix[index] + sums[index]
        for index in range(len(indices) - 1, -1, -1):
            suffix[index] = sums[index] + suffix[index + 1]
        for index, rows in enumerate(indices):
            # omit this parent from both sums to avoid cancellation or self dependence
            reference[rows] = softmax(prefix[index] + suffix[index + 1])
            parent_count[rows] = len(indices) - 1
    return reference, parent_count


def _logsumexp(values, axis=-1):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)), axis=axis)


def _model_log_prior(candidate):
    suffix_prior = float(candidate['suffix_prior'])
    outlier_prior = float(candidate['outlier_prior'])
    prior = np.array([1.0 - suffix_prior - outlier_prior] + [suffix_prior / 7.0] * 7 + [outlier_prior])
    assert (prior >= 0).all() and prior.sum() > 0
    with np.errstate(divide='ignore'):
        return np.log(prior)


def _model_posteriors(model_emissions, assessable, candidate, hidden_cell=None):
    mask = assessable.copy()
    if hidden_cell is not None:
        mask[:, hidden_cell] = False
    # keep the hidden observation out of every sum, even via subtraction
    selected = np.where(mask[:, None, :], model_emissions, 0.0)
    model_scores = selected.sum(axis=-1) + _model_log_prior(candidate)
    probabilities = np.exp(model_scores - model_scores.max(axis=-1, keepdims=True))
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def score_weights(unit_log_scores, references, other_parent_count, candidate, config):
    """return one retained-evidence probability per read-cell."""
    scores = np.asarray(unit_log_scores, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    counts = np.asarray(other_parent_count)
    assert scores.ndim == 3 and scores.shape[1:] == (8, 4)
    assert references.shape == scores.shape and counts.shape == scores.shape[:2]
    assert np.isfinite(scores).all() and np.isfinite(references).all()
    assert (references >= 0).all() and np.allclose(references.sum(axis=-1), 1.0)
    settings = config['weighting']
    with np.errstate(divide='ignore'):
        log_agreement = _logsumexp(scores + np.log(references))
    log_alternative = _logsumexp(scores) - np.log(4.0)
    emissions = np.empty((len(scores), 9, 8), dtype=np.float64)
    emissions[:, 0] = log_agreement
    for boundary in range(1, 8):
        emissions[:, boundary, :boundary] = log_agreement[:, :boundary]
        emissions[:, boundary, boundary:] = log_alternative[:, boundary:]
    emissions[:, 8] = log_alternative
    assessable = counts >= int(settings['min_other_parents'])
    other_assessable = assessable.sum(axis=1, keepdims=True) - assessable
    fallback = (~assessable) | (other_assessable < int(settings['min_other_cells']))
    leave_cell = np.empty((len(scores), 8, 9), dtype=np.float64)
    weights = np.ones(scores.shape[:2], dtype=np.float64)
    for cell in range(8):
        posterior = _model_posteriors(emissions, assessable, candidate, hidden_cell=cell)
        leave_cell[:, cell] = posterior
        # boundary b keeps cell j exactly when j < b
        retained = posterior[:, 0] + posterior[:, cell + 1:8].sum(axis=1)
        # bound only floating-point summation roundoff at probability endpoints
        retained = np.clip(retained, 0.0, 1.0)
        weights[:, cell] = np.where(fallback[:, cell], 1.0, retained)
    if candidate['name'] == 'none':
        weights.fill(1.0)
    full = _model_posteriors(emissions, assessable, candidate)
    diagnostic = _model_posteriors(emissions, assessable, settings['diagnostic_candidate'])
    diagnostic_assessable = assessable.sum(axis=1) >= int(settings['min_other_cells'])
    suffix_inconsistent = diagnostic_assessable & (
        diagnostic[:, 1:8].sum(axis=1) > float(settings['suffix_diagnostic_threshold']))
    assert np.isfinite(weights).all() and ((weights >= 0) & (weights <= 1)).all()
    return dict(weights=weights, log_emissions_agreement=log_agreement,
                log_emissions_alternative=log_alternative,
                leave_cell_model_posteriors=leave_cell, full_model_posteriors=full,
                diagnostic_full_model_posteriors=diagnostic,
                assessable=assessable, other_assessable_count=other_assessable,
                fallback=fallback, diagnostic_assessable=diagnostic_assessable,
                suffix_inconsistent=suffix_inconsistent)


def _metrics(aggregate_scores, inventory, states, resolved_probability):
    scores = aggregate_scores[inventory]
    posterior = softmax(scores)
    truth = np.array([states[int(identity)] for identity in inventory], dtype=np.int64)
    assert truth.shape == scores.shape[:2] and ((truth >= 0) & (truth < 4)).all()
    expected = np.eye(4)[truth]
    brier = ((posterior - expected) ** 2).sum(axis=-1)
    log_posterior = scores - _logsumexp(scores)[:, :, None]
    log_score = np.take_along_axis(log_posterior, truth[:, :, None], axis=-1)[..., 0]
    labels = posterior.argmax(axis=-1)
    correct = labels == truth
    ordered = np.sort(posterior, axis=-1)
    resolved = (ordered[..., -1] >= resolved_probability) & (ordered[..., -1] > ordered[..., -2])
    return dict(n_cells=int(truth.size), brier_sum=float(brier.sum()),
                log_score_sum=float(log_score.sum()), mean_brier=float(brier.mean()),
                mean_log_score=float(log_score.mean()), map_correct=int(correct.sum()),
                correct=int((correct & resolved).sum()), wrong=int((~correct & resolved).sum()),
                unresolved=int((~resolved).sum()))


def select_weighting(fold_records, config):
    """score the fixed held-out cell inventory before format coupling."""
    priority = {'none': 0, 'weak': 1, 'moderate': 2}
    candidates = sorted(config['weighting']['candidates'], key=lambda row: priority[row['name']])
    metrics, reconstruction = [], []
    for fold in fold_records:
        scores = np.asarray(fold['unit_log_scores'], dtype=np.float64)
        identities = np.asarray(fold['identities'])
        parents = np.asarray(fold['parents'])
        inventory = np.asarray(fold['inventory_identities'], dtype=np.int64)
        assert len(set(inventory)) == len(inventory)
        assert set(int(identity) for identity in identities).issubset(set(inventory))
        reference, counts = build_references(scores, identities, parents)
        saved = dict(fold_id=fold['fold_id'], inventory_identities=inventory,
                     identities=identities, parents=parents, unit_log_scores=scores,
                     references=reference, other_parent_count=counts, candidates={})
        if 'unit_indices' in fold:
            saved['unit_indices'] = np.asarray(fold['unit_indices'])
        for candidate in candidates:
            result = score_weights(scores, reference, counts, candidate, config)
            weights = None if candidate['name'] == 'none' else result['weights']
            summed, depth = aggregate(scores, identities, weights=weights)
            row = _metrics(summed, inventory, fold['states'], config['inference']['resolved_probability'])
            metrics.append(dict(row, candidate=candidate['name'], fold_id=fold['fold_id']))
            saved['candidates'][candidate['name']] = dict(result, aggregate_scores=summed,
                                                        depth=depth, posteriors=softmax(summed))
        reconstruction.append(saved)
    totals = []
    for candidate in candidates:
        rows = [row for row in metrics if row['candidate'] == candidate['name']]
        n_cells = sum(row['n_cells'] for row in rows)
        assert n_cells > 0
        total = {key: sum(row[key] for row in rows) for key in
                 ('n_cells', 'brier_sum', 'log_score_sum', 'map_correct', 'correct', 'wrong', 'unresolved')}
        totals.append(dict(total, candidate=candidate['name'],
                           mean_brier=total['brier_sum'] / n_cells,
                           mean_log_score=total['log_score_sum'] / n_cells))
    # exact ties follow candidate order
    winner = min(totals, key=lambda row: row['mean_brier'])['candidate']
    return dict(winner=winner, selected_candidate=next(row for row in candidates if row['name'] == winner),
                candidate_metrics=totals, fold_metrics=metrics, fold_records=reconstruction)
