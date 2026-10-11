"""pure observation probabilities and ordered evidence aggregation."""

import math

import numpy as np


BASES = "ACT"
LENGTHS = (3, 7, 10)
STATES = np.asarray(((3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)), dtype=np.int64)
REQUIRED_ROWS = (("A", 3), ("A", 7), ("C", 3), ("C", 7), ("C", 10),
                 ("T", 3), ("T", 7), ("T", 10))


def validate_counts(counts):
    """retain the frozen positive integer observation alphabet."""
    counts = np.asarray(counts)
    if counts.ndim != 3 or counts.shape[1:] != (8, 3):
        raise ValueError("counts must have shape [units, 8, 3]")
    if not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("observed runs must be integers")
    if ((counts < 1) | (counts > 20)).any():
        raise ValueError("observed runs must be from 1 through 20")
    return counts.astype(np.int64, copy=False)


def validate_states(states, n_units):
    states = np.asarray(states)
    if states.shape != (n_units, 8) or not np.issubdtype(states.dtype, np.integer):
        raise ValueError("states must be an integer array with shape [units, 8]")
    if ((states < 0) | (states > 3)).any():
        raise ValueError("states must be from 0 through 3")
    return states.astype(np.int64, copy=False)


def global_counts(counts, states):
    counts = validate_counts(counts)
    states = validate_states(states, len(counts))
    histogram = np.zeros((3, 3, 20), dtype=np.int64)
    desired = STATES[states]
    for base in range(3):
        for length_index, length in enumerate(LENGTHS):
            observed = counts[..., base][desired[..., base] == length]
            histogram[base, length_index] = np.bincount(observed - 1, minlength=20)
    return histogram


def global_probabilities(histogram, pseudocount=0.5):
    histogram = np.asarray(histogram)
    if histogram.shape != (3, 3, 20) or not np.issubdtype(histogram.dtype, np.integer):
        raise ValueError("global histogram must have shape [3, 3, 20]")
    if (histogram < 0).any() or not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("histogram and pseudocount must be nonnegative and positive")
    return (histogram + pseudocount) / (histogram.sum(axis=-1, keepdims=True) + 20 * pseudocount)


def global_from_document(document):
    """copy the frozen rows without recomputing their probabilities."""
    if document["observed_bins"] != list(range(1, 21)):
        raise ValueError("global observation alphabet differs from the frozen alphabet")
    if document["pseudocount_per_bin"] != 0.5:
        raise ValueError("global pseudocount differs from the frozen pseudocount")
    if not np.array_equal(document["states_ACT"], STATES):
        raise ValueError("global state mapping differs from the frozen mapping")
    rows = {(row["base"], row["designed_length"]): row for row in document["rows"]}
    if set(rows) != set(REQUIRED_ROWS) or len(document["rows"]) != len(REQUIRED_ROWS):
        raise ValueError("global document must contain the eight required rows")
    probabilities = np.full((3, 3, 20), 0.05, dtype=np.float64)
    for (base, length), row in rows.items():
        values = np.asarray(row["probabilities"], dtype=np.float64)
        histogram = np.asarray(row["observed_histogram"])
        if histogram.shape != (20,) or not np.issubdtype(histogram.dtype, np.integer):
            raise ValueError("frozen global histogram is invalid")
        if (histogram < 0).any() or histogram.sum() != row["run_observations"]:
            raise ValueError("frozen global support is invalid")
        expected = (histogram + 0.5) / (histogram.sum() + 10)
        if values.shape != (20,) or not np.array_equal(values, expected):
            raise ValueError("frozen global probabilities differ from their saved counts")
        probabilities[BASES.index(base), LENGTHS.index(length)] = values
    return probabilities


def position_counts(counts, states, identities, parents):
    counts = validate_counts(counts)
    states = validate_states(states, len(counts))
    identities = np.asarray(identities)
    parents = np.asarray(parents)
    if identities.shape != (len(counts),) or not np.issubdtype(identities.dtype, np.integer):
        raise ValueError("identities must match the observation units")
    if parents.shape != (len(counts),):
        raise ValueError("parents must match the observation units")
    histogram = np.zeros((8, 3, 3, 20), dtype=np.int64)
    desired = STATES[states]
    support_rows = []
    for position in range(8):
        for base in range(3):
            for length_index, length in enumerate(LENGTHS):
                selected = desired[:, position, base] == length
                observed = counts[selected, position, base]
                histogram[position, base, length_index] = np.bincount(observed - 1, minlength=20)
                support_rows.append(dict(
                    position=position, base=BASES[base], designed_length=length,
                    observations=int(selected.sum()),
                    pilot_designs=len(set(identities[selected].tolist())),
                    raw_parents=len(set(parents[selected].tolist())),
                    required=(BASES[base], length) in REQUIRED_ROWS))
    return histogram, support_rows


def support_flags(support_rows, support):
    flags = np.zeros((8, 3, 3), dtype=bool)
    keys = set()
    for row in support_rows:
        index = (int(row["position"]), BASES.index(row["base"]), LENGTHS.index(int(row["designed_length"])))
        if index in keys:
            raise ValueError("duplicate position support row")
        keys.add(index)
        flags[index] = ((row["base"], int(row["designed_length"])) in REQUIRED_ROWS
                        and row["observations"] >= support["min_observations"]
                        and row["pilot_designs"] >= support["min_pilot_designs"]
                        and row["raw_parents"] >= support["min_raw_parents"])
    if len(keys) != 72:
        raise ValueError("position support must describe all 72 rows")
    return flags


def position_model(histogram, global_w, lambda_value, support_rows, support):
    histogram = np.asarray(histogram)
    global_w = np.asarray(global_w, dtype=np.float64)
    if histogram.shape != (8, 3, 3, 20) or not np.issubdtype(histogram.dtype, np.integer):
        raise ValueError("position histogram must have shape [8, 3, 3, 20]")
    if (histogram < 0).any() or global_w.shape != (3, 3, 20):
        raise ValueError("invalid position histogram or global model")
    if not np.isfinite(global_w).all() or (global_w <= 0).any():
        raise ValueError("global probabilities must be positive and finite")
    model = np.broadcast_to(global_w, histogram.shape).copy()
    flags = support_flags(support_rows, support)
    if lambda_value == "infinity":
        return model
    if not isinstance(lambda_value, (int, float)) or not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda must be positive and finite or exactly infinity")
    estimated = (histogram + lambda_value * global_w) / (histogram.sum(axis=-1, keepdims=True) + lambda_value)
    model[flags] = estimated[flags]
    return model


def cell_log_scores(counts, probabilities):
    """add base scores in the canonical order without a prior constant."""
    counts = validate_counts(counts)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape == (3, 3, 20):
        probabilities = np.broadcast_to(probabilities, (8, 3, 3, 20))
    if probabilities.shape != (8, 3, 3, 20):
        raise ValueError("probabilities must be global or position conditioned")
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("probabilities must be positive and finite")
    log_probabilities = np.log(probabilities)
    scores = np.zeros((len(counts), 8, 4), dtype=np.float64)
    for state, desired in enumerate(STATES):
        for base in range(3):
            length_index = LENGTHS.index(int(desired[base]))
            scores[..., state] += log_probabilities[np.arange(8), base, length_index, counts[..., base] - 1]
    return scores


def aggregate(scores, identities, weights=None, n_addresses=480):
    scores = np.asarray(scores, dtype=np.float64)
    identities = np.asarray(identities)
    if scores.ndim != 3 or scores.shape[1:] != (8, 4) or not np.isfinite(scores).all():
        raise ValueError("scores must have shape [units, 8, 4] and be finite")
    if identities.shape != (len(scores),) or not np.issubdtype(identities.dtype, np.integer):
        raise ValueError("identities must match the score units")
    if ((identities < 0) | (identities >= n_addresses)).any():
        raise ValueError("identity lies outside the address allocation")
    contributions = scores
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != scores.shape[:2] or not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
            raise ValueError("weights must have shape [units, 8] and be in [0, 1]")
        # equality arm skips multiplication
        if not np.array_equal(weights, np.ones_like(weights)):
            contributions = weights[..., None] * scores
    summed = np.zeros((n_addresses, 8, 4), dtype=np.float64)
    depth = np.zeros((n_addresses, 8), dtype=np.int64)
    np.add.at(summed, identities, contributions)
    np.add.at(depth, identities, np.ones(scores.shape[:2], dtype=np.int64))
    return summed, depth


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim < 1 or scores.shape[-1] != 4 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and end in four states")
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return probability / probability.sum(axis=-1, keepdims=True)


def discriminability(probabilities):
    """enumerate mutual information under a uniform four state prior."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (3, 3, 20):
        raise ValueError("discriminability requires one position model")
    joint = []
    for desired in STATES:
        rows = [probabilities[base, LENGTHS.index(int(length))] for base, length in enumerate(desired)]
        joint.append((rows[0][:, None, None] * rows[1][None, :, None] * rows[2][None, None, :]).reshape(-1))
    joint = np.asarray(joint)
    marginal = joint.mean(axis=0)
    return float(np.sum(joint * np.log(joint / marginal), dtype=np.float64) / 4)
