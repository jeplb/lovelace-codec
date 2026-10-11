"""Independent NEW_ABC_ONLY template-switching tests from frozen inputs."""

from pathlib import Path
import csv
import gzip
import itertools
import json

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tsv_records import read_table, write_output

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / "frozen_inputs"
STATES = np.array([(3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)])
COHORT = "NEW_ABC_ONLY"


def write_table(name, rows):
    rows = list(rows)
    if not rows:
        return
    opener = gzip.open if name.endswith(".gz") else open
    with opener(OUT / name, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_inputs():
    sequences = {}
    identity = None
    for line in (INPUT / "Lovelace_S12_480.fasta").read_text().splitlines():
        if line.startswith(">"):
            identity = int(line.rsplit("_", 1)[1])
            sequences[identity] = ""
        else:
            sequences[identity] += line.strip()
    design = []
    for identity in range(480):
        sequence = sequences[identity]
        if len(sequence) != 298:
            raise ValueError("unexpected designed strand length")
        runs = [(base, len(list(group))) for base, group in itertools.groupby(sequence[109:269])]
        if "".join(base for base, _ in runs) != "ACT" * 8:
            raise ValueError("unexpected designed payload frame")
        cells = np.array([length for _, length in runs]).reshape(8, 3)
        distances = np.abs(cells[:, None, :] - STATES[None, :, :]).sum(2)
        if np.any(distances.min(1)):
            raise ValueError("nonlegal designed state")
        design.append(distances.argmin(1))
    units = read_table(INPUT / "F_complete_new_units.tsv")
    observed = np.array([unit["observed_direct"] for unit in units])
    if observed.shape != (9059, 8, 3):
        raise ValueError("unexpected frozen unit shape")
    identity = np.array([unit["identity"] for unit in units])
    design = np.array(design)
    truth = design[identity]
    distance = np.abs(observed[:, :, None, :] - STATES[None, None, :, :]).sum(3)
    valid = (distance == distance.min(2)[:, :, None]).sum(2) == 1
    called = distance.argmin(2)
    exact_swap = (distance.min(2) == 0) & (called != truth)
    nearest_swap = valid & (called != truth)
    called[~valid] = -1
    return units, design, observed, identity, truth, called, valid, exact_swap, nearest_swap


def suffix_length(swap, valid=None):
    counts = swap.sum(1)
    suffix = np.all(swap == (np.arange(swap.shape[1])[None, :] >= (swap.shape[1] - counts[:, None])), 1)
    suffix &= counts > 0
    if valid is not None:
        suffix &= valid.all(1)
    return np.where(suffix, counts, 0)


def transition_counts(swap, valid):
    eligible = valid[:, :-1] & valid[:, 1:]
    previous = swap[:, :-1]
    following = swap[:, 1:]
    return np.array([[(eligible & (previous == a) & (following == b)).sum() for b in [0, 1]] for a in [0, 1]])


def transition_difference(counts):
    denominators = counts.sum(1)
    if np.any(denominators == 0):
        return float("nan")
    return float(counts[1, 1] / denominators[1] - counts[0, 1] / denominators[0])


def transition_outputs(units, identities, exact, nearest, valid):
    orientation = np.array([unit["orientation"] for unit in units])
    data = identities < 448
    rows = []
    for cohort, selected in [("data", data), ("all", np.ones(len(units), bool))]:
        for label, swap, eligible in [("exact_legal", exact, np.ones_like(valid)), ("nearest_unique", nearest, valid)]:
            for direction in ["all", "forward", "reverse_complement"]:
                keep = selected & ((orientation == direction) if direction != "all" else True)
                for position in [0] + list(range(1, 8)):
                    cols = slice(None) if position == 0 else slice(position - 1, position)
                    pairvalid = eligible[keep, :-1][:, cols] & eligible[keep, 1:][:, cols]
                    previous = swap[keep, :-1][:, cols]
                    following = swap[keep, 1:][:, cols]
                    for a in [0, 1]:
                        n = int((pairvalid & (previous == a)).sum())
                        y = int((pairvalid & (previous == a) & following).sum())
                        rows.append(dict(cohort=COHORT, subset=cohort, definition=label, orientation=direction,
                                         from_cell=position, previous_swap=a, pairs=n, next_swaps=y,
                                         probability=y / n if n else ""))
    write_table("swap_transitions.tsv", rows)
    # collapse repeated parents before the position-adjusted independence null
    unique = np.array(list(dict.fromkeys(unit["raw_parent_id"] for unit in units)))
    first = {unit["raw_parent_id"]: i for i, unit in reversed(list(enumerate(units)))}
    indices = np.array([first[parent] for parent in unique if identities[first[parent]] < 448])
    groups = []
    for identity in np.unique(identities[indices]):
        for direction in ["forward", "reverse_complement"]:
            groups.append(np.where((identities[indices] == identity) & (orientation[indices] == direction))[0])
    groups = [group for group in groups if len(group)]
    rng = np.random.default_rng(48115)
    null_rows = []
    for label, swap, eligible in [("exact_legal", exact, np.ones_like(valid)), ("nearest_unique", nearest, valid)]:
        x = swap[indices]
        v = eligible[indices]
        observed = transition_difference(transition_counts(x, v))
        samples = []
        for replicate in range(999):
            shuffled = np.empty_like(x)
            shuffled_valid = np.empty_like(v)
            for group in groups:
                for cell in range(8):
                    order = rng.permutation(group)
                    shuffled[group, cell] = x[order, cell]
                    shuffled_valid[group, cell] = v[order, cell]
            difference = transition_difference(transition_counts(shuffled, shuffled_valid))
            samples.append(difference)
            null_rows.append(dict(cohort=COHORT, subset="data_unique_parent", definition=label,
                                 replicate=replicate, difference=difference))
        samples = np.array(samples)
        rowsummary = dict(cohort=COHORT, subset="data_unique_parent", definition=label,
                          units=len(indices), observed_difference=observed, null_mean=float(samples.mean()),
                          null_q025=float(np.quantile(samples, .025)), null_q975=float(np.quantile(samples, .975)),
                          replicates=len(samples), p_one_sided=(1 + int((samples >= observed).sum())) / (len(samples) + 1))
        write_table("swap_permutation_summary_" + label + ".tsv", [rowsummary])
    write_table("swap_permutation_null.tsv", null_rows)


def empirical_substitution(called, truth, valid, orientation, selected):
    frequencies = {}
    for direction in ["forward", "reverse_complement"]:
        for cell in range(8):
            for state in range(4):
                keep = selected & (orientation == direction) & valid[:, cell] & (truth[:, cell] == state) & (called[:, cell] != state)
                counts = np.bincount(called[keep, cell], minlength=4).astype(float)
                # an empty stratum has no empirical distribution, so use the uniform null
                if counts.sum() == 0:
                    counts[:] = 1
                    counts[state] = 0
                frequencies[direction, cell, state] = counts / counts.sum()
    return frequencies


def tail_outputs(units, design, identities, truth, called, valid, exact, nearest):
    orientation = np.array([unit["orientation"] for unit in units])
    data = identities < 448
    rows = []
    summary_rows = []
    flagged = {}
    for label, swap, eligibility in [("exact_legal", exact, None), ("nearest_unique", nearest, valid)]:
        frequencies = empirical_substitution(called, truth, exact if label == "exact_legal" else valid, orientation, data)
        length = suffix_length(swap, eligibility)
        donor_flag = np.zeros(len(units), bool)
        for index in np.where(length >= 4)[0]:
            size = int(length[index])
            identity = int(identities[index])
            donors = np.where(np.arange(480) != identity)[0]
            tail = called[index, -size:]
            matches = (design[donors, -size:] == tail).sum(1)
            best = int(matches.max())
            winners = donors[matches == best]
            unique_tails = np.unique(design[donors, -size:], axis=0)
            all_wrong = np.all(unique_tails != truth[index, -size:], 1)
            compatible = unique_tails[all_wrong]
            uniform_null = len(compatible) / 3 ** size
            empirical_null = 0.0
            for donor_tail in compatible:
                empirical_null += float(np.prod([frequencies[orientation[index], cell, int(truth[index, cell])][state]
                                                 for cell, state in zip(range(8 - size, 8), donor_tail)]))
            exact_match = best == size
            donor_flag[index] = exact_match
            # compare a one-cell registration shift within the addressed template
            earlier = int((tail[1:] == truth[index, 8 - size:-1]).sum())
            later = int((tail[:-1] == truth[index, 9 - size:]).sum())
            rows.append(dict(cohort=COHORT, subset="data" if identity < 448 else "pilot", definition=label,
                             unit_id=units[index]["unit_id"], parent_id=units[index]["raw_parent_id"],
                             identity=identity, orientation=orientation[index], suffix_length=size,
                             called_tail="".join(map(str, tail)), best_matches=best, exact_other_match=int(exact_match),
                             best_other_design=int(winners[0]), all_best_other_designs=",".join(map(str, winners)),
                             tied_best_designs=len(winners), unique_other_tails=len(unique_tails),
                             unique_uniform_4state_probability=len(unique_tails) / 4 ** size,
                             requested_480_union_bound=480 / 4 ** size, capped_480_union_bound=min(1.0, 480 / 4 ** size),
                             other479_union_bound=479 / 4 ** size, conditional_wrong_uniform_null=uniform_null,
                             conditional_wrong_empirical_null=empirical_null,
                             earlier_self_shift_matches=earlier, later_self_shift_matches=later, self_shift_comparable_cells=size - 1))
        flagged[label] = donor_flag
        for subset, selected in [("data", data), ("all", np.ones(len(units), bool))]:
            for size in [0, 4, 5, 6, 7, 8]:
                eligible_rows = [row for row in rows if row["definition"] == label and
                                 (subset == "all" or row["subset"] == "data") and
                                 (size == 0 or row["suffix_length"] == size)]
                swap_units = int((selected & swap.any(1)).sum())
                complete_swap_units = int((selected & swap.any(1) & (valid.all(1) if eligibility is not None else True)).sum())
                suffix_units = int((selected & (length > 0)).sum())
                count = len(eligible_rows)
                summary_rows.append(dict(cohort=COHORT, subset=subset, definition=label, suffix_length=size,
                                         swap_units=swap_units, fully_called_swap_units=complete_swap_units,
                                         suffix_units=suffix_units, suffix_fraction_all_swap_units=suffix_units / swap_units,
                                         suffix_fraction_fully_called_swap_units=suffix_units / complete_swap_units,
                                         long_suffix_units=count, exact_donor_matches=sum(row["exact_other_match"] for row in eligible_rows),
                                         expected_conditional_uniform_matches=sum(row["conditional_wrong_uniform_null"] for row in eligible_rows),
                                         expected_conditional_empirical_matches=sum(row["conditional_wrong_empirical_null"] for row in eligible_rows),
                                         mean_conditional_uniform_probability=np.mean([row["conditional_wrong_uniform_null"] for row in eligible_rows]) if count else "",
                                         mean_conditional_empirical_probability=np.mean([row["conditional_wrong_empirical_null"] for row in eligible_rows]) if count else ""))
    write_table("swap_tail_matches.tsv", rows)
    write_table("swap_suffix_summary.tsv", summary_rows)
    return flagged


def mismatch_histogram(design, identities, truth, called, valid, cuts, weights=None):
    errors = valid & (called != truth)
    histogram = np.zeros((len(identities), 9))
    for identity in np.unique(identities):
        indices = np.where(identities == identity)[0]
        donors = np.where(np.arange(480) != identity)[0]
        donor_error = valid[indices, None, :] & (called[indices, None, :] != design[donors][None, :, :])
        suffix = np.flip(np.cumsum(np.flip(donor_error, 2), axis=2), 2)
        prefix = np.column_stack([np.zeros(len(indices), int), np.cumsum(errors[indices], 1)])
        donor_weights = np.ones(len(donors)) if weights is None else np.asarray(weights[donors], dtype=float)
        donor_weights /= donor_weights.sum()
        for cut in cuts:
            mismatch = suffix[:, :, cut] + prefix[:, cut, None]
            for count in range(9):
                histogram[indices, count] += ((mismatch == count) * donor_weights[None, :]).sum(1) / len(cuts)
    return histogram


def component_likelihood(epsilon, nvalid, nerror, histogram):
    clean = (1 - epsilon) ** (nvalid - nerror) * (epsilon / 3) ** nerror
    counts = np.arange(9)[None, :]
    chimera = (histogram * (1 - epsilon) ** (nvalid[:, None] - counts) * (epsilon / 3) ** counts).sum(1)
    return clean, chimera


def fit_mixture(nvalid, nerror, histogram, selected, profile=False):
    def objective(parameters):
        epsilon, fraction = parameters
        clean, chimera = component_likelihood(epsilon, nvalid[selected], nerror[selected], histogram[selected])
        return float(-np.log((1 - fraction) * clean + fraction * chimera).sum())
    fit = minimize(objective, [.025, .4], method="L-BFGS-B", bounds=[(1e-7, .75), (1e-7, 1 - 1e-7)])
    if not fit.success:
        raise RuntimeError(fit.message)
    epsilon, fraction = map(float, fit.x)
    clean, chimera = component_likelihood(epsilon, nvalid, nerror, histogram)
    posterior = fraction * chimera / ((1 - fraction) * clean + fraction * chimera)
    result = dict(epsilon=epsilon, chimera_fraction=fraction, negative_log_likelihood=float(fit.fun),
                  selected_units=int(selected.sum()), inferred_chimera_units=int((selected & (posterior >= .5)).sum()),
                  posterior_expected_chimera_units=float(posterior[selected].sum()),
                  profile_lower="", profile_upper="")
    if profile:
        threshold = float(fit.fun + chi2.ppf(.95, 1) / 2)
        def excess(pi):
            fit_epsilon = minimize_scalar(lambda value: objective([value, pi]), bounds=(1e-7, .75), method="bounded")
            return fit_epsilon.fun - threshold
        result["profile_lower"] = float(brentq(excess, 1e-7, fraction)) if excess(1e-7) > 0 else 0.0
        result["profile_upper"] = float(brentq(excess, fraction, 1 - 1e-7)) if excess(1 - 1e-7) > 0 else 1.0
    return result, posterior, clean, chimera


def best_latent(design, identities, truth, called, valid, cuts):
    rows = []
    for index, identity in enumerate(identities):
        donors = np.where(np.arange(480) != identity)[0]
        errors = valid[index] & (called[index] != truth[index])
        donor_errors = valid[index][None, :] & (called[index][None, :] != design[donors])
        candidates = np.array([errors[:cut].sum() + donor_errors[:, cut:].sum(1) for cut in cuts])
        best = int(candidates.min())
        choices = np.argwhere(candidates == best)
        cut_index, donor_index = choices[0]
        rows.append((cuts[int(cut_index)], int(donors[donor_index]), len(choices), best))
    return np.array(rows)


def parser_agreement_summary(units, identities, valid, exact, nearest, flagged, histogram):
    path = OUT / "parser_strict_agreement.tsv"
    if not path.exists():
        return None
    with path.open() as handle:
        agreement = {row["unit_id"]: all(row[key] == "True" for key in
                     ["orientation_match", "identity_match", "bounds_match", "payload_runs_match"])
                     for row in csv.DictReader(handle, delimiter="\t")}
    selected = (identities < 448) & np.array([agreement.get(unit["unit_id"], False) for unit in units])
    result, _, _, _ = fit_mixture(valid.sum(1), nearest.sum(1), histogram, selected, profile=True)
    result.update(cohort=COHORT, subset="data_independent_parser_agreement", model="internal_1to7", donor_prior="uniform_other479")
    selected_units = {unit["unit_id"] for unit, keep in zip(units, selected) if keep}
    with (OUT / "swap_tail_matches.tsv").open() as handle:
        tails = [row for row in csv.DictReader(handle, delimiter="\t") if row["unit_id"] in selected_units]
    rows = []
    for label, swap, eligible in [("exact_legal", exact, np.ones_like(valid)), ("nearest_unique", nearest, valid)]:
        counts = transition_counts(swap[selected], eligible[selected])
        length = suffix_length(swap, valid if label == "nearest_unique" else None)
        row = dict(cohort=COHORT, subset=result["subset"], definition=label, units=int(selected.sum()),
                   swap_units=int((selected & swap.any(1)).sum()), suffix_units=int((selected & (length > 0)).sum()),
                   long_suffix_units=int((selected & (length >= 4)).sum()),
                   exact_tail_donor_matches=int((selected & flagged[label]).sum()),
                   expected_conditional_uniform_matches=sum(float(row["conditional_wrong_uniform_null"]) for row in tails if row["definition"] == label),
                   expected_conditional_empirical_matches=sum(float(row["conditional_wrong_empirical_null"]) for row in tails if row["definition"] == label),
                   previous0_next0=int(counts[0, 0]), previous0_next1=int(counts[0, 1]),
                   previous1_next0=int(counts[1, 0]), previous1_next1=int(counts[1, 1]),
                   probability_after_nonswap=float(counts[0, 1] / counts[0].sum()),
                   probability_after_swap=float(counts[1, 1] / counts[1].sum()),
                   chimera_fraction=result["chimera_fraction"], epsilon=result["epsilon"],
                   profile_lower=result["profile_lower"], profile_upper=result["profile_upper"])
        rows.append(row)
    write_table("chimera_parser_agreement.tsv", rows)
    return result


def gradient_rows(observed, truth_states, called, valid, posterior, flagged, selected, subset, leave_out=None):
    rows = []
    exact_run = observed != STATES[truth_states]
    wrong = valid & (called != truth_states)
    strategies = ["before", "posterior_below_half", "soft_clean", "remove_exact_tail_donor", "remove_nearest_tail_donor"]
    if leave_out is not None:
        strategies += ["leave_cell_out_below_half", "leave_cell_out_soft_clean"]
    for strategy in strategies:
        weights = selected.astype(float)
        if strategy == "posterior_below_half":
            weights *= posterior < .5
        elif strategy == "soft_clean":
            weights *= 1 - posterior
        elif strategy == "remove_exact_tail_donor":
            weights *= ~flagged["exact_legal"]
        elif strategy == "remove_nearest_tail_donor":
            weights *= ~flagged["nearest_unique"]
        for cell in range(8):
            cell_weights = weights.copy()
            if strategy == "leave_cell_out_below_half":
                cell_weights *= leave_out[:, cell] < .5
            elif strategy == "leave_cell_out_soft_clean":
                cell_weights *= 1 - leave_out[:, cell]
            denominator = float((cell_weights * valid[:, cell]).sum())
            rows.append(dict(cohort=COHORT, subset=subset, strategy=strategy, cell=cell + 1,
                             effective_units=float(cell_weights.sum()), exact_run_error=float((cell_weights[:, None] * exact_run[:, cell]).sum() / (3 * cell_weights.sum())),
                             nearest_state_error=float((cell_weights * wrong[:, cell]).sum() / denominator),
                             valid_state_weight=denominator))
    return rows


def main():
    units, design, observed, identities, truth, called, valid, exact, nearest = load_inputs()
    orientations = np.array([unit["orientation"] for unit in units])
    data = identities < 448
    transition_outputs(units, identities, exact, nearest, valid)
    flagged = tail_outputs(units, design, identities, truth, called, valid, exact, nearest)
    nvalid = valid.sum(1)
    nerror = nearest.sum(1)
    fits = []
    profile_rows = []
    histograms = {}
    posteriors = {}
    for cut_label, cuts in [("internal_1to7", list(range(1, 8))), ("including_before_cell1_0to7", list(range(8)))]:
        histogram = mismatch_histogram(design, identities, truth, called, valid, cuts)
        histograms[cut_label] = histogram
        for subset, selected in [("data", data), ("all", np.ones(len(units), bool))]:
            result, posterior, clean, chimera = fit_mixture(nvalid, nerror, histogram, selected, profile=True)
            result.update(cohort=COHORT, subset=subset, model=cut_label, donor_prior="uniform_other479")
            fits.append(result)
            posteriors[cut_label, subset] = posterior
            if cut_label == "internal_1to7" and subset == "data":
                primary = posterior
                primary_clean = clean
                primary_chimera = chimera
        if cut_label == "internal_1to7":
            for direction in ["forward", "reverse_complement"]:
                selected = data & (orientations == direction)
                result, _, _, _ = fit_mixture(nvalid, nerror, histogram, selected, profile=True)
                result.update(cohort=COHORT, subset="data_" + direction, model=cut_label, donor_prior="uniform_other479")
                fits.append(result)
            first = np.zeros(len(units), bool)
            seen = set()
            for index, unit in enumerate(units):
                if unit["raw_parent_id"] not in seen:
                    first[index] = True
                    seen.add(unit["raw_parent_id"])
            result, _, _, _ = fit_mixture(nvalid, nerror, histogram, data & first, profile=True)
            result.update(cohort=COHORT, subset="data_unique_parent", model=cut_label, donor_prior="uniform_other479")
            fits.append(result)
    empirical_depth = np.bincount(identities, minlength=480) + 1
    weighted_histogram = mismatch_histogram(design, identities, truth, called, valid, list(range(1, 8)), empirical_depth)
    result, weighted_posterior, _, _ = fit_mixture(nvalid, nerror, weighted_histogram, data, profile=True)
    result.update(cohort=COHORT, subset="data", model="internal_1to7", donor_prior="observed_strict_depth_plus_one")
    fits.append(result)
    # squared-distance calls check sensitivity to mapping nonlegal measured runs to states
    squared = ((observed[:, :, None, :] - STATES[None, None, :, :]) ** 2).sum(3)
    square_valid = (squared == squared.min(2)[:, :, None]).sum(2) == 1
    square_called = squared.argmin(2)
    square_called[~square_valid] = -1
    square_error = (square_valid & (square_called != truth)).sum(1)
    square_hist = mismatch_histogram(design, identities, truth, square_called, square_valid, list(range(1, 8)))
    result, squared_posterior, _, _ = fit_mixture(square_valid.sum(1), square_error, square_hist, data, profile=True)
    result.update(cohort=COHORT, subset="data_squared_distance_calls", model="internal_1to7", donor_prior="uniform_other479")
    fits.append(result)
    parser_fit = parser_agreement_summary(units, identities, valid, exact, nearest, flagged, histograms["internal_1to7"])
    if parser_fit is not None:
        fits.append(parser_fit)
    write_table("chimera_model_fit.tsv", fits)
    independent = []
    for subset, selected in [("data", data), ("all", np.ones(len(units), bool))]:
        for model in ["iid_clean", "position_specific_independent"]:
            p = nearest[selected].sum(0) / valid[selected].sum(0) if model == "position_specific_independent" else float(nerror[selected].sum() / nvalid[selected].sum())
            probabilities = np.where(nearest[selected], p / 3, 1 - p)
            nll = float(-np.log(probabilities)[valid[selected]].sum())
            independent.append(dict(cohort=COHORT, subset=subset, model=model, parameters=8 if model == "position_specific_independent" else 1,
                                    negative_log_likelihood=nll, fitted_error_rates=",".join(map(str, np.atleast_1d(p)))))
    write_table("chimera_independent_models.tsv", independent)
    latent = best_latent(design, identities, truth, called, valid, list(range(1, 8)))
    unit_rows = []
    for index, unit in enumerate(units):
        unit_rows.append(dict(cohort=COHORT, unit_id=unit["unit_id"], parent_id=unit["raw_parent_id"], identity=int(identities[index]),
                              orientation=orientations[index], role=unit["role"], valid_cells=int(nvalid[index]), wrong_nearest_cells=int(nerror[index]),
                              posterior=float(primary[index]), posterior_all_fit=float(posteriors["internal_1to7", "all"][index]),
                              posterior_allow_start=float(posteriors["including_before_cell1_0to7", "data"][index]),
                              posterior_depth_prior=float(weighted_posterior[index]), posterior_squared_calls=float(squared_posterior[index]),
                              clean_likelihood=float(primary_clean[index]), chimera_likelihood=float(primary_chimera[index]),
                              map_crossover_after_cell=int(latent[index, 0]), map_donor_identity=int(latent[index, 1]),
                              map_latent_ties=int(latent[index, 2]), map_latent_mismatches=int(latent[index, 3]),
                              exact_tail_donor=int(flagged["exact_legal"][index]), nearest_tail_donor=int(flagged["nearest_unique"][index])))
    write_table("chimera_units.tsv", unit_rows)
    calls_rows = []
    for index, unit in enumerate(units):
        for cell in range(8):
            calls_rows.append(dict(cohort=COHORT, unit_id=unit["unit_id"], parent_id=unit["raw_parent_id"], identity=int(identities[index]),
                                   orientation=orientations[index], role=unit["role"], cell=cell + 1, true_state=int(truth[index, cell]),
                                   called_state=int(called[index, cell]), unique_nearest=int(valid[index, cell]),
                                   exact_wrong_legal=int(exact[index, cell]), nearest_wrong=int(nearest[index, cell]),
                                   observed_a=int(observed[index, cell, 0]), observed_c=int(observed[index, cell, 1]), observed_t=int(observed[index, cell, 2])))
    write_table("strict_cell_calls.tsv.gz", calls_rows)
    primary_fit = next(row for row in fits if row["subset"] == "data" and row["model"] == "internal_1to7" and row["donor_prior"] == "uniform_other479")
    leave_out = np.empty((len(units), 8))
    for cell in range(8):
        mask = valid.copy()
        mask[:, cell] = False
        masked_errors = (mask & (called != truth)).sum(1)
        masked_histogram = mismatch_histogram(design, identities, truth, called, mask, list(range(1, 8)))
        clean, chimera = component_likelihood(primary_fit["epsilon"], mask.sum(1), masked_errors, masked_histogram)
        fraction = primary_fit["chimera_fraction"]
        leave_out[:, cell] = fraction * chimera / ((1 - fraction) * clean + fraction * chimera)
    gradient = gradient_rows(observed, truth, called, valid, primary, flagged, data, "data", leave_out)
    gradient += gradient_rows(observed, truth, called, valid, posteriors["internal_1to7", "all"], flagged, np.ones(len(units), bool), "all")
    write_table("chimera_gradients.tsv", gradient)
    np.savez_compressed(OUT / "chimera_arrays.npz", cohort=COHORT, unit_id=np.array([u["unit_id"] for u in units]),
                        parent_id=np.array([u["raw_parent_id"] for u in units]), identity=identities, orientation=orientations,
                        role=np.array([u["role"] for u in units]), observed=observed, truth=STATES[truth], true_state=truth,
                        called_state=called, valid_call=valid, exact_swap=exact, nearest_swap=nearest, posterior=primary,
                        posterior_all=posteriors["internal_1to7", "all"], exact_tail_donor=flagged["exact_legal"],
                        nearest_tail_donor=flagged["nearest_unique"], leave_cell_out_posterior=leave_out, internal_mismatch_histogram=histograms["internal_1to7"])
    summary = dict(cohort=COHORT, units=len(units), data_units=int(data.sum()), unique_parents=len(set(u["raw_parent_id"] for u in units)),
                   ambiguous_nearest_cells=int((~valid).sum()), model_assumptions=[
                       "clean cells have one shared symmetric substitution rate",
                       "chimera uses uniform internal crossover after cells 1 through 7",
                       "donor is uniform over the other 479 actual designed payloads",
                       "chimera shares the clean residual substitution rate",
                       "ambiguous nearest-state observations are marginalized as missing",
                       "iid per-unit likelihood ignores eight duplicate-parent pairs; unique-parent sensitivity is saved"],
                   limits=[
                       "the mixture fraction is model dependent and does not directly prove PCR switching",
                       "a foreign template may share true states after crossover, so a chimera need not make a contiguous wrong suffix",
                       "hard posterior removal selects on the tested errors and can flatten the gradient by construction",
                       "uniform crossover and donor assumptions can mimic template deletion or duplicated-cell errors",
                       "the 480 divided by four-to-the-L expression is a union bound under uniform independent symbols, not an exact match probability",
                       "real library tails repeat and conditional wrong-state empirical frequencies include the tested observations",
                       "conditional likelihood drops tie information, whose rate may depend on latent template and run error"])
    write_output(OUT / "chimera_summary.tsv", summary)
    print(json.dumps(summary))
    print(json.dumps(fits))


if __name__ == "__main__":
    main()
