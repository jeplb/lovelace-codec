"""independent raw-read registration audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import edlib
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tsv_records import append_table_rows, iter_table, write_output


COHORT = "NEW_ABC_ONLY"
OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / "frozen_inputs"
MANIFEST = OUT.parents[2] / "codebook/address_manifest.csv"
P1 = "CCCAATTAATACGTTGCCTACTCCG"
P2 = "CGCTATCAGAACAAGAGAAGCACTA"
STATES = np.array([(3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)])


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def rle(sequence):
    runs = []
    for i, base in enumerate(sequence):
        if runs and base == runs[-1]["base"]:
            runs[-1]["length"] += 1
            runs[-1]["end"] = i + 1
        else:
            runs.append(dict(base=base, length=1, start=i, end=i + 1))
    return runs


def compressed(sequence):
    return "".join(run["base"] for run in rle(sequence))


def nearest_address(pattern, patterns):
    distances = [edlib.align(pattern, p, mode="NW", task="distance")["editDistance"]
                 for p in patterns]
    minimum = min(distances)
    identities = [i for i, d in enumerate(distances) if d == minimum]
    return minimum, identities


def primer_pair(sequence, max_edits=5):
    candidates = []
    for orientation, seq in (("forward", sequence),
                             ("reverse_complement", reverse_complement(sequence))):
        left = edlib.align(P1, seq, mode="HW", task="locations", k=max_edits)
        right = edlib.align(P2, seq, mode="HW", task="locations", k=max_edits)
        if left["editDistance"] < 0 or right["editDistance"] < 0:
            continue
        for left_start, left_end in left["locations"]:
            for right_start, right_end in right["locations"]:
                if left_end < right_start:
                    candidates.append(dict(
                        orientation=orientation, sequence=seq,
                        left_edits=left["editDistance"], right_edits=right["editDistance"],
                        primer_start=left_start, address_start=left_end + 1,
                        primer_right_start=right_start, primer_end=right_end + 1))
    if not candidates:
        return None, "primers_missing_or_out_of_order"
    best_score = min(c["left_edits"] + c["right_edits"] for c in candidates)
    best = [c for c in candidates if c["left_edits"] + c["right_edits"] == best_score]
    if len({c["orientation"] for c in best}) > 1:
        return None, "orientation_tie"
    # equally good primer endpoints prefer the outermost complete interval
    return min(best, key=lambda c: (c["primer_start"], -c["primer_end"],
                                   -c["address_start"], c["primer_right_start"])), "resolved"


class Parser:
    def __init__(self, patterns, max_edits=5, address_threshold=2):
        self.patterns = tuple(patterns)
        self.max_edits = max_edits
        self.address_threshold = address_threshold
        self.address_call = lru_cache(maxsize=None)(
            lambda pattern: nearest_address(pattern, self.patterns))

    def parse(self, sequence):
        pair, reason = primer_pair(sequence, self.max_edits)
        result = dict(cohort=COHORT, orientation_supported=pair is not None,
                      address_supported=False, payload_resolved=False, complete_ACT8=False,
                      any_ACT=False, reason=reason, address_distance=None,
                      address_nearest_unique=False)
        if pair is None:
            return result
        seq = pair.pop("sequence")
        result.update(pair)
        runs = rle(seq)
        anchors = [run for run in runs if run["base"] == "G"
                   and run["start"] >= pair["address_start"]
                   and run["end"] <= pair["primer_right_start"] + 2]
        addresses = []
        for anchor in anchors:
            address_seq = seq[pair["address_start"]:anchor["start"]]
            pattern = compressed(address_seq)
            # distances through three edits require an address length of 13 to 19 runs
            if not 13 <= len(pattern) <= 19:
                continue
            distance, identities = self.address_call(pattern)
            addresses.append(dict(anchor=anchor, address_pattern_observed=pattern,
                                  address_distance=distance, address_nearest=identities))
        if not addresses:
            result["reason"] = "no_address_window"
            return result
        best_distance = min(a["address_distance"] for a in addresses)
        addresses = [a for a in addresses if a["address_distance"] == best_distance]
        identities = {i for a in addresses for i in a["address_nearest"]}
        result.update(address_distance=best_distance,
                      address_nearest_unique=len(identities) == 1,
                      address_nearest=sorted(identities))
        if len(identities) != 1:
            result["reason"] = "address_tie"
            return result
        result["identity"] = next(iter(identities))
        result["address_supported"] = best_distance <= self.address_threshold
        if not result["address_supported"]:
            result["reason"] = "address_distance_above_threshold"
            return result
        payloads = []
        for address in addresses:
            left = address["anchor"]
            for right in anchors:
                if right["start"] <= left["end"] or right["start"] >= pair["primer_right_start"]:
                    continue
                # the delimiter must adjoin the fitted right primer within its edit budget
                primer_gap = pair["primer_right_start"] - right["end"]
                if not -2 <= primer_gap <= self.max_edits:
                    continue
                payload = seq[left["end"]:right["start"]]
                payload_runs = rle(payload)
                bases = "".join(run["base"] for run in payload_runs)
                distance = edlib.align(bases, "ACT" * 8, mode="NW", task="distance")["editDistance"]
                score = (distance, abs(primer_gap),
                         abs(left["length"] - 4) + abs(right["length"] - 4))
                payloads.append(dict(score=score, address=address, left=left, right=right,
                                     payload_sequence=payload, payload_run_bases=bases,
                                     payload_run_lengths=[run["length"] for run in payload_runs],
                                     payload_runs=payload_runs, payload_compressed_edit_distance=distance))
        if not payloads:
            result["reason"] = "no_right_G_anchor_at_primer"
            return result
        score = min(p["score"] for p in payloads)
        best = [p for p in payloads if p["score"] == score]
        coordinates = {(p["left"]["end"], p["right"]["start"]) for p in best}
        if len(coordinates) != 1:
            result["reason"] = "payload_endpoint_tie"
            return result
        selected = best[0]
        bases = selected["payload_run_bases"]
        pure = bases == "ACT" * (len(bases) // 3) and bool(bases)
        result.update(payload_resolved=True, reason="resolved",
                      address_end=selected["left"]["start"],
                      address_pattern_observed=selected["address"]["address_pattern_observed"],
                      payload_start=selected["left"]["end"], payload_end=selected["right"]["start"],
                      payload_run_bases=bases, payload_run_lengths=selected["payload_run_lengths"],
                      payload_runs=selected["payload_runs"],
                      payload_compressed_edit_distance=selected["payload_compressed_edit_distance"],
                      pure_ACT=bool(pure), cell_count=len(bases) // 3 if pure else None,
                      direct_ACT_triplets=sum(bases[i:i+3] == "ACT" for i in range(len(bases)-2)),
                      complete_ACT8=bases == "ACT" * 8, any_ACT="ACT" in bases)
        return result


def write_tsv(filename, rows):
    if not rows:
        raise ValueError(filename)
    with (OUT / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_reference():
    with (INPUT / "read_audit.tsv").open() as handle:
        audit = list(csv.DictReader(handle, delimiter="\t"))
    references = []
    frozen_h4 = []
    primer_threshold = Counter()
    with gzip.open(INPUT / "read_evidence.jsonl.gz", "rt") as handle:
        for ix, line in enumerate(handle):
            evidence = json.loads(line)
            fits = evidence["primer_fits"]
            within_five = any(f["left_edits"] <= 5 and f["right_edits"] <= 5
                              for f in fits.values())
            native_supported = audit[ix]["native_orientation_supported"] == "True"
            primer_threshold["either_orientation_both_primers_within_five"] += within_five
            primer_threshold["native_supported_and_both_primers_within_five"] += native_supported and within_five
            if native_supported:
                fit = fits[audit[ix]["orientation"]]
                maximum = max(fit["left_edits"], fit["right_edits"])
                primer_threshold[f"native_supported_max_primer_edits_{maximum}"] += 1
            addresses = evidence["supported_addresses"]
            resolved = [(a, a["payload_evidence"]["resolved_payload"]) for a in addresses
                        if a["payload_evidence"].get("resolved_payload")]
            ref = dict(orientation_supported=audit[ix]["native_orientation_supported"] == "True",
                       orientation=audit[ix]["orientation"], address_supported=bool(addresses),
                       payload_resolved=bool(resolved),
                       complete_ACT8=any(p["complete_ACT8"] for a, p in resolved),
                       any_ACT=any("ACT" in p["payload_run_bases"] for a, p in resolved),
                       address_candidates=[dict(identity=a["identity"], orientation=a["orientation"],
                                                address_start=a["address_start"], address_end=a["address_end"])
                                           for a in addresses],
                       payload_candidates=[dict(identity=a["identity"], orientation=a["orientation"],
                                                payload_start=p["payload_start"], payload_end=p["payload_end"],
                                                payload_run_bases=p["payload_run_bases"],
                                                payload_run_lengths=p["payload_run_counts"])
                                           for a, p in resolved])
            references.append(ref)
            for a, p in resolved:
                bases = p["payload_run_bases"]
                if bases in ("ACT" * 7, "ACT" * 9):
                    frozen_h4.append(dict(record_index=ix, identity=a["identity"],
                                          raw_parent_id=evidence["raw_parent_id"],
                                          orientation=a["orientation"], cell_count=len(bases) // 3,
                                          payload_run_lengths=p["payload_run_counts"]))
    return references, frozen_h4, dict(primer_threshold)


def design_cells():
    designs = {}
    for line in (INPUT / "Lovelace_S12_480.fasta").read_text().splitlines():
        if line.startswith(">"):
            identity = int(line.rsplit("_", 1)[1])
        else:
            designs[identity] = np.array([run["length"] for run in rle(line[109:269])]).reshape(8, 3)
    return designs


def h4_registration(records, designs, population):
    counts = defaultdict(lambda: np.zeros(5))
    for row in records:
        n = row["cell_count"]
        observed = np.array(row["payload_run_lengths"]).reshape(n, 3)
        truth = designs[row["identity"]]
        for registration in ("5prime", "3prime"):
            for j, triplet in enumerate(observed):
                position = j if registration == "5prime" else j + 8 - n
                if not 0 <= position < 8:
                    continue
                dist = np.sum((STATES - triplet) ** 2, axis=1)
                nearest = np.flatnonzero(dist == dist.min())
                correct = sum(np.array_equal(STATES[s], truth[position]) for s in nearest) / len(nearest)
                values = np.array([1, np.array_equal(triplet, truth[position]), 1 - correct,
                                   triplet[0] != truth[position, 0] and triplet[0] in (3, 7),
                                   len(nearest) > 1], dtype=float)
                for orientation in ("all", row["orientation"]):
                    counts[(n, registration, position, orientation)] += values
    return [dict(cohort=COHORT, population=population, cell_count=n, registration=reg,
                 cell_position=pos, orientation=ori, denominator=int(v[0]),
                 exact_cells=int(v[1]), exact_fraction=v[1]/v[0],
                 nearest_state_error_count=v[2], nearest_state_error_fraction=v[2]/v[0],
                 A_legal_run_swap_count=int(v[3]), A_legal_run_swap_fraction=v[3]/v[0],
                 nearest_state_tie_count=int(v[4]))
            for (n, reg, pos, ori), v in sorted(counts.items())]


def h4_paired(records, designs, population, replicates=1000):
    rows = []
    for row in records:
        n = row["cell_count"]
        observed = np.array(row["payload_run_lengths"]).reshape(n, 3)
        truth = designs[row["identity"]]
        score = {}
        for registration in ("5prime", "3prime"):
            exact, error, denominator = 0, 0.0, 0
            for j, triplet in enumerate(observed):
                pos = j if registration == "5prime" else j + 8 - n
                if not 0 <= pos < 8:
                    continue
                distances = np.sum((STATES - triplet) ** 2, axis=1)
                nearest = np.flatnonzero(distances == distances.min())
                exact += np.array_equal(triplet, truth[pos])
                error += 1 - sum(np.array_equal(STATES[s], truth[pos]) for s in nearest) / len(nearest)
                denominator += 1
            score[registration] = (exact / denominator, error / denominator, denominator)
        rows.append(dict(cohort=COHORT, population=population, record_index=row["record_index"],
                         raw_parent_id=row["raw_parent_id"], identity=row["identity"],
                         orientation=row["orientation"], cell_count=n,
                         denominator_per_registration=score["5prime"][2],
                         exact_fraction_5prime=score["5prime"][0], exact_fraction_3prime=score["3prime"][0],
                         nearest_state_error_5prime=score["5prime"][1],
                         nearest_state_error_3prime=score["3prime"][1],
                         exact_difference_3prime_minus_5prime=score["3prime"][0]-score["5prime"][0],
                         state_error_difference_3prime_minus_5prime=score["3prime"][1]-score["5prime"][1]))
    summaries = []
    rng = np.random.default_rng(983214)
    for n in (7, 9, "7_or_9"):
        for orientation in ("all", "forward", "reverse_complement"):
            selected = [r for r in rows if (n == "7_or_9" or r["cell_count"] == n)
                        and (orientation == "all" or r["orientation"] == orientation)]
            clusters = defaultdict(lambda: np.zeros(3))
            for r in selected:
                clusters[r["raw_parent_id"]] += (1, r["exact_difference_3prime_minus_5prime"],
                                                 r["state_error_difference_3prime_minus_5prime"])
            values = np.array(list(clusters.values()))
            if len(values) == 0:
                continue
            draws = rng.integers(len(values), size=(replicates, len(values)))
            totals = values[draws].sum(axis=1)
            ci = np.percentile(totals[:, 1:] / totals[:, :1], [2.5, 97.5], axis=0)
            summaries.append(dict(cohort=COHORT, population=population, cell_count=n, orientation=orientation,
                                  candidate_count=len(selected), distinct_raw_parents=len(values),
                                  exact_fraction_5prime=np.mean([r["exact_fraction_5prime"] for r in selected]),
                                  exact_fraction_3prime=np.mean([r["exact_fraction_3prime"] for r in selected]),
                                  nearest_state_error_5prime=np.mean([r["nearest_state_error_5prime"] for r in selected]),
                                  nearest_state_error_3prime=np.mean([r["nearest_state_error_3prime"] for r in selected]),
                                  exact_difference_3prime_minus_5prime=np.mean([r["exact_difference_3prime_minus_5prime"] for r in selected]),
                                  exact_difference_ci_low=ci[0, 0], exact_difference_ci_high=ci[1, 0],
                                  state_error_difference_3prime_minus_5prime=np.mean([r["state_error_difference_3prime_minus_5prime"] for r in selected]),
                                  state_error_difference_ci_low=ci[0, 1], state_error_difference_ci_high=ci[1, 1],
                                  fraction_3prime_better_exact=np.mean([r["exact_difference_3prime_minus_5prime"] > 0 for r in selected]),
                                  fraction_3prime_better_state_error=np.mean([r["state_error_difference_3prime_minus_5prime"] < 0 for r in selected]),
                                  fraction_5prime_better_state_error=np.mean([r["state_error_difference_3prime_minus_5prime"] > 0 for r in selected]),
                                  bootstrap_replicates=replicates,
                                  bootstrap_cluster="raw_parent_id", nearest_state_tie_rule="fractional equal credit"))
    return rows, summaries


def run(limit=None):
    with MANIFEST.open() as handle:
        manifest = list(csv.DictReader(handle))
    patterns = [row["address_pattern"] for row in manifest]
    parser = Parser(patterns)
    references, frozen_h4, frozen_threshold = load_frozen_reference()
    results = []
    parents = set()
    with gzip.open(INPUT / "001_ABC_reads.fastq.gz", "rt") as handle, \
         gzip.open(OUT / "parser_records.tsv.gz", "wt", newline="") as output:
        ix = 0
        while True:
            header = handle.readline().rstrip()
            if not header:
                break
            sequence = handle.readline().strip()
            plus, quality = handle.readline(), handle.readline().strip()
            if not plus.startswith("+") or len(sequence) != len(quality):
                raise ValueError(f"invalid FASTQ record {ix}")
            if limit is not None and ix >= limit:
                break
            read_id = header.split()[0][1:]
            pi = re.search(r"(?:\s)pi:Z:(\S+)", header)
            parent = pi.group(1) if pi else read_id
            result = parser.parse(sequence)
            result.update(record_index=ix, read_id=read_id, raw_parent_id=parent,
                          raw_read_length=len(sequence))
            parents.add(parent)
            if ix == 0:
                parser_fields = list(result)
            append_table_rows(output, [result], parser_fields, header=(ix == 0))
            results.append(result)
            ix += 1
            if ix % 10000 == 0:
                print(f"parsed {ix} NEW_ABC_ONLY records", flush=True)
    n = len(results)
    funnel = []
    for key in ("orientation_supported", "address_supported", "payload_resolved", "any_ACT", "complete_ACT8"):
        matrix = Counter((r[key], ref[key]) for r, ref in zip(results, references))
        tp, fp, fn, tn = (matrix[(True, True)], matrix[(True, False)],
                          matrix[(False, True)], matrix[(False, False)])
        funnel.append(dict(cohort=COHORT, step=key, independent_count=tp+fp,
                           frozen_count=tp+fn, denominator=n, true_positive=tp,
                           false_positive=fp, false_negative=fn, true_negative=tn,
                           agreement_fraction=(tp+tn)/n, precision=tp/(tp+fp) if tp+fp else 0,
                           recall=tp/(tp+fn) if tp+fn else 0))
    write_tsv("parser_funnel.tsv", funnel)
    agreements = []
    for kind, flag, fields in (("orientation", "orientation_supported", ("orientation",)),
                               ("address", "address_supported", ("identity", "orientation")),
                               ("address_bounds", "address_supported", ("identity", "orientation", "address_start", "address_end")),
                               ("payload", "payload_resolved", ("identity", "orientation", "payload_start", "payload_end")),
                               ("payload_runs", "payload_resolved", ("identity", "orientation", "payload_run_bases", "payload_run_lengths"))):
        overlap = match = 0
        for result, ref in zip(results, references):
            if not result[flag] or not ref[flag]:
                continue
            overlap += 1
            candidates = [ref] if kind == "orientation" else ref[
                "address_candidates" if kind.startswith("address") else "payload_candidates"]
            match += any(all(result.get(field) == c.get(field) for field in fields) for c in candidates)
        agreements.append(dict(cohort=COHORT, comparison=kind, overlapping_records=overlap,
                               matched_records=match, mismatched_records=overlap-match,
                               match_fraction=match/overlap if overlap else 0))
    write_tsv("parser_agreement.tsv", agreements)
    sensitivity = [dict(cohort=COHORT, address_edit_threshold=t,
                        supported_records=sum(r["orientation_supported"] and r["address_nearest_unique"]
                                              and r["address_distance"] <= t for r in results),
                        denominator=n) for t in range(4)]
    write_tsv("parser_address_sensitivity.tsv", sensitivity)
    strict_rows = []
    for unit in iter_table(INPUT / "F_complete_new_units.tsv"):
        ix = int(re.search(r"F_NEW_ABC_(\d+):", unit["unit_id"]).group(1))
        if ix >= n:
            continue
        result = results[ix]
        run_match = (result.get("payload_run_bases") == "ACT" * 8 and
                     result.get("payload_run_lengths") == [v for c in unit["observed_direct"] for v in c])
        strict_rows.append(dict(cohort=COHORT, unit_id=unit["unit_id"], record_index=ix,
                                identity=unit["identity"], role=unit["role"],
                                independent_orientation_supported=result["orientation_supported"],
                                orientation_match=result.get("orientation") == unit["orientation"],
                                independent_address_supported=result["address_supported"],
                                identity_match=result.get("identity") == unit["identity"],
                                independent_payload_resolved=result["payload_resolved"],
                                bounds_match=result.get("payload_start") == unit["oriented_payload_start"]
                                and result.get("payload_end") == unit["oriented_payload_end"],
                                payload_runs_match=run_match))
    write_tsv("parser_strict_agreement.tsv", strict_rows)
    h4_independent = [r for r in results if r["payload_resolved"]
                      and r.get("cell_count") in (7, 9)]
    designs = design_cells()
    h4_rows = h4_registration(h4_independent, designs, "independent_parser")
    if limit is not None:
        frozen_h4 = [r for r in frozen_h4 if r["record_index"] < n]
    h4_rows += h4_registration(frozen_h4, designs, "frozen_endpoint_sensitivity")
    write_tsv("h4_registration.tsv", h4_rows)
    paired, paired_summary = h4_paired(h4_independent, designs, "independent_parser")
    frozen_paired, frozen_paired_summary = h4_paired(frozen_h4, designs, "frozen_endpoint_sensitivity")
    write_tsv("h4_pairs.tsv", paired + frozen_paired)
    write_tsv("h4_paired_summary.tsv", paired_summary + frozen_paired_summary)
    h4_summary = []
    for population, resolved_count, affected, records in (
            ("independent_parser", sum(r["payload_resolved"] for r in results),
             len(h4_independent), h4_independent),
            ("frozen_endpoint_sensitivity", sum(r["payload_resolved"] for r in references[:n]),
             len({r["record_index"] for r in frozen_h4}), frozen_h4)):
        for cell_count in (7, 9, "7_or_9"):
            subset = [r for r in records if cell_count == "7_or_9" or r["cell_count"] == cell_count]
            count = len({r["record_index"] for r in subset})
            h4_summary.append(dict(cohort=COHORT, population=population, cell_count=cell_count,
                                   affected_record_count=count, affected_candidate_count=len(subset),
                                   payload_resolved_denominator=resolved_count,
                                   affected_fraction=count/resolved_count if resolved_count else 0))
    write_tsv("h4_population.tsv", h4_summary)
    scope_rows = []
    for population, resolved in (("independent_parser", [(r["record_index"], [r]) for r in results if r["payload_resolved"]]),
                                 ("frozen_endpoint_sensitivity", [(i, r["payload_candidates"]) for i, r in enumerate(references[:n]) if r["payload_resolved"]])):
        for triplets in (7, 9, "7_or_9"):
            pure_ids, interrupted_ids = set(), set()
            for ix, candidates in resolved:
                for candidate in candidates:
                    bases = candidate["payload_run_bases"]
                    count = sum(bases[i:i+3] == "ACT" for i in range(len(bases)-2))
                    if count not in (7, 9) or triplets != "7_or_9" and count != triplets:
                        continue
                    if bases == "ACT" * count:
                        pure_ids.add(ix)
                    else:
                        interrupted_ids.add(ix)
            scope_rows.append(dict(cohort=COHORT, population=population,
                                   direct_ACT_triplet_count=triplets, pure_records=len(pure_ids),
                                   interrupted_records=len(interrupted_ids),
                                   union_records=len(pure_ids | interrupted_ids),
                                   payload_resolved_denominator=len(resolved),
                                   union_fraction=len(pure_ids | interrupted_ids)/len(resolved),
                                   interpretation="interrupted ACT triplets are counted without assigning a complete cell frame"))
    write_tsv("h4_scope.tsv", scope_rows)
    broad = Counter((r.get("pure_ACT", False), r.get("cell_count"), r.get("direct_ACT_triplets"))
                    for r in results if r["payload_resolved"])
    write_tsv("parser_payload_structure.tsv", [dict(cohort=COHORT, pure_ACT=pure,
              complete_cell_count=cell, direct_ACT_triplet_count=triplet, records=count)
              for (pure, cell, triplet), count in sorted(broad.items(), key=lambda x: str(x[0]))])
    summary = dict(cohort=COHORT, raw_reads=n, unique_raw_parents=len(parents),
                   primer_edit_threshold=5, address_edit_threshold=2,
                   address_sensitivity_thresholds=[0, 1, 2, 3],
                   reasons=dict(Counter(r["reason"] for r in results)),
                   strict_units_compared=len(strict_rows),
                   frozen_primer_threshold_comparison=frozen_threshold,
                   strict_matches={k:sum(r[k] for r in strict_rows) for k in strict_rows[0]
                                   if k.endswith("match") or k.startswith("independent_")},
                   assumptions=["both full primers aligned with edlib HW and at most five edits each",
                                "orientation chosen by minimum total primer edits, equal orientation scores rejected",
                                "unique nearest compressed manifest address with at most two edits",
                                "address windows of 13 to 19 compressed runs permit sensitivity through three edits",
                                "G delimiter may have any positive length",
                                "right G delimiter within five bases of fitted right primer, or two bases of overlap",
                                "payload anchors minimize compressed ACT8 edit distance, primer gap, then G length departure",
                                "equal endpoint scores at distinct coordinates rejected",
                                "all raw payload runs retained without forcing twenty four runs"],
                   no_decoder_refit=True)
    write_output(OUT / "parser_summary.tsv", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--limit", type=int)
    run(arg.parse_args().limit)
