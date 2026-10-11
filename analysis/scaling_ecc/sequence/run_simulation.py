"""molecular simulations and explicit synthesis and read-budget comparisons."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from ..outer.core import K, PARITIES, HEADER, make_library, decode_library, score_library
from .codecs import Codec
from tsv_records import read_output, write_output
from .payloads import load_frozen_payloads, object_metrics
from .noise import mutate_many, allocate_reads, conditions

HERE = Path(__file__).resolve().parent


def seed(*parts):
    message = "|".join(map(str, (2026101002,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def inputs():
    values = load_frozen_payloads()
    for row in read_output(HERE / "payloads/random_payload_manifest.tsv"):
        payload = (HERE / "payloads" / row["payload_file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        values.append(dict(codec=row["codec"], name=row["name"], payload=payload,
                           metadata=row["metadata"], sources=[]))
    return values


def physical_library(obj, codec, arm, r):
    lib = make_library(obj["payload"], obj["codec"], codec.packet_bytes,
                       r if arm == "RS" else 0, obj["metadata"])
    packets = [dict(p) for p in lib.packets]
    targets = list(range(len(packets)))
    if arm == "repetition":
        for stripe in range(lib.stripes):
            for j in range(r):
                slot = (j + stripe * r) % K
                source = stripe * K + slot
                packets.append(dict(lib.packets[source], identity=len(packets), is_copy=True))
                targets.append(source)
    sequences = [codec.encode(p["identity"], p["bytes"]) for p in packets]
    assert max(map(len, sequences)) <= 300
    return lib, packets, targets, sequences


def export_library(stem, lib, packets, targets, sequences, codec):
    folder = HERE / "libraries"
    folder.mkdir(exist_ok=True)
    (folder / (stem + ".fasta")).write_text("".join(">packet_{} stripe_{} slot_{} role_{}\n{}\n".format(
        p["identity"], p["stripe"], p["slot"], "copy" if p.get("is_copy") else "parity" if p["is_parity"] else "data", seq)
        for p, seq in zip(packets, sequences)))
    with (folder / (stem + "_mapping.csv")).open("w") as handle:
        fields = ["identity", "stripe", "slot", "role", "logical_packet", "byte_hex", "strand_nt", "payload_nt", "address_nt", "primers_nt", "marker_nt", "padding_nt"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p, target, seq in zip(packets, targets, sequences):
            alloc = codec.allocation(p["bytes"])
            writer.writerow(dict(identity=p["identity"], stripe=p["stripe"], slot=p["slot"],
                                 role="copy" if p.get("is_copy") else "parity" if p["is_parity"] else "data",
                                 logical_packet=target, byte_hex=p["bytes"].hex(), strand_nt=len(seq),
                                 payload_nt=alloc["data_nt"], address_nt=alloc["address_nt"], primers_nt=alloc["primers_nt"],
                                 marker_nt=alloc["marker_nt"], padding_nt=alloc["padding_nt"]))
    for role, predicate in (("data", lambda packet: not packet["is_parity"]), ("parity", lambda packet: packet["is_parity"])):
        (folder / (stem + "_" + role + ".fasta")).write_text("".join(f">packet_{packet['identity']}\n{sequence}\n" for packet, sequence in zip(packets, sequences) if predicate(packet)))
    meta_length = len(lib.envelope) - HEADER.size - len(lib.payload)
    mapping = []
    for i in range(len(lib.data) * lib.width):
        kind = "header" if i < HEADER.size else "metadata" if i < HEADER.size + meta_length else "payload" if i < len(lib.envelope) else "padding"
        mapping.append(dict(stream_byte=i, stripe=i // (K * lib.width), slot=(i // lib.width) % K,
                            byte_column=i % lib.width, region=kind,
                            payload_byte=i - HEADER.size - meta_length if kind == "payload" else ""))
    with (folder / (stem + "_bytes.csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)


def packet_estimates(result):
    return [[None if e else int(v) for v, e in zip(p, erased)]
            for p, erased in zip(result.packets, result.erased)]


def simulate(obj, codec, lib, packets, targets, sequences, arm, r, condition, replicate, mode, reference_count):
    cname, psub, pindel, pdrop, shared = condition
    realization_seed = seed(obj["codec"], obj["name"], cname, psub, pindel, pdrop, replicate, mode)
    rng = np.random.default_rng(realization_seed)
    allocated = allocate_reads(len(sequences), reference_count, mode, rng)
    dropped = rng.random(len(sequences)) < pdrop
    reads = []
    for seq, count, dropout in zip(sequences, allocated, dropped):
        if dropout:
            continue
        if shared:
            template = mutate_many(seq, 1, psub, 0, rng)[0]
            reads.extend([template] * int(count))
        else:
            reads.extend(mutate_many(seq, int(count), psub, pindel, rng))
    # pooling sees sequence strings only, with a public allocated address list
    codec._cache.clear()
    result = codec.decode_reads(reads, list(range(len(sequences))),
                                group_targets=targets if arm == "repetition" else None)
    estimates = packet_estimates(result)
    decoded = decode_library(estimates, lib.codec, lib.r, lib.width)
    score = score_library(lib, decoded)
    offset = len(lib.envelope) - len(lib.payload)
    values = decoded["values"][offset:offset + len(lib.payload)]
    metrics = object_metrics(obj["codec"], values, decoded["metadata"], obj["payload"], obj["metadata"])
    req = 0
    for start in range(0, len(estimates), K + lib.r):
        for col in range(lib.width):
            errors = erasures = 0
            for j, p in enumerate(lib.packets[start:start + K + lib.r]):
                value = estimates[start + j][col]
                erasures += value is None
                errors += value is not None and value != p["bytes"][col]
            req = max(req, 2 * errors + erasures)
    parity_positions = [i for i, p in enumerate(packets) if p["is_parity"]]
    score.update(codec=obj["codec"], payload=obj["name"], arm=arm, r=r, condition=cname,
                 level=pdrop if cname == "dropout" else pindel if cname == "indel" else psub,
                 replicate=replicate, budget=mode, seed=realization_seed,
                 physical_strands=len(sequences), designed_nt=sum(map(len, sequences)),
                 requested_reads=int(allocated.sum()), generated_reads=len(reads),
                 sequenced_nt=sum(map(len, reads)), dropped_strands=int(dropped.sum()),
                 parity_strands=len(parity_positions), parity_nt=sum(len(sequences[i]) for i in parity_positions),
                 max_column_requirement=req, assigned_reads=int(result.coverage.sum()),
                 unassigned_reads=len(reads) - int(result.coverage.sum()))
    score.update(metrics)
    return score, np.asarray([-1 if v is None else v for v in values], dtype=np.int16), result.statuses


def run(selected="lovelace", refresh_repetition=False):
    rows, outputs, diagnostics = [], [], []
    suffix = "_" + selected if selected else ""
    prior = HERE / ("realizations" + suffix + ".csv")
    if prior.exists():
        import pandas as pd
        rows = pd.read_csv(prior).to_dict("records")
        saved = np.load(HERE / ("predictions" + suffix + ".npz"))
        outputs = [saved["values"][start:stop].copy() for start, stop in zip(saved["offsets"][:-1], saved["offsets"][1:])]
        diagnostics = read_output(HERE / ("decoder_diagnostics" + suffix + ".tsv"))
        assert len(rows) == len(outputs) == len(diagnostics)
        for row in rows:
            row["exact_payload"] = row.get("exact_payload", row["exact_object"])
            row["exact_object"] = bool(row["exact_payload"] and row["metadata_recovered"])
        if refresh_repetition:
            archive = HERE / "preliminary_byte_vote"
            archive.mkdir(exist_ok=True)
            for name in ("realizations" + suffix + ".csv", "predictions" + suffix + ".npz", "decoder_diagnostics" + suffix + ".tsv"):
                shutil.copy2(HERE / name, archive / name)
            keep = [i for i, row in enumerate(rows) if row["arm"] != "repetition"]
            rows, outputs, diagnostics = ([collection[i] for i in keep] for collection in (rows, outputs, diagnostics))
    completed = {(row["codec"], row["payload"], row["arm"], row["r"], row["budget"], row["condition"], row["level"])
                 for row in rows}
    written_rows = len(rows)
    all_inputs = [obj for obj in inputs() if selected is None or obj["codec"] == selected]
    for obj in all_inputs:
        codec = Codec(obj["codec"])
        reference = make_library(obj["payload"], obj["codec"], codec.packet_bytes, 0, obj["metadata"])
        for r in PARITIES:
            for arm in (("unprotected",) if r == 0 else ("RS", "repetition")):
                lib, packets, targets, sequences = physical_library(obj, codec, arm, r)
                stem = "{}_{}_{}_r{}".format(obj["codec"], obj["name"], arm, r)
                export_library(stem, lib, packets, targets, sequences, codec)
                # check actual sequence readout before corrupting any proposed library
                clean = codec.decode_reads(sequences, list(range(len(sequences))),
                                            group_targets=targets if arm == "repetition" else None)
                estimates = packet_estimates(clean)
                assert score_library(lib, decode_library(estimates, lib.codec, lib.r, lib.width))["exact_object"], stem
                for mode in ("per_strand", "total"):
                    for condition in conditions():
                        cname, psub, pindel, pdrop, shared = condition
                        level = pdrop if cname == "dropout" else pindel if cname == "indel" else psub
                        key = (obj["codec"], obj["name"], arm, r, mode, cname, level)
                        if key in completed:
                            assert sum((row["codec"], row["payload"], row["arm"], row["r"], row["budget"], row["condition"], row["level"]) == key for row in rows) == 20
                            continue
                        start = time.perf_counter()
                        for replicate in range(20):
                            score, values, statuses = simulate(obj, codec, lib, packets, targets, sequences, arm, r, condition,
                                                               replicate, mode, len(reference.packets))
                            rows.append(score)
                            outputs.append(values)
                            diagnostics.append(statuses)
                        print(json.dumps(dict(payload=obj["name"], codec=obj["codec"], arm=arm, r=r, budget=mode,
                                              condition=condition[0], level=rows[-1]["level"],
                                              exact=sum(x["exact_object"] for x in rows[-20:]),
                                              seconds=round(time.perf_counter() - start, 3))), flush=True)
                    if len(rows) > written_rows:
                        write_outputs(rows, outputs, diagnostics, selected)
                        written_rows = len(rows)


def write_outputs(rows, outputs, diagnostics, selected):
    suffix = "_" + selected if selected else ""
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (HERE / ("realizations" + suffix + ".csv")).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    # flatten variable-length objects and keep row offsets
    lengths = np.asarray([len(x) for x in outputs], dtype=np.int64)
    np.savez_compressed(HERE / ("predictions" + suffix + ".npz"), values=np.concatenate(outputs),
                        offsets=np.concatenate(([0], np.cumsum(lengths))))
    write_output(HERE / ("decoder_diagnostics" + suffix + ".tsv"), diagnostics)


def summarize():
    import pandas as pd
    files = sorted(HERE.glob("realizations_*.csv"))
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame.to_csv(HERE / "realizations.csv", index=False)
    groups = ["codec", "payload", "arm", "r", "budget", "condition", "level"]
    summary = frame.groupby(groups, dropna=False).agg(trials=("exact_object", "size"),
        successes=("exact_object", "sum"), incorrect_accepted=("incorrect_accepted", "sum"),
        exact_payloads=("exact_payload", "sum"), metadata_recovered=("metadata_recovered", "sum"),
        exact_stripes=("exact_stripes", "sum"), total_stripes=("stripes", "sum"), wrong_bits=("wrong_bits", "sum"),
        declared_failures=("declared_failure", "sum"), missing_bytes=("missing_bytes", "sum"),
        wrong_bytes=("wrong_bytes", "sum"), designed_nt=("designed_nt", "mean"),
        sequenced_nt=("sequenced_nt", "mean"), requested_reads=("requested_reads", "mean"),
        generated_reads=("generated_reads", "mean"), parity_nt=("parity_nt", "mean")).reset_index()
    summary["recovery"] = summary.successes / summary.trials
    summary["ci_low"] = [0 if k == 0 else beta.ppf(.025, k, n-k+1) for k,n in zip(summary.successes,summary.trials)]
    summary["ci_high"] = [1 if k == n else beta.ppf(.975, k+1, n-k) for k,n in zip(summary.successes,summary.trials)]
    summary.to_csv(HERE / "recovery_summary.csv", index=False)
    costs = frame.groupby(["codec", "payload", "arm", "r", "budget"]).agg(
        physical_strands=("physical_strands", "first"), designed_nt=("designed_nt", "first"),
        parity_nt=("parity_nt", "first"), requested_reads=("requested_reads", "first"),
        mean_sequenced_nt=("sequenced_nt", "mean"), mean_generated_reads=("generated_reads", "mean")).reset_index()
    reference = costs[(costs.arm == "unprotected") & (costs.r == 0)][["codec", "payload", "budget", "designed_nt", "requested_reads"]]
    costs = costs.merge(reference, on=["codec", "payload", "budget"], suffixes=("", "_reference"))
    costs["synthesis_ratio"] = costs.designed_nt / costs.designed_nt_reference
    costs["requested_read_ratio"] = costs.requested_reads / costs.requested_reads_reference
    costs["parity_nt_fraction"] = costs.parity_nt / costs.designed_nt
    costs.to_csv(HERE / "physical_resource_costs.csv", index=False)
    print(json.dumps({"realizations": len(frame), "groups": len(summary), "incorrect_accepted": int(frame.incorrect_accepted.sum()),
                      "zero_error_failures": int((~frame[frame.condition == "baseline"].exact_object).sum())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=("lovelace",), default="lovelace")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--refresh-repetition", action="store_true")
    args = parser.parse_args()
    summarize() if args.summarize else run(args.codec, args.refresh_repetition)
