"""controlled outer-code experiments, independent of molecular decoding."""
import csv
import json
import random
from pathlib import Path
from .core import K, PARITIES, encode_stripe, decode_stripe
from tsv_records import write_output

ROOT = Path(__file__).resolve().parent


def run():
    rng = random.Random(2026101003)
    rows = []
    for r in PARITIES:
        mixes = [(e, s) for e in range(r // 2 + 1) for s in range(r - 2 * e + 1)]
        mixes += [(0, r + 1), (r // 2 + 1, 0), (r // 2, 2)]
        for e, s in sorted(set(mixes)):
            for placement in ("data_first", "parity_first", "random"):
                for replicate in range(20):
                    data = [bytes(rng.randrange(256) for _ in range(4)) for _ in range(K)]
                    encoded = encode_stripe(data, r)
                    order = list(range(K + r))
                    if placement == "parity_first":
                        order = order[K:] + order[:K]
                    elif placement == "random":
                        rng.shuffle(order)
                    estimates = [list(p) for p in encoded]
                    for i in order[:e]:
                        estimates[i] = [v ^ rng.randrange(1, 256) for v in estimates[i]]
                    for i in order[e:e + s]:
                        estimates[i] = None
                    recovered, failures = decode_stripe(estimates, r, 4)
                    exact = all(all(v == t for v, t in zip(p, q)) for p, q in zip(recovered, data))
                    rows.append(dict(r=r, errors=e, erasures=s, placement=placement,
                                     replicate=replicate, guaranteed=2 * e + s <= r,
                                     exact=exact, declared_rs_failure=failures > 0,
                                     incorrect_return=not exact and not failures and all(v is not None for p in recovered for v in p),
                                     missing_return=any(v is None for p in recovered for v in p),
                                     corrupted_parity_packets=sum(i >= K for i in order[:e + s])))
                    if 2 * e + s <= r:
                        assert exact and not failures
    # a different valid codeword is an explicit beyond-boundary miscorrection control
    for r in PARITIES[1:]:
        for replicate in range(20):
            data = [bytes(rng.randrange(256) for _ in range(4)) for _ in range(K)]
            original = encode_stripe(data, r)
            altered = list(data)
            altered[0] = bytes(v ^ 0x65 for v in altered[0])
            replacement = encode_stripe(altered, r)
            errors = sum(p != q for p, q in zip(original, replacement))
            recovered, failures = decode_stripe([list(p) for p in replacement], r, 4)
            assert not failures and [bytes(p) for p in recovered] == altered
            rows.append(dict(r=r, errors=errors, erasures=0, placement="valid_codeword_replacement",
                             replicate=replicate, guaranteed=False, exact=False,
                             declared_rs_failure=False, incorrect_return=True, missing_return=False,
                             corrupted_parity_packets=sum(p != q for p, q in zip(original[K:], replacement[K:]))))
    with (ROOT / "packet_realizations.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"trials": len(rows), "guaranteed_trials": sum(x["guaranteed"] for x in rows),
               "guaranteed_failures": sum(x["guaranteed"] and not x["exact"] for x in rows),
               "beyond_boundary_declared_failures": sum(not x["guaranteed"] and x["declared_rs_failure"] for x in rows),
               "beyond_boundary_incorrect_returns": sum(not x["guaranteed"] and x["incorrect_return"] for x in rows)}
    summary["beyond_boundary_missing_returns"] = sum(not x["guaranteed"] and x["missing_return"] for x in rows)
    summary["rs_random_miscorrections"] = sum(x["r"] > 0 and x["placement"] != "valid_codeword_replacement" and x["incorrect_return"] for x in rows)
    summary["rs_constructed_miscorrections"] = sum(x["placement"] == "valid_codeword_replacement" and x["incorrect_return"] for x in rows)
    write_output(ROOT / "packet_summary.tsv", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
