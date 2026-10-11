"""measure native lovelace operations in fresh processes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[key] = "1"
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tsv_records import read_output
from cell_model import cell_log_scores, aggregate
from inference import decode_arm
from analysis.scaling_ecc.outer.core import encode_stripe, decode_stripe


def measure(stage, size, repeat):
    rng = np.random.default_rng(10102026)
    if stage == "lovelace_cell_likelihoods":
        counts = rng.integers(1, 21, (size, 8, 3), dtype=np.int64)
        probabilities = np.full((3, 3, 20), .05)
        fn = lambda: cell_log_scores(counts, probabilities)
    else:
        with np.load(ROOT / "data/prepared/expected_decode/predictions/weight_inputs.npz", allow_pickle=False) as saved:
            available, available_ids = saved["unit_log_scores"], saved["identities"]
        if stage == "lovelace_aggregation":
            scores, ids = np.resize(available, (size, 8, 4)), np.resize(available_ids, size)
            fn = lambda: aggregate(scores, ids, n_addresses=480)
        else:
            config = read_output(ROOT / "codebook/model_config.tsv")
            scores, depth = aggregate(available, available_ids, n_addresses=480)
            fn = lambda: [decode_arm(scores, depth > 0, config, dict(header_mask="v2", coupling=True), config["header_mask"]["proven_bits"]) for _ in range(size)]
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    started = time.perf_counter()
    result = fn()
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    factor = 1 if sys.platform == "darwin" else 1024
    return dict(stage=stage, problem_size=size, repetition=repeat, elapsed_seconds=seconds, peak_resident_bytes=int(after * factor), resident_before_bytes=int(before * factor), traced_operation_peak_bytes=int(peak), thread_count=1, python=sys.version.split()[0], measured=True, output_items=len(result))


def main():
    output = []
    for stage, sizes in (("lovelace_cell_likelihoods", (1024, 4096, 16384)), ("lovelace_aggregation", (1024, 4096, 16384)), ("lovelace_coefficient_inference", (1, 4, 16))):
        for size in sizes:
            for repeat in range(3):
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", stage, "--size", str(size), "--repeat", str(repeat)], capture_output=True, text=True, check=True)
                output.append(json.loads(result.stdout.strip().splitlines()[-1]))
    with (HERE / "repeated_runtime_memory_measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage")
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    if args.stage:
        print(json.dumps(measure(args.stage, args.size, args.repeat)))
    else:
        main()
