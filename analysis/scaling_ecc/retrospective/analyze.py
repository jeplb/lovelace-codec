"""replay native resolved values and erasures with newly computed parity."""
import csv
import hashlib
from pathlib import Path
import numpy as np
from ..outer.core import make_library, decode_library, score_library
from tsv_records import write_output
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"

def table(path, delimiter=','):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))

def write_table(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def stripe_rows(codec, population, truth, values, width):
    padded_length = ((len(truth) + 32 * width - 1) // (32 * width)) * 32 * width
    padded = truth.ljust(padded_length, b'\0')
    estimates = values + [0] * (padded_length - len(values))
    output = []
    for stripe in range(padded_length // (32 * width)):
        for column in range(width):
            positions = [stripe * 32 * width + slot * width + column for slot in range(32)]
            erased = sum(estimates[i] is None for i in positions)
            errors = sum(estimates[i] is not None and estimates[i] != padded[i] for i in positions)
            output.append(dict(codec=codec, population=population, packet_bytes=width,
                stripe=stripe, byte_column=column, unknown_errors=errors, declared_erasures=erased,
                correction_requirement=2 * errors + erased, data_packets=32,
                source_bytes=len(truth), padding_bytes=padded_length - len(truth)))
    return output

def replay(codec, population, truth, values, width):
    output = []
    for parity in (0, 4, 8, 16, 24):
        lib = make_library(truth, codec, width, parity)
        offset = len(lib.envelope) - len(truth)
        for draw in range(20):
            rng = np.random.default_rng(10102026 + draw + 100 * parity)
            estimates = []
            for packet in lib.packets:
                if packet['is_parity']:
                    estimate = list(packet['bytes'])
                    newly_synthesized = True
                else:
                    address = packet['stripe'] * 32 + packet['slot']
                    start = address * width
                    estimate = [values[i - offset] if offset <= i < offset + len(truth) else b
                                for i, b in enumerate(packet['bytes'], start)]
                    newly_synthesized = start < offset
                if newly_synthesized:
                    if rng.random() < .05:
                        estimate = [None] * width
                    else:
                        estimate = [int(v) ^ int(rng.integers(1, 256)) if v is not None and rng.random() < .01 else v
                                    for v in estimate]
                estimates.append(estimate)
            decoded = decode_library(estimates, codec, parity, width)
            output.append(dict(codec=codec, population=population, parity_packets=parity, realization=draw,
                computational_scenario=True, new_parity_packet_dropout=.05, new_parity_unknown_byte_error=.01,
                **score_library(lib, decoded)))
    return output

def lovelace_residual(image, method):
    truth = (INPUTS / f"designed_{image}.bin").read_bytes()
    stream = (INPUTS / f"{method}_{image}.bin").read_bytes()
    erased_bits = np.zeros(len(stream) * 8, bool)
    for row in table(INPUTS / "labels.tsv", "\t"):
        if row["image"] != image or row["arm"] != method or row["resolved"] == "True":
            continue
        if row["kind"] == "payload":
            start, stop = 64 * 8 + int(row["scalar_index"]) * 6, 64 * 8 + (int(row["scalar_index"]) + 1) * 6
        else:
            local_address = int(row["address"]) - 224 * "AB".index(image)
            start = 16 * local_address + 2 * int(row["cell"])
            stop = start + 2
        erased_bits[start:stop] = True
    erased = erased_bits.reshape(-1, 8).any(axis=1)
    return truth, [None if erased[i] else value for i, value in enumerate(stream)]


def main():
    stripe, mapping, replay_rows = [], [], []
    for image in "AB":
        for method in ("control", "position", "position_weighted"):
            truth, values = lovelace_residual(image, method)
            population = f"image {image} {method}"
            stripe.extend(stripe_rows("lovelace", population, truth, values, 2))
            replay_rows.extend(replay("lovelace", population, truth, values, 2))
            mapping.extend(dict(codec="lovelace", population=population, native_byte=i, native_strand=i // 2, byte_column=i % 2, stripe=(i // 2) // 32, stripe_slot=(i // 2) % 32, declared_erasure=value is None, unknown_error=value is not None and value != truth[i]) for i, value in enumerate(values))
    write_table(HERE / "stripe_residuals.csv", stripe)
    write_table(HERE / "native_byte_mapping.csv", mapping)
    write_table(HERE / "computational_replay_realizations.csv", replay_rows)
    summary = dict(native_streams=6, original_lovelace_checksum_passes=0, no_oracle_erasure=True, computational_replay_realizations=len(replay_rows), max_requirement_by_population={population:max(row["correction_requirement"] for row in stripe if row["population"] == population) for population in sorted({row["population"] for row in stripe})})
    write_output(HERE / "summary.tsv", summary)


if __name__ == "__main__":
    main()
