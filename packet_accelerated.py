"""compiled public extraction and vectorized equivalent likelihood aggregation."""

from collections import Counter
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent / "analysis/scaling_ecc/sequence"
LIBRARY = HERE / ("accelerated.dylib" if sys.platform == "darwin" else "accelerated.so")
STATUS = ("primer_failure", "length_failure", "marker_failure", "address_failure", "extraction_failure", "eligible")


def build():
    command = ["clang++", "-std=c++17", "-O3", "-shared", "-fPIC", str(HERE / "accelerated.cpp"), "-o", str(LIBRARY)]
    subprocess.run(command, check=True, capture_output=True)
    return command


@lru_cache(None)
def library():
    native = ctypes.CDLL(str(LIBRARY))
    native.decode_batch.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    native.decode_batch.restype = None
    return native


def decode_observations(codec, reads):
    records = [str(s).encode("ascii") for s in reads]
    offsets = np.concatenate(([0], np.cumsum([len(s) for s in records], dtype=np.uint64))).astype(np.uint64)
    blob = b"".join(records)
    count = len(records)
    identities = np.full(count, -1, np.int32)
    observations = np.zeros((count, 48), np.uint8)
    statuses = np.zeros(count, np.uint8)
    mode = 2
    library().decode_batch(blob, offsets.ctypes.data, count, mode, codec.address_runs,
                           codec.packet_bytes, identities.ctypes.data, observations.ctypes.data, statuses.ctypes.data)
    return identities, observations, statuses


def decode_reads(codec, reads, addresses, group_targets=None):
    from packet_codec import DecodeResult, STATES, _packet_index

    multiplicities = Counter(map(str, reads))
    ids, observed, status = decode_observations(codec, multiplicities)
    weights = np.fromiter(multiplicities.values(), dtype=np.int64)
    index, packet_count = _packet_index(addresses, group_targets)
    positions = np.asarray([index.get(int(identity), -1) for identity in ids], np.int32)
    statuses = {label: int(weights[status == i].sum()) for i, label in enumerate(STATUS) if (status == i).any()}
    unallocated = (ids >= 0) & (positions < 0)
    if unallocated.any():
        statuses["unallocated_address"] = int(weights[unallocated].sum())
    eligible = (status == 5) & (positions >= 0)
    where, counts, obs = positions[eligible], weights[eligible], observed[eligible]
    symbol_count = codec.packet_bytes * 4
    state_count = 4
    scores = np.zeros((packet_count, symbol_count, state_count), np.float64)
    coverage = np.zeros(packet_count, np.int32)
    np.add.at(coverage, where, counts)
    cells = obs[:, :symbol_count * 3].reshape(-1, symbol_count, 3)
    values = np.zeros((len(where), symbol_count, 4), np.float64)
    for base in range(3):
        values += codec.model[base, STATES[:, base] - 3][:, cells[:, :, base]].transpose(1, 2, 0)
    values *= counts[:, None, None]
    np.add.at(scores, where, values)
    top = scores.max(axis=-1)
    unresolved = (scores == top[..., None]).sum(axis=-1) != 1
    unresolved |= coverage[:, None] == 0
    symbols = scores.argmax(axis=-1).astype(np.uint8)
    bits_per_symbol = 2
    shifts = np.arange(bits_per_symbol - 1, -1, -1)
    binary = (symbols[:, :, None] >> shifts) & 1
    packets = np.packbits(binary.reshape(packet_count, codec.packet_bytes * 8), axis=1, bitorder="big")
    erased = unresolved.reshape(packet_count, codec.packet_bytes, 8 // bits_per_symbol).any(axis=2)
    return DecodeResult(packets, erased, coverage, statuses)


if __name__ == "__main__":
    print(" ".join(build()))
