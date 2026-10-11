"""public lovelace packet readout without fourier constraints."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np

from stream_format import BASES as ALPHABET, STATES, P1, P2
HERE = Path(__file__).resolve().parent / "analysis/scaling_ecc/sequence"
PRIMERS = {"lovelace": (P1, P2)}


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def transitions(bits, initial="A"):
    state = ALPHABET.index(initial)
    result = []
    for bit in bits:
        state = (state + 1 + int(bit)) % 3
        result.append(ALPHABET[state])
    return "".join(result)


def transition_bits(pattern, initial="A"):
    previous = ALPHABET.index(initial)
    result = []
    for base in pattern:
        value = ALPHABET.index(base)
        step = (value - previous) % 3
        if step == 0:
            return None
        result.append(step - 1)
        previous = value
    return result


def address_dimensions(n):
    p = (n - 1).bit_length()
    parity = tuple(1 << j for j in range(p) if (1 << j) < n)
    data = tuple(i for i in range(1, n) if i not in parity)
    return parity, data


@lru_cache(maxsize=16384)
def address_bits(identity, n=16):
    parity, data = address_dimensions(n)
    if not 0 <= identity < 2 ** len(data):
        raise ValueError("identity exceeds shortened SECDED capacity")
    bits = [0] * (n + 1)
    for shift, position in enumerate(data):
        bits[position] = (identity >> shift) & 1
    for position in parity:
        bits[position] = sum(bits[i] for i in range(1, n) if i & position) % 2
    bits[n] = sum(bits[1:n]) % 2
    # extend the native 16-bit lovelace mask with zeros for longer words
    return tuple(bits[i] ^ ((0xF6CF >> (i - 1)) & 1) for i in range(1, n + 1))


def decode_address(observed):
    n = len(observed)
    bits = [0] + [int(v) ^ ((0xF6CF >> j) & 1) for j, v in enumerate(observed)]
    syndrome = 0
    for i in range(1, n):
        if bits[i]:
            syndrome ^= i
    overall = sum(bits[1:]) % 2
    if syndrome and not overall:
        return None
    if syndrome:
        if syndrome >= n:
            return None
        bits[syndrome] ^= 1
    elif overall:
        bits[n] ^= 1
    _, data = address_dimensions(n)
    return sum(bits[position] << shift for shift, position in enumerate(data))


def runs(sequence):
    # ignore isolated g noise before merging singleton runs.
    observed = []
    for base in sequence:
        if base not in ALPHABET:
            continue
        if observed and observed[-1][0] == base:
            observed[-1] = (base, observed[-1][1] + 1)
        else:
            observed.append((base, 1))
    return merge_noise_runs(observed)


def merge_noise_runs(observed):
    changed = True
    while changed and len(observed) > 2:
        changed = False
        merged = [observed[0]]
        i = 1
        while i < len(observed):
            if i + 1 < len(observed) and observed[i][1] == 1 and merged[-1][0] == observed[i + 1][0]:
                merged[-1] = (merged[-1][0], merged[-1][1] + 1 + observed[i + 1][1])
                i += 2
                changed = True
            else:
                merged.append(observed[i])
                i += 1
        observed = merged
    return observed


def _gtail(payload):
    n = len(payload)
    i = n - 1
    while i >= 0 and payload[i] == "G":
        i -= 1
    if n - i - 1 >= 3:
        return i + 1
    region = payload[-30:]
    for j in range(len(region) - 2):
        if region[j:j + 3] == "GGG":
            return n - len(region) + j
    return -1


def _address_end(payload, observed, n):
    target = sum(length for _, length in observed[:n])
    count = 0
    for i, base in enumerate(payload):
        count += base in ALPHABET
        if count >= target:
            return i + 1
    return len(payload)


def _bytes_from_symbols(symbols, bits_per_symbol):
    vector = np.asarray(symbols, dtype=np.uint8)
    shifts = np.arange(bits_per_symbol - 1, -1, -1)
    binary = (vector[:, None] >> shifts) & 1
    return np.packbits(binary.reshape(-1), bitorder="big")


def _packet_index(addresses, group_targets=None):
    identities = list(map(int, addresses))
    targets = list(range(len(identities))) if group_targets is None else list(map(int, group_targets))
    if len(targets) != len(identities) or any(target < 0 for target in targets):
        raise ValueError("public copy groups must provide one nonnegative target per address")
    return {identity: target for identity, target in zip(identities, targets)}, max(targets, default=-1) + 1


@dataclass
class DecodeResult:
    packets: np.ndarray
    erased: np.ndarray
    coverage: np.ndarray
    statuses: dict

    @property
    def byte_erasures(self):
        return self.erased


class Codec:
    """proposed synthesis and readout, with no ground-truth input to decoding."""

    def __init__(self, name, address_runs=None, packet_bytes=None, accelerated=True):
        self.name = name.lower()
        if self.name not in PRIMERS:
            raise ValueError("unknown block representation")
        self.address_runs = int(address_runs or 16)
        self.packet_bytes = int(packet_bytes or 2)
        self.primers = PRIMERS[self.name]
        self._cache = {}
        self.accelerated = bool(accelerated)
        self.model = None
        self.model = np.log(np.maximum(np.load(HERE / "inputs/lovelace_public_prior.npy", allow_pickle=False), 1e-30))

    @property
    def address_capacity(self):
        return 2 ** len(address_dimensions(self.address_runs)[1])

    def capacity_bytes(self, limit=300):
        return max(0, (int(limit) - 50 - 5 * self.address_runs - 8) // 80)

    def encode(self, identity, payload_bytes):
        data = bytes(payload_bytes)
        if len(data) != self.packet_bytes:
            raise ValueError("packet has the wrong byte width")
        pattern = transitions(address_bits(int(identity), self.address_runs))
        address = "".join(base * 5 for base in pattern)
        binary = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="big")
        digits = binary.reshape(-1, 2) @ np.asarray([2, 1])
        payload = "".join("".join(base * int(length) for base, length in zip(ALPHABET, STATES[d])) for d in digits)
        body = address + "GGGG" + payload + "GGGG"
        return self.primers[0] + body + self.primers[1]

    encode_packet = encode

    def allocation(self, payload_bytes):
        strand = self.encode(0, payload_bytes)
        padding_nt = 0
        primers_nt = sum(map(len, self.primers))
        marker_nt = 8
        return dict(strand_nt=len(strand), primers_nt=primers_nt, address_nt=5 * self.address_runs,
                    marker_nt=marker_nt, padding_nt=padding_nt,
                    data_nt=len(strand) - primers_nt - 5 * self.address_runs - marker_nt - padding_nt)

    def _trim(self, sequence):
        fwd, rev = self.primers
        fcore, rcore = fwd[5:20], rev[5:20]
        for seq in (sequence, reverse_complement(sequence)):
            start = seq[:60].find(fcore)
            if start < 0:
                continue
            payload_start = max(0, start - 5 + len(fwd))
            tail_start = max(0, len(seq) - 60)
            stop = seq[tail_start:].find(rcore)
            payload_end = tail_start + stop - 5 if stop >= 0 else len(seq)
            if payload_end > payload_start:
                return seq[payload_start:payload_end]
        return None

    def decode_read(self, sequence):
        """return assigned address and observations, with no true template identity."""
        cached = self._cache.get(sequence)
        if cached is not None:
            return cached
        body = self._trim(sequence)
        result = (None, None, "primer_failure")
        if body is not None:
            result = self._decode_body(body)
        if len(self._cache) < 65536:
            self._cache[sequence] = result
        return result

    decode = decode_read

    def _decode_body(self, body):
        if not 100 <= len(body) <= 400:
            return None, None, "length_failure"
        marker = re.search("G{3,}", body[:5 * self.address_runs + 30])
        if marker is None:
            return None, None, "marker_failure"
        address_runs = runs(body[:marker.start()])
        address_end = marker.end()
        if len(address_runs) < self.address_runs:
            return None, None, "address_failure"
        pattern = "".join(base for base, _ in address_runs[:self.address_runs])
        observed_bits = transition_bits(pattern)
        identity = decode_address(observed_bits) if observed_bits is not None else None
        if identity is None:
            return None, None, "address_failure"
        tail = _gtail(body)
        if tail <= address_end:
            return identity, None, "extraction_failure"
        payload_runs = runs(body[address_end:tail])
        expected = self.packet_bytes * 12
        if len(payload_runs) != expected or "".join(base for base, _ in payload_runs) != "ACT" * (self.packet_bytes * 4):
            return identity, None, "extraction_failure"
        return identity, tuple(min(length, 20) for _, length in payload_runs), "eligible"

    def decode_reads(self, reads, addresses, packet_bytes=None, group_targets=None):
        """pool observations by public address or known copy group before calling."""
        if packet_bytes is not None and int(packet_bytes) != self.packet_bytes:
            raise ValueError("readout byte width differs from public packet layout")
        if self.accelerated and self.address_runs <= 24 and self.packet_bytes <= 6:
            from packet_accelerated import LIBRARY, decode_reads
            if LIBRARY.exists():
                return decode_reads(self, reads, addresses, group_targets=group_targets)
        index, packet_count = _packet_index(addresses, group_targets)
        symbol_count = self.packet_bytes * 4
        state_count = 4
        scores = np.zeros((packet_count, symbol_count, state_count), np.float64)
        coverage = np.zeros(packet_count, np.int32)
        statuses = Counter()
        multiplicities = Counter(map(str, reads))
        for read, count in multiplicities.items():
            identity, observed, status = self.decode_read(str(read))
            statuses[status] += count
            position = index.get(identity)
            if position is None or observed is None:
                if identity is not None and position is None:
                    statuses["unallocated_address"] += count
                continue
            coverage[position] += count
            obs = np.asarray(observed, np.int64)
            cells = obs.reshape(-1, 3)
            cell_scores = np.zeros((symbol_count, 4), np.float64)
            for base in range(3):
                cell_scores += self.model[base, STATES[:, base] - 3][:, cells[:, base]].T
            scores[position] += count * cell_scores
        top = scores.max(axis=-1)
        unresolved = (scores == top[..., None]).sum(axis=-1) != 1
        unresolved |= coverage[:, None] == 0
        symbols = scores.argmax(axis=-1).astype(np.uint8)
        packets = np.zeros((packet_count, self.packet_bytes), np.uint8)
        erased = np.ones_like(packets, dtype=bool)
        bits_per_symbol = 2
        for i in range(packet_count):
            packets[i] = _bytes_from_symbols(symbols[i], bits_per_symbol)
            erased[i] = unresolved[i].reshape(self.packet_bytes, 8 // bits_per_symbol).any(axis=1)
        return DecodeResult(packets, erased, coverage, dict(statuses))

    decode_packets = decode_reads
