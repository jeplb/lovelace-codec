"""systematic packet stripes and protected file envelope."""
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache

import reedsolo

K = 32
PARITIES = (0, 4, 8, 16, 24)
MAGIC = b"BLOCKECC"
HEADER = struct.Struct(">8sBBBBQ I 32s")
CODEC_IDS = {"lovelace": 3}


@lru_cache(None)
def rs(r):
    return reedsolo.RSCodec(r, nsize=255, fcr=0, prim=0x11d,
                            generator=2, c_exp=8)


def encode_stripe(data, r):
    assert len(data) == K and len({len(p) for p in data}) == 1
    if not r:
        return list(data)
    width = len(data[0])
    encoded = [bytearray(width) for _ in range(K + r)]
    for j in range(width):
        word = rs(r).encode(bytearray(p[j] for p in data))
        assert len(word) == K + r
        for i, value in enumerate(word):
            encoded[i][j] = value
    return [bytes(p) for p in encoded]


def decode_stripe(estimates, r, width):
    """none marks a declared byte erasure, never a scorer-detected error."""
    assert len(estimates) == K + r
    output = [[None] * width for _ in range(K)]
    failures = 0
    for j in range(width):
        erased = [i for i, p in enumerate(estimates) if p is None or p[j] is None]
        word = bytearray(0 if i in erased else p[j]
                         for i, p in enumerate(estimates))
        if not r:
            decoded = [None if i in erased else word[i] for i in range(K)]
        else:
            try:
                decoded = rs(r).decode(word, erase_pos=erased)[0]
            except reedsolo.ReedSolomonError:
                failures += 1
                # keep observed data for residual scoring. do not repair it
                decoded = [None if i in erased else word[i] for i in range(K)]
        for i, value in enumerate(decoded):
            output[i][j] = value
    return output, failures


@dataclass
class Library:
    codec: str
    r: int
    width: int
    envelope: bytes
    payload: bytes
    metadata: dict
    packets: list
    data: list

    @property
    def stripes(self):
        return len(self.data) // K


def make_library(payload, codec, width, r, metadata=None):
    metadata = metadata or {}
    meta = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(meta + payload).digest()
    header = HEADER.pack(MAGIC, 1, CODEC_IDS[codec], r, width,
                         len(payload), len(meta), digest)
    envelope = header + meta + payload
    count = math.ceil(len(envelope) / (K * width)) * K
    padded = envelope.ljust(count * width, b"\0")
    data = [padded[i:i + width] for i in range(0, len(padded), width)]
    packets = []
    for stripe in range(count // K):
        words = encode_stripe(data[stripe * K:(stripe + 1) * K], r)
        for slot, word in enumerate(words):
            packets.append({"identity": stripe * (K + r) + slot,
                            "stripe": stripe, "slot": slot,
                            "is_parity": slot >= K, "bytes": word})
    return Library(codec, r, width, envelope, payload, metadata, packets, data)


def parse_envelope(values, codec, r, width):
    result = {"accepted": False, "payload": None, "metadata": None,
              "reason": "missing header"}
    if len(values) < HEADER.size or any(x is None for x in values[:HEADER.size]):
        return result
    magic, version, cid, rr, ww, length, mlength, digest = HEADER.unpack(bytes(values[:HEADER.size]))
    if (magic, version, cid, rr, ww) != (MAGIC, 1, CODEC_IDS[codec], r, width):
        result["reason"] = "invalid header"
        return result
    end = HEADER.size + mlength + length
    if end > len(values):
        result["reason"] = "invalid lengths"
        return result
    meta_values = values[HEADER.size:HEADER.size + mlength]
    body_values = values[HEADER.size + mlength:end]
    result["payload"] = body_values
    if any(x is None for x in meta_values):
        result["reason"] = "missing metadata"
        return result
    meta = bytes(meta_values)
    try:
        result["metadata"] = json.loads(meta)
    except (ValueError, UnicodeDecodeError):
        result["reason"] = "invalid metadata"
        return result
    if any(x is None for x in body_values):
        result["reason"] = "missing payload"
        return result
    body = bytes(body_values)
    result["payload"] = body
    result["accepted"] = hashlib.sha256(meta + body).digest() == digest
    result["reason"] = "accepted" if result["accepted"] else "digest mismatch"
    return result


def decode_library(estimates, codec, r, width):
    """decoder receives estimates and public format parameters only."""
    assert len(estimates) % (K + r) == 0
    values, stripes, failures = [], [], 0
    for start in range(0, len(estimates), K + r):
        data, failed = decode_stripe(estimates[start:start + K + r], r, width)
        flat = [v for packet in data for v in packet]
        values.extend(flat)
        stripes.append(flat)
        failures += failed
    parsed = parse_envelope(values, codec, r, width)
    parsed.update(values=values, stripes=stripes, failed_columns=failures)
    return parsed


def score_library(library, decoded):
    """ground truth exists only in this scoring function."""
    body = decoded["payload"]
    # known layout is for residual scoring only if a damaged header blocks parsing
    offset = len(library.envelope) - len(library.payload)
    scored = decoded["values"][offset:offset + len(library.payload)]
    missing = sum(v is None for v in scored)
    wrong = sum(v is not None and v != t for v, t in zip(scored, library.payload))
    bit_errors = sum((v ^ t).bit_count() if hasattr(int, "bit_count") else bin(v ^ t).count("1")
                     for v, t in zip(scored, library.payload) if v is not None)
    stripe_exact = 0
    for i, stripe in enumerate(decoded["stripes"]):
        expected = b"".join(library.data[i * K:(i + 1) * K])
        stripe_exact += all(v == t for v, t in zip(stripe, expected))
    exact_payload = body is not None and all(v is not None for v in body) and bytes(body) == library.payload
    exact = exact_payload and decoded["metadata"] == library.metadata
    return {"exact_object": exact, "exact_payload": exact_payload, "accepted": decoded["accepted"],
            "incorrect_accepted": decoded["accepted"] and not exact,
            "declared_failure": not decoded["accepted"],
            "missing_bytes": missing, "wrong_bytes": wrong,
            "wrong_bits": bit_errors, "payload_bytes": len(library.payload),
            "exact_stripes": stripe_exact, "stripes": library.stripes,
            "failed_columns": decoded["failed_columns"],
            "metadata_recovered": decoded["metadata"] == library.metadata}
