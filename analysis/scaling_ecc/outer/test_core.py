import random
import pytest
from .core import K, make_library, decode_library, score_library, encode_stripe, decode_stripe, PARITIES


@pytest.mark.parametrize("r", PARITIES)
def test_systematic_and_roundtrip(r):
    lib = make_library(bytes(range(256)) * 2, "lovelace", 4, r, {"kind": "random"})
    assert [p["bytes"] for p in lib.packets if not p["is_parity"]] == lib.data
    result = decode_library([list(p["bytes"]) for p in lib.packets], lib.codec, r, lib.width)
    assert score_library(lib, result)["exact_object"]
    assert result["accepted"]


@pytest.mark.parametrize("r", PARITIES[1:])
def test_every_boundary_mix(r):
    rng = random.Random(100 + r)
    data = [rng.randbytes(3) if hasattr(rng, "randbytes") else bytes(rng.randrange(256) for _ in range(3)) for _ in range(K)]
    encoded = encode_stripe(data, r)
    for errors in range(r // 2 + 1):
        erasures = r - 2 * errors
        for placement in ("data_first", "parity_first", "random"):
            order = list(range(K + r))
            if placement == "parity_first":
                order = order[K:] + order[:K]
            elif placement == "random":
                rng.shuffle(order)
            estimates = [list(p) for p in encoded]
            for index in order[:errors]:
                estimates[index] = [value ^ 0x65 for value in estimates[index]]
            for index in order[errors:errors + erasures]:
                estimates[index] = None
            result, failed = decode_stripe(estimates, r, 3)
            assert not failed
            assert [bytes(p) for p in result] == data


def test_integrity_rejects_unknown_errors_and_missing_stays_missing():
    lib = make_library(b"unknown arbitrary data" * 9, "lovelace", 4, 0)
    estimates = [list(p["bytes"]) for p in lib.packets]
    estimates[-4] = None
    estimates[17][1] ^= 1
    result = decode_library(estimates, "lovelace", 0, 4)
    assert not result["accepted"]
    assert None in result["values"]


def test_correct_payload_with_wrong_metadata_is_not_exact_object():
    lib = make_library(b"unchanged payload", "lovelace", 2, 0, {"scale": 1})
    estimates = [list(p["bytes"]) for p in lib.packets]
    marker = lib.envelope.index(b"1", 56)
    estimates[marker // 2][marker % 2] = ord("2")
    result = decode_library(estimates, "lovelace", 0, 2)
    scores = score_library(lib, result)
    assert scores["exact_payload"]
    assert not scores["exact_object"] and not scores["metadata_recovered"]
    assert not result["accepted"]


def test_valid_wrong_rs_word_is_rejected_by_encoded_file_digest():
    lib = make_library(bytes(range(256)) * 2, "lovelace", 6, 4)
    altered = list(lib.data[:K])
    altered[20] = bytes([altered[20][0] ^ 1]) + altered[20][1:]
    replacement = encode_stripe(altered, 4)
    estimates = [list(p["bytes"]) for p in lib.packets]
    estimates[:K + 4] = [list(p) for p in replacement]
    result = decode_library(estimates, "lovelace", 4, 6)
    assert result["failed_columns"] == 0
    assert result["reason"] == "digest mismatch"
    assert not score_library(lib, result)["exact_object"]
