"""verify the compiled extension against independent public reference readout."""

import builtins
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from packet_codec import Codec, address_bits, decode_address, reverse_complement
from packet_accelerated import LIBRARY, STATUS, decode_observations
from analysis.scaling_ecc.sequence.noise import mutate_many
from analysis.scaling_ecc.sequence.payloads import load_frozen_payloads


def assert_same_readout(expected, actual):
    for field in ("packets", "erased", "coverage"):
        np.testing.assert_array_equal(getattr(expected, field), getattr(actual, field))
    assert expected.statuses == actual.statuses


def test_compiled_extraction_and_aggregation_match_reference_under_noise():
    if not LIBRARY.exists():
        pytest.skip("optional compiled accelerator is absent")
    reference = Codec("lovelace", accelerated=False)
    compiled = Codec("lovelace", accelerated=True)
    rng = np.random.default_rng(674342)
    reads = []
    for identity in range(12):
        data = rng.integers(0, 256, 2, dtype=np.uint8).tobytes()
        sequence = reference.encode(identity, data)
        for probability in (0, .005, .01, .02, .05):
            reads.extend(mutate_many(sequence, 4, probability, 0, rng))
            reads.extend(mutate_many(sequence, 4, 0, probability, rng))
            reads.extend(mutate_many(sequence, 4, probability, probability, rng))
        reads.extend([sequence[:100], sequence[10:], reverse_complement(sequence)])
    identities, observations, statuses = decode_observations(reference, reads)
    for index, read in enumerate(reads):
        identity, observed, status = reference.decode_read(read)
        assert (identity if identity is not None else -1) == int(identities[index])
        assert status == STATUS[int(statuses[index])]
        if observed is not None:
            assert tuple(observations[index, :len(observed)]) == observed
    assert_same_readout(reference.decode_reads(reads, range(12)), compiled.decode_reads(reads, range(12)))


def test_copy_groups_pool_observations_before_calls_without_truth_access():
    reference = Codec("lovelace", accelerated=False)
    compiled = Codec("lovelace", accelerated=True)
    payload, altered = b"\x27\x27", b"\x7b\x27"
    original = reference.encode(5, payload)
    copy = reference.encode(12, altered)
    same_address_altered = reference.encode(5, altered)
    expected = reference.decode_reads([original] * 16 + [same_address_altered] * 16, [5])
    for codec in (reference, compiled):
        with patch.object(builtins, "open", side_effect=AssertionError("truth read attempted")), patch.object(Path, "open", side_effect=AssertionError("truth path read attempted")):
            actual = codec.decode_reads([original] * 16 + [copy] * 16, [5, 12], group_targets=[0, 0])
        assert actual.coverage.tolist() == [32]
        assert_same_readout(expected, actual)


def test_grouped_compiled_and_reference_noisy_readout_match():
    rng = np.random.default_rng(584602)
    reference = Codec("lovelace", accelerated=False)
    compiled = Codec("lovelace", accelerated=True)
    targets = [0, 1, 2, 0, 1, 2]
    payloads = [rng.integers(0, 256, 2, dtype=np.uint8).tobytes() for _ in range(3)]
    reads = []
    for identity, target in enumerate(targets):
        reads.extend(mutate_many(reference.encode(identity, payloads[target]), 16, .01, .01, rng))
    assert_same_readout(reference.decode_reads(reads, range(6), group_targets=targets), compiled.decode_reads(reads, range(6), group_targets=targets))


def test_native_packet_strands_equal_every_image_design():
    codec = Codec("lovelace")
    root = Path(__file__).resolve().parents[3]
    lines = (root / "codebook/Lovelace_S12_480.fasta").read_text().splitlines()
    objects = load_frozen_payloads()
    for index, obj in enumerate(objects):
        content = bytes.fromhex(obj["metadata"]["native_header_hex"]) + obj["payload"]
        for local in range(224):
            identity = 224 * index + local
            assert codec.encode(identity, content[2 * local:2 * local + 2]) == lines[2 * identity + 1]


def test_address_code_corrects_single_and_rejects_double_errors():
    for length in (16, 18):
        for identity in (0, 1, 127, 511, 1023):
            vector = list(address_bits(identity, length))
            assert decode_address(vector) == identity
            for index in range(length):
                damaged = vector.copy()
                damaged[index] ^= 1
                assert decode_address(damaged) == identity
            for left in range(length):
                for right in range(left):
                    damaged = vector.copy()
                    damaged[left] ^= 1
                    damaged[right] ^= 1
                    assert decode_address(damaged) is None


def test_unobserved_packet_erases_and_unknown_wrong_packet_remains_resolved():
    codec = Codec("lovelace")
    wrong = b"\x7b\x7b"
    result = codec.decode_reads([codec.encode(0, wrong)] * 16, [0, 1])
    assert result.packets[0].tobytes() == wrong
    assert not result.erased[0].any()
    assert result.erased[1].all()
    assert result.coverage.tolist() == [16, 0]
