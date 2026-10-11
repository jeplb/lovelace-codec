"""independent analytic checks on conjugacy, error targets, and stream coverage."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('error_propagation', HERE / 'analyze.py')
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def read(name):
    with (HERE / name).open() as handle:
        return list(csv.DictReader(handle))


def test_real_inverse_obeys_conjugacy_and_not_frequency_rank():
    groups = ANALYSIS.spectrum_groups()
    for grid in [(9, 10), (16, 17), (20, 19)]:
        block = np.zeros((16, 16), np.complex128)
        block[grid[0] - 8, grid[1] - 8] = 3 + 4j
        rows, parts, _, _ = ANALYSIS.group_decomposition(block, groups)
        assert np.isclose(sum(r['normalized_pixel_mse'] for r in rows), 25 / (2 * 1024))
        assert np.isclose(np.mean(parts.sum(0) ** 2), 25 / (2 * 1024))
    block = np.zeros((16, 16), np.complex128)
    block[9 - 8, 10 - 8] = 3 + 4j
    block[23 - 8, 22 - 8] = -(3 - 4j)
    _, parts, _, _ = ANALYSIS.group_decomposition(block, groups)
    np.testing.assert_array_equal(parts.sum(0), np.zeros((32, 32)))


def test_crop_boundary_and_self_conjugate_cases():
    groups = ANALYSIS.spectrum_groups()
    assert len(groups) == 144
    assert sum(g['kind'] == 'crop_boundary' for g in groups) == 31
    block = np.zeros((16, 16), np.complex128)
    block[0, 6] = 3 + 4j
    rows, _, _, _ = ANALYSIS.group_decomposition(block, groups)
    active = [r for r in rows if r['normalized_pixel_mse'] > 0]
    assert len(active) == 1 and active[0]['group_kind'] == 'crop_boundary'
    assert np.isclose(active[0]['normalized_pixel_mse'], 25 / (2 * 1024))
    block[:] = 0
    block[8, 8] = 3 + 4j
    rows, _, _, _ = ANALYSIS.group_decomposition(block, groups)
    assert np.isclose(sum(r['normalized_pixel_mse'] for r in rows), 9 / 1024)
    assert np.isclose(sum(r['normalized_imaginary_mse'] for r in rows), 16 / 1024)


def test_stage_errors_and_interactions_recover_total():
    arrays = np.load(HERE / 'error_arrays.npz')
    summary = read('stream_summary.csv')
    for image in 'AB':
        for arm in ANALYSIS.ARMS:
            crop = arrays[f'crop_error_{image}']
            quant = arrays[f'quantization_error_{image}']
            readout = arrays[f'readout_error_{image}_{arm}']
            total = arrays[f'total_error_{image}_{arm}']
            np.testing.assert_allclose(crop + quant + readout, total, atol=2e-16, rtol=1e-12)
            expected_mse = np.mean(total ** 2)
            variance_sum = sum(np.mean(v ** 2) for v in [crop, quant, readout])
            cross = 2 * sum(np.mean(a * b) for a, b in [(crop, quant), (crop, readout), (quant, readout)])
            assert np.isclose(variance_sum + cross, expected_mse, rtol=1e-12, atol=1e-16)
            row = next(r for r in summary if r['image'] == image and r['method'] == arm)
            assert np.isclose(float(row['total_mse']), expected_mse, rtol=1e-12)
            assert abs(cross) > 1e-6
            physical_groups = arrays[f'physical_group_pixel_errors_{image}_{arm}']
            mean = arrays[f'mean_error_{image}_{arm}']
            np.testing.assert_allclose(physical_groups.sum(0) + mean, readout, rtol=1e-12, atol=4e-16)
            assert np.isclose(np.sum(np.mean(physical_groups ** 2, axis=(1, 2))) + np.mean(mean ** 2) +
                2 * np.mean(physical_groups.sum(0) * mean), np.mean(readout ** 2), rtol=1e-12, atol=1e-16)


def test_scalar_ledger_bit_digit_and_packing_checks():
    ledger = read('scalar_error_ledger.csv')
    assert len(ledger) == 6 * 512
    for row in ledger:
        expected, estimated = int(row['expected_label']), int(row['estimated_label'])
        wrong = json.loads(row['incorrect_bit_positions_msb_first'])
        assert len(wrong) == bin(expected ^ estimated).count('1')
        assert sum(json.loads(row['digit_label_error_contributions'])) == estimated - expected
        cells = json.loads(row['cell_locations'])
        image_offset = 0 if row['image'] == 'A' else 224
        scalar = int(row['scalar_index'])
        assert [8 * (a - image_offset) + c for a, c in cells] == list(range(256 + 3 * scalar, 259 + 3 * scalar))
    assert any(len({a for a, _ in json.loads(row['cell_locations'])}) == 2 for row in ledger)


def test_all_six_current_outcomes_and_group_changes():
    summary = read('stream_summary.csv')
    counts = {('A', 'control'): (475, 466), ('A', 'position'): (474, 466),
              ('A', 'position_weighted'): (496, 479), ('B', 'control'): (486, 478),
              ('B', 'position'): (486, 475), ('B', 'position_weighted'): (500, 492)}
    for row in summary:
        assert (int(row['matching_scalar_labels']), int(row['resolved_correct_labels'])) == counts[(row['image'], row['method'])]
        assert row['checksum_pass'] == 'False'
    changes = read('fourier_group_changes.csv')
    for image in 'AB':
        weighted = next(r for r in summary if r['image'] == image and r['method'] == 'position_weighted')
        control = next(r for r in summary if r['image'] == image and r['method'] == 'control')
        delta = sum(float(r['normalized_pixel_mse_change']) for r in changes if r['image'] == image and
                    r['comparison_method'] == 'position_weighted')
        assert np.isclose(delta, float(weighted['normalized_pixel_rmse_to_designed']) ** 2 -
                          float(control['normalized_pixel_rmse_to_designed']) ** 2, rtol=1e-12, atol=1e-17)
        assert (delta > 0) == (image == 'B')


def test_checksum_includes_metadata_and_excludes_carried_digest():
    codec = ANALYSIS.load_codec(ANALYSIS.INTEGRATION)
    prefix = ANALYSIS.INTEGRATION / 'evidence/codec/source/S12/image_0_header.bin'
    payload = ANALYSIS.INTEGRATION / 'evidence/codec/source/S12/image_0_payload.bin'
    truth = prefix.read_bytes() + payload.read_bytes()
    assert codec.inspect(truth, 6)[1]
    for changed_byte in [12, 20, 64]:
        content = bytearray(truth)
        content[changed_byte] ^= 1
        assert hashlib.sha256(content[:32] + content[64:]).digest() != truth[32:64]
        assert not codec.inspect(content, 6)[1]
    content = bytearray(truth)
    content[32] ^= 1
    assert hashlib.sha256(content[:32] + content[64:]).digest() == truth[32:64]
    assert not codec.inspect(content, 6)[1]
    rows = read('checksum_failure_table.csv')
    assert len(rows) == 6 and all(r['overall_checksum_pass'] == 'False' for r in rows)
    assert all(int(r['covered_bytes']) == 416 for r in rows)


def test_metadata_diagnostics_do_not_replace_read_derived_images():
    arrays = np.load(HERE / 'error_arrays.npz')
    for image in 'AB':
        for arm in ANALYSIS.ARMS:
            np.testing.assert_array_equal(arrays[f'diagnostic_{image}_{arm}_estimated_decoded'], arrays[f'pixels_{image}_{arm}'])
            np.testing.assert_array_equal(arrays[f'diagnostic_{image}_{arm}_designed_designed'], arrays[f'designed_{image}'])
            coefficient = arrays[f'coefficient_only_error_{image}_{arm}']
            metadata = arrays[f'metadata_only_error_{image}_{arm}']
            interaction = arrays[f'mixed_interaction_{image}_{arm}']
            np.testing.assert_allclose(coefficient + metadata + interaction, arrays[f'readout_error_{image}_{arm}'],
                                       rtol=1e-12, atol=2e-16)
