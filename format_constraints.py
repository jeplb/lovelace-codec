"""derive public format constraints without reading encoded image contents."""

import csv
from pathlib import Path
import struct

import numpy as np


HEADER_STRUCT = struct.Struct('>4sBBBBBBHddI32s')
HEADER_CELLS = HEADER_STRUCT.size * 4
PAYLOAD_BYTES = 64 * 6
PAYLOAD_CELLS = PAYLOAD_BYTES * 4
CELLS_PER_OLIGO = 8
OLIGOS_PER_IMAGE = (HEADER_CELLS + PAYLOAD_CELLS) // CELLS_PER_OLIGO


def pairing_rows():
    rows = []
    for index in range(16 * 16):
        row, column = divmod(index, 16)
        freq_u, freq_v = row - 8, column - 8
        kind = 'unpaired_minus8' if -8 in (freq_u, freq_v) else 'paired'
        if (freq_u, freq_v) == (0, 0):
            kind = 'dc'
        partner = 272 - index if kind != 'unpaired_minus8' else None
        rows.append(dict(coefficient_index=index, freq_u=freq_u, freq_v=freq_v,
                         partner_index=partner, **{'class': kind}))
    assert sum(row['class'] == 'paired' for row in rows) == 224
    assert sum(row['class'] == 'unpaired_minus8' for row in rows) == 31
    assert sum(row['class'] == 'dc' for row in rows) == 1
    return rows


def scalar_rows():
    rows = []
    for image_index, image in enumerate(('A', 'B')):
        for scalar in range(16 * 16 * 2):
            positions = [HEADER_CELLS + scalar * 3 + offset for offset in range(3)]
            addresses = [image_index * OLIGOS_PER_IMAGE + pos // 8 for pos in positions]
            cells = [pos % 8 for pos in positions]
            row = dict(image=image, coefficient_index=scalar // 2,
                       part=('real', 'imag')[scalar % 2], scalar_index=scalar,
                       global_address=addresses[0], cell_start=cells[0],
                       cell_end=cells[-1], end_address=addresses[-1],
                       triple_addresses=','.join(map(str, addresses)),
                       triple_cells=','.join(map(str, cells)))
            row.update({'address_%d' % i: addresses[i] for i in range(3)})
            row.update({'cell_%d' % i: cells[i] for i in range(3)})
            rows.append(row)
    assert HEADER_CELLS == 256
    assert PAYLOAD_CELLS == 1536
    assert OLIGOS_PER_IMAGE == 224
    assert rows[0]['global_address'] == 32 and rows[-1]['end_address'] == 447
    return rows


def header_rows():
    # widths from the public big-endian struct, including two-byte flags
    fields = [('4s', b'LVIM', True), ('B', 1, True), ('B', 32, True),
              ('B', 32, True), ('B', 16, True), ('B', 16, True),
              ('B', 6, True), ('H', 1, True), ('d', 0.0, False),
              ('d', 0.0, False), ('I', PAYLOAD_BYTES, True), ('32s', bytes(32), False)]
    packed = HEADER_STRUCT.pack(*(value for _, value, _ in fields))
    known_bytes = []
    for field, _, known in fields:
        known_bytes.extend([known] * struct.calcsize('>' + field))
    rows = []
    for image_index, image in enumerate(('A', 'B')):
        for position in range(HEADER_CELLS):
            byte_index, digit_index = divmod(position, 4)
            known = known_bytes[byte_index]
            digit = (packed[byte_index] >> (6 - 2 * digit_index)) & 3
            rows.append(dict(image=image,
                             address=image_index * OLIGOS_PER_IMAGE + position // 8,
                             cell=position % 8, known_bits=2 if known else 0,
                             known_mask=3 if known else 0,
                             known_digit=digit if known else None))
    return rows


def header_counts():
    counts = {}
    rows = header_rows()
    for image in ('A', 'B'):
        oligos = {}
        for row in rows:
            if row['image'] == image:
                oligos.setdefault(row['address'], []).append(row['known_bits'])
        totals = [sum(values) for values in oligos.values()]
        counts[image] = dict(fully_known=sum(value == 16 for value in totals),
                             partially_known=sum(0 < value < 16 for value in totals),
                             unknown=sum(value == 0 for value in totals),
                             known_bits=sum(totals), unknown_bits=512 - sum(totals))
    return counts


def rescue_rows():
    pairs = {row['coefficient_index']: row for row in pairing_rows()}
    scalars = {(row['image'], row['coefficient_index'], row['part']): row
               for row in scalar_rows()}
    rows = []
    for scalar in scalars.values():
        pair = pairs[scalar['coefficient_index']]
        for offset in range(3):
            if scalar['address_%d' % offset] != 442:
                continue
            partner = scalars[(scalar['image'], pair['partner_index'], scalar['part'])]
            rows.append(dict(address=442, cell=scalar['cell_%d' % offset],
                             image=scalar['image'], coefficient_index=scalar['coefficient_index'],
                             part=scalar['part'], digit_position=offset,
                             freq_u=pair['freq_u'], freq_v=pair['freq_v'],
                             partner_index=pair['partner_index'],
                             partner_address=partner['address_%d' % offset],
                             partner_cell=partner['cell_%d' % offset]))
    rows.sort(key=lambda row: row['cell'])
    expected = [(442, cell, 274, cell, 7, 0) for cell in range(6)]
    expected += [(442, cell, 273, cell - 4, 7, 1) for cell in range(6, 8)]
    actual = [(row['address'], row['cell'], row['partner_address'], row['partner_cell'],
               row['freq_u'], row['freq_v']) for row in rows]
    assert actual == expected
    return rows


def quantize(x):
    # same encoder order at floating-point rounding boundaries
    return np.clip(np.floor((np.asarray(x, dtype=np.float64) + 1) * 31.5 + .5),
                   0, 63).astype(np.uint8)


def build_psi(config):
    model = config['quantizer']
    labels = np.arange(64)
    difference = np.abs(labels[:, None] - labels[None, :])
    sums = labels[:, None] + labels[None, :]
    real = np.full((64, 64), model['epsilon'], dtype=np.float64)
    imag = np.full((64, 64), model['epsilon'], dtype=np.float64)
    real[difference == 0] = model['real_equal_weight']
    real[(difference > 0) & (difference <= model['real_tolerance'])] = model['adjacent_weight']
    for value in model['imag_support']:
        weight = model['imag_central_weight'] if value == 63 else model['imag_edge_weight']
        imag[sums == value] = weight
    return real, imag


def redundancy_rows(config):
    model = config['quantizer']
    labels = np.arange(64)
    real_count = int(np.sum(np.abs(labels[:, None] - labels[None, :]) <= model['real_tolerance']))
    imag_count = int(np.sum(np.isin(labels[:, None] + labels[None, :], model['imag_support'])))
    count = real_count * imag_count
    rows = []
    for image in ('A', 'B'):
        for pair in pairing_rows():
            if pair['class'] == 'paired' and pair['coefficient_index'] < pair['partner_index']:
                rows.append(dict(image=image, coefficient_index=pair['coefficient_index'],
                                 partner_index=pair['partner_index'],
                                 real_allowed_combinations=real_count,
                                 imag_allowed_combinations=imag_count,
                                 allowed_complex_combinations=count,
                                 implied_bit_reduction=float(24 - np.log2(count)),
                                 interpretation='combinatorial'))
    return rows


def scale_gate(scale, probability, config):
    rule = config['scale_gate']
    return bool(np.isfinite(scale) and np.isfinite(probability)
                and probability >= rule['p_gate'] and scale >= rule['s_min'])


gate = scale_gate


def _write_table(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def emit_artifacts(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tables = {'pairing_map.tsv': pairing_rows(), 'scalar_layout.tsv': scalar_rows(),
              'header_mask.tsv': header_rows(), 'design_442_rescue_map.tsv': rescue_rows(),
              'combinatorial_redundancy.tsv': redundancy_rows(config)}
    for name, rows in tables.items():
        _write_table(output / name, rows)
    real, imag = build_psi(config)
    np.save(output / 'psi_real.npy', real, allow_pickle=False)
    np.save(output / 'psi_imag.npy', imag, allow_pickle=False)
    return dict(header=header_counts(), paired_coefficients=224, unique_pairs_per_image=112,
                unpaired_coefficients=31, dc_coefficients=1, header_cells_per_image=HEADER_CELLS,
                payload_cells_per_image=PAYLOAD_CELLS, scalars_per_image=512,
                rescue_cells=len(tables['design_442_rescue_map.tsv']))
