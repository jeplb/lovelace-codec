"""derive amended header bits and sum the frozen scale domain event."""

import numpy as np
from format_constraints import header_rows as header_rows_v1


MAX_FINITE_HEX = '7fefffffffffffff'
PROVEN_BIT_LOCATIONS = {
    'mean_sign': (12, 0),
    'mean_exponent_msb': (12, 1),
    'scale_sign': (20, 0),
}


def domain_probability(posteriors, threshold_hex):
    """sum an inclusive unsigned binary64 interval with whole-cell states."""
    probabilities = np.asarray(posteriors, dtype=np.float64)
    if probabilities.shape != (32, 4):
        raise ValueError('scale posterior must have shape (32, 4)')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError('scale probabilities must be finite and nonnegative')
    if not np.allclose(probabilities.sum(axis=1), 1., rtol=0., atol=1.e-12):
        raise ValueError('scale cell posteriors must already be normalized')
    text = str(threshold_hex).lower().removeprefix('0x')
    if len(text) != 16 or any(char not in '0123456789abcdef' for char in text):
        raise ValueError('threshold must be a 64-bit hexadecimal pattern')
    lower, upper = int(text, 16), int(MAX_FINITE_HEX, 16)
    if not 0 <= lower <= upper:
        raise ValueError('threshold must be nonnegative and finite')
    lower_digits = [(lower >> shift) & 3 for shift in range(62, -1, -2)]
    upper_digits = [(upper >> shift) & 3 for shift in range(62, -1, -2)]
    states = np.zeros((2, 2), dtype=np.float64)
    states[1, 1] = 1.
    for index, (lower_digit, upper_digit) in enumerate(zip(lower_digits, upper_digits)):
        next_states = np.zeros_like(states)
        for lower_tight in (0, 1):
            for upper_tight in (0, 1):
                low = lower_digit if lower_tight else 0
                high = upper_digit if upper_tight else 3
                for digit in range(low, high + 1):
                    next_lower = int(lower_tight and digit == lower_digit)
                    next_upper = int(upper_tight and digit == upper_digit)
                    next_states[next_lower, next_upper] += (
                        states[lower_tight, upper_tight] * probabilities[index, digit])
        states = next_states
    return float(states.sum(dtype=np.float64))


def header_rows_v2(proven_bits):
    """add only independently proven zero bits to the frozen v1 mask."""
    proven_bits = set(proven_bits)
    if not proven_bits <= PROVEN_BIT_LOCATIONS.keys():
        raise ValueError('unknown header bit proof')
    rows = header_rows_v1()
    for row in rows:
        image_offset = 0 if row['image'] == 'A' else 224
        position = (row['address'] - image_offset) * 8 + row['cell']
        for name in proven_bits:
            byte_offset, bit_offset = PROVEN_BIT_LOCATIONS[name]
            if position == byte_offset * 4 + bit_offset // 2:
                mask = 2 >> (bit_offset % 2)
                row['known_mask'] |= mask
                row['known_digit'] = (row['known_digit'] or 0) & ~mask
        mask, digit = row['known_mask'], row['known_digit'] or 0
        row['known_bits'] = (mask & 1) + (mask >> 1)
        row['allowed_states'] = ','.join(str(state) for state in range(4)
                                         if (state & mask) == digit)
        for bit_offset in (0, 1):
            bit_mask = 2 >> bit_offset
            row[f'bit_{bit_offset}_provenance'] = (
                'format_supplied' if mask & bit_mask else 'read_dependent')
    return rows
