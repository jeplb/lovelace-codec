"""apply the frozen v2 format decoder to changed observation scores."""

import numpy as np

from inference import decode_arm
from cell_model import aggregate


ARMS = ('control', 'position', 'position_weighted')


def _posterior(scores):
    values = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def decode_one(unit_log_scores, identities, weights, v2config,
               unitparents=None, arm='control', coupling=True, partner_enabled=True):
    """aggregate unchanged recipient contributions and decode one arm."""
    if arm not in ARMS:
        raise ValueError('undeclared observation arm')
    identities = np.asarray(identities, dtype=np.int64)
    if unitparents is not None:
        pairs = list(zip(map(str, unitparents), map(int, identities)))
        if len(set(pairs)) != len(pairs):
            raise ValueError('duplicate parent and recipient contribution')
    weights = np.asarray(weights, dtype=np.float64)
    scores, depth = aggregate(unit_log_scores, identities, weights,
                              n_addresses=480)
    supported = depth > 0
    retained = np.zeros((480, 8), dtype=np.float64)
    np.add.at(retained, identities, weights)
    preformat = _posterior(scores)
    result = decode_arm(scores, supported, v2config,
                        dict(header_mask='v2', coupling=coupling),
                        v2config['header_mask']['proven_bits'], partner_enabled=partner_enabled)
    for row in result['label_rows']:
        row['arm'] = arm
        if row['kind'] == 'payload':
            # the three scalar digits can span adjacent recipient oligos.
            start = 256 + 3 * int(row['scalar_index'])
            base = 'AB'.index(row['image']) * 224
            locations = [(base + (start + j) // 8, (start + j) % 8)
                         for j in range(3)]
            row['retained_weight_cells'] = [float(retained[a, j])
                                            for a, j in locations]
            row['depth_cells'] = [int(depth[a, j]) for a, j in locations]
        else:
            address, cell = int(row['address']), int(row['cell'])
            row['retained_weight_cells'] = [float(retained[address, cell])]
            row['depth_cells'] = [int(depth[address, cell])]
    final = preformat.copy()
    final_calls = scores.argmax(axis=-1)
    for row in result['cell_rows']:
        address, cell = int(row['address']), int(row['cell'])
        row.update(arm=arm, depth=int(depth[address, cell]),
                   retained_weight=float(retained[address, cell]))
        final[address, cell] = row['posterior_vector']
        final_calls[address, cell] = int(row['digit'])
    for row in result['bit_rows']:
        row['arm'] = arm
    for row in result['gates']:
        row['arm'] = arm
    result.update(preformat_posteriors=preformat, final_posteriors=final,
                  preformat_log_scores=scores, depth=depth,
                  retained_weight=retained, supported=supported,
                  preformat_calls=scores.argmax(axis=-1), final_calls=final_calls)
    return result
