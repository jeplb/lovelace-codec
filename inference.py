"""pure three-arm inference under the frozen v1 likelihood model."""

import struct

import numpy as np

from scalar_inference import (DIGITS, confidence, digits_to_bytes, joint_pair,
                            normalize, provenance, resolved)
from format_constraints import build_psi, header_rows, pairing_rows, scalar_rows
from header_constraints import domain_probability, header_rows_v2


ARMS = {
    'same_mask_control': dict(header_mask='v2', coupling=False),
    'gate_only': dict(header_mask='v1', coupling=True),
    'primary': dict(header_mask='v2', coupling=True),
}


def _support_fields(mask):
    count = int(np.sum(mask))
    return dict(read_support=bool(np.all(mask)),
                read_support_mask=[bool(value) for value in mask],
                supported_cell_count=count,
                support_class=('none' if count == 0 else
                               'complete' if count == 3 else 'partial'))


def _confidence_fields(posterior, selected, baseline_p, baseline, config,
                       read_support, coupled=False, fixed=False,
                       baseline_fixed=False):
    probability, margin = confidence(posterior, selected)
    base_probability, base_margin = confidence(baseline_p, baseline)
    return dict(posterior=probability, posterior_margin=margin,
                resolved=resolved(posterior, selected, config),
                posterior_vector=posterior.tolist(),
                provenance=provenance(posterior, selected, baseline,
                                      read_support, config, coupled, fixed),
                baseline_posterior=base_probability,
                baseline_posterior_margin=base_margin,
                baseline_resolved=resolved(baseline_p, baseline, config),
                baseline_posterior_vector=baseline_p.tolist(),
                baseline_provenance=provenance(baseline_p, baseline, baseline,
                                               read_support, config,
                                               fixed=baseline_fixed))


def decode_arm(scores, supported, config, arm_config, proven_bits=(), partner_enabled=True):
    """decode one predeclared arm without reading files or reference truth."""
    arm = ARMS[arm_config] if isinstance(arm_config, str) else arm_config
    scores = np.array(scores[:448], dtype=np.float64, copy=True)
    supported = np.asarray(supported[:448], dtype=bool)
    assert scores.shape == (448, 8, 4) and supported.shape == (448, 8)
    assert np.isfinite(scores).all()
    assert arm['header_mask'] in ('v1', 'v2')
    scores[~supported] = config['inference']['unsupported_log_likelihood']
    headers = (header_rows_v2(proven_bits) if arm['header_mask'] == 'v2'
               else header_rows())
    for row in headers:
        mask = int(row['known_mask'])
        if mask:
            known = int(row['known_digit'])
            scores[row['address'], row['cell'],
                   (np.arange(4) & mask) != (known & mask)] = -np.inf

    cell_p = np.stack([normalize(value) for value in scores.reshape(-1, 4)])
    cell_p = cell_p.reshape(448, 8, 4)
    base_cells = scores.argmax(axis=-1)
    layout = scalar_rows()
    scalar_scores = np.zeros((2, 512, 64), dtype=np.float64)
    scalar_support = np.zeros((2, 512, 3), dtype=bool)
    for row in layout:
        image, index = 'AB'.index(row['image']), row['scalar_index']
        addresses = [row[f'address_{j}'] for j in range(3)]
        cells = [row[f'cell_{j}'] for j in range(3)]
        scalar_scores[image, index] = sum(
            scores[address, cell, DIGITS[:, j]]
            for j, (address, cell) in enumerate(zip(addresses, cells)))
        scalar_support[image, index] = supported[addresses, cells]
    base_labels = scalar_scores.argmax(axis=-1)
    base_p = np.stack([normalize(value)
                       for value in scalar_scores.reshape(-1, 64)])
    base_p = base_p.reshape(2, 512, 64)

    header_posteriors, gates = [], []
    for image in range(2):
        start = image * 224
        digits = base_cells[start:start + 32].reshape(-1)
        posterior = cell_p[start:start + 32].reshape(256, 4)
        header_posteriors.append(posterior.copy())
        packed = digits_to_bytes(digits)
        scale = struct.unpack('>d', packed[20:28])[0]
        scale_p = posterior[80:112]
        p_exact = float(np.prod(scale_p[np.arange(32), digits[80:112]]))
        p_domain = domain_probability(scale_p,
                                      config['scale_gate']['threshold_bits_hex'])
        domain_passes = bool(p_domain >= config['scale_gate']['p_gate'])
        gates.append(dict(image='AB'[image], header_mask=arm['header_mask'],
                          coupling_enabled=bool(arm['coupling']),
                          scale=scale if np.isfinite(scale) else None,
                          scale_hex=packed[20:28].hex(),
                          p_exact_word=p_exact, p_domain=p_domain,
                          accumulation_dtype=config['scale_gate'].get('accumulation_dtype', 'float64'),
                          domain_passes=domain_passes,
                          passes=bool(arm['coupling'] and domain_passes)))

    labels, posteriors = base_labels.copy(), base_p.copy()
    coupled = np.zeros((2, 512), dtype=bool)
    dc_fixed = np.zeros((2, 512), dtype=bool)
    pairs = pairing_rows()
    pair_map = {row['coefficient_index']: row for row in pairs}
    psi_real, psi_imag = build_psi(config)
    for image, gate in enumerate(gates):
        if not gate['passes']:
            continue
        for row in pairs:
            index, partner = row['coefficient_index'], row['partner_index']
            if partner_enabled and row['class'] == 'paired' and index < partner:
                for part, psi in enumerate((psi_real, psi_imag)):
                    first, second = 2 * index + part, 2 * partner + part
                    chosen, first_p, second_p = joint_pair(
                        scalar_scores[image, first],
                        scalar_scores[image, second], psi)
                    labels[image, first], labels[image, second] = chosen
                    posteriors[image, first] = first_p
                    posteriors[image, second] = second_p
                    coupled[image, first] = coupled[image, second] = True
            elif row['class'] == 'dc':
                for part, key in enumerate(('dc_real_support', 'dc_imag_support')):
                    scalar = 2 * index + part
                    potential = np.full(64, config['quantizer']['epsilon'])
                    potential[config['quantizer'][key]] = config['quantizer']['dc_allowed_weight']
                    log_p = scalar_scores[image, scalar] + np.log(potential)
                    labels[image, scalar] = log_p.argmax()
                    posteriors[image, scalar] = normalize(log_p)
                    dc_fixed[image, scalar] = True

    branch_cells, branch_p = base_cells.copy(), cell_p.copy()
    label_rows, cell_rows, bit_rows = [], [], []
    for row in layout:
        image, index = 'AB'.index(row['image']), row['scalar_index']
        label, baseline = int(labels[image, index]), int(base_labels[image, index])
        posterior, baseline_p = posteriors[image, index], base_p[image, index]
        uses_pair, uses_dc = bool(coupled[image, index]), bool(dc_fixed[image, index])
        evidence = _support_fields(scalar_support[image, index])
        pair = pair_map[row['coefficient_index']]
        partner = pair['partner_index']
        pair_id = (f"{row['image']}_{min(row['coefficient_index'], partner)}_"
                   f"{max(row['coefficient_index'], partner)}"
                   if pair['class'] == 'paired' else '')
        compatibility = ''
        if pair['class'] == 'paired':
            partner_label = int(base_labels[image, 2 * partner + index % 2])
            compatibility = bool(
                abs(baseline - partner_label) <= config['quantizer']['real_tolerance']
                if row['part'] == 'real' else
                baseline + partner_label in config['quantizer']['imag_support'])
        values = dict(image=row['image'], kind='payload',
                      coefficient_index=row['coefficient_index'], part=row['part'],
                      scalar_index=index, address=row['global_address'],
                      cell=row['cell_start'], label=label, baseline_label=baseline,
                      pairing_class=pair['class'], partner_index=partner,
                      partner_scalar_index=2 * partner + index % 2 if partner is not None else '',
                      complex_pair_id=pair_id,
                      baseline_pair_compatibility=compatibility,
                      coupled=uses_pair, dc_fixed=uses_dc,
                      known_bits=0, known_mask=0)
        values.update(evidence)
        values.update(_confidence_fields(posterior, label, baseline_p, baseline,
                                        config, evidence['read_support'],
                                        uses_pair, uses_dc))
        label_rows.append(values)
        for j in range(3):
            address, cell = row[f'address_{j}'], row[f'cell_{j}']
            digit = int(DIGITS[label, j])
            branch_cells[address, cell] = digit
            marginal = np.bincount(DIGITS[:, j], weights=posterior, minlength=4)
            baseline_marginal = np.bincount(DIGITS[:, j], weights=baseline_p,
                                            minlength=4)
            branch_p[address, cell] = marginal
            values = dict(image=row['image'], kind='payload',
                          coefficient_index=row['coefficient_index'], part=row['part'],
                          scalar_index=index, digit_position=j, address=address,
                          cell=cell, digit=digit,
                          baseline_digit=int(base_cells[address, cell]),
                          pairing_class=pair['class'], complex_pair_id=pair_id,
                          read_support=bool(supported[address, cell]),
                          coupled=uses_pair, dc_fixed=uses_dc,
                          known_bits=0, known_mask=0)
            values.update(_confidence_fields(marginal, digit, baseline_marginal,
                                            int(base_cells[address, cell]), config,
                                            bool(supported[address, cell]),
                                            uses_pair, uses_dc))
            cell_rows.append(values)

    for row in headers:
        address, cell = row['address'], row['cell']
        digit, posterior = int(branch_cells[address, cell]), branch_p[address, cell]
        mask = int(row['known_mask'])
        read_support = bool(supported[address, cell])
        # a partial mask supplies only its known bit, not the whole cell.
        fixed = mask == 3
        common = dict(image=row['image'], kind='header',
                      coefficient_index='', part='', scalar_index='',
                      address=address, cell=cell, pairing_class='header',
                      complex_pair_id='', read_support=read_support,
                      coupled=False, dc_fixed=False,
                      known_bits=int(row['known_bits']), known_mask=mask)
        common.update(_confidence_fields(posterior, digit, posterior, digit,
                                         config, read_support, fixed=fixed,
                                         baseline_fixed=fixed))
        cell_rows.append(dict(common, digit=digit, baseline_digit=digit,
                              digit_position=''))
        label_rows.append(dict(common, label=digit, baseline_label=digit,
                               read_support_mask=[], supported_cell_count='',
                               support_class='', partner_index='',
                               partner_scalar_index='',
                               baseline_pair_compatibility=''))
        image_index = 'AB'.index(row['image'])
        for bit_offset, bit_mask in enumerate((2, 1)):
            known = bool(mask & bit_mask)
            selected = int(bool(digit & bit_mask))
            states = (np.arange(4) & bit_mask != 0).astype(int)
            marginal = np.bincount(states, weights=posterior, minlength=2)
            fields = _confidence_fields(marginal, selected, marginal, selected,
                                         config, read_support, fixed=known,
                                         baseline_fixed=known)
            bit_rows.append(dict(image=row['image'], address=address, cell=cell,
                                 bit_offset=bit_offset,
                                 bit_position=address * 16 + cell * 2 + bit_offset,
                                 header_bit_position=(address - image_index * 224) * 16
                                 + cell * 2 + bit_offset,
                                 known=known, bit=selected, read_support=read_support,
                                 **fields))

    streams = {image: digits_to_bytes(branch_cells[index * 224:(index + 1) * 224].reshape(-1))
               for index, image in enumerate('AB')}
    return dict(streams=streams, label_rows=label_rows, cell_rows=cell_rows,
                bit_rows=bit_rows, gates=gates,
                header_posteriors=np.stack(header_posteriors),
                baseline_cell_posteriors=cell_p.copy())
