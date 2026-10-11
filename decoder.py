"""decode the prepared ordinary experimental observations with frozen pilot models."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cell_model import cell_log_scores
from checksum import digest_report
from observation_decoder import ARMS, decode_one
from reconstruction import reconstruct_stream
from suffix_weighting import build_references, score_weights
from tsv_records import read_output, write_output

ROOT = Path(__file__).resolve().parent


def load_record(path):
    return read_output(path)


def write_record(path, value):
    write_output(path, value)


def write_tsv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def decode_observations(counts, identities, parents, model, selection, config,
                        coupling=True, partner_enabled=True):
    global_scores = cell_log_scores(counts, model['global_probabilities'])
    position_scores = cell_log_scores(counts, model['selected_probabilities'])
    reference, parent_counts = build_references(position_scores, identities, parents)
    weighted = score_weights(position_scores, reference, parent_counts,
                             selection['selected_candidate'], config)
    results = {}
    for arm in ARMS:
        scores = global_scores if arm == 'control' else position_scores
        weights = weighted['weights'] if arm == 'position_weighted' else np.ones(scores.shape[:2])
        results[arm] = decode_one(scores, identities, weights, config,
                                  unitparents=parents, arm=arm, coupling=coupling, partner_enabled=partner_enabled)
    return results, dict(global_unit_log_scores=global_scores,
                         unit_log_scores=position_scores, reference=reference,
                         other_parent_counts=parent_counts, **weighted)


def save_decoding(results, diagnostics, output, coupling, partner_enabled=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels, cells, bits, gates, checksums = [], [], [], [], []
    arrays = {}
    for arm, result in results.items():
        for name, rows in (('label_rows', labels), ('cell_rows', cells),
                           ('bit_rows', bits), ('gates', gates)):
            rows.extend(result[name])
        for image, content in result['streams'].items():
            path = output / f'{arm}_{image}.bin'
            path.write_bytes(content)
            checksums.append(dict(arm=arm, image=image, **digest_report(content)))
            # keep unresolved map estimates and failed checksums in the figures
            reconstruction = reconstruct_stream(content)
            arrays[f'{arm}_{image}'] = reconstruction['image']
        saved = {key: result[key] for key in (
            'preformat_posteriors', 'final_posteriors', 'preformat_log_scores',
            'preformat_calls', 'final_calls', 'depth', 'retained_weight',
            'supported', 'header_posteriors', 'baseline_cell_posteriors')}
        np.savez_compressed(output / f'{arm}.npz', **saved)
    for name, rows in (('labels.tsv', labels), ('cells.tsv', cells),
                       ('bits.tsv', bits), ('gates.tsv', gates)):
        write_tsv(output / name, rows)
    np.savez_compressed(output / 'rendered_arrays.npz', **arrays)
    np.savez_compressed(output / 'weight_inputs.npz', **diagnostics)
    summary = dict(coupling_enabled=bool(coupling), partner_inference_enabled=bool(partner_enabled), arms=list(results),
                   checksums=checksums, unresolved_estimates_retained=True,
                   labels={arm: dict(payload_labels=sum(row['kind'] == 'payload' for row in result['label_rows']),
                                     resolved_payload_labels=sum(bool(row['resolved']) for row in result['label_rows'] if row['kind'] == 'payload'))
                           for arm, result in results.items()})
    write_record(output / 'decode_summary.tsv', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, default=ROOT / 'data/prepared/F_clean.npz')
    parser.add_argument('--model', type=Path, default=ROOT / 'data/prepared/model.npz')
    parser.add_argument('--selection', type=Path, default=ROOT / 'data/prepared/selection.tsv')
    parser.add_argument('--config', type=Path, default=ROOT / 'codebook/model_config.tsv')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/results/decoding')
    parser.add_argument('--partner-disabled', action='store_true')
    args = parser.parse_args()
    with np.load(args.observations, allow_pickle=False) as observations:
        counts = observations['counts'].copy()
        identities = observations['identity'].copy()
        parents = observations['raw_parent_id'].copy()
    with np.load(args.model, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in ('global_probabilities', 'selected_probabilities')}
    results, diagnostics = decode_observations(counts, identities, parents,
                                             model, load_record(args.selection),
                                             load_record(args.config),
                                             partner_enabled=not args.partner_disabled)
    summary = save_decoding(results, diagnostics, args.output, True, not args.partner_disabled)
    print(json.dumps(dict(streams=len(summary['checksums']),
                          checksum_passes=sum(row['digest_matches'] for row in summary['checksums']),
                          coupling_enabled=True,
                          partner_inference_enabled=not args.partner_disabled), sort_keys=True))


if __name__ == '__main__':
    main()
