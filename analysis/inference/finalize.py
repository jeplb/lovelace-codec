"""complete measured coefficient diagnostics and write the supplementary section."""


import numpy as np

try:
    from .analyze import ARMS, CURRENT, HERE, INTEGRATION, read_tsv, sha, write_record, write_tsv
except ImportError:
    from analyze import ARMS, CURRENT, HERE, INTEGRATION, read_tsv, sha, write_record, write_tsv
from tsv_records import read_output


def main():
    selection = read_output(CURRENT / 'fitting/selection.tsv')
    write_record(HERE / 'selected_model.tsv', {key: selection[key]
        for key in ('selected_lambda', 'selected_candidate', 'selected_name', 'config_sha256')})
    scales = {image: float(np.load(INTEGRATION / f'evidence/codec/source/S12/image_{index}.npz')['scale'])
        for index, image in enumerate('AB')}
    ledger = read_tsv(HERE / 'scalar_evidence_ledger.tsv')
    for row in ledger:
        encoded, emitted = int(row['encoded_six_bit_label']), int(row['emitted_six_bit_label'])
        scale = scales[row['image']]
        # same frozen column order for fresh and repeated exports
        row.pop('label_spacing_component_error', None)
        row.update(encoded_normalized_component=-1 + 2 * encoded / 63,
            estimated_normalized_component=-1 + 2 * emitted / 63,
            normalized_component_error=(-1 + 2 * emitted / 63) - (-1 + 2 * encoded / 63),
            designed_coefficient_scale=scale,
            encoded_dequantized_component=-scale + 2 * scale * encoded / 63,
            estimated_dequantized_component_designed_scale=-scale + 2 * scale * emitted / 63,
            dequantized_component_error_designed_scale=(-scale + 2 * scale * emitted / 63) - (-scale + 2 * scale * encoded / 63),
            label_spacing_component_error=2 * (emitted - encoded) / 63)
    write_tsv(HERE / 'scalar_evidence_ledger.tsv', ledger)
    ablation = read_tsv(HERE / 'partner_ablation.tsv')
    for row in ablation:
        arm, image = row['method'], row['image']
        selected = sorted([item for item in ledger if item['method'] == arm and item['image'] == image],
                          key=lambda item: int(item['scalar_index']))
        content = (CURRENT / f'predictions/{arm}_{image}.bin' if row['partner_information'] == 'True'
            else HERE / f'no_partner_{arm}_{image}.bin').read_bytes()
        labels = np.unpackbits(np.frombuffer(content[64:], dtype=np.uint8), bitorder='big').reshape(512, 6) @ (2 ** np.arange(5, -1, -1))
        truth = np.array([int(item['encoded_six_bit_label']) for item in selected])
        support = np.array([item['direct_support'] == 'complete' for item in selected])
        errors = (-1 + 2 * labels.astype(float) / 63) - (-1 + 2 * truth.astype(float) / 63)
        scale = scales[image]
        physical_errors = (-scale + 2 * scale * labels.astype(float) / 63) - (-scale + 2 * scale * truth.astype(float) / 63)
        row.update(coefficient_error_available_components=int(support.sum()),
            coefficient_rmse_normalized=float(np.sqrt(np.mean(errors ** 2))),
            coefficient_rmse_available_normalized=float(np.sqrt(np.mean(errors[support] ** 2))),
            coefficient_rmse_all_designed_scale=float(np.sqrt(np.mean(physical_errors ** 2))),
            coefficient_rmse_available_designed_scale=float(np.sqrt(np.mean(physical_errors[support] ** 2))),
            numerical_coefficient_error_scale='designed coefficient scale')
    write_tsv(HERE / 'partner_ablation.tsv', ablation)
    sample_lookup = {(float(row['fraction']), int(row['draw'])): row for row in read_tsv(HERE / 'sample_summary.tsv')}
    draw_rows = read_tsv(HERE / 'read_sampling_draws.tsv')
    for row in draw_rows:
        sample = sample_lookup[float(row['fraction']), int(row['draw'])]
        row.update(parent_set_sha256=sample['parent_set_sha256'],
            sampled_parent_count=sample['sampled_parent_count'], admitted_unit_count=sample['admitted_unit_count'],
            admitted_parent_count=sample['admitted_parent_count'])
    write_tsv(HERE / 'read_sampling_draws.tsv', draw_rows)
    findings = read_output(HERE / 'findings.tsv')
    for group, enabled in (('full_data', 'True'), ('partner_disabled', 'False')):
        for row in findings[group]:
            companion = next(item for item in ablation if item['method'] == row['method'] and item['image'] == row['image']
                             and item['partner_information'] == enabled)
            for field in ('coefficient_rmse_normalized', 'coefficient_rmse_available_normalized', 'coefficient_rmse_all_designed_scale',
                          'coefficient_rmse_available_designed_scale'):
                row[field] = float(companion[field])
            row['coefficient_error_available_components'] = int(companion['coefficient_error_available_components'])
    saved = np.load(HERE / 'sampling_predictions.npz', allow_pickle=False)
    full = saved['streams'][saved['fractions'] == 1]
    equal_streams = 0
    for method_index, arm in enumerate(ARMS):
        for image_index, image in enumerate('AB'):
            expected = np.frombuffer((CURRENT / f'predictions/{arm}_{image}.bin').read_bytes(), dtype=np.uint8)
            np.testing.assert_array_equal(full[:, method_index, image_index], np.tile(expected, (50, 1)))
            equal_streams += 50
    findings['full_data_sampled_stream_checks'] = equal_streams
    write_record(HERE / 'findings.tsv', findings)
    figure_sources = []
    for panel, filename in (('a', 'support_resolution.tsv'), ('b', 'partner_ablation.tsv'), ('c', 'read_sampling_summary.tsv')):
        for row in read_tsv(HERE / filename):
            if panel == 'c' and row['metric'] not in ('resolved_labels', 'exact_scalar_matches', 'image_rmse_normalized_designed'):
                continue
            actual_panel = 'd' if row.get('metric') == 'image_rmse_normalized_designed' else panel
            figure_sources.append(dict(panel=actual_panel, **row))
    write_tsv(HERE / 'SuppFigure_Lovelace_Inference_source_data.tsv', figure_sources)


if __name__ == '__main__':
    main()
