"""encode the numerical image pair and rebuild the canonical S12 library."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from stream_format import BITS, STRAND_NT, address_pattern, address_word, encode_images, image_content, pilots

ROOT = Path(__file__).resolve().parent




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'codebook/source_images/images_float64.npy')
    parser.add_argument('--output', type=Path, default=ROOT / 'codebook/Lovelace_S12_480.fasta')
    parser.add_argument('--resources', type=Path, default=ROOT / 'analysis/results/encoding')
    args = parser.parse_args()
    images = np.load(args.images, allow_pickle=False)
    fasta = encode_images(images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(fasta)
    args.resources.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        content = image_content(image)
        (args.resources / f'image_{index}_header.bin').write_bytes(content[:64])
        (args.resources / f'image_{index}_payload.bin').write_bytes(content[64:])
        (args.resources / f'image_{index}_stream.bin').write_bytes(content)
    with (args.resources / 'address_manifest.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\r\n')
        writer.writerow(['address', 'header', 'role', 'byte_start', 'address_word', 'address_pattern', 'length'])
        for identity in range(480):
            writer.writerow([identity, f'lovelace_{identity:04d}', f'image_{identity // 224}' if identity < 448 else 'fit_pilot' if identity < 472 else 'qc_pilot', 2 * (identity % 224) if identity < 448 else '', address_word(identity), address_pattern(identity), STRAND_NT])
    with (args.resources / 'pilot_manifest.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['address', 'role'] + [f'cell_{cell}' for cell in range(8)])
        for offset, row in enumerate(pilots()):
            writer.writerow([448 + offset, 'fit' if offset < 24 else 'qc'] + row.tolist())
    print(json.dumps(dict(designs=480, strand_nt=STRAND_NT, fasta_sha256=hashlib.sha256(fasta).hexdigest()), sort_keys=True))


if __name__ == '__main__':
    main()
