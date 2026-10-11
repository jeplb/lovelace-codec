"""S12 SHA256 covers the prefix followed by the coefficient payload."""

import argparse
import hashlib
import json
from pathlib import Path

from stream_format import inspect



def digest_report(content, bits=6):
    valid, matches, fields, payload = inspect(content, bits)
    content = bytes(content)
    computed = hashlib.sha256(content[:32] + payload).hexdigest()
    return dict(header_valid=valid, digest_matches=matches,
                stored_digest=content[32:64].hex(), computed_digest=computed,
                covered_prefix_bytes=[0, 32], covered_payload_bytes=[64, len(content)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('streams', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.streams:
        print(json.dumps(dict(stream=path.name, **digest_report(path.read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
