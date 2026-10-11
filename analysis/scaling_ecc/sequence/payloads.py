"""frozen lovelace inputs and scorer-only reconstruction."""
import hashlib
from pathlib import Path
import numpy as np
from fourier_codec import dequantize, inverse
from stream_format import unpack
from tsv_records import read_output

def load_frozen_payloads(destination=None):
    """load the archived study inputs without reopening historical sources."""
    destination = Path(destination) if destination is not None else Path(__file__).resolve().parent / "payloads"
    manifest = read_output(destination / "payload_source_manifest.tsv")
    objects = []
    for row in manifest:
        stem = row["codec"] + "_" + row["name"]
        payload = (destination / (stem + ".bin")).read_bytes()
        metadata = read_output(destination / (stem + ".metadata.tsv"))
        if len(payload) != row["payload_bytes"] or hashlib.sha256(payload).hexdigest() != row["payload_sha256"]:
            raise ValueError("frozen stored-object bytes do not match the study manifest")
        if metadata != row["metadata"]:
            raise ValueError("frozen reconstruction metadata do not match the study manifest")
        objects.append(dict(codec=row["codec"], name=row["name"], payload=payload,
                            metadata=metadata, sources=row["sources"]))
    return objects

def unpack_levels(payload, bits, shape):
    binary = np.unpackbits(np.frombuffer(bytes(payload), np.uint8), bitorder="big")
    count = int(np.prod(shape))
    if len(binary) < count * bits:
        raise ValueError("stored stream is incomplete")
    binary = binary[:count * bits].reshape(count, bits)
    return (binary @ 2 ** np.arange(bits - 1, -1, -1)).astype(np.uint8).reshape(shape)

def reconstruct_lovelace(payload, metadata):
    labels = unpack(bytes(payload), bits=6)
    scale, mean = float(metadata["scale"]), float(metadata["mean"])
    if not np.isfinite(scale) or not 0 < scale <= 32 or not np.isfinite(mean) or not 0 <= mean <= 1:
        raise ValueError("recovered reconstruction metadata is invalid")
    image, imaginary = inverse(dequantize(labels, scale, bits=6), mean)
    return image, labels, imaginary

def object_metrics(codec, observed_payload, observed_metadata, truth_payload, truth_metadata):
    """score frozen recovered outputs, without repairing missing or wrong bytes."""
    metrics = {}
    if observed_payload is None or len(observed_payload) != len(truth_payload) or any(v is None for v in observed_payload):
        return metrics
    if "shape" not in truth_metadata or not isinstance(observed_metadata, dict):
        return metrics
    if codec == "lovelace":
        expected_image, expected_labels, _ = reconstruct_lovelace(truth_payload, truth_metadata)
        try:
            observed_image, observed_labels, imaginary = reconstruct_lovelace(observed_payload, observed_metadata)
        except (KeyError, TypeError, ValueError):
            return metrics
        metrics.update(component_error_fraction=float((observed_labels != expected_labels).mean()),
                       component_mae=float(np.mean(abs(observed_labels.astype(float) - expected_labels))),
                       image_rmse=float(np.sqrt(np.mean((observed_image - expected_image) ** 2))),
                       imaginary_rms=imaginary,
                       reconstruction_metadata_exact=bool(observed_metadata == truth_metadata))
    return metrics
