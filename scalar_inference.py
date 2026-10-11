"""canonical scalar confidence, resolution and joint pair inference."""

import numpy as np

DIGITS = np.stack((np.arange(64) // 16, np.arange(64) // 4 % 4,
                   np.arange(64) % 4), axis=1)

def normalize(log_scores):
    weights = np.exp(log_scores - np.max(log_scores))
    return weights / weights.sum()


def confidence(posterior, selected):
    other = np.delete(posterior, selected).max()
    probability = float(posterior[selected])
    margin = probability - float(other)
    return probability, margin


def resolved(posterior, selected, config):
    p, margin = confidence(posterior, selected)
    return p >= config['inference']['resolved_probability'] and margin > config['inference']['tie_tolerance']


def provenance(posterior, selected, baseline, read_support, config,
               coupled=False, fixed=False):
    if not resolved(posterior, selected, config):
        return 'unresolved'
    if fixed:
        return 'format_supplied'
    if coupled and not read_support:
        return 'partner_supplied'
    if coupled and selected != baseline:
        return 'partner_changed'
    return 'read_supported' if read_support else 'unresolved'


def joint_pair(first, second, psi):
    log_joint = first[:, None] + second[None, :] + np.log(psi)
    joint = normalize(log_joint)
    selected = np.unravel_index(np.argmax(log_joint), log_joint.shape)
    return selected, joint.sum(axis=1), joint.sum(axis=0)


def digits_to_bytes(digits):
    groups = np.asarray(digits, dtype=np.uint8).reshape(-1, 4)
    return (groups @ np.array([64, 16, 4, 1], dtype=np.uint8)).astype(np.uint8).tobytes()

