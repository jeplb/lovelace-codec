"""recover unclipped images and compute canonical reference metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from checksum import digest_report, inspect
from fourier_codec import dequantize, inverse
from stream_format import unpack

def ssim(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    ua=uniform_filter(a,7);ub=uniform_filter(b,7)
    va=(uniform_filter(a*a,7)-ua*ua)*49/48
    vb=(uniform_filter(b*b,7)-ub*ub)*49/48
    cov=(uniform_filter(a*b,7)-ua*ub)*49/48
    m=((2*ua*ub+.01**2)*(2*cov+.03**2))/((ua*ua+ub*ub+.01**2)*(va+vb+.03**2))
    return float(m[3:-3,3:-3].mean())


def metrics(reconstruction,correct,wrong):
    mse=float(np.mean((reconstruction-correct)**2));wrong_mse=float(np.mean((reconstruction-wrong)**2))
    return dict(mse=mse,psnr_db=float(-10*np.log10(mse)) if mse else None,
                ssim=ssim(reconstruction,correct),wrong_mse=wrong_mse,wrong_ssim=ssim(reconstruction,wrong),
                phase_control_pass=bool(mse<=.5*wrong_mse))


def coefficient_metrics(observed,reference):
    mask=abs(reference)>.01*abs(reference).max()
    delta=np.angle(observed*np.conj(reference))
    return dict(complex_mse=float(np.mean(abs(observed-reference)**2)),
        magnitude_mse=float(np.mean((abs(observed)-abs(reference))**2)),
        phase_mae_rad=float(np.mean(abs(delta[mask]))),phase_evaluated_coefficients=int(mask.sum()))



def reconstruct_stream(content):
    valid, matches, fields, payload = inspect(content, 6)
    if not valid:
        raise ValueError('invalid S12 header or payload length')
    codes = unpack(payload, 6)
    block = dequantize(codes, fields[9], 6)
    image, imaginary_rms = inverse(block, fields[8])
    return dict(image=image, codes=codes, coefficients=block,
                header_valid=valid, digest_matches=matches,
                mean=fields[8], scale=fields[9], imaginary_rms=imaginary_rms)



def from_components(values, mean=0.0, scale=1.0):
    values = np.asarray(values, dtype=np.float64).reshape(16, 16, 2)
    block = scale * (values[..., 0] + 1j * values[..., 1])
    return inverse(block, mean)[0]

def pixel_metrics(observed, reference):
    delta = np.asarray(observed, np.float64) - np.asarray(reference, np.float64)
    mse = float(np.mean(delta ** 2))
    return dict(mae=float(np.mean(abs(delta))), mse=mse,
                rmse=float(np.sqrt(mse)),
                psnr_db=float(-10 * np.log10(mse)) if mse else None,
                ssim=ssim(observed, reference))


def zero_reference(mean=0.0):
    return inverse(np.zeros((16, 16), np.complex128), mean)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('streams', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('analysis/results/reconstruction'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.streams:
        value = reconstruct_stream(path.read_bytes())
        np.save(args.output / (path.stem + '.npy'), value['image'], allow_pickle=False)
        print(json.dumps(dict(stream=path.name, **digest_report(path.read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
