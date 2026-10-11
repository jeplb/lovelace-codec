"""Reproduce the original Fourier proof-of-concept source images.

Install dependencies with `python3 -m pip install numpy scipy pillow`.
Run `python3 generate_images.py` to regenerate source_images, or use
`--output-dir PATH` to write a separate copy.

This implements the original construction documented in the Lovelace image
preregistration. The historical repositories saved the recipe and images,
but did not include their generator script. No saved image is read here.

Image A is an asymmetric arrow with a notch, disk and fine bars. It is drawn
at eightfold resolution, area-averaged, then smoothed with a periodic Gaussian
of standard deviation 0.8 pixel. Image B shifts A by three rows and five
columns, preserving Fourier magnitudes while changing phases.

The float64 arrays are the encoder inputs. PNGs are rounded display previews.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def generate_images():
    """Return the original pair of 32 by 32 float64 images in A, B order."""
    coordinates = (np.arange(32 * 8) + 0.5) / 8
    x, y = np.meshgrid(coordinates, coordinates)
    image = np.full(x.shape, 0.06, dtype=np.float64)

    # original seven-vertex polygon, shaft plus triangular head
    shaft = (x >= 4) & (x < 17) & (y >= 12) & (y < 18)
    head = (x >= 17) & (11 * np.abs(y - 15) <= 9 * (28 - x))
    image[shaft | head] = 0.82
    image[(x >= 7) & (x < 11) & (y >= 12) & (y < 15)] = 0.06
    image[(x - 8) ** 2 + (y - 25) ** 2 < 2.2**2] = 0.62
    bars = ((x >= 22) & (x < 23) & (y >= 25) & (y < 28)) | (
        (x >= 25) & (x < 26) & (y >= 26) & (y < 29)
    )
    image[bars] = 0.95

    image = image.reshape(32, 8, 32, 8).mean(axis=(1, 3))
    image = gaussian_filter(image, sigma=0.8, mode="wrap")
    return np.stack((image, np.roll(image, (3, 5), axis=(0, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "codebook/source_images",
    )
    args = parser.parse_args()
    images = generate_images()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "images_float64.npy", images, allow_pickle=False)
    for label, image in zip("AB", images):
        pixels = np.rint(image * 255).astype(np.uint8)
        Image.fromarray(pixels).save(args.output_dir / f"image_{label}_preview.png")
    print(f"Wrote the original source arrays and PNG previews to {args.output_dir}")


if __name__ == "__main__":
    main()
