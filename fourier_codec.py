"""canonical Fourier representation and endpoint inclusive quantization."""

import numpy as np

def forward(image):
    mean = float(np.mean(image))
    full = np.fft.fftshift(np.fft.fft2(image-mean,norm='ortho'))
    block = full[8:24,8:24].copy()
    scale = float(max(abs(block.real).max(),abs(block.imag).max()))
    return mean, scale, full, block


def inverse(block, mean):
    full=np.zeros((32,32),dtype=np.complex128);full[8:24,8:24]=np.asarray(block).reshape(16,16)
    z=np.fft.ifft2(np.fft.ifftshift(full),norm='ortho')
    return z.real+mean, float(np.sqrt(np.mean(z.imag**2)))


def quantize(block,scale,bits):
    coords=np.stack((block.real,block.imag),axis=-1)
    if scale==0: return np.full(coords.shape,2**(bits-1),np.uint8)
    return np.clip(np.floor((coords/scale+1)*((2**bits-1)/2)+.5),0,2**bits-1).astype(np.uint8)


def dequantize(codes,scale,bits):
    coords=-scale+2*scale*np.asarray(codes,dtype=np.float64)/(2**bits-1)
    return coords[...,0]+1j*coords[...,1]

