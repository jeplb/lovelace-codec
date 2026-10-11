"""canonical S12 metadata, payload packing, addresses and strand construction."""

import hashlib
import struct
import numpy as np
from fourier_codec import forward, quantize

PREFIX = struct.Struct('>4sBBBBBBHddI')
HEADER_BYTES = 64
BITS = 6
BASES = 'ACT'
P1 = 'CCCAATTAATACGTTGCCTACTCCG'
P2 = 'CGCTATCAGAACAAGAGAAGCACTA'
STRAND_NT = 298
STATES = np.asarray(((3, 7, 10), (3, 10, 7), (7, 3, 10), (7, 10, 3)), dtype=np.int64)

def pack(codes,bits):
    c=np.asarray(codes,np.uint8).reshape(-1)
    assert c.size==512 and c.max()<2**bits
    binary=(c[:,None]>>np.arange(bits-1,-1,-1))&1
    return np.packbits(binary.reshape(-1),bitorder='big').tobytes()


def unpack(payload,bits):
    if len(payload)!=64*bits:raise ValueError('Bad fixed payload length')
    binary=np.unpackbits(np.frombuffer(payload,np.uint8),bitorder='big').reshape(512,bits)
    return (binary@2**np.arange(bits-1,-1,-1)).astype(np.uint8).reshape(16,16,2)


def make_header(payload,mean,scale,bits):
    prefix=PREFIX.pack(b'LVIM',1,32,32,16,16,bits,1,mean,scale,len(payload))
    assert len(prefix)==32
    return prefix+hashlib.sha256(prefix+payload).digest()


def address_word(message):
    if not 0<=message<2048:raise ValueError('11-bit address required')
    bits=[0]*17
    positions=[i for i in range(1,16) if i not in (1,2,4,8)]
    for shift,pos in enumerate(positions):bits[pos]=(message>>shift)&1
    for parity in (1,2,4,8):bits[parity]=sum(bits[i] for i in range(1,16) if i&parity)%2
    bits[16]=sum(bits[1:16])%2
    return sum(bits[i]<<(i-1) for i in range(1,17))^0xF6CF


def address_pattern(message):
    word,state,result=address_word(message),0,''
    for shift in range(16):
        state=(state+1+((word>>shift)&1))%3;result+=BASES[state]
    return result


def pilots():return np.asarray([[(i+j)%4 for j in range(8)] for i in range(32)],np.uint8)


def strand(identity,digits,states=STATES):
    seq=P1+''.join(b*5 for b in address_pattern(identity))+'GGGG'
    seq+=''.join(''.join(b*int(n) for b,n in zip(BASES,states[int(s)])) for s in digits)
    seq+='GGGG'+P2
    assert len(seq)==STRAND_NT
    return seq


def encode_images(images):
    """Return exact FASTA bytes for a pair of 32 by 32 numerical images."""
    images = np.asarray(images, dtype=np.float64)
    if images.shape != (2, 32, 32):
        raise ValueError("The library requires images with shape (2, 32, 32)")
    digits = []
    for image in images:
        content = np.frombuffer(image_content(image), dtype=np.uint8)
        pairs = ((content[:, None] >> np.array([6, 4, 2, 0])) & 3).astype(np.uint8)
        digits.extend(pairs.reshape(-1, 8))
    digits.extend([(pilot + cell) % 4 for cell in range(8)] for pilot in range(32))
    # explicit crlf keeps the ordered fasta bytes on every operating system
    records = [
        f">lovelace_{identity:04d}\r\n{strand(identity, row)}\r\n"
        for identity, row in enumerate(digits)
    ]
    return "".join(records).encode("ascii")


def inspect(content,bits):
    content=bytes(content);header=content[:64];payload=content[64:]
    fields=PREFIX.unpack(header[:32]) if len(header)==64 else None
    valid=bool(fields and fields[:8]==(b'LVIM',1,32,32,16,16,bits,1) and
        np.isfinite(fields[8]) and 0<=fields[8]<=1 and np.isfinite(fields[9]) and 0<fields[9]<=32 and
        fields[10]==64*bits and len(payload)==64*bits)
    success=valid and header[32:]==hashlib.sha256(header[:32]+payload).digest()
    return valid,bool(success),fields,payload



def image_content(image):
    image = np.asarray(image, dtype=np.float64)
    if image.shape != (32, 32):
        raise ValueError('each source image must have shape (32, 32)')
    if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
        raise ValueError('image values must be finite and within [0, 1]')
    mean, scale, full, block = forward(image)
    if scale == 0:
        raise ValueError('the S12 header requires a positive retained Fourier scale')
    codes = quantize(block, scale, BITS)
    payload = pack(codes, BITS)
    return make_header(payload, mean, scale, BITS) + payload

