#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Copyright (c) 2008-2010, The Regents of the University of California
# Produced by the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
    lzw is used to decode strings compressed using the lzw algorythm.
"""

import struct

import numpy

def decodelzw(encoded):
    """Decompress LZW (Lempel-Ziv-Welch) encoded TIFF strip (byte string).

    The strip must begin with a CLEAR code and end with an EOI code.

    This is an implementation of the LZW decoding algorithm described in (1).
    It is not compatible with old style LZW compressed files like quad-lzw.tif.

    """
    unpack = struct.unpack
    def next_code():
        """Return integer of `bitw` bits at `bitcount` position in encoded."""
        start = bitcount // 8
        s = encoded[start:start+4]
        try:
            code = unpack('>I', s)[0]
        except Exception:
            code = unpack('>I', s + '\x00'*(4-len(s)))[0]
        code = code << (bitcount % 8)
        code = code & mask
        return code >> shr

    switchbitch = { # code: bit-width, shr-bits, bit-mask
        255: (9, 23, int(9*'1'+'0'*23, 2)),
        511: (10, 22, int(10*'1'+'0'*22, 2)),
        1023: (11, 21, int(11*'1'+'0'*21, 2)),
        2047: (12, 20, int(12*'1'+'0'*20, 2)), }
    bitw, shr, mask = switchbitch[255]
    bitcount = 0

    if len(encoded) < 4:
        raise ValueError("strip must be at least 4 characters long")

    if next_code() != 256:
        raise ValueError("strip must begin with CLEAR code")

    code = 0
    result = []
    while 1:
        code = next_code() # ~5% faster when inlining this function
        bitcount += bitw
        if code == 257: # EOI
            break
        if code == 256: # CLEAR
            table = [chr(i) for i in xrange(256)]
            table.extend((0, 0))
            lentable = 258
            bitw, shr, mask = switchbitch[255]
            code = next_code()
            bitcount += bitw
            if code == 257: # EOI
                break
            result.append(table[code])
        else:
            if code < lentable:
                decoded = table[code]
                newcode = table[oldcode] + decoded[0]
            else:
                newcode = table[oldcode]
                newcode += newcode[0]
                decoded = newcode
            result.append(decoded)
            table.append(newcode)
            lentable += 1
        oldcode = code
        if lentable in switchbitch:
            bitw, shr, mask = switchbitch[lentable]

    if code != 257:
        raise ValueError("unexpected end of stream")
    return ''.join(result)



def unpackints(data, dtype, intsize, runlen=0):
    """Decompress byte string to array of integers of any bit size <= 32.

    data : str

    dtype : numpy.dtype or str
        A numpy boolean or integer type.

    intsize : int
        Number of bits per integer.

    runlen : int
        Number of consecutive integers, after which to start at next byte

    """
    # bitarray
    if intsize == 1:
        data = numpy.fromstring(data, '|B')
        data = numpy.unpackbits(data)
        if runlen % 8 != 0:
            data = data.reshape(-1, runlen+(8-runlen%8))
            data = data[:, :runlen].reshape(-1)
        return data.astype(dtype)

    dtype = numpy.dtype(dtype)

    if 32 < intsize < 1:
        raise ValueError("intsize out of range")

    if dtype.kind not in "biu":
        raise ValueError("invalid dtype")

    if intsize > dtype.itemsize * 8:
        raise ValueError("dtype.itemsize too small")

    for i in (8, 16, 32):
        if intsize <= i:
            itembytes = i // 8
            break

    if runlen == 0:
        runlen = len(data) // itembytes
    skipbits = runlen*intsize % 8
    if skipbits:
        skipbits = 8 - skipbits
    shrbits = itembytes*8 - intsize
    bitmask = int(intsize*'1'+'0'*shrbits, 2)
    if dtype.byteorder == '|':
        dtypestr = '=' + dtype.char
    else:
        dtypestr = dtype.byteorder + dtype.char
    unpack = struct.unpack

    l = runlen * (len(data)*8 // (runlen*intsize + skipbits))
    result = numpy.empty((l,), dtype)

    bitcount = 0
    for i in xrange(len(result)):
        start = bitcount // 8
        s = data[start:start+itembytes]
        try:
            code = unpack(dtypestr, s)[0]
        except Exception:
            code = unpack(dtypestr, s + '\x00'*(itembytes-len(s)))[0]
        code = code << (bitcount % 8)
        code = code & bitmask
        result[i] = code >> shrbits
        bitcount += intsize
        if (i+1) % runlen == 0:
            bitcount += skipbits

    return result
