#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
lsmparse is used to parse the lsm files in order to extract the image matrices
'''

from struct import unpack

import numpy

import lzw

def read_tag(fid, position, method='Normal'):
    '''
    Read the tags in header
    '''
    fid.seek(position)
    if method == 'Normal':
        tag = unpack('=6H', fid.read(2 * 6))
    elif method == 'Long':
        tag = unpack('=3I', fid.read(4 * 3))
    elif method == 'LSMInfo':
        fid.seek(position)
        tag = {}
        nb_tag = 10
        _temp = unpack('=' + str(nb_tag) + 'I', fid.read(4 * nb_tag))
        tag['Magic Number'] = hex(int(_temp[0]))
        tag['Structure Size'] = int(_temp[1])
        tag['Dimension X'] = int(_temp[2])
        tag['Dimension Y'] = int(_temp[3])
        tag['Dimension Z'] = int(_temp[4])
        tag['ThumbnailX'] = int(_temp[8])
        tag['Thumbnail Y'] = int(_temp[9])
        _temp = unpack('=3d', fid.read(24))
        tag['Voxel Size X'] = _temp[0]
        tag['Voxel Size Y'] = _temp[1]
        tag['Voxel Size Z'] = _temp[2]
        nb_tag = 6
        _temp = unpack('=' + str(nb_tag) + 'I', fid.read(4 * nb_tag))
        tag['Scan Type'] = int(_temp[0])
        tag['Data Type'] = int(_temp[1])
        tag['Offset Vector Overlay'] = int(_temp[2])
        tag['Offset Input Lut'] = int(_temp[3])
        tag['Offset Output Lut'] = int(_temp[4])
        tag['Offset Channel Color'] = int(_temp[5])
        _temp = unpack('=d', fid.readline(8))
        tag['Time Interval'] = int(_temp[0]) ## float 64 <-- a changer
        nb_tag = 9
        _temp = unpack('=' + str(nb_tag) + 'I', fid.read(4 * nb_tag))
        tag['Offset Channel Data Type'] = int(_temp[0])
        tag['Offset Scan Information'] = int(_temp[1])
        tag['Offset Ks Data'] = int(_temp[2])
        tag['Offset Time Stamps'] = int(_temp[3])
        tag['Offset Event List'] = int(_temp[4])
        tag['Offset ROI'] = int(_temp[5])
        tag['Offset Bleach ROI'] = int(_temp[6])
        tag['Offset Next Recording'] = int(_temp[7])
        tag['Reserved'] = int(_temp[8])
    return tag
    
def convert_type(tiff_type):
    '''
    Converts the tiff type in python type
    '''
    if tiff_type == 1:
        nb_bytes = 1
    elif tiff_type == 2:
        nb_bytes = 1
    elif tiff_type == 3:
        nb_bytes = 2
    elif tiff_type == 4:
        nb_bytes = 4
    elif tiff_type == 5:
        nb_bytes = 8
    elif tiff_type == 11:
        nb_bytes = 4
    elif tiff_type == 12:
        nb_bytes = 8
    else:
        nb_bytes = None
    return nb_bytes
    
def read_image_header(fid, start_position):
    '''
    Read the image header
    '''
    header = dict()
    # Default values.
    header['Tiff Sample Format'] = 1
    header['Predictor'] = 1
    fid.seek(start_position)
    nb_tag = unpack('=H', fid.read(2))[0]
    for i in xrange(nb_tag):
        current_tag_position = start_position + 2 + i * 12
        tag = read_tag(fid, current_tag_position)
        if tag[0] == 254:
            header['New Subfile Type'] = tag[4]
        elif tag[0] == 256:
            header['Width'] = tag[4]
        elif tag[0] == 257:
            header['Length'] = tag[4]
        elif tag[0] == 258:
            nb_bytes = convert_type(tag[1])
            if nb_bytes*tag[2] > 4:      # This is an offset
                new_tag = read_tag(fid, tag[4])
                header['Bit / Sample'] = new_tag[1]
            else:
                header['Bit / Sample'] = tag[4]
        elif tag[0] == 259:
            header['Compression'] = tag[4]
        elif tag[0] == 262:
            header['Photometric Interpretation'] = tag[4]
        elif tag[0] == 273:
            nb_bytes = convert_type(tag[1])
            if nb_bytes * tag[2] > 4:    # This is an offset
                new_tag = read_tag(fid, tag[4], 'Long')
                header['Strip Offset'] = new_tag[0]
            else:
                new_tag = read_tag(fid, current_tag_position, 'Long')
                header['Strip Offset'] = new_tag[2]
            header['Strip Number'] = tag[2]
        elif tag[0] == 277:
            header['Sample / Pixel'] = tag[4]
        elif tag[0] == 279:
            new_tag = read_tag(fid, current_tag_position, 'Long')
            nb_bytes = convert_type(tag[1])
            header['Strip Byte Counts'] = new_tag[2]
            if nb_bytes * tag[2] > 4:    # This is an offset
                new_tag = read_tag(fid, tag[4], 'Long')
                header['Strip Byte Counts'] = new_tag[0]
            else:
                header['Strip Byte Counts'] = new_tag[2]
        elif tag[0] == 317:
            header['Predictor'] = tag[4]
        elif tag[0] == 320:
            header['Colormap'] = tag[4]
        elif tag[0] == 339:
            header['Tiff Sample Format'] = tag[4]
        elif tag[0] == 34412:
            header['CZ LSM info offset'] = tag[4]
            new_tag = read_tag(fid, header['CZ LSM info offset'], 'LSMInfo')
            header['CZ LSM info'] = new_tag
    if header.has_key('CZ LSM info'):
        header['Image Size X'] = (header['Width'] *
                                  header['CZ LSM info']['Voxel Size X'] *
                                  1e6)
        header['Image Size Y'] = (header['Length'] *
                                  header['CZ LSM info']['Voxel Size Y'] *
                                  1e6)
    fid.seek(start_position + 2 + nb_tag * 12)
    return header
    
def read_stack(fid, width, length, bits_per_samples, compression):
    '''
    Reads one stack of the image file
    '''
    if bits_per_samples == 8:
        s = fid.read(length * width)
        order = 'B'
    elif bits_per_samples == 16:
        s = fid.read(length * width * 2)
        order = 'H'
    if compression == 5:
        s = lzw.decodelzw(s)
    data = numpy.fromstring(s, order)
    return data
    
def read_image(fid, headers, cz_info):
    '''
    Read the images and return them as matrices
    '''
    if cz_info['Scan Type'] == 0:
        if headers[0]['Tiff Sample Format'] == 1:
            data = list()
            for channel in range(headers[0]['Sample / Pixel']):
                data.append(numpy.empty((headers[0]['Width'],
                                         headers[0]['Length'],
                                         len(headers))))
            z_image = 0
            for this_header in headers:
                offset = this_header['Strip Offset']
                fid.seek(offset)
                for channel in range(this_header['Sample / Pixel']):
                    this_stack = read_stack(fid,
                                           this_header['Width'],
                                           this_header['Length'],
                                           this_header['Bit / Sample'])
                    data[channel][:, :, z_image] = this_stack
                z_image += 1
        return data
        
def find_num_after(string, pattern):
    '''
    Finds a number in a string after the given pattern.
    
    >>> find_num_after('96toto43','toto')
    >>> 43
    
    '''
    index = string.find(pattern)
    if index + 1:
        index = index + len(pattern)
    else :
        return -1
    number = string[index:index+1]
    while number.isalnum():
        number = string[index:index + len(number) + 1]
    number = number[0:-1]
    return int(number)
    
def find_num_before(string, pattern):
    '''
    Finds a number in a string before the given pattern.

    >>> find_num_before('19taratata96toto54','toto')
    >>> 96
    '''
    index = string.find(pattern)
    if index + 1:
        index = string.find(pattern) - 1
    else:
        return -1
    number = string[index:index+1]
    while number.isalnum():
        number = string[index-len(number):index+1]
    number = number[1:]
    return int(number)
