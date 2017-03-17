#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
The module lsmreader is a module to read images recorded with a
Zeiss microscope LSM 510

To load an image, simply points the file you want to load.

>>> imageFile = Lsmimage(LSM_FILE)
>>> imageFile.open()

To get the image data of the first channel :

>>> imageData = imageFile.get_image(stack=0,channel=0)

If you have several stacks :

>>> imageData = imageFile.get_image(stack=10,channel=0)

To get the histogram of the image :

>>> [x, y] = imageFile.get_hist()

To avoid all the pixels that have a value below than 1000 (for example).

>>> imageThresh = imageFile.get_threshold(1000)

This command will produce the same matrix as the one from get_image, but with a
mask that prevent pixels below the threshold to be considered.
'''
import pdb
from struct import unpack

import numpy
from scipy import misc, ndimage

import lsmparse

__author__  = "Charles Roduit <charles.roduit@gmail.com>"
__date__ = "25 juillet 2008"
__license__ = "GNU Public License (GPL) version 3"
__version__ = "0.1"
__credits__ = "Higly inspired by LSM_Reader plugin of ImageJ and TiffReader of\
Cytosim"


class Lsmimage:
    '''
    Opens the zeiss microscope image
    self.header contains the header of the opened file
    
    self.image contain the array that difine the image.
    '''
    
    def __new__(cls):
        pass
        
    def __init__(self, filename, time = None):
        self.image = dict()
        self.filename = filename
        self.time = time
        self.angle = 0
        self.stack = 0
        self.channel = 0
        self.nbstack = 0
        self.counter = 0 # will vary between 0 and 100 to show the evolution of
                         # internal state
        self.header = {'Image':[], 'Thumbnail':[], 'CZ LSM info':[]}
        
    def __cmp__(self, other):
        '''
        Sort the files depending on the aquisition time
        '''
        return cmp(self.time, other.time)
        
    def open(self):
        '''
        Load the image in the memory
        '''
        __fid = open(self.filename, 'rb')
        # The first part of the file gives basic informations
        # most important ar im_id, which has to be 42, and im_offset which
        # informs on the position of the first image header.
        [byte_order, im_id, im_offset] = unpack('3H', __fid.read(6))
        byte_order = hex(byte_order) # Only Intel Byte order are supported
        if im_id == 42:
            # There is as much headers as stacks. We scan all the headers. At
            # the end of each header, there is the iformation on the position 
            # of the next header, that we store in im_offset. The last header
            # has an im_offset = 0.
            # Each header is important as they contain the information of the
            # position of the corresponding image (offset).
            while im_offset:
                tmp_header = lsmparse.read_image_header(__fid, im_offset)
                if tmp_header['New Subfile Type']:
                    self.header['Thumbnail'].append(tmp_header)
                else:
                    self.header['Image'].append(tmp_header)
                if tmp_header.has_key('CZ LSM info'):
                    self.header['CZ LSM info'] = tmp_header['CZ LSM info']
                im_offset = unpack('H', __fid.read(2))[0]
            # We send all headers corresponding to the images, we don't care
            # about thmubnails.
            self.image['data'] = self.__read_image(__fid,
                                                    self.header['Image'],
                                                    self.header['CZ LSM info'])
            # print self.image['data']
            self.nbstack = self.image['data'][0].shape[2]
            
    def add_stack(self, matrix_list, where='top'):
        '''
        Adds a stack to the image data
        '''
        new_matrix_list = list()
        for channel_nbr in range(len(self.image['data'])):
            image_shape = list(self.image['data'][channel_nbr].shape)
            image_shape[2] = image_shape[2]+1
            new_image = numpy.zeros(image_shape)
            if where == 'top':
                new_image[:, :, :image_shape[2]-1] = \
                                      self.image['data'][channel_nbr]
                new_image[:, :, image_shape[2]-1] = matrix_list[channel_nbr]
            elif where == 'bottom':
                new_image[:, :, 1:] = self.image['data'][channel_nbr]
                new_image[:, :, 0] = matrix_list[channel_nbr]
            new_matrix_list.append(new_image)
        self.image['data'] = new_matrix_list
        self.nbstack += 1
        
    def close(self):
        '''
        Close the image by cleaning the memory. The image can still be opened
        using the open method
        '''
        del(self.image['data'])
        del(self.header)
        if self.image.has_key('rotated'):
            del(self.image['rotated'])
            
    def threshold(self, value):
        '''
        Computes the threshold of all the images
        '''
        self.image['Threshold'] = list()
        for image in self.image['data']:
            self.image['Threshold'].append(image > value)
            
    def __fft(self, input_image):
        '''
        Computes the fast Fourrier transform of the image sent in argument        
        The method returns the module and the phase image.
        '''
        fft_image = numpy.fft.fftpack.fft2(input_image)
        tmp_module_image = numpy.sqrt(fft_image.real ** 2 + fft_image.imag ** 2)
        tmp_phase_image = numpy.arctan2(fft_image.real, fft_image.imag)
        # reorder the images...
        module_image = numpy.fft.fftshift(tmp_module_image)
        phase_image = numpy.fft.fftshift(tmp_phase_image)
        return [module_image, phase_image]
        
    def get_blob(self):
        '''
        Gets the blob of the last displayed image.
        '''
        matrix = self.get_thresholded_image()
        return ndimage.label(matrix)
        
    def get_thresholded_image(self, value = None):
        '''
        return the thresholded image of the original image. Value is the min
        value to be considered. If no value is send, the distribution of the
        pixel intensity in the whole image is considered
        '''
        return (self.get_image() > self.get_thresholded_value(value))
        
    def get_thresholded_value(self, value=None):
        '''
        return the threshold below which the noise is estimated.
        Value (default = 1.2) is the portion of the pixels to consider.
        '''
        if value == None:
            value = 1.2
            
        original_image = self.get_image()
        vector = original_image.copy()
        vector = numpy.reshape(vector, vector.size)
        vector.sort()
        return vector[numpy.ceil(len(vector) / value)]
        
    def get_fft(self, stack=-1, channel=-1, precision=1):
        '''
        Computes the fast Fourrier transform of the image. You can specify the
        stack and the channel you want to compute. If stack and channel are not
        specified, it computes the fft on the latest displayed image.
        
        The method returns the module and the phase image.
        '''
        initial_image = self.get_image(stack, channel, precision, angle = 0)
        [module_image, phase_image] = self.__fft(initial_image)
        module_image = self.__rotate_image(module_image, precision = precision)
        phase_image = self.__rotate_image(phase_image, precision = precision)
        return [module_image, phase_image]
        
    def get_hist(self, stack=0, length=100):
        '''
        Returns the histogram of the image pixel values
        '''
        vect_y, vect_x = numpy.histogram(self.image['data'][stack], length)
        return vect_x, vect_y
        
    def get_image(self, stack=-1, channel=-1, precision=1, angle=None):
        '''
        Returns the array matrix of the image
        '''
        if angle == None:
            angle = self.angle
        if stack + 1:
            self.stack = stack
        if channel + 1:
            self.channel = channel
        if angle or (precision != 1):
            return self.__rotate_image(self.image['data']
                                                [self.channel]
                                                [:, :, self.stack],
                                      angle, precision)
        else:
            return (self.image['data'][self.channel][:, :, self.stack])
            
    def __rotate_image(self, matrix, angle=None, precision=1):
        '''
        Rotates the given matrix to a specified angle (in degree).
        matrix      : 2D numpy array
        [angle]     : Rotation angle of the matrix. If none is specified, it
                      will use the previously defined angle
        [precision] : relative size of the matrix used for the rotation. For
                      example, with a precision set to 0.5, the matrix that will
                      rotate has half the size of the original
        '''
        if angle == None:
            angle = self.angle
        # The rotation is done by converting the matrix in a PIL object
        # and then applying the rotation to it to finally reconvert
        # into a numpy matrix.
        #
        # A mask is generated to reject pixels that are outside the
        # original image. It finally results in a masked array. Thanks
        # to that, we can do matrix manipulation without taking care
        # of suplementary pixels generated by the rotation.
        
        tmp_image = misc.toimage(matrix, mode = 'F')
        new_size = (numpy.array(tmp_image.size) * precision).round()
        tmp_image = tmp_image.resize(new_size)
        tmp_mask = misc.toimage(numpy.ones(new_size).transpose(),
                                mode = 'F')
        array_image = misc.fromimage(tmp_image.rotate(angle, expand = 1))
        array_mask = abs(misc.fromimage(tmp_mask.rotate(angle,
                                                        expand = 1))
                        -1)
        return numpy.ma.array(array_image, mask=array_mask)
            
    def get_projection(self, precision=1, threshold=None):
        '''
        Returns the projection in the x axis of the image
        '''
        if threshold != None:
            matrix = self.get_threshold(threshold, precision)
        else:
            matrix = self.get_image(precision = precision)
        return matrix.sum(0)
        
    def get_rotation(self):
        '''
        Returns the current rotation of the image
        '''
        return self.angle
        
    def get_threshold(self, threshold, precision=1):
        '''
        Returns an binary matrix where 0 depicts pixels lower than 
        the threshold and 1 values higher or equal.
        '''
        current_image = self.get_image(precision = precision)
        thresh = (current_image <= threshold)
        try:
            new_mask = current_image.mask + thresh
        except AttributeError:
            new_mask=thresh
        return numpy.ma.array(current_image, mask = new_mask)
        
    def set_rotation(self, angle):
        '''
        Set the rotation of the image
        '''
        self.angle = angle
    
    def __read_image(self, fid, headers, cz_info):     
        '''
        Read the images and return them as matrices
        '''
        self.counter = 0
        totalImages = 0
        currentImage = 0
        for this_header in headers:
            totalImages = totalImages + this_header['Sample / Pixel']
        print cz_info
        # raw_input()
        # if cz_info['Scan Type'] == 0:
        if cz_info.has_key('Scan Type'):
            if headers[0]['Tiff Sample Format'] == 1:
                data = list()
                ntyp = eval("numpy.dtype(numpy.uint%d)" % headers[0]['Bit / Sample'])
                for channel in range(headers[0]['Sample / Pixel']):
                    data.append(numpy.empty((headers[0]['Width'] *
                                             headers[0]['Length'] *
                         len(headers)),ntyp))
                z_image = 0
                size_image = this_header['Width'] * this_header['Length']
                for this_header in headers:
                    offset = this_header['Strip Offset']
                    fid.seek(offset)
                    for channel in range(this_header['Sample / Pixel']):
                        this_stack = lsmparse.read_stack(fid,
                                               this_header['Width'],
                                               this_header['Length'],
                                               this_header['Bit / Sample'],
                                               this_header['Compression'])
                        data[channel][size_image * z_image : size_image * (z_image + 1)] = this_stack
                        currentImage += 1
                        self.counter = currentImage / float(totalImages)
                    z_image += 1
            for channel in range(this_header['Sample / Pixel']):
                data[channel].shape = (len(headers), this_header['Width'], this_header['Length'])
                if headers[0]['Predictor'] == 2:
                    if headers[0]['Bit / Sample'] == 8:
                        dtype = 'B'
                    elif headers[0]['Bit / Sample'] == 16:
                        dtype = 'H'
                    numpy.cumsum(data[channel], axis=2, dtype=dtype, out=data[channel])
                data[channel] = data[channel].transpose(1,2,0)
            return data

if __name__ == '__main__':
    import doctest
    LSM_FILE = '../../Example_file/INRIA.lsm'
    doctest.testmod()

