# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Nuclei Quantification
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Sophie Ribes <sophie.ribes@inria.fr>,
#                            Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np
import pandas as pd

import scipy.ndimage as nd

from vplants.tissue_nukem_3d.epidermal_maps import compute_local_2d_signal

def image_layer_slices(positions, images, microscope_orientation=1):

    X = positions[:,0]
    Y = positions[:,1]
    Z = positions[:,2]
    
    size = np.array(images[0].shape)
    resolution = microscope_orientation*np.array(images[0].resolution)

    xx, yy = np.mgrid[0:size[0]*resolution[0]:resolution[0],0:size[1]*resolution[1]:resolution[1]]       
    extent = yy.min(),yy.max(),xx.min(),xx.max()
    
    zz = compute_local_2d_signal(np.transpose([X,Y]),np.transpose([xx,yy],(1,2,0)),Z)
    outer_mask = np.where(np.isnan(zz))
    zz[outer_mask] = 0

    image_coords = (np.transpose([xx,yy,zz],(1,2,0))/resolution).astype(int)
    image_coords = tuple(np.transpose(np.concatenate(image_coords)))

    image_slices = []
    for img in images:
        img_slice = np.transpose(img[image_coords].reshape(xx.shape))
        img_slice[outer_mask] = 0
        image_slices += [img_slice]
    
    return image_slices,xx,yy
                
