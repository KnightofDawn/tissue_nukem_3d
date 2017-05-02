import numpy as np

from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imsave

import os

def read_lsm_image(lsm_file, channel_names=None):
    import lsmreader

    lsm_img = lsmreader.Lsmimage(lsm_file)
    lsm_img.open()

    voxelsize = tuple([float(np.around(lsm_img.header['CZ LSM info']['Voxel Size '+dim]*1000000,decimals=3)) for dim in ['X','Y','Z']])
    n_channels = len(lsm_img.image['data'])

    filename = lsm_file.split('/')[-1]
    print filename," : ",n_channels," Channels ",voxelsize

    if n_channels > 1:
        if channel_names is None:
            channel_names = ["CH"+str(i) for i in range(n_channels)]
        img = {}
        for i_channel,channel_name in enumerate(channel_names):
            img[channel_name] = SpatialImage(lsm_img.image['data'][i_channel],voxelsize=voxelsize)
    else:
        img = SpatialImage(lsm_img.image['data'][0],voxelsize=voxelsize)

    return img


def read_czi_image(czi_file, channel_names=None):
    from czifile import CziFile

    czi_img = CziFile(czi_file)

    czi_channels = np.transpose(czi_img.asarray()[0,:,0,:,:,:,0],(0,2,3,1))

    voxelsize = {}
    for s in czi_img.segments():
        if s.SID == "ZISRAWMETADATA":
                metadata = s.data().split('\n')

                for i,row in enumerate(metadata):
                    if "Distance Id" in row:
                        s = metadata[i+1]
                        voxelsize[row.split('"')[1]] = np.around(float(s[s.find('>')+1:s.find('>')+s[s.find('>'):].find('<')])*1e6,decimals=3)

    voxelsize = tuple([voxelsize[dim] for dim in ['X','Y','Z']])
    n_channels = czi_channels.shape[0]

    print czi_file.split('/')[-1]," : ",n_channels," Channels ",voxelsize

    if n_channels > 1:
        if channel_names is None:
            channel_names = ["CH"+str(i) for i in range(n_channels)]
        img = {}
        for i_channel,channel_name in enumerate(channel_names):
            img[channel_name] = SpatialImage(czi_channels[i_channel],voxelsize=voxelsize)
    else:
        img = SpatialImage(czi_channels[0],voxelsize=voxelsize)

    return img

def read_tiff_image(tiff_file, channel_names=None):
    from tifffile import TiffFile

    tiff_img = TiffFile(tiff_file)
    tiff_channels = tiff_img.asarray()

    n_channels = 1 if tiff_channels.ndim==3 else tiff_channels.shape[1]

    if n_channels > 1:
        if channel_names is None:
            channel_names = ["CH"+str(i) for i in range(n_channels)]
        img = {}
        for i_channel,channel_name in enumerate(channel_names):
            img[channel_name] = SpatialImage(np.transpose(tiff_channels[:,i_channel],(1,2,0)))
    else:
        img = SpatialImage(np.transpose(tiff_channels,(1,2,0)))

    return img



def export_microscopy_image_as_inr(image_file, channel_names=None, saving_directory=None, saving_filename=None):
    if image_file.split('.')[-1] == 'lsm':
        img = read_lsm_image(image_file,channel_names)
    elif image_file.split('.')[-1] == 'czi':
        img = read_czi_image(image_file,channel_names)

    n_channels = len(img) if isinstance(img,dict) else 1

    filename = os.path.split(image_file)[1]
    if saving_directory is None:
        saving_directory = os.path.dirname(image_file)
    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)
    if saving_filename is None:
        saving_filename = filename

    if n_channels > 1:
        channel_names = img.keys()
        for i_channel,channel_name in enumerate(channel_names):
            inr_filename = saving_directory+"/"+os.path.splitext(saving_filename)[0]+"_" +channel_name+".inr.gz"
            imsave(inr_filename,img[channel_name])
    else:
        inr_filename = saving_directory+"/"+os.path.splitext(saving_filename)[0]+".inr.gz"
        imsave(inr_filename,img)

    return


def imread(image_file, channel_names=None):
    from openalea.image.serial.all import imread as oa_imread

    if image_file.split('.')[-1] == 'lsm':
        return read_lsm_image(image_file,channel_names)
    elif image_file.split('.')[-1] == 'czi':
        return read_czi_image(image_file,channel_names)
    elif image_file.split('.')[-1] in ['tif','tiff']:
        return read_tiff_image(image_file,channel_names)
    else:
        return oa_imread(image_file)





