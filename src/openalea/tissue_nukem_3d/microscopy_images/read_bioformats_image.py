import numpy as np
import bioformats
import javabridge

from openalea.image.spatial_image import SpatialImage
from bioformats.omexml import get_int_attr, get_float_attr

import sys  
reload(sys)  
sys.setdefaultencoding('utf8')


def imread(filename, channel_names=None):
    javabridge.start_vm(class_path=bioformats.JARS)

    reader = bioformats.get_image_reader(path=filename,key=0)
    n_series = reader.rdr.getSeriesCount()

    metadata_xml = bioformats.get_omexml_metadata(path=filename).encode('utf-8')

    img_metadata = bioformats.OMEXML(metadata_xml)

    bit_dtypes = {}
    bit_dtypes[8] = np.uint8
    bit_dtypes[16] = np.uint16

    img_series = {}
    for s in range(n_series)[:1]:
        reader.rdr.setSeries(s)
        img = np.array([[reader.read(c=None,z=z,t=t) for z in xrange(reader.rdr.getSizeZ())] for t in xrange(reader.rdr.getSizeT())])    

        if reader.rdr.getSizeC()==1:
            img = img[:,:,:,:,np.newaxis]

        # img = np.transpose(img,(0,4,2,3,1))
        img = np.transpose(img,(1,4,3,2,0))
        
        series_name = img_metadata.image(s).Name
        
        vx = get_float_attr(img_metadata.image(s).Pixels.node,"PhysicalSizeX")
        vy = get_float_attr(img_metadata.image(s).Pixels.node,"PhysicalSizeY")
        vz = get_float_attr(img_metadata.image(s).Pixels.node,"PhysicalSizeZ")

        vx = 1 if vx is None else vx
        vy = 1 if vy is None else vy
        vz = 1 if vz is None else vz
        print (vx,vy,vz)
        
        bits = get_int_attr(img_metadata.image(s).Pixels.node,"SignificantBits")
        
        if img.shape[0]>1:
            for t in xrange(img.shape[0]):
                img_series[series_name+"_T"+str(t).zfill(2)] = {}
                
                for c in xrange(img.shape[1]):
                    image = SpatialImage(img[t,c],voxelsize=(vx,vy,vz))
                    if (image.max()<=1.0):
                        image = image*(np.power(2,bits)-1)
                    image = image.astype(bit_dtypes[bits])
                    
                    if channel_names is not None and len(channel_names)==img.shape[1]:
                        channel_id = channel_names[c]
                    else:
                        channel_id = img_metadata.image(s).Pixels.Channel(c).ID
                    if channel_id is None:
                        channel_id = "C"+str(c)
                    img_series[series_name+"_T"+str(t).zfill(2)][channel_id] = image
                
                if img.shape[1]==1:
                    img_series[series_name+"_T"+str(t).zfill(2)] = img_series[series_name+"_T"+str(t).zfill(2)].values()[0]
        else:
            img_series[series_name] = {}
            for c in xrange(img.shape[1]):
                image = np.copy(img[0,c])
                if (image.max()<=1.0):
                    image = image*(np.power(2,bits)-1)
                image = image.astype(bit_dtypes[bits])
                image = SpatialImage(image,voxelsize=(vx,vy,vz))
                if channel_names is not None and len(channel_names)==img.shape[1]:
                    channel_id = channel_names[c]
                else:
                    channel_id = img_metadata.image(s).Pixels.Channel(c).Name
                if channel_id is None:
                    channel_id = img_metadata.image(s).Pixels.Channel(c).ID
                    if channel_id is None:
                        channel_id = "C"+str(c)
                img_series[series_name][channel_id] = image
            if img.shape[1]==1:
                img_series[series_name] = img_series[series_name].values()[0]

    if n_series == 1:
        return img_series.values()[0]
    else:
        return img_series
                
