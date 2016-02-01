import numpy as np
from scipy import ndimage as nd

from openalea.image.spatial_image           import SpatialImage
from openalea.image.serial.all              import imread, imsave

from openalea.deploy.shared_data import shared_data
import vplants.meshing_data

from vplants.meshing.property_topomesh_analysis     import *
from vplants.meshing.intersection_tools     import inside_triangle, intersecting_segment, intersecting_triangle
from vplants.meshing.evaluation_tools       import jaccard_index
from vplants.meshing.tetrahedrization_tools import tetrahedra_dual_topomesh, tetrahedra_from_triangulation, tetra_geometric_features, triangle_geometric_features, triangulated_interface_topomesh
from vplants.meshing.topomesh_from_image    import *

from vplants.meshing.triangular_mesh import TriangularMesh, _points_repr_geom_, _points_repr_vtk_

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface

from openalea.container.array_dict             import array_dict
from openalea.container.property_topomesh      import PropertyTopomesh

from time                                   import time, sleep
import csv
import pickle
from copy import deepcopy

#sequence_name = "L1PMN_1.1_150129_sam02"
#sequence_name = "WUS-PI_1.1_140327_sam00"
#sequence_name = "DR5v2-PI_D2_150217_sam01"
#sequence_name = "r2DII_2.2_141204_sam08" 
#sequence_name = "r2DII_1.2_141202_sam03"
#sequence_name = "r2DII_2.2_141204_sam09" 
sequence_name = "r2DII_1.2_141202_sam03" 
#sequence_name = "r2DII_3.2_141127_sam04" 
#sequence_name = "r2DII_1.2_141202_sam02" 
#sequence_name = "r2DII_1.2_141202_sam04"
#sequence_name = "r2DII_1.2_141202_sam11" 

filenames = [sequence_name+"_t00",
             sequence_name+"_t04",
             sequence_name+"_t08"]
dirname = shared_data(vplants.meshing_data)

tag_name = "tdT"
#tag_name = "H2BTFP"
#tag_name = "PI"

signal_name = "DIIV"
#signal_name = "PMCit"
#signal_name = "DR5"
#signal_name = "WUS"

world.clear()
#viewer.ren.SetBackground(1,1,1)
    
for i,filename in enumerate(filenames):
    filetime = filename[-4:]
    
    try:
        tag_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+tag_name+".inr.gz"
        tag_img = imread(tag_file)
        #position=np.array(tag_img.shape)/2. + np.array([((int(filetime[2:])+4)%24)/4,-2*((int(filetime[2:])+4)/24),0])*np.array(tag_img.shape)
        position=np.array(tag_img.shape)/2. + np.array([((int(filetime[2:])+4)%24)/4,-((int(filetime[2:])+4)/24),0])*np.array(tag_img.shape)
        
        if tag_img.dtype==np.uint16:
            world.add(tag_img,'nuclei_image'+filetime,position=position,resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey',intensity_range=(2000,40000))
        else:
            world.add(tag_img,'nuclei_image'+filetime,position=position,resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey',intensity_range=(20,255))
   
        #position=np.array(tag_img.shape)/2. + np.array([((int(filetime[2:])+4)%24)/4,-2*((int(filetime[2:])+4)/24)-1,0])*np.array(tag_img.shape)
        position=np.array(tag_img.shape)/2. + np.array([((int(filetime[2:])+4)%24)/4,-((int(filetime[2:])+4)/24),0])*np.array(tag_img.shape)
        #venus_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
        venus_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_seg.inr.gz"
        venus_img = imread(venus_file)
        
        world.add(venus_img,'DII_venus'+filetime,position=position,resolution=np.array(venus_img.resolution)*np.array([-1.,-1.,-1.]),colormap='glasbey')
      
      
        if venus_img.dtype==np.uint16:
            world.add(venus_img,'DII_venus'+filetime,position=position,resolution=np.array(venus_img.resolution)*np.array([-1.,-1.,-1.]),colormap='green',intensity_range=(4000,30000))
        else:
            world.add(venus_img,'DII_venus'+filetime,position=position,resolution=np.array(venus_img.resolution)*np.array([-1.,-1.,-1.]),colormap='green',intensity_range=(30,200))
        
    except:
        pass
    
    
    
    