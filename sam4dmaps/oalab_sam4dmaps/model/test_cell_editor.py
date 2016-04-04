import numpy as np
from scipy import ndimage as nd

from openalea.image.serial.all              import imread, imsave
from openalea.container                     import array_dict

from openalea.deploy.shared_data import shared_data

from vplants.meshing.triangular_mesh import TriangularMesh
from vplants.sam4dmaps.nuclei_detection import detect_nuclei, write_nuclei_points, read_nuclei_points
from vplants.sam4dmaps.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image

import pickle
import csv
from copy import deepcopy

world.clear()

import vplants.meshing_data

#filename = "DR5N_6.1_151124_sam02_z1.04_t00"
#filename = "r2DII_1.2_141202_sam03_t08"
#filename = "DR5N_6.1_151124_sam01_z0.50_t00"
#filename = "DR5N_6.1_151124_sam01_z0.80_t00"
filename = "DR5N_6.1_151124_sam03_z0.80_t00"
dirname = shared_data(vplants.meshing_data)

reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
reference_img = imread(reference_file)

size = np.array(reference_img.shape)
resolution = np.array(reference_img.resolution)


import matplotlib.pyplot as plt
from vplants.meshing.cute_plot import histo_plot, smooth_plot
figure = plt.figure(0) 
#figure.clf()
histo_plot(figure,reference_img.ravel(),np.array([0,0,1.0]),"Intensity","Voxels (%)",cumul=True,bar=False,smooth_factor=1000,spline_order=5)
figure.gca().set_xlim(0,2000)
plt.show(block=False)

position = np.array([0,0,0])
world.add(reference_img,'reference_image',position=position/resolution,resolution=resolution,colormap='invert_grey')

signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_DR5.inr.gz"
signal_img = imread(signal_file) 
world.add(signal_img,'signal_image',position=position/resolution,resolution=resolution,colormap='Blues')


segmented_filename = dirname+"/nuclei_images/"+filename+"/segmented_cells.csv"
try:
    segmented_positions = array_dict(read_nuclei_points(segmented_filename))
except:
    detected_filename = dirname+"/nuclei_images/"+filename+"/initial_cells.csv"
    try:
        positions = array_dict(read_nuclei_points(detected_filename))
    except:
        positions = detect_nuclei(reference_img,threshold=3500,size_range_start=0.5,size_range_end=0.8)
        write_nuclei_points(positions,detected_filename)
    
    from openalea.mesh.property_topomesh_creation import vertex_topomesh
    detected_nuclei = vertex_topomesh(positions)
    world.add(detected_nuclei,"detected_nuclei")
    
    positions = deepcopy(segmented_positions)
    image_coords = tuple(np.transpose((positions.values()/resolution).astype(int)))
    intensity_min = np.percentile(reference_img[image_coords],0)
    
    segmented_img = nuclei_active_region_segmentation(reference_img, positions, display=False, intensity_min=intensity_min)
    world.add(segmented_img,"segmented_nuclei",colormap="glasbey",alphamap="constant",bg_id=1)
    
    new_segmented_positions = nuclei_positions_from_segmented_image(segmented_img)
    
    combined_positions = array_dict([new_segmented_positions[p] if point_status[p]!=1 else segmented_positions[p] for p in new_segmented_positions.keys()],new_segmented_positions.keys())
    
    detected_cells = TriangularMesh()
    detected_cells.points = combined_positions
    detected_cells.point_data = array_dict([point_status[p]for p in new_segmented_positions.keys()],new_segmented_positions.keys())
    world.add(detected_cells,'new_segmented_cells',colormap='leaf',intensity_range=(0,1),position=position)
    raw_input()
    
    segmentation_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_seg.inr.gz"
    imsave(segmentation_file,segmented_img)
    
    segmented_positions = nuclei_positions_from_segmented_image(segmented_img)
    write_nuclei_points(segmented_positions,segmented_filename)


    nuclei_filename = dirname+"/nuclei_images/"+filename+"/"+filename+"_centered_cells.csv"
    write_nuclei_points(world['new_segmented_cells'].data,nuclei_filename,data_name='certainty')

detected_cells = TriangularMesh()
detected_cells.points = segmented_positions
detected_cells.point_data = dict([(c,0.5) for c in segmented_positions.keys()])
world.add(detected_cells,'segmented_cells',colormap='leaf',intensity_range=(0,1),position=position)
raw_input()

nuclei_filename = dirname+"/nuclei_images/"+filename+"/"+filename+"_edited_cells.csv"
segmented_positions, point_status = read_nuclei_points(nuclei_filename,return_data=True)
detected_cells = TriangularMesh()
detected_cells.points = segmented_positions
detected_cells.point_data = point_status
world.add(detected_cells,'segmented_cells',colormap='leaf',intensity_range=(0,1),position=position)
raw_input()

segmented_img = nuclei_active_region_segmentation(reference_img, segmented_positions, display=False)

world.add(segmented_img,"segmentation",colormap='glasbey',bg_id=1,alphamap='constant')

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
img_graph = graph_from_image(segmented_img,spatio_temporal_properties=['barycenter','volume'],ignore_cells_at_stack_margins=False,property_as_real=True)

cell_volumes = array_dict(img_graph.vertex_property('volume'))
cell_volumes[1] = 0

volume_img = cell_volumes.values(segmented_img).astype(np.uint16)
world.add(volume_img,"volumes",resolution=resolution,colormap='jet',bg_id=0,alphamap='constant')





(np.array(point_status.values()) != 0.5).sum()

write_nuclei_points(world['segmented_cells'].data,nuclei_filename,data_name='certainty')




