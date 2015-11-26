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

filename = "DR5N_6.1_151124_sam02_z1.04_t00"
dirname = shared_data(vplants.meshing_data)

reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
reference_img = imread(reference_file)

size = np.array(reference_img.shape)
resolution = np.array(reference_img.resolution)
position = np.array([0,0,0])

world.add(reference_img,'reference_image',position=position/resolution,resolution=resolution,colormap='invert_grey')

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
    
    segmented_img = nuclei_active_region_segmentation(reference_img, positions, display=False)
    segmentation_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_seg.inr.gz"
    imsave(segmentation_file,segmented_img)
    
    segmented_positions = nuclei_positions_from_segmented_image(segmented_img)
    write_nuclei_points(segmented_positions,segmented_filename)

detected_cells = TriangularMesh()
detected_cells.points = segmented_positions
detected_cells.point_data = dict([(c,0.5) for c in segmented_positions.keys()])
world.add(detected_cells,'segmented_cells',colormap='leaf',intensity_range=(0,1),position=position)
raw_input()


nuclei_filename = dirname+"/nuclei_images/"+filename+"/"+filename+"_edited_cells.csv"
write_nuclei_points(world['segmented_cells'].data,nuclei_filename,data_name='certainty')




