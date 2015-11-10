import numpy as np

from openalea.image.serial.all              import imread, imsave

from openalea.deploy.shared_data import shared_data

from vplants.sam4dmaps.nuclei_detection        import detect_nuclei, compute_fluorescence_ratios, read_nuclei_points, write_nuclei_points
from vplants.meshing.triangular_mesh import TriangularMesh
from vplants.meshing.cute_plot           import simple_plot, density_plot, smooth_plot, histo_plot, bar_plot, violin_plot, spider_plot
from vplants.meshing.array_tools import array_unique


import matplotlib.pyplot as plt

import pickle
import csv
from copy import deepcopy


filename = "DR5N_7.1_150415_sam04_t00"

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)

#signal_name = 'DIIV'
signal_name = 'DR5'

signal_colors = {}
signal_colors['DIIV'] = 'Greens'
signal_colors['DR5'] = 'Oranges'


world.clear()

signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
signal_img = imread(signal_file)
    
tag_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
tag_img = imread(tag_file)
    
world.add(tag_img,'nuclei_image',colormap='invert_grey')
world.add(signal_img,signal_name+"_image",colormap=signal_colors[signal_name])

inputfile = dirname+"/nuclei_images/"+filename+"/"+filename+"_cells.csv"

try:
    positions = read_nuclei_points(inputfile)
except:
    positions = detect_nuclei(tag_img,threshold=3000,size_range_start=0.4,size_range_end=0.7)
    write_nuclei_points(positions,inputfile)
        
detected_cells = TriangularMesh()
detected_cells.points = positions
world.add(detected_cells,'detected_cells',colormap=signal_colors[signal_name])


neighborhood_sigma = 2.0

resolution = np.array(tag_img.resolution)
intensity_max = np.percentile(tag_img,99)
image_neighborhood = np.array(np.ceil(neighborhood_sigma/np.abs(np.array(resolution))),int)


nuclei_correlations = {}

for i in positions.keys():
    coords = np.array(positions[i]/resolution,int)
    
    neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
    neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + coords
    neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),np.array(tag_img.shape)-1)
    neighborhood_coords = array_unique(neighborhood_coords)
    
    neighborhood_positions = neighborhood_coords*resolution
    neighborhood_distances = np.linalg.norm(neighborhood_positions - positions[i],axis=1)
    neighborhood_intensities =  tag_img[tuple(np.transpose(neighborhood_coords))]

    neighborhood_intensities = (neighborhood_intensities/intensity_max)*100.

    print neighborhood_intensities.max(), neighborhood_intensities.mean()
    
    distance_intensity_covariance = np.cov(neighborhood_distances,neighborhood_intensities)
    distance_intensity_correlation = distance_intensity_covariance[0,1]/np.sqrt(distance_intensity_covariance[0,0]*distance_intensity_covariance[1,1])
    print "Distance x Intensity correlation = ",distance_intensity_correlation
    
    nuclei_correlations[i] = distance_intensity_correlation
    
    # #Uncomment the following lines for individual evaluation
    
    # slice_image = tag_img[coords[0]-image_neighborhood[0]:coords[0]+image_neighborhood[0]+1,coords[1]-image_neighborhood[1]:coords[1]+image_neighborhood[1]+1,coords[2]-1:coords[2]+2]
    # slice_image = slice_image.max(axis=2)
    
    # figure = plt.figure(0)
    # figure.clf()
    # histo_plot(figure,neighborhood_intensities,color=np.array([0,0,0]),cumul=True,bar=False,xlabel="Intensity (%)",ylabel="Number of voxels (%)",spline_order=1)
    
    # figure = plt.figure(1)
    # figure.clf()
    # import vplants.meshing.cute_plot
    # reload(vplants.meshing.cute_plot)
    # from vplants.meshing.cute_plot import simple_plot
    # simple_plot(figure,neighborhood_distances,neighborhood_intensities,color=np.array([0,0,0]),xlabel="Distance",ylabel="Intensity (%)")
   
    # figure = plt.figure(2)
    # plt.clf()
    # plt.imshow(slice_image,cmap='gray')
    # plt.show(block=False)
    
    # current_cell = TriangularMesh
    # current_cell.points = dict([(i,positions[i])])
    # world.add(current_cell,"current_cell",colormap=signal_colors[signal_name])
    
    # raw_input()
    
detected_cells.point_data = nuclei_correlations
world.add(detected_cells,'nuclei_correlations',colormap='curvature',intensity_range=(-1,0))

figure = plt.figure(2)
figure.clf()
histo_plot(figure,-np.array(nuclei_correlations.values()),color=np.array([0.8,0,0]))
plt.show(block=False)

cell_ratios = compute_fluorescence_ratios(tag_img,signal_img,positions,nuclei_sigma=3.0)

detected_cells.point_data = cell_ratios
world.add(detected_cells,'fluorescence_ratios',colormap=signal_colors[signal_name])

