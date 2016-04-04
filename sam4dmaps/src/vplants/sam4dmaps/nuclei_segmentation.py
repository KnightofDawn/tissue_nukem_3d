# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Mersitem 4D Maps
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>,
#                            Julien Mille <julien.mille@liris.cnrs.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np
import scipy.ndimage as nd

from openalea.image.spatial_image import SpatialImage
from openalea.container import array_dict
from vplants.sam4dmaps.nuclei_detection import array_unique

from time import time

def seed_image_from_points(size, resolution, positions, point_radius=1.0):
    """
    Generate a SpatialImage of a given shape with labelled spherical regions around points
    """

    seed_img = np.ones(tuple(size),np.uint16)

    size = np.array(size)
    resolution = np.array(resolution)

    for p in positions.keys():
        image_neighborhood = np.array(np.ceil(point_radius/resolution),int)
        neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
        neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + np.array(positions[p]/resolution,int)
        neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),size-1)
        neighborhood_coords = array_unique(neighborhood_coords)
        
        neighborhood_distance = np.linalg.norm(neighborhood_coords*resolution - positions[p],axis=1)
        neighborhood_coords = neighborhood_coords[neighborhood_distance<=point_radius]
        neighborhood_coords = tuple(np.transpose(neighborhood_coords))
        
        seed_img[neighborhood_coords] = p

    return SpatialImage(seed_img,resolution=resolution)


def inside_image(points, img):
    inside = np.ones(len(points),bool)
    inside = inside & (np.array(points)[:,0] >= 0)
    inside = inside & (np.array(points)[:,1] >= 0)
    inside = inside & (np.array(points)[:,2] >= 0)
    inside = inside & (np.array(points)[:,0] < img.shape[0])
    inside = inside & (np.array(points)[:,1] < img.shape[1])
    inside = inside & (np.array(points)[:,2] < img.shape[2])
    return inside


neighborhood_6 = [[0,0,-1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]]
neighborhood_18 = [[0,0,-1],[1,0,-1],[0,1,-1],[-1,0,-1],[0,-1,-1],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],[0,0,1],[1,0,1],[0,1,1],[-1,0,1],[0,-1,1],]


def active_regions_energy_gradient_descent(initial_regions_img, reference_img, omega_energies=dict(intensity=1.0,gradient=1.5,smoothness=10000.0), intensity_min=20000., gradient_img=None):
    """
    3D extension of the multiple region extension of the binary level set implementation :
    Step of growth using estimated energy gradient descent
    """

    regions_img = np.copy(initial_regions_img)
    size = np.array(regions_img.shape)

    candidate_points = []
    candidate_labels = []
    candidate_inside = []
    candidate_current_labels = []
    candidate_potential_labels = []
    
    outer_margin = nd.binary_dilation(regions_img>1) - (regions_img>1)
    outer_margin_points = list(np.transpose(np.where(outer_margin)))
    
    neighbor_labels = []
    for n in neighborhood_18:
        neighbor_points = np.array(outer_margin_points) + np.array(n)
        neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),size-1)
        neighbor_labels += [regions_img[tuple(np.transpose(neighbor_points))]]
    
    neighbor_labels = np.transpose(neighbor_labels)
    neighbor_labels = [ list(set(l).difference({1})) for l in neighbor_labels]
    outer_margin_labels = [l[0] for l in neighbor_labels]
    outer_margin_current_labels = [1 for l in neighbor_labels]
    outer_margin_potential_labels = [l[0] for l in neighbor_labels]
    
    candidate_points += outer_margin_points
    candidate_labels += outer_margin_labels
    candidate_inside += [False for p in outer_margin_labels]
    candidate_current_labels += outer_margin_current_labels
    candidate_potential_labels += outer_margin_potential_labels
    
    inner_margin = (regions_img>1) - nd.binary_erosion(regions_img>1) 
    inner_margin_points = list(np.transpose(np.where(inner_margin)))
    inner_margin_labels = list(regions_img[inner_margin])
    inner_margin_current_labels = list(regions_img[inner_margin])
    inner_margin_potential_labels = [1 for l in inner_margin_current_labels]
    
    candidate_points += inner_margin_points
    candidate_labels += inner_margin_labels
    candidate_inside += [True for p in inner_margin_labels]
    candidate_current_labels += inner_margin_current_labels
    candidate_potential_labels += inner_margin_potential_labels
    
    region_points = list(np.transpose(np.where(regions_img>1)))
    
    neighbor_labels = []
    for n in neighborhood_18:
        neighbor_points = np.array(region_points) + np.array(n)
        neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),size-1)
        neighbor_labels += [regions_img[tuple(np.transpose(neighbor_points))]]
    neighbor_labels = np.transpose(neighbor_labels)
    neighbor_labels = [ list(set(l).difference({1})) for l in neighbor_labels]
    
    neighbor_number = np.array(map(len,neighbor_labels))
    frontier_points = list(np.array(region_points)[neighbor_number>1])
    
    if len(frontier_points)>0:
        frontier_neighbor_labels = np.array(neighbor_labels)[neighbor_number>1]
        frontier_labels =  [list(set(l).difference({c}))[0] for l,c in zip(frontier_neighbor_labels,regions_img[tuple(np.transpose(frontier_points))])]
        frontier_current_labels = list(regions_img[tuple(np.transpose(frontier_points))])
        frontier_potential_labels = frontier_labels
        
        candidate_points += frontier_points
        candidate_labels += frontier_labels
        candidate_inside += [True for p in frontier_labels]
        candidate_current_labels += frontier_current_labels
        candidate_potential_labels += frontier_potential_labels
    
    
    candidate_points = np.array(candidate_points)
    candidate_labels = np.array(candidate_labels)
    candidate_inside = np.array(candidate_inside)
    candidate_current_labels = np.array(candidate_current_labels)
    candidate_potential_labels = np.array(candidate_potential_labels)
    
    speeds = np.zeros(len(candidate_labels),float)
    
    if omega_energies.has_key('intensity'):
        #intensity_speeds = i_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float)
        intensity_speeds = np.zeros(len(candidate_points)) 
        intensity_speeds -= (candidate_current_labels>1)*(intensity_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float))
        intensity_speeds += (candidate_potential_labels>1)*(intensity_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float))
        speeds += omega_energies['intensity']*intensity_speeds
    
    if omega_energies.has_key('gradient'):
        #gradient_speeds = -np.array(gradient[tuple(np.transpose(candidate_points))],float)
        gradient_speeds = np.zeros(len(candidate_points)) 
        gradient_speeds -= (candidate_current_labels>1)*(-np.array(gradient_img[tuple(np.transpose(candidate_points))],float))
        gradient_speeds += (candidate_potential_labels>1)*(-np.array(gradient_img[tuple(np.transpose(candidate_points))],float))
        speeds += omega_energies['gradient']*gradient_speeds
        
    if omega_energies.has_key('smoothness'):
        neighbor_labels = []
        for n in neighborhood_6:
            neighbor_points = np.array(candidate_points) + np.array(n)
            neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),np.array(regions_img.shape)-1)
            neighbor_labels += [regions_img[tuple(np.transpose(neighbor_points))]]
        neighbor_labels = np.transpose(neighbor_labels)
        smoothness_speeds = 1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_labels)])
        
        smoothness_speeds = np.zeros(len(candidate_points)) 
        smoothness_speeds -= (candidate_current_labels>1)*(1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_current_labels)]))
        smoothness_speeds += (candidate_potential_labels>1)*(1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_potential_labels)]))
        
        speeds += omega_energies['smoothness']*smoothness_speeds
    
    probability_coef = 1.0
    change_probabilities = np.exp(np.minimum(-speeds/probability_coef-1,0))
    change_decision = (np.random.rand(len(speeds))<=change_probabilities)
    
    voxels_to_change = tuple(np.transpose(candidate_points[change_decision]))
    labels_to_change = candidate_potential_labels[change_decision]
    regions_img[voxels_to_change] = labels_to_change

    return regions_img


def nuclei_active_region_segmentation(reference_img, positions, omega_energies=dict(intensity=1.0,gradient=1.5,smoothness=10000.0), intensity_min=20000., iterations=10, display=False):
    """
    3D extension of the multiple region extension of the binary level set implementation
    """
    
    segmentation_start_time = time()
    print "--> Active region segmentation (",len(positions)," regions )"

    size = np.array(reference_img.shape)
    resolution = np.array(reference_img.resolution) 

    if display:
        from openalea.core.world import World
        world = World()

    if omega_energies.has_key('gradient'):
        from scipy.ndimage.filters import gaussian_gradient_magnitude
        start_time = time()
        print "  --> Computing image gradient"
        gradient = gaussian_gradient_magnitude(np.array(reference_img,np.float64),sigma=0.5/np.array(reference_img.resolution))
        gradient_img = SpatialImage(np.array(gradient,np.uint16),resolution=reference_img.resolution)
        end_time = time()
        print "  <-- Computing image gradient         [",end_time-start_time," s]"

    start_time = time()
    print "  --> Creating seed image"
    seed_img = seed_image_from_points(size, resolution, positions, point_radius=1.0)
    regions_img = np.copy(seed_img)
    end_time = time()
    print "  <-- Creating seed image              [",end_time-start_time," s]"

    if display:
        world.add(seed_img,'active_regions_seeds',colormap='glasbey',resolution=resolution,alphamap='constant',volume=False,cut_planes=True)
        raw_input()

    for iteration in xrange(iterations):
        start_time = time()
        print "  --> Active region energy gradient descent : iteration",iteration
        previous_regions_img = np.copy(regions_img)
        regions_img = active_regions_energy_gradient_descent(regions_img,reference_img,omega_energies=omega_energies,intensity_min=intensity_min,gradient_img=gradient)
        change = ((regions_img-previous_regions_img) != 0.).sum() / float((regions_img > 1.).sum())
        end_time = time()
        print "  --> Active region energy gradient descent : iteration",iteration,"  (Evolution : ",int(100*change)," %)  ","[",end_time-start_time," s]"

        if display:
            world.add(regions_img,'active_regions',colormap='invert_grey',resolution=resolution,intensity_range=(1,2))

    segmented_img = SpatialImage(regions_img,resolution=reference_img.resolution)
    if display:
        world.add(segmented_img,'active_regions',colormap='glasbey',resolution=resolution,alphamap='constant')
        raw_input()

    segmentation_end_time = time()
    print "<-- Active region segmentation (",len(np.unique(segmented_img))-1," regions )    [",segmentation_end_time-segmentation_start_time," s]"

    return segmented_img

    
def nuclei_positions_from_segmented_image(segmented_img, background_label=1):
    """
    """
    resolution = np.array(segmented_img.resolution)
    segmented_cells = np.array([c for c in np.unique(segmented_img) if c!= background_label])
    segmented_positions = array_dict(np.array(nd.center_of_mass(np.ones_like(segmented_img),segmented_img,index=segmented_cells))*resolution,segmented_cells)

    return segmented_positions



   







