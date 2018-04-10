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
import scipy.ndimage as nd

from openalea.container import array_dict

from copy import deepcopy
import pickle

# from openalea.draco_stem.draco.adjacency_complex_optimization import delaunay_tetrahedrization_topomesh, tetrahedrization_clean_surface
# from openalea.draco_stem.draco.dual_reconstruction import topomesh_triangle_split

from openalea.cellcomplex.property_topomesh.property_topomesh_creation import vertex_topomesh, triangle_topomesh
from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
from openalea.cellcomplex.property_topomesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components, cut_surface_topomesh, clean_topomesh
from openalea.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_vertices_deformation, topomesh_triangle_split, property_topomesh_isotropic_remeshing

from openalea.cellcomplex.property_topomesh.utils.implicit_surfaces import implicit_surface_topomesh
from openalea.cellcomplex.property_topomesh.utils.delaunay_tools import delaunay_triangulation
from openalea.cellcomplex.property_topomesh.utils.array_tools import array_unique



def nuclei_density_function(nuclei_positions,cell_radius,k=0.1):
    import numpy as np
    
    def density_func(x,y,z,return_potential=False):
        
        max_radius = cell_radius
        # max_radius = 0.

        points = np.array(nuclei_positions.values())


        if len((x+y+z).shape) == 0:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0],2) +  np.power(y[np.newaxis] - points[:,1],2) + np.power(z[np.newaxis] - points[:,2],2),0.5)
        elif len((x+y+z).shape) == 1:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 2:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 3:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis,np.newaxis],2),0.5)


        density_potential = 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))

        if len(density_potential.shape)==1 and density_potential.shape[0]==1:
            density = density_potential.sum()
        else:
            density = density_potential.sum(axis=0)

        # density = np.zeros_like(x+y+z)
        # for p in nuclei_positions.keys():
        #     cell_distances = np.power(np.power(x-nuclei_positions[p][0],2) + np.power(y-nuclei_positions[p][1],2) + np.power(z-nuclei_positions[p][2],2),0.5)
        #     density += 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))
        
        if return_potential:
            return density, density_potential
        else:
            return density
    return density_func


def nuclei_surface_topomesh(nuclei_topomesh, size, voxelsize, cell_radius=5.0, subsampling=6., density_k=0.25):

    nuclei_positions = nuclei_topomesh.wisp_property('barycenter',0)

    from time import time
    start_time = time()

    size_offset = 0.25

    grid_voxelsize = np.sign(voxelsize)*subsampling
    #x,y,z = np.ogrid[0:size[0]*voxelsize[0]:grid_voxelsize[0],0:size[1]*voxelsize[1]:grid_voxelsize[1],0:size[2]*voxelsize[2]:grid_voxelsize[2]]
    #grid_size = size
    x,y,z = np.ogrid[-size_offset*size[0]*voxelsize[0]:(1+size_offset)*size[0]*voxelsize[0]:grid_voxelsize[0],-size_offset*size[1]*voxelsize[1]:(1+size_offset)*size[1]*voxelsize[1]:grid_voxelsize[1],-size_offset*size[2]*voxelsize[2]:(1+size_offset)*size[2]*voxelsize[2]:grid_voxelsize[2]]
    grid_size = (1+2.*size_offset)*size

    end_time = time()
    print "--> Generating grid     [",end_time - start_time,"s]"

    start_time = time()
    nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(x,y,z)
    end_time = time()
    print "--> Computing density   [",end_time - start_time,"s]"

    start_time = time()
    surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,voxelsize,iso=0.5,center=False)
    surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - size_offset*voxelsize*size,surface_topomesh.wisp_property('barycenter',0).keys()))
    surface_topomesh = topomesh_triangle_split(surface_topomesh)
    property_topomesh_vertices_deformation(surface_topomesh,iterations=20,omega_forces=dict(taubin_smoothing=0.65),sigma_deformation=2.0)
    end_time = time()
    print "--> Generating topomesh [",end_time - start_time,"s]"

    start_time = time()
    surface_points = surface_topomesh.wisp_property('barycenter',0).values()
    surface_density, surface_potential = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(surface_points[:,0],surface_points[:,1],surface_points[:,2],return_potential=True)
    surface_membership = np.transpose(surface_potential)/surface_density[:,np.newaxis]
    end_time = time()
    print "--> Computing surface membership [",end_time - start_time,"s]"

    start_time = time()
    # print np.argmax(surface_membership,axis=1).shape
    # print np.argmax(surface_membership,axis=0).shape
    # print nuclei_topomesh.nb_wisps(0)
    # print surface_topomesh.nb_wisps(0)

    surface_topomesh.update_wisp_property('cell',0,array_dict(np.array(nuclei_positions.keys())[np.argmax(surface_membership,axis=1)],list(surface_topomesh.wisps(0))))
    for property_name in nuclei_topomesh.wisp_property_names(0):
        if not property_name in ['barycenter']:
        #if not property_name in []:
            if nuclei_topomesh.wisp_property(property_name,0).values().ndim == 1:
                # print (surface_membership*nuclei_topomesh.wisp_property(property_name,0).values()[np.newaxis,:]).sum(axis=1).shape
                surface_topomesh.update_wisp_property(property_name,0,array_dict((surface_membership*nuclei_topomesh.wisp_property(property_name,0).values()[np.newaxis,:]).sum(axis=1),list(surface_topomesh.wisps(0))))
            elif nuclei_topomesh.wisp_property(property_name,0).values().ndim == 2:
                surface_topomesh.update_wisp_property(property_name,0,array_dict((surface_membership[:,:,np.newaxis]*nuclei_topomesh.wisp_property(property_name,0).values()[np.newaxis,:]).sum(axis=1),list(surface_topomesh.wisps(0))))
    end_time = time()
    print "--> Updating properties [",end_time - start_time,"s]"

    return surface_topomesh


def spherical_structuring_element(radius=1.0, voxelsize=(1.,1.,1.)):
    neighborhood = np.array(np.ceil(radius/np.array(voxelsize)),int)
    structuring_element = np.zeros(tuple(2*neighborhood+1),np.uint8)

    neighborhood_coords = np.mgrid[-neighborhood[0]:neighborhood[0]+1,-neighborhood[1]:neighborhood[1]+1,-neighborhood[2]:neighborhood[2]+1]
    neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + neighborhood
    neighborhood_coords = array_unique(neighborhood_coords)
        
    neighborhood_distance = np.linalg.norm(neighborhood_coords*voxelsize - neighborhood*voxelsize,axis=1)
    neighborhood_coords = neighborhood_coords[neighborhood_distance<=radius]
    neighborhood_coords = tuple(np.transpose(neighborhood_coords))
    structuring_element[neighborhood_coords] = 1

    return structuring_element


def nuclei_image_surface_topomesh(nuclei_img, nuclei_sigma=2., density_voxelsize=1., intensity_threshold=2000., microscope_orientation=1, maximal_length=10., remeshing_iterations=10, erosion_radius=0.0):
    voxelsize = np.array(nuclei_img.voxelsize)
    size = np.array(nuclei_img.shape)
    subsampling = np.ceil(density_voxelsize/voxelsize).astype(int)

    # nuclei_density = nd.gaussian_filter(nuclei_img,nuclei_sigma/voxelsize)/(2.*intensity_threshold)
    nuclei_density = (nd.gaussian_filter(nuclei_img,nuclei_sigma/voxelsize) > (2.*intensity_threshold)).astype(np.uint8)
    nuclei_density = nuclei_density[::subsampling[0],::subsampling[1],::subsampling[2]]

    pad_shape = np.transpose(np.tile(np.array(nuclei_density.shape)/2,(2,1)))
    nuclei_density = np.pad(nuclei_density,pad_shape,mode='constant')

    if erosion_radius>0:
            structuring_element = spherical_structuring_element(erosion_radius,voxelsize*subsampling)
            nuclei_density = nd.binary_erosion(nuclei_density,structuring_element).astype(np.uint8)

    surface_topomesh = implicit_surface_topomesh(nuclei_density,np.array(nuclei_density).shape,microscope_orientation*voxelsize*subsampling,smoothing=50,decimation=100,iso=0.5,center=False)
    surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.25*microscope_orientation*voxelsize*subsampling*np.array(nuclei_density.shape),surface_topomesh.wisp_property('barycenter',0).keys()))
    
    if remeshing_iterations>0:
        surface_topomesh = property_topomesh_isotropic_remeshing(surface_topomesh,maximal_length=maximal_length,iterations=remeshing_iterations)

    return surface_topomesh


def up_facing_surface_topomesh(input_surface_topomesh, normal_method='density', nuclei_positions=None, down_facing_threshold=-0., connected=True):
    
    assert (normal_method != 'density') or (nuclei_positions is not None)
    assert normal_method in ['density','orientation']

    if nuclei_positions is not None:
        positions = array_dict(nuclei_positions)
    else:
        positions = None

    surface_topomesh = deepcopy(input_surface_topomesh)
    
    compute_topomesh_property(surface_topomesh,'normal',2,normal_method=normal_method,object_positions=positions)
    compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    down_facing = surface_topomesh.wisp_property('normal',0).values()[:,2] < down_facing_threshold
    surface_topomesh.update_wisp_property('downward',0,array_dict(down_facing,keys=list(surface_topomesh.wisps(0))))

    triangle_down_facing = np.any(surface_topomesh.wisp_property('downward',0).values(surface_topomesh.wisp_property('vertices',2).values(list(surface_topomesh.wisps(2)))),axis=1)
    triangle_down_facing = triangle_down_facing.astype(float)
    surface_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(surface_topomesh.wisps(2))))

    for i,t in enumerate(surface_topomesh.wisps(2)):
        if surface_topomesh.wisp_property('downward',2)[t] == 1:
            triangle_neighbors = np.array(list(surface_topomesh.border_neighbors(2,t)),np.uint16)
            if np.any(surface_topomesh.wisp_property('downward',2).values(triangle_neighbors)==0):
                triangle_down_facing[i] = 0.5
    surface_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(surface_topomesh.wisps(2))))

    triangles_to_remove = np.array(list(surface_topomesh.wisps(2)))[surface_topomesh.wisp_property('downward',2).values() == 1]
    for t in triangles_to_remove:
        surface_topomesh.remove_wisp(2,t)
        
    edges_to_remove = np.array(list(surface_topomesh.wisps(1)))[np.array([surface_topomesh.nb_regions(1,e)==0 for e in surface_topomesh.wisps(1)])]
    for e in edges_to_remove:
        surface_topomesh.remove_wisp(1,e)

    vertices_to_remove = np.array(list(surface_topomesh.wisps(0)))[np.array([surface_topomesh.nb_regions(0,v)==0 for v in surface_topomesh.wisps(0)])]
    for v in vertices_to_remove:
        surface_topomesh.remove_wisp(0,v)
        
    #top_surface_topomesh = cut_surface_topomesh(surface_topomesh,z_cut=0.8*size[2]*voxelsize[2])
    if connected:
        top_surface_topomesh = topomesh_connected_components(surface_topomesh)[0]
    else:
        top_surface_topomesh = surface_topomesh

    return top_surface_topomesh

# def nuclei_layer(nuclei_positions, size, voxelsize, maximal_distance=12., maximal_eccentricity=0.8, return_topomesh=False, display=False):
def nuclei_layer(nuclei_positions, nuclei_image, microscope_orientation=1, surface_mode="image", density_voxelsize=1., return_topomesh=False):
    
    size = np.array(nuclei_image.shape)
    voxelsize = microscope_orientation*np.array(nuclei_image.voxelsize)

    positions = array_dict(nuclei_positions)

    # if display:
    #     from openalea.core.world import World
    #     world = World()


    # #grid_voxelsize = voxelsize*[12,12,4]
    # grid_voxelsize = np.sign(voxelsize)*[4.,4.,4.]
    # #x,y,z = np.ogrid[0:size[0]*voxelsize[0]:grid_voxelsize[0],0:size[1]*voxelsize[1]:grid_voxelsize[1],0:size[2]*voxelsize[2]:grid_voxelsize[2]]
    # #grid_size = size
    # x,y,z = np.ogrid[-0.5*size[0]*voxelsize[0]:1.5*size[0]*voxelsize[0]:grid_voxelsize[0],-0.5*size[1]*voxelsize[1]:1.5*size[1]*voxelsize[1]:grid_voxelsize[1],-0.5*size[2]*voxelsize[2]:1.5*size[2]*voxelsize[2]:grid_voxelsize[2]]
    # grid_size = 2*size
    
    # #nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(x,y,z) for p in positions.keys()])
    # #nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    # #nuclei_density = np.sum(nuclei_potential,axis=3)

    # nuclei_density = nuclei_density_function(positions,cell_radius=5,k=1.0)(x,y,z) 

    # surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,voxelsize,iso=0.5,center=False)
    # surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*voxelsize*size,surface_topomesh.wisp_property('barycenter',0).keys()))

    # # if display:
    # #     world.add(surface_topomesh,'surface')
    # #     raw_input()
    # #     world.remove('surface')
    # # return None,None,surface_topomesh

    # triangulation_topomesh = delaunay_tetrahedrization_topomesh(positions,clean_surface=False)
    # delaunay_topomesh = deepcopy(triangulation_topomesh)

    # # triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','sliver'],surface_topomesh=surface_topomesh)
    # triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','distance','sliver'],surface_topomesh=surface_topomesh,maximal_distance=maximal_distance,maximal_eccentricity=maximal_eccentricity)
    # #triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','distance','sliver'],surface_topomesh=surface_topomesh,maximal_distance=maximal_distance,maximal_eccentricity=maximal_eccentricity)

    # # if display:
    # #     world.add(triangulation_topomesh,'triangulation')
    # #     raw_input()
    # #     world.remove('triangulation')
    # # return None,triangulation_topomesh,surface_topomesh

    # L1_triangulation_topomesh = epidermis_topomesh(triangulation_topomesh)

    # compute_topomesh_property(L1_triangulation_topomesh,'normal',2,normal_method="density",object_positions=positions)

    # compute_topomesh_vertex_property_from_faces(L1_triangulation_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    # down_facing = L1_triangulation_topomesh.wisp_property('normal',0).values()[:,2] < -0.0
    # L1_triangulation_topomesh.update_wisp_property('downward',0,array_dict(down_facing,keys=list(L1_triangulation_topomesh.wisps(0))))

    # triangle_down_facing = np.any(L1_triangulation_topomesh.wisp_property('downward',0).values(L1_triangulation_topomesh.wisp_property('vertices',2).values(list(L1_triangulation_topomesh.wisps(2)))),axis=1)
    # triangle_down_facing = triangle_down_facing.astype(float)
    # L1_triangulation_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(L1_triangulation_topomesh.wisps(2))))
    
    # for i,t in enumerate(L1_triangulation_topomesh.wisps(2)):
    #     if L1_triangulation_topomesh.wisp_property('downward',2)[t] == 1:
    #         triangle_neighbors = list(L1_triangulation_topomesh.border_neighbors(2,t))
    #         if np.any(L1_triangulation_topomesh.wisp_property('downward',2).values(triangle_neighbors)==0):
    #             triangle_down_facing[i] = 0.5
    # L1_triangulation_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(L1_triangulation_topomesh.wisps(2))))
    
    # triangles_to_remove = np.array(list(L1_triangulation_topomesh.wisps(2)))[L1_triangulation_topomesh.wisp_property('downward',2).values() == 1]
    # for t in triangles_to_remove:
    #     L1_triangulation_topomesh.remove_wisp(2,t)
        
    # edges_to_remove = np.array(list(L1_triangulation_topomesh.wisps(1)))[np.array([L1_triangulation_topomesh.nb_regions(1,e)==0 for e in L1_triangulation_topomesh.wisps(1)])]
    # for e in edges_to_remove:
    #     L1_triangulation_topomesh.remove_wisp(1,e)

    # vertices_to_remove = np.array(list(L1_triangulation_topomesh.wisps(0)))[np.array([L1_triangulation_topomesh.nb_regions(0,v)==0 for v in L1_triangulation_topomesh.wisps(0)])]
    # for v in vertices_to_remove:
    #     L1_triangulation_topomesh.remove_wisp(0,v)

    # L1_triangulation_topomesh = topomesh_connected_components(L1_triangulation_topomesh)[0]

    # cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
    # for c in L1_triangulation_topomesh.wisps(0):
    #     cell_layer[c] = 1
    # for c in triangulation_topomesh.wisps(0):
    #     if cell_layer[c] != 1:
    #         if np.any(cell_layer.values(list(triangulation_topomesh.region_neighbors(0,c))) == 1):
    #             cell_layer[c] = 2

    if surface_mode == 'density':
        grid_voxelsize = microscope_orientation*np.ones(3,float)*density_voxelsize
        x,y,z = np.ogrid[-0.5*size[0]*voxelsize[0]:1.5*size[0]*voxelsize[0]:grid_voxelsize[0],-0.5*size[1]*voxelsize[1]:1.5*size[1]*voxelsize[1]:grid_voxelsize[1],-0.5*size[2]*voxelsize[2]:1.5*size[2]*voxelsize[2]:grid_voxelsize[2]]
        grid_size = 2*size

        nuclei_density = nuclei_density_function(positions,cell_radius=5,k=1.0)(x,y,z) 

        surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,voxelsize,iso=0.5,center=False)
        surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*voxelsize*size,surface_topomesh.wisp_property('barycenter',0).keys()))
    elif surface_mode == 'image':
        surface_topomesh = nuclei_image_surface_topomesh(nuclei_image,microscope_orientation=microscope_orientation,density_voxelsize=density_voxelsize,intensity_threshold=1000.,nuclei_sigma=1,maximal_length=6.,remeshing_iterations=20)

    top_surface_topomesh = up_facing_surface_topomesh(surface_topomesh,nuclei_positions=nuclei_positions,connected=True)
    top_surface_topomesh = topomesh_triangle_split(top_surface_topomesh)

    #world.add(top_surface_topomesh,'top_surface')
    
    surface_points = top_surface_topomesh.wisp_property('barycenter',0).values(list(top_surface_topomesh.wisps(0)))
    surface_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(surface_points[:,0],surface_points[:,1],surface_points[:,2]) for p in positions.keys()])
    surface_cells = np.array(positions.keys())[np.argmax(surface_potential,axis=0)]
    
    L1_cells = np.unique(surface_cells)
    
    cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
    for c in L1_cells:
        cell_layer[c] = 1

    if return_topomesh:
        # triangulation_topomesh.update_wisp_property('layer',0,cell_layer)
        # return cell_layer, triangulation_topomesh, surface_topomesh
        return cell_layer, top_surface_topomesh
    else:
        return cell_layer


def nuclei_topomesh_curvature(topomesh, surface_subdivision=1, return_topomesh=False, projection_center=None):
    L1_cells = topomesh.wisp_property('layer',0).keys()[topomesh.wisp_property('layer',0).values()==1]
    cell_barycenters = array_dict(topomesh.wisp_property('barycenter',0).values(L1_cells),L1_cells)
    
    if projection_center is None:
        center = cell_barycenters.values().mean(axis=0) 
        center[2] -= 2.*(cell_barycenters.values()-cell_barycenters.values().mean(axis=0))[:,2].max()
    else:
        center = np.array(projection_center)
    
    cell_vectors = cell_barycenters.values() - center
    cell_r = np.linalg.norm(cell_vectors,axis=1)
    cell_rx = np.linalg.norm(cell_vectors[:,np.array([0,2])],axis=1)
    cell_ry = np.linalg.norm(cell_vectors[:,np.array([1,2])],axis=1)
        
    cell_phi = np.sign(cell_vectors[:,0])*np.arccos(cell_vectors[:,2]/cell_rx)
    cell_psi = np.sign(cell_vectors[:,1])*np.arccos(cell_vectors[:,2]/cell_ry)
        
    cell_flat_barycenters = deepcopy(cell_barycenters)
    for i,c in enumerate(cell_barycenters.keys()):
        cell_flat_barycenters[c][0] = cell_phi[i]
        cell_flat_barycenters[c][1] = cell_psi[i]
        cell_flat_barycenters[c][2] = 0.

    triangles = np.array(cell_barycenters.keys())[delaunay_triangulation(np.array([cell_flat_barycenters[c] for c in cell_barycenters.keys()]))]
    cell_topomesh = triangle_topomesh(triangles, cell_barycenters)
        
    maximal_length = 15.
    
    compute_topomesh_property(cell_topomesh,'length',1)
    compute_topomesh_property(cell_topomesh,'triangles',1)
    
    boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
    distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
    edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
    
    while len(edges_to_remove) > 0:
        triangles_to_remove = np.concatenate(cell_topomesh.wisp_property('triangles',1).values(edges_to_remove))
        for t in triangles_to_remove:
            cell_topomesh.remove_wisp(2,t)
        
        clean_topomesh(cell_topomesh)
        
        compute_topomesh_property(cell_topomesh,'triangles',1)
    
        boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
        distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
        edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
    
    cell_topomesh = topomesh_connected_components(cell_topomesh)[0]
    property_topomesh_vertices_deformation(cell_topomesh,iterations=5)
    
    for i in xrange(surface_subdivision):
        cell_topomesh = topomesh_triangle_split(cell_topomesh)
    compute_topomesh_property(cell_topomesh,'vertices',2)
    
    property_topomesh_vertices_deformation(cell_topomesh,iterations=15)
    
    compute_topomesh_property(cell_topomesh,'barycenter',2)
    compute_topomesh_property(cell_topomesh,'normal',2,normal_method='orientation')
    
    compute_topomesh_vertex_property_from_faces(cell_topomesh,'normal',adjacency_sigma=2,neighborhood=5)
    compute_topomesh_property(cell_topomesh,'mean_curvature',2)
    compute_topomesh_vertex_property_from_faces(cell_topomesh,'mean_curvature',adjacency_sigma=2,neighborhood=5)
    compute_topomesh_vertex_property_from_faces(cell_topomesh,'gaussian_curvature',adjacency_sigma=2,neighborhood=5)
    
    cell_points = cell_barycenters.values()

    for curvature_property in ['mean_curvature','gaussian_curvature']:

        surface_curvature = cell_topomesh.wisp_property(curvature_property,0).values(list(cell_topomesh.wisps(0)))
    
        cell_potential = np.array([nuclei_density_function(dict([(p,cell_topomesh.wisp_property('barycenter',0)[p])]),cell_radius=5,k=1.0)(cell_points[:,0],cell_points[:,1],cell_points[:,2]) for p in cell_topomesh.wisps(0)])
        cell_curvature = (cell_potential*surface_curvature[:,np.newaxis]).sum(axis=0)/cell_potential.sum(axis=0)
        cell_curvature = dict(zip(cell_barycenters.keys(),cell_curvature))
        topomesh.update_wisp_property(curvature_property,0,array_dict([cell_curvature[c] if c in cell_barycenters.keys() else 0 for c in topomesh.wisps(0)],list(topomesh.wisps(0))))
    
    if return_topomesh:
        return cell_topomesh
    else:
        return

def nuclei_surface_topomesh_curvature(topomesh, surface_topomesh=None, nuclei_img=None, microscope_orientation=-1, return_topomesh=False):
    assert (surface_topomesh is not None) or (nuclei_img is not None)

    L1_cells = topomesh.wisp_property('layer',0).keys()[topomesh.wisp_property('layer',0).values()==1]
    cell_barycenters = array_dict(topomesh.wisp_property('barycenter',0).values(L1_cells),L1_cells)

    if surface_topomesh is None:
        size = np.array(reference_img.shape)
        voxelsize = np.array(reference_img.voxelsize)

        surface_topomesh = nuclei_image_surface_topomesh(reference_img, intensity_threshold=1000., microscope_orientation=microscope_orientation, remeshing_iterations=2, erosion_radius=1.)
        surface_topomesh = up_facing_surface_topomesh(surface_topomesh,normal_method='orientation',connected=True)

    compute_topomesh_property(surface_topomesh,'normal',2,normal_method='orientation')
    compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',neighborhood=3,adjacency_sigma=1.2)
    compute_topomesh_property(surface_topomesh,'mean_curvature',2)

    for property_name in ['mean_curvature','gaussian_curvature']:
        compute_topomesh_vertex_property_from_faces(surface_topomesh,property_name,neighborhood=3,adjacency_sigma=1.2)

    surface_points = surface_topomesh.wisp_property('barycenter',0).values(list(surface_topomesh.wisps(0)))
    density, potential = nuclei_density_function(cell_barycenters,cell_radius=5,k=0.33)(surface_points[:,0],surface_points[:,1],surface_points[:,2],return_potential=True)

    for property_name in ['mean_curvature','gaussian_curvature']:
        property_data = surface_topomesh.wisp_property(property_name,0).values(list(surface_topomesh.wisps(0)))
        cell_property = array_dict(np.sum(potential*property_data[np.newaxis,:],axis=1)/np.sum(potential,axis=1),keys=cell_barycenters.keys())
        topomesh.update_wisp_property(property_name,0,array_dict([cell_property[c] if c in cell_barycenters.keys() else 0 for c in topomesh.wisps(0)],list(topomesh.wisps(0))))
    
    if return_topomesh:
        return topomesh, surface_topomesh
    else:
        return topomesh


def nuclei_curvature(nuclei_positions, cell_layer, size, voxelsize, surface_topomesh=None):
    topomesh = vertex_topomesh(nuclei_positions)
    topomesh.update_wisp_property('layer',0,cell_layer)
    nuclei_topomesh_curvature(topomesh)
    return topomesh.wisp_property('mean_curvature',0)

    # positions = array_dict(nuclei_positions)

    # if surface_topomesh is None:
    #     grid_voxelsize = voxelsize*[12,12,4]
    #     #x,y,z = np.ogrid[0:size[0]*voxelsize[0]:grid_voxelsize[0],0:size[1]*voxelsize[1]:grid_voxelsize[1],0:size[2]*voxelsize[2]:grid_voxelsize[2]]
    #     #grid_size = size
    #     x,y,z = np.ogrid[-0.5*size[0]*voxelsize[0]:1.5*size[0]*voxelsize[0]:grid_voxelsize[0],-0.5*size[1]*voxelsize[1]:1.5*size[1]*voxelsize[1]:grid_voxelsize[1],-0.5*size[2]*voxelsize[2]:1.5*size[2]*voxelsize[2]:grid_voxelsize[2]]
    #     grid_size = 2*size
    #     # nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(x,y,z) for p in positions.keys()])
    #     # nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    #     # nuclei_density = np.sum(nuclei_potential,axis=3)
    #     nuclei_density = nuclei_density_function(positions,cell_radius=8,k=1.0)(x,y,z) 
    #     surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,voxelsize,iso=0.5,center=False)
    #     surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*voxelsize*size,surface_topomesh.wisp_property('barycenter',0).keys()))

    # property_topomesh_vertices_deformation(surface_topomesh,iterations=50,omega_forces=dict(taubin_smoothing=0.65),sigma_deformation=2.0)
        
    # compute_topomesh_property(surface_topomesh,'normal',2,normal_method="density",object_positions=positions)
    # compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
    # compute_topomesh_property(surface_topomesh,'mean_curvature',2)
    # compute_topomesh_vertex_property_from_faces(surface_topomesh,'mean_curvature',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    # L1_cells = cell_layer.keys()[cell_layer.values()==1]
    # L1_positions = array_dict(positions.values(L1_cells),L1_cells)
    # s_x,s_y,s_z = tuple(np.transpose(surface_topomesh.wisp_property('barycenter',0).values()))

    # _, L1_nuclei_potential = nuclei_density_function(L1_positions,cell_radius=8,k=1.0)(s_x,s_y,s_z,return_potential=True)
    # # L1_nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(s_x,s_y,s_z) for p in L1_cells])
    # L1_nuclei_density = np.sum(L1_nuclei_potential,axis=1)
    # L1_nuclei_membership = L1_nuclei_potential/L1_nuclei_density[...,np.newaxis]
    
    # surface_curvature = surface_topomesh.wisp_property('mean_curvature',0).values()
    # L1_nuclei_curvature = (L1_nuclei_membership*surface_curvature[np.newaxis,:]).sum(axis=1)
    # L1_nuclei_curvature = array_dict(L1_nuclei_curvature,L1_cells)
    
    # cell_curvature = array_dict([L1_nuclei_curvature[p] if p in L1_cells else 0. for p in positions.keys()],positions.keys())

    # return cell_curvature
    






