import numpy as np
import scipy.ndimage as nd

from openalea.container import array_dict

from copy import deepcopy
import pickle

from openalea.draco_stem.draco.adjacency_complex_optimization import delaunay_tetrahedrization_topomesh, tetrahedrization_clean_surface
from openalea.draco_stem.draco.dual_reconstruction import topomesh_triangle_split

from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
from openalea.cellcomplex.property_topomesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components
from openalea.cellcomplex.property_topomesh.utils.implicit_surfaces import implicit_surface_topomesh
from openalea.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_vertices_deformation


from vplants.sam4dmaps.sam_model_tools import nuclei_density_function


def nuclei_surface_topomesh(nuclei_topomesh, size, resolution, cell_radius=5.0, subsampling=6., density_k=0.25):

    nuclei_positions = nuclei_topomesh.wisp_property('barycenter',0)

    from time import time
    start_time = time()

    size_offset = 0.25

    grid_resolution = np.sign(resolution)*subsampling
    #x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
    #grid_size = size
    x,y,z = np.ogrid[-size_offset*size[0]*resolution[0]:(1+size_offset)*size[0]*resolution[0]:grid_resolution[0],-size_offset*size[1]*resolution[1]:(1+size_offset)*size[1]*resolution[1]:grid_resolution[1],-size_offset*size[2]*resolution[2]:(1+size_offset)*size[2]*resolution[2]:grid_resolution[2]]
    grid_size = (1+2.*size_offset)*size

    end_time = time()
    print "--> Generating grid     [",end_time - start_time,"s]"

    start_time = time()
    nuclei_density = nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=density_k)(x,y,z)
    end_time = time()
    print "--> Computing density   [",end_time - start_time,"s]"

    start_time = time()
    surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution,iso=0.5,center=False)
    surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - size_offset*resolution*size,surface_topomesh.wisp_property('barycenter',0).keys()))
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



def nuclei_layer(nuclei_positions, size, resolution, maximal_distance=12., maximal_eccentricity=0.8, return_topomesh=False, display=False):
    positions = array_dict(nuclei_positions)

    if display:
        from openalea.core.world import World
        world = World()


    #grid_resolution = resolution*[12,12,4]
    grid_resolution = np.sign(resolution)*[4.,4.,4.]
    #x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
    #grid_size = size
    x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:grid_resolution[2]]
    grid_size = 2*size
    
    #nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(x,y,z) for p in positions.keys()])
    #nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    #nuclei_density = np.sum(nuclei_potential,axis=3)

    nuclei_density = nuclei_density_function(positions,cell_radius=5,k=1.0)(x,y,z) 

    surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution,iso=0.5,center=False)
    surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*resolution*size,surface_topomesh.wisp_property('barycenter',0).keys()))

    # if display:
    #     world.add(surface_topomesh,'surface')
    #     raw_input()
    #     world.remove('surface')
    # return None,None,surface_topomesh

    triangulation_topomesh = delaunay_tetrahedrization_topomesh(positions,clean_surface=False)
    delaunay_topomesh = deepcopy(triangulation_topomesh)

    # triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','sliver'],surface_topomesh=surface_topomesh)
    triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','distance','sliver'],surface_topomesh=surface_topomesh,maximal_distance=maximal_distance,maximal_eccentricity=maximal_eccentricity)
    #triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','distance','sliver'],surface_topomesh=surface_topomesh,maximal_distance=maximal_distance,maximal_eccentricity=maximal_eccentricity)

    # if display:
    #     world.add(triangulation_topomesh,'triangulation')
    #     raw_input()
    #     world.remove('triangulation')
    # return None,triangulation_topomesh,surface_topomesh

    L1_triangulation_topomesh = epidermis_topomesh(triangulation_topomesh)

    compute_topomesh_property(L1_triangulation_topomesh,'normal',2,normal_method="density",object_positions=positions)

    compute_topomesh_vertex_property_from_faces(L1_triangulation_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    down_facing = L1_triangulation_topomesh.wisp_property('normal',0).values()[:,2] < -0.0
    L1_triangulation_topomesh.update_wisp_property('downward',0,array_dict(down_facing,keys=list(L1_triangulation_topomesh.wisps(0))))

    triangle_down_facing = np.any(L1_triangulation_topomesh.wisp_property('downward',0).values(L1_triangulation_topomesh.wisp_property('vertices',2).values(list(L1_triangulation_topomesh.wisps(2)))),axis=1)
    triangle_down_facing = triangle_down_facing.astype(float)
    L1_triangulation_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(L1_triangulation_topomesh.wisps(2))))
    
    for i,t in enumerate(L1_triangulation_topomesh.wisps(2)):
        if L1_triangulation_topomesh.wisp_property('downward',2)[t] == 1:
            triangle_neighbors = list(L1_triangulation_topomesh.border_neighbors(2,t))
            if np.any(L1_triangulation_topomesh.wisp_property('downward',2).values(triangle_neighbors)==0):
                triangle_down_facing[i] = 0.5
    L1_triangulation_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(L1_triangulation_topomesh.wisps(2))))
    
    triangles_to_remove = np.array(list(L1_triangulation_topomesh.wisps(2)))[L1_triangulation_topomesh.wisp_property('downward',2).values() == 1]
    for t in triangles_to_remove:
        L1_triangulation_topomesh.remove_wisp(2,t)
        
    edges_to_remove = np.array(list(L1_triangulation_topomesh.wisps(1)))[np.array([L1_triangulation_topomesh.nb_regions(1,e)==0 for e in L1_triangulation_topomesh.wisps(1)])]
    for e in edges_to_remove:
        L1_triangulation_topomesh.remove_wisp(1,e)

    vertices_to_remove = np.array(list(L1_triangulation_topomesh.wisps(0)))[np.array([L1_triangulation_topomesh.nb_regions(0,v)==0 for v in L1_triangulation_topomesh.wisps(0)])]
    for v in vertices_to_remove:
        L1_triangulation_topomesh.remove_wisp(0,v)

    L1_triangulation_topomesh = topomesh_connected_components(L1_triangulation_topomesh)[0]

    cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
    for c in L1_triangulation_topomesh.wisps(0):
        cell_layer[c] = 1
    for c in triangulation_topomesh.wisps(0):
        if cell_layer[c] != 1:
            if np.any(cell_layer.values(list(triangulation_topomesh.region_neighbors(0,c))) == 1):
                cell_layer[c] = 2

    if return_topomesh:
        triangulation_topomesh.update_wisp_property('layer',0,cell_layer)
        return cell_layer, triangulation_topomesh, surface_topomesh
    else:
        return cell_layer

def nuclei_curvature(nuclei_positions, cell_layer, size, resolution, surface_topomesh=None):
    
    positions = array_dict(nuclei_positions)

    if surface_topomesh is None:
        grid_resolution = resolution*[12,12,4]
        #x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
        #grid_size = size
        x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:grid_resolution[2]]
        grid_size = 2*size
        # nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(x,y,z) for p in positions.keys()])
        # nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
        # nuclei_density = np.sum(nuclei_potential,axis=3)
        nuclei_density = nuclei_density_function(positions,cell_radius=8,k=1.0)(x,y,z) 
        surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution,iso=0.5,center=False)
        surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*resolution*size,surface_topomesh.wisp_property('barycenter',0).keys()))

    property_topomesh_vertices_deformation(surface_topomesh,iterations=50,omega_forces=dict(taubin_smoothing=0.65),sigma_deformation=2.0)
        
    compute_topomesh_property(surface_topomesh,'normal',2,normal_method="density",object_positions=positions)
    compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
    compute_topomesh_property(surface_topomesh,'mean_curvature',2)
    compute_topomesh_vertex_property_from_faces(surface_topomesh,'mean_curvature',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    L1_cells = cell_layer.keys()[cell_layer.values()==1]
    L1_positions = array_dict(positions.values(L1_cells),L1_cells)
    s_x,s_y,s_z = tuple(np.transpose(surface_topomesh.wisp_property('barycenter',0).values()))

    _, L1_nuclei_potential = nuclei_density_function(L1_positions,cell_radius=8,k=1.0)(s_x,s_y,s_z,return_potential=True)
    # L1_nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=8,k=1.0)(s_x,s_y,s_z) for p in L1_cells])
    L1_nuclei_density = np.sum(L1_nuclei_potential,axis=1)
    L1_nuclei_membership = L1_nuclei_potential/L1_nuclei_density[...,np.newaxis]
    
    surface_curvature = surface_topomesh.wisp_property('mean_curvature',0).values()
    L1_nuclei_curvature = (L1_nuclei_membership*surface_curvature[np.newaxis,:]).sum(axis=1)
    L1_nuclei_curvature = array_dict(L1_nuclei_curvature,L1_cells)
    
    cell_curvature = array_dict([L1_nuclei_curvature[p] if p in L1_cells else 0. for p in positions.keys()],positions.keys())

    return cell_curvature
    






