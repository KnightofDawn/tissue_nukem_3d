import numpy as np
import scipy.ndimage as nd

from openalea.container import array_dict

from copy import deepcopy
import pickle

from openalea.draco_stem.draco.adjacency_complex_optimization import delaunay_tetrahedrization_topomesh, tetrahedrization_clean_surface

from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, epidermis_topomesh, compute_topomesh_vertex_property_from_faces
from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function


def nuclei_layer(nuclei_positions, size, resolution):
    positions = array_dict(nuclei_positions)

    triangulation_topomesh = delaunay_tetrahedrization_topomesh(positions,clean_surface=False)
    delaunay_topomesh = deepcopy(triangulation_topomesh)

    grid_resolution = resolution*[8,8,4]
    x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
    grid_size = size
    nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(x,y,z) for p in positions.keys()])
    nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    nuclei_density = np.sum(nuclei_potential,axis=3)
    surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution,iso=0.5,center=False)

    triangulation_topomesh = tetrahedrization_clean_surface(delaunay_topomesh,surface_cleaning_criteria=['surface','distance','sliver'],surface_topomesh=surface_topomesh,maximal_distance=12.,maximal_eccentricity=0.8)

    L1_triangulation_topomesh = epidermis_topomesh(triangulation_topomesh)

    compute_topomesh_property(L1_triangulation_topomesh,'normal',2,normal_method="density",object_positions=positions)
    compute_topomesh_vertex_property_from_faces(L1_triangulation_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)

    down_facing = L1_triangulation_topomesh.wisp_property('normal',0).values()[:,2] < -0.0
    L1_triangulation_topomesh.update_wisp_property('downward',0,array_dict(down_facing,keys=list(L1_triangulation_topomesh.wisps(0))))

    triangle_down_facing = np.any(L1_triangulation_topomesh.wisp_property('downward',0).values(L1_triangulation_topomesh.wisp_property('vertices',2).values(list(L1_triangulation_topomesh.wisps(2)))),axis=1)
    L1_triangulation_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(L1_triangulation_topomesh.wisps(2))))

    triangles_to_remove = np.array(list(L1_triangulation_topomesh.wisps(2)))[L1_triangulation_topomesh.wisp_property('downward',2).values()]
    for t in triangles_to_remove:
        L1_triangulation_topomesh.remove_wisp(2,t)
        
    edges_to_remove = np.array(list(L1_triangulation_topomesh.wisps(1)))[np.array([L1_triangulation_topomesh.nb_regions(1,e)==0 for e in L1_triangulation_topomesh.wisps(1)])]
    for e in edges_to_remove:
        L1_triangulation_topomesh.remove_wisp(1,e)

    vertices_to_remove = np.array(list(L1_triangulation_topomesh.wisps(0)))[np.array([L1_triangulation_topomesh.nb_regions(0,v)==0 for v in L1_triangulation_topomesh.wisps(0)])]
    for v in vertices_to_remove:
        L1_triangulation_topomesh.remove_wisp(0,v)

    cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
    for c in L1_triangulation_topomesh.wisps(0):
        cell_layer[c] = 1
    for c in triangulation_topomesh.wisps(0):
        if cell_layer[c] != 1:
            if np.any(cell_layer.values(list(triangulation_topomesh.region_neighbors(0,c))) == 1):
                cell_layer[c] = 2

    return cell_layer

