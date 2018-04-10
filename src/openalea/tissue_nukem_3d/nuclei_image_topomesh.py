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

from openalea.tissue_nukem_3d.nuclei_detection import detect_nuclei, compute_fluorescence_ratios
from openalea.tissue_nukem_3d.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image
from openalea.tissue_nukem_3d.nuclei_mesh_tools import nuclei_layer, nuclei_curvature, nuclei_topomesh_curvature, nuclei_surface_topomesh_curvature, nuclei_density_function
        
from openalea.container import array_dict

from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
from openalea.mesh.property_topomesh_creation import vertex_topomesh, triangle_topomesh
from openalea.mesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components, cut_surface_topomesh, clean_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.property_topomesh_optimization import property_topomesh_vertices_deformation, topomesh_triangle_split
from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe
from openalea.mesh.utils.delaunay_tools import delaunay_triangulation

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage

from copy import deepcopy


def nuclei_detection(reference_img, threshold=1000, radius_range=(0.8,1.4), step=0.2, segmentation_centering=False, subsampling=4, microscope_orientation=1):

    size = np.array(reference_img.shape)
    voxelsize = microscope_orientation*np.array(reference_img.voxelsize)


    positions = detect_nuclei(reference_img,threshold=threshold,radius_range=radius_range, step=step)
    positions = array_dict(positions)
    positions = array_dict(positions.values(),positions.keys()+2).to_dict()

    if segmentation_centering:
        nuclei_img = deepcopy(reference_img)
        image_coords = tuple(np.transpose((positions.values()/voxelsize).astype(int)))

        if subsampling>1:
            #nuclei_img = nd.gaussian_filter(nuclei_img,sigma=subsampling/4.)[::subsampling,::subsampling,::subsampling]
            nuclei_img = nd.gaussian_filter1d(nd.gaussian_filter1d(nuclei_img,sigma=subsampling/8.,axis=0),sigma=subsampling/8.,axis=1)[::subsampling,::subsampling,:]
            nuclei_img = SpatialImage(nuclei_img,voxelsize=(subsampling*reference_img.voxelsize[0],subsampling*reference_img.voxelsize[1],reference_img.voxelsize[2]))
            image_coords = tuple(np.transpose((positions.values()/(microscope_orientation*np.array(nuclei_img.voxelsize))).astype(int)))

        intensity_min = np.percentile(nuclei_img[image_coords],0)-1
        segmented_img = nuclei_active_region_segmentation(nuclei_img, positions, display=False, omega_energies=dict(intensity=subsampling,gradient=1.5,smoothness=10000.0*np.power(subsampling,1.5)), intensity_min=intensity_min)

        positions = nuclei_positions_from_segmented_image(segmented_img)
    
    positions = array_dict(positions)
    positions = array_dict(positions.values()*microscope_orientation,positions.keys()).to_dict()

    return positions


def nuclei_image_topomesh(image_dict, reference_name='TagBFP', signal_names=['DIIV','CLV3'], compute_ratios=[True,False], microscope_orientation=1, radius_range=(0.8,1.4), threshold=1000, subsampling=4, surface_voxelsize=1, nuclei_sigma=2.0, compute_layer=True, surface_mode='image', compute_curvature=True, return_surface=False):
    """Compute a point cloud PropertyTopomesh with image nuclei.

    The function runs a nuclei detection on the reference channel of
    the image, and then computes the signal intensities from the other
    channels (ratiometric or not) for each detected point. Additionnally,
    it can estimate the first layer of cells and compute the curvature
    on the surface.


    Args:
        image_dict (dict):
            An SatialImage dictionary with channel names as keys.
        reference_name (str):
            The name of the channel containing nuclei marker.
        signal_names (list):
            A list of the channel names on which to quantify signal.
        compute_ratios (list):
            A list of booleans stating if signal is to be computed 
            as channel intensity (False) or as a ratio between channel
            intensity and reference channel intensity (True) for each
            channel name in signal_names argument.
        microscope_orientation (int):
            1 for an upright microscope, -1 for an inverted one, in
            order to estimate consistently the upper layer of cells.
        radius_range (tuple):
            A tuple of two lengths (in microns) corresponding to the 
            typical size of nuclei to be detected in the image.
        threshold (int):
            A response intensity threshold for the detection, to 
            eliminate weak local maxima of response intensity. 
        subsampling (int):
            A downsampling factor to make the region growing part of 
            the detection faster.
        nuclei_sigma (float):
            The standard deviation (in microns) of the gaussian kernel
            used to quantify the signals.
        compute_layer (bool):
            Whether to estimate the first layer of cells based on the
            point cloud geometry.
        compute_curvature (bool):
            Whether to estimate surface curvature on the epidermal 
            layer of cells (compute_layer must be set to True)

    Returns:
        topomesh (:class:`openaela.mesh.PropertyTopomesh`):
            A point cloud structure with properties defined for each
            quantified signal, cell layer (if compute_layer argument 
            is set to True) and mean and gaussian curvatures (if 
            compute_curvature argument is set to true)
    """

    try:
        assert isinstance(image_dict,dict)
    except:
        print "Image channels should be passed as a Python dictionary with channel names as keys"
        raise
    assert reference_name in image_dict.keys()
    assert np.all([s in image_dict.keys() for s in signal_names])
    assert len(compute_ratios)==len(signal_names)

    reference_img = image_dict[reference_name]
    size = np.array(reference_img.shape)
    voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

    positions = nuclei_detection(reference_img, threshold=threshold, radius_range=radius_range, subsampling=subsampling, microscope_orientation=1)

    signal_values = {}
    for signal_name, compute_ratio in zip(signal_names,compute_ratios):
        signal_img = image_dict[signal_name]

        ratio_img = reference_img if compute_ratio else np.ones_like(reference_img)
        signal_values[signal_name] = compute_fluorescence_ratios(ratio_img,signal_img,positions)
        
    positions = array_dict(positions)
  

    positions = array_dict(positions.values()*microscope_orientation,positions.keys()).to_dict()
    topomesh = vertex_topomesh(positions)
    
    for signal_name in signal_names:
        topomesh.update_wisp_property(signal_name,0,signal_values[signal_name])
    
    # cell_layer, surface_topomesh = nuclei_layer(positions,reference_img,microscope_orientation=microscope_orientation,density_voxelsize=surface_voxelsize,return_topomesh=True) 
    cell_layer, surface_topomesh = nuclei_layer(positions,reference_img,microscope_orientation=microscope_orientation,density_voxelsize=surface_voxelsize,surface_mode=surface_mode,return_topomesh=True) 
    # cell_layer, surface_topomesh = nuclei_layer(positions,size,np.array(reference_img.voxelsize),return_topomesh=True) 
    topomesh.update_wisp_property('layer',0,cell_layer)

    # compute_curvature = False
    if compute_curvature:
        # triangulation_topomesh = nuclei_topomesh_curvature(topomesh,surface_subdivision=0,return_topomesh=True)
        topomesh = nuclei_surface_topomesh_curvature(topomesh,surface_topomesh=surface_topomesh)

        from copy import deepcopy
        topomesh._borders[1] = deepcopy(triangulation_topomesh._borders[1])
        topomesh._regions[0] = deepcopy(triangulation_topomesh._regions[0])
        for v in cell_layer.keys():
            if not v in topomesh._regions[0]:
                topomesh._regions[0][v] = []
        topomesh._borders[2] = deepcopy(triangulation_topomesh._borders[2])
        topomesh._regions[1] = deepcopy(triangulation_topomesh._regions[1])
        for t in topomesh.wisps(2):
            topomesh._regions[2][t] = []
        topomesh.add_wisps(3,0)
        for t in topomesh.wisps(2):
            topomesh.link(3,0,t)

    if return_surface:
        return topomesh, surface_topomesh
    else:
        return topomesh
        

# def nuclei_image_file_topomesh(filename, dirname=None, reference_name='TagBFP', signal_names=['DIIV','CLV3'], compute_ratios=[True,False], compute_curvature=True, microscope_orientation=-1, recompute=False, threshold=1000, size_range_start=0.6, size_range_end=0.9, subsampling=4):
#     '''
#     TODO
#     '''

#     if dirname is None:
#         from openalea.deploy.shared_data import shared_data
#         import vplants.meshing_data
#         dirname = shared_data(vplants.meshing_data)+"/nuclei_images"

#     reference_file = dirname+"/"+filename+"/"+filename+"_"+reference_name+".inr.gz"
#     reference_img = imread(reference_file)

#     size = np.array(reference_img.shape)
#     voxelsize = microscope_orientation*np.array(reference_img.voxelsize)

#     topomesh_file = dirname+"/"+filename+"/"+filename+"_nuclei_signal_curvature_topomesh.ply"

#     try:
#         assert not recompute 
#         topomesh = read_ply_property_topomesh(topomesh_file)
#         positions = topomesh.wisp_property('barycenter',0)
#     except:
#         positions = detect_nuclei(reference_img,threshold=threshold,size_range_start=size_range_start,size_range_end=size_range_end)
        
#         detection_topomesh = vertex_topomesh(array_dict(array_dict(positions).values()*microscope_orientation,array_dict(positions).keys()).to_dict())
#         detection_topomesh_file = dirname+"/"+filename+"/"+filename+"_detected_nuclei_topomesh.ply"
#         save_ply_property_topomesh(detection_topomesh,detection_topomesh_file,properties_to_save=dict([(0,[]),(1,[]),(2,[]),(3,[])]),color_faces=False)

#         from copy import deepcopy
#         nuclei_img = deepcopy(reference_img)
#         image_coords = tuple(np.transpose((positions.values()/voxelsize).astype(int)))
#         print "Reference image : ",nuclei_img.shape,nuclei_img.voxelsize

#         if subsampling>1:
#             #nuclei_img = nd.gaussian_filter(nuclei_img,sigma=subsampling/4.)[::subsampling,::subsampling,::subsampling]
#             nuclei_img = nd.gaussian_filter1d(nd.gaussian_filter1d(nuclei_img,sigma=subsampling/8.,axis=0),sigma=subsampling/8.,axis=1)[::subsampling,::subsampling,:]
#             nuclei_img = SpatialImage(nuclei_img,voxelsize=(subsampling*reference_img.voxelsize[0],subsampling*reference_img.voxelsize[1],reference_img.voxelsize[2]))
#             image_coords = tuple(np.transpose((positions.values()/(microscope_orientation*np.array(nuclei_img.voxelsize))).astype(int)))
        
#             print "Subsampled image : ",nuclei_img.shape,nuclei_img.voxelsize

#         intensity_min = np.percentile(nuclei_img[image_coords],0)
#         segmented_img = nuclei_active_region_segmentation(nuclei_img, positions, display=False, omega_energies=dict(intensity=subsampling,gradient=1.5,smoothness=10000.0*np.power(subsampling,1.5)), intensity_min=intensity_min)

#         positions = nuclei_positions_from_segmented_image(segmented_img)

#         segmentation_file = dirname+"/"+filename+"/"+filename+"_nuclei_seg.inr.gz"
#         imsave(segmentation_file,segmented_img)
        
#         positions = array_dict(positions)
#         positions = array_dict(positions.values(),positions.keys()+2).to_dict()
        
#         signal_values = {}
#         for signal_name, compute_ratio in zip(signal_names,compute_ratios):
#             signal_file = dirname+"/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
#             signal_img = imread(signal_file)

#             ratio_img = reference_img if compute_ratio else np.ones_like(reference_img)

#             signal_values[signal_name] = compute_fluorescence_ratios(ratio_img,signal_img,positions)
        
#         positions = array_dict(positions)
#         positions = array_dict(positions.values()*microscope_orientation,positions.keys()).to_dict()
        
#         topomesh = vertex_topomesh(positions)
#         for signal_name in signal_names:
#             topomesh.update_wisp_property(signal_name,0,signal_values[signal_name])
        
#         save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names),(1,[]),(2,[]),(3,[])]),color_faces=False)

#     surface_topomesh = None
#     if not (topomesh.has_wisp_property('layer',0)):
#         # cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,voxelsize,maximal_distance=10,return_topomesh=True,display=False)
#         # topomesh.update_wisp_property('layer',0,cell_layer)
#         grid_voxelsize = np.sign(voxelsize)*[4.,4.,4.]
#         x,y,z = np.ogrid[-0.5*size[0]*voxelsize[0]:1.5*size[0]*voxelsize[0]:grid_voxelsize[0],-0.5*size[1]*voxelsize[1]:1.5*size[1]*voxelsize[1]:grid_voxelsize[1],-0.5*size[2]*voxelsize[2]:1.5*size[2]*voxelsize[2]:grid_voxelsize[2]]
#         grid_size = 2*size
 
#         nuclei_density = nuclei_density_function(positions,cell_radius=5,k=1.0)(x,y,z) 

#         surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,voxelsize,iso=0.5,center=False)
#         surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*voxelsize*size,surface_topomesh.wisp_property('barycenter',0).keys()))

#         compute_topomesh_property(surface_topomesh,'normal',2,normal_method="density",object_positions=positions)
#         compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)

#         down_facing = surface_topomesh.wisp_property('normal',0).values()[:,2] < -0.0
#         surface_topomesh.update_wisp_property('downward',0,array_dict(down_facing,keys=list(surface_topomesh.wisps(0))))

#         triangle_down_facing = np.any(surface_topomesh.wisp_property('downward',0).values(surface_topomesh.wisp_property('vertices',2).values(list(surface_topomesh.wisps(2)))),axis=1)
#         triangle_down_facing = triangle_down_facing.astype(float)
#         surface_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(surface_topomesh.wisps(2))))
        
#         for i,t in enumerate(surface_topomesh.wisps(2)):
#             if surface_topomesh.wisp_property('downward',2)[t] == 1:
#                 triangle_neighbors = list(surface_topomesh.border_neighbors(2,t))
#                 if np.any(surface_topomesh.wisp_property('downward',2).values(triangle_neighbors)==0):
#                     triangle_down_facing[i] = 0.5
#         surface_topomesh.update_wisp_property('downward',2,array_dict(triangle_down_facing,keys=list(surface_topomesh.wisps(2))))
        
#         triangles_to_remove = np.array(list(surface_topomesh.wisps(2)))[surface_topomesh.wisp_property('downward',2).values() == 1]
#         for t in triangles_to_remove:
#             surface_topomesh.remove_wisp(2,t)
            
#         edges_to_remove = np.array(list(surface_topomesh.wisps(1)))[np.array([surface_topomesh.nb_regions(1,e)==0 for e in surface_topomesh.wisps(1)])]
#         for e in edges_to_remove:
#             surface_topomesh.remove_wisp(1,e)

#         vertices_to_remove = np.array(list(surface_topomesh.wisps(0)))[np.array([surface_topomesh.nb_regions(0,v)==0 for v in surface_topomesh.wisps(0)])]
#         for v in vertices_to_remove:
#             surface_topomesh.remove_wisp(0,v)
            
#         #top_surface_topomesh = cut_surface_topomesh(surface_topomesh,z_cut=0.8*size[2]*voxelsize[2])
#         top_surface_topomesh = topomesh_connected_components(surface_topomesh)[0]
#         top_surface_topomesh = topomesh_triangle_split(top_surface_topomesh)
        

#         #world.add(top_surface_topomesh,'top_surface')
        
#         surface_points = top_surface_topomesh.wisp_property('barycenter',0).values(list(top_surface_topomesh.wisps(0)))
#         surface_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(surface_points[:,0],surface_points[:,1],surface_points[:,2]) for p in positions.keys()])
#         surface_cells = np.array(positions.keys())[np.argmax(surface_potential,axis=0)]
        
#         L1_cells = np.unique(surface_cells)
        
#         cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
#         for c in L1_cells:
#             cell_layer[c] = 1
#         topomesh.update_wisp_property('layer',0,cell_layer)

#         save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
#     if compute_curvature and not (topomesh.has_wisp_property("mean_curvature",0)):
#         # if surface_topomesh is None:
#         #     cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,voxelsize,maximal_distance=10,return_topomesh=True,display=False)
#         #     topomesh.update_wisp_property('layer',0,cell_layer)

#         # cell_curvature = nuclei_curvature(positions,cell_layer,size,voxelsize,surface_topomesh)
#         # topomesh.update_wisp_property('mean_curvature',0,cell_curvature)

#         L1_cells = topomesh.wisp_property('layer',0).keys()[topomesh.wisp_property('layer',0).values()==1]
#         cell_barycenters = array_dict(topomesh.wisp_property('barycenter',0).values(L1_cells),L1_cells)
        
#         center = cell_barycenters.values().mean(axis=0) 
#         center[2] -= 3.*(cell_barycenters.values()-cell_barycenters.values().mean(axis=0))[:,2].max()
        
#         cell_vectors = cell_barycenters.values() - center
#         cell_r = np.linalg.norm(cell_vectors,axis=1)
#         cell_rx = np.linalg.norm(cell_vectors[:,np.array([0,2])],axis=1)
#         cell_ry = np.linalg.norm(cell_vectors[:,np.array([1,2])],axis=1)
            
#         cell_phi = np.sign(cell_vectors[:,0])*np.arccos(cell_vectors[:,2]/cell_rx)
#         cell_psi = np.sign(cell_vectors[:,1])*np.arccos(cell_vectors[:,2]/cell_ry)
            
#         cell_flat_barycenters = deepcopy(cell_barycenters)
#         for i,c in enumerate(cell_barycenters.keys()):
#             cell_flat_barycenters[c][0] = cell_phi[i]
#             cell_flat_barycenters[c][1] = cell_psi[i]
#             cell_flat_barycenters[c][2] = 0.
    
#         triangles = np.array(cell_barycenters.keys())[delaunay_triangulation(np.array([cell_flat_barycenters[c] for c in cell_barycenters.keys()]))]
#         cell_topomesh = triangle_topomesh(triangles, cell_barycenters)
            
#         maximal_length = 15.
        
#         compute_topomesh_property(cell_topomesh,'length',1)
#         compute_topomesh_property(cell_topomesh,'triangles',1)
        
#         boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
#         distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
#         edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
        
#         while len(edges_to_remove) > 0:
#             triangles_to_remove = np.concatenate(cell_topomesh.wisp_property('triangles',1).values(edges_to_remove))
#             for t in triangles_to_remove:
#                 cell_topomesh.remove_wisp(2,t)
            
#             clean_topomesh(cell_topomesh)
            
#             compute_topomesh_property(cell_topomesh,'triangles',1)
        
#             boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
#             distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
#             edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
            
        
#         cell_topomesh = topomesh_connected_components(cell_topomesh)[0]
#         property_topomesh_vertices_deformation(cell_topomesh,iterations=5)
        
#         cell_topomesh = topomesh_triangle_split(cell_topomesh)
#         compute_topomesh_property(cell_topomesh,'vertices',2)
        
#         property_topomesh_vertices_deformation(cell_topomesh,iterations=15)
        
#         compute_topomesh_property(cell_topomesh,'barycenter',2)
#         compute_topomesh_property(cell_topomesh,'normal',2,normal_method='orientation')
        
#         compute_topomesh_vertex_property_from_faces(cell_topomesh,'normal',adjacency_sigma=2,neighborhood=5)
#         compute_topomesh_property(cell_topomesh,'mean_curvature',2)
#         compute_topomesh_vertex_property_from_faces(cell_topomesh,'mean_curvature',adjacency_sigma=2,neighborhood=5)
#         compute_topomesh_vertex_property_from_faces(cell_topomesh,'gaussian_curvature',adjacency_sigma=2,neighborhood=5)
        
#         cell_points = cell_barycenters.values()

#         for curvature_property in ['mean_curvature','gaussian_curvature']:

#             surface_curvature = cell_topomesh.wisp_property(curvature_property,0).values(list(cell_topomesh.wisps(0)))
        
#             cell_potential = np.array([nuclei_density_function(dict([(p,cell_topomesh.wisp_property('barycenter',0)[p])]),cell_radius=5,k=1.0)(cell_points[:,0],cell_points[:,1],cell_points[:,2]) for p in cell_topomesh.wisps(0)])
#             cell_curvature = (cell_potential*surface_curvature[:,np.newaxis]).sum(axis=0)/cell_potential.sum(axis=0)
#             cell_curvature = dict(zip(cell_barycenters.keys(),cell_curvature))
#             topomesh.update_wisp_property(curvature_property,0,array_dict([cell_curvature[c] if c in cell_barycenters.keys() else 0 for c in topomesh.wisps(0)],list(topomesh.wisps(0))))
        
#         save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer','mean_curvature','gaussian_curvature']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
#     df = topomesh_to_dataframe(topomesh,0)
#     df.to_csv(dirname+"/"+filename+"/"+filename+"_signal_data.csv")    

#     return topomesh
            
    



