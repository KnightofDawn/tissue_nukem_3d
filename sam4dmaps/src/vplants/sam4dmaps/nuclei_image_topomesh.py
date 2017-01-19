import numpy as np
import scipy.ndimage as nd

from vplants.sam4dmaps.nuclei_detection import detect_nuclei, compute_fluorescence_ratios
from vplants.sam4dmaps.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image

from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer, nuclei_curvature 
from vplants.sam4dmaps.sam_model_tools import nuclei_density_function
        
from openalea.container import array_dict
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_extraction import epidermis_topomesh, topomesh_connected_components, cut_surface_topomesh
from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage


def nuclei_image_topomesh(filename, dirname=None, reference_name='TagBFP', signal_names=['DIIV','CLV3'], compute_ratios=[True,False], compute_curvature=False, microscope_orientation=-1, recompute=False, threshold=1000, size_range_start=0.6, size_range_end=0.9, subsampling=4):
    if dirname is None:
        from openalea.deploy.shared_data import shared_data
        import vplants.meshing_data
        dirname = shared_data(vplants.meshing_data)+"/nuclei_images"

    reference_file = dirname+"/"+filename+"/"+filename+"_"+reference_name+".inr.gz"
    reference_img = imread(reference_file)

    size = np.array(reference_img.shape)
    resolution = microscope_orientation*np.array(reference_img.resolution)

    topomesh_file = dirname+"/"+filename+"/"+filename+"_nuclei_signal_curvature_topomesh.ply"

    try:
        assert not recompute 
        topomesh = read_ply_property_topomesh(topomesh_file)
        positions = topomesh.wisp_property('barycenter',0)
    except:
        positions = detect_nuclei(reference_img,threshold=threshold,size_range_start=size_range_start,size_range_end=size_range_end)
        
        detection_topomesh = vertex_topomesh(array_dict(array_dict(positions).values()*microscope_orientation,array_dict(positions).keys()).to_dict())
        detection_topomesh_file = dirname+"/"+filename+"/"+filename+"_detected_nuclei_topomesh.ply"
        save_ply_property_topomesh(detection_topomesh,detection_topomesh_file,properties_to_save=dict([(0,[]),(1,[]),(2,[]),(3,[])]),color_faces=False)

        from copy import deepcopy
        nuclei_img = deepcopy(reference_img)
        image_coords = tuple(np.transpose((positions.values()/resolution).astype(int)))
        print "Reference image : ",nuclei_img.shape,nuclei_img.resolution

        if subsampling>1:
            #nuclei_img = nd.gaussian_filter(nuclei_img,sigma=subsampling/4.)[::subsampling,::subsampling,::subsampling]
            nuclei_img = nd.gaussian_filter1d(nd.gaussian_filter1d(nuclei_img,sigma=subsampling/8.,axis=0),sigma=subsampling/8.,axis=1)[::subsampling,::subsampling,:]
            nuclei_img = SpatialImage(nuclei_img,resolution=(subsampling*reference_img.resolution[0],subsampling*reference_img.resolution[1],reference_img.resolution[2]))
            image_coords = tuple(np.transpose((positions.values()/(microscope_orientation*np.array(nuclei_img.resolution))).astype(int)))
        
            print "Subsampled image : ",nuclei_img.shape,nuclei_img.resolution

        intensity_min = np.percentile(nuclei_img[image_coords],0)
        segmented_img = nuclei_active_region_segmentation(nuclei_img, positions, display=False, omega_energies=dict(intensity=subsampling,gradient=1.5,smoothness=10000.0*np.power(subsampling,1.5)), intensity_min=intensity_min)
        positions = nuclei_positions_from_segmented_image(segmented_img)

        segmentation_file = dirname+"/"+filename+"/"+filename+"_nuclei_seg.inr.gz"
        imsave(segmentation_file,segmented_img)
        
        positions = array_dict(positions)
        positions = array_dict(positions.values(),positions.keys()+2).to_dict()
        
        signal_values = {}
        for signal_name, compute_ratio in zip(signal_names,compute_ratios):
            signal_file = dirname+"/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
            signal_img = imread(signal_file)

            ratio_img = reference_img if compute_ratio else np.ones_like(reference_img)

            signal_values[signal_name] = compute_fluorescence_ratios(ratio_img,signal_img,positions)
        
        positions = array_dict(positions)
        positions = array_dict(positions.values()*microscope_orientation,positions.keys()).to_dict()
        
        topomesh = vertex_topomesh(positions)
        for signal_name in signal_names:
            topomesh.update_wisp_property(signal_name,0,signal_values[signal_name])
        
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names),(1,[]),(2,[]),(3,[])]),color_faces=False)

    surface_topomesh = None
    if not (topomesh.has_wisp_property('layer',0)):
        # cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,resolution,maximal_distance=10,return_topomesh=True,display=False)
        # topomesh.update_wisp_property('layer',0,cell_layer)
        grid_resolution = np.sign(resolution)*[4.,4.,4.]
        x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:grid_resolution[2]]
        grid_size = 2*size
 
        nuclei_density = nuclei_density_function(positions,cell_radius=5,k=1.0)(x,y,z) 

        surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution,iso=0.5,center=False)
        surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_topomesh.wisp_property('barycenter',0).values() - 0.5*resolution*size,surface_topomesh.wisp_property('barycenter',0).keys()))

        top_surface_topomesh = cut_surface_topomesh(surface_topomesh,z_cut=0.8*size[2]*resolution[2])
        top_surface_topomesh = topomesh_connected_components(top_surface_topomesh)[0]

        world.add(top_surface_topomesh,'top_surface')
        
        surface_points = top_surface_topomesh.wisp_property('barycenter',0).values(list(top_surface_topomesh.wisps(0)))
        surface_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(surface_points[:,0],surface_points[:,1],surface_points[:,2]) for p in positions.keys()])
        surface_cells = np.array(positions.keys())[np.argmax(surface_potential,axis=0)]
        
        L1_cells = np.unique(surface_cells)
        
        cell_layer = array_dict(np.zeros_like(positions.keys()),positions.keys())
        for c in L1_cells:
            cell_layer[c] = 1
        topomesh.update_wisp_property('layer',0,cell_layer)

        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
    if compute_curvature and not (topomesh.has_wisp_property("mean_curvature",0)):
        if surface_topomesh is None:
            cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,resolution,maximal_distance=10,return_topomesh=True,display=False)
            topomesh.update_wisp_property('layer',0,cell_layer)

        cell_curvature = nuclei_curvature(positions,cell_layer,size,resolution,surface_topomesh)
        topomesh.update_wisp_property('mean_curvature',0,cell_curvature)
        
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,signal_names+['layer','mean_curvature']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
    df = topomesh_to_dataframe(topomesh,0)
    df.to_csv(dirname+"/"+filename+"/"+filename+"_signal_data.csv")    

    return topomesh
            
    



