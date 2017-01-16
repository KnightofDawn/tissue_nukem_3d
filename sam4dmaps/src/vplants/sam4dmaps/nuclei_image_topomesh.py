import numpy as np

from vplants.sam4dmaps.nuclei_detection import detect_nuclei, compute_fluorescence_ratios
from vplants.sam4dmaps.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image

from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer, nuclei_curvature 
        
from openalea.container import array_dict
from openalea.mesh.property_topomesh_creation import vertex_topomesh

from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage


def nuclei_image_topomesh(filename, dirname=None, reference_name='TagBFP', signal_names=['DIIV','CLV3'], compute_ratios=[True,False], compute_curvature=False, microscope_orientation=-1, recompute=False, threshold=1000, size_range_start=0.6, size_range_end=0.9):
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
        
        image_coords = tuple(np.transpose((positions.values()/resolution).astype(int)))
        intensity_min = np.percentile(reference_img[image_coords],0)
        segmented_img = nuclei_active_region_segmentation(reference_img, positions, display=False, intensity_min=intensity_min)
        positions = nuclei_positions_from_segmented_image(segmented_img)
        
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
        cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,resolution,maximal_distance=10,return_topomesh=True,display=False)
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
            
    



