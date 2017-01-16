import numpy as np
import pandas as pd

from vplants.sam4dmaps.nuclei_detection import detect_nuclei, compute_fluorescence_ratios
from vplants.sam4dmaps.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image

from openalea.container import array_dict
from openalea.mesh import TriangularMesh
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage

from openalea.oalab.colormap.colormap_def import load_colormaps

import pickle
from copy import deepcopy

tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

#filename = "r2DII_3.2_141127_sam08_t24"
#filename = "DR5N_5.2_150415_sam01_t00"
#filename = "r2DII_1.2_141202_sam06_t04"
#filename = "r2DII_1.2_141202_sam03_t32"
#previous_filename = "r2DII_2.2_141204_sam01_t28"
#previous_filename = "r2DII_1.2_141202_sam03_t28"
#previous_filename = "r2DII_2.2_141204_sam07_t04"
#previous_filename = "r2DII_3.2_141127_sam08_t00"
#filename = str(argv[1])

filenames = [] 

# filenames += ["qDII-CLV3-PI-LD_11c_160518_sam01_t00"]
# filenames += ["qDII-CLV3-PI-LD_11c_160518_sam04_t00"]
#filenames += ["qDII-CLV3-LD_11c_160404_sam02_t00"]
#filenames += ["qDII-CLV3-PI-LD_11c_160519_sam03_t00"]
#filenames += ["qDII-CLV3-PI-LD_11c_160518_sam02_t00"]
# filenames += ["qDII-CLV3-PI-SD_11c_160522_sam01_t00"] #regionalext seg_fault?
#filenames += ["qDII-CLV3-PI-SD_11c_160523_sam02_t00"]

# filenames += ["qDII-CLV3-SD_11c_160623_sam01_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam02_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam03_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam04_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam05_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam06_t00"]
# filenames += ["qDII-CLV3-SD_11c_160623_sam07_t00"]

# filenames += ["qDII-CLV3-LD_11c_160623_sam01_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam02_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam03_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam04_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam05_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam06_t00"]
# filenames += ["qDII-CLV3-LD_11c_160623_sam07_t00"]

# filenames += ["qDII-CLV3-SD_11c_160628_sam02_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam03_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam04_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam10_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam12_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam15_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam16_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam18_t00"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam20_t00"]

# filenames += ["qDII-CLV3-SD_11c_160628_sam02_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam03_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam04_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam10_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam12_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam15_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam16_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam18_t06"]
# filenames += ["qDII-CLV3-SD_11c_160628_sam20_t06"]

for t in [0, 6, 10]:
    for sam in [1, 2, 3, 5, 6, 7, 9, 10, 13, 14, 15, 16, 18]:
        filenames += ["qDII-CLV3-LD_11c_160906_sam"+str(sam).zfill(2)+"_t"+str(t).zfill(2)]
        

from openalea.deploy.shared_data import shared_data
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)

reference_name = 'TagBFP'

microscope_orientation = -1

signal_name = 'DIIV'
#signal_name = 'DR5'

signal_colors = {}
signal_colors['DIIV'] = 'green'
signal_colors['DR5'] = 'Blues'
signal_colors['Auxin'] = 'viridis'
signal_colors['CLV3'] = 'purple'
signal_colors['CLV3_ratio'] = 'purple'
signal_colors['mean_curvature'] = 'curvature'

identity_name = 'CLV3'


world.clear()
nuclei_positions = {}
nuclei_signals = {}
meristem_models = {}

for filename in filenames:

    signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
    #signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_DR5.inr.gz"
    signal_img = imread(signal_file)
    
    identity_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+identity_name+".inr.gz"
    identity_img = imread(identity_file)
    
    reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+reference_name+".inr.gz"
    reference_img = imread(reference_file)
    
    size = np.array(reference_img.shape)
    resolution = microscope_orientation*np.array(reference_img.resolution)
    
    noorgan_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_NoOrgans_"+reference_name+".tif"
    noorgan_img = SpatialImage(imread(noorgan_file),resolution=reference_img.resolution)
    
    
    #world.add(identity_img,'identity_image',resolution=resolution,colormap='Purples')
    #world.add(reference_img,'nuclei_image',resolution=resolution,colormap='Blues')
    #world.add(signal_img,'signal_image',resolution=resolution,colormap=signal_colors[signal_name])
    
    #topomesh_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_nuclei_signal_curvature_topomesh.ply"
    topomesh_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_no_organs_nuclei_signal_curvature_topomesh.ply"
    
    try:
        topomesh = read_ply_property_topomesh(topomesh_file)
        positions = topomesh.wisp_property('barycenter',0)
    except:
        #positions = detect_nuclei(reference_img,threshold=1000,size_range_start=0.6,size_range_end=0.9)
        positions = detect_nuclei(noorgan_img,threshold=1000,size_range_start=0.6,size_range_end=0.9)
        
        image_coords = tuple(np.transpose((positions.values()/resolution).astype(int)))
        intensity_min = np.percentile(reference_img[image_coords],0)
        segmented_img = nuclei_active_region_segmentation(reference_img, positions, display=False, intensity_min=intensity_min)
        positions = nuclei_positions_from_segmented_image(segmented_img)
        
        positions = array_dict(positions)
        positions = array_dict(positions.values(),positions.keys()+2).to_dict()
        
        signal_ratios = compute_fluorescence_ratios(reference_img,signal_img,positions)
        identity_values = compute_fluorescence_ratios(np.ones_like(reference_img),identity_img,positions)
        
        positions = array_dict(positions)
        positions = array_dict(positions.values()*microscope_orientation,positions.keys()).to_dict()
        
        import vplants.sam4dmaps.nuclei_mesh_tools
        reload(vplants.sam4dmaps.nuclei_mesh_tools)
        from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer, nuclei_curvature 
        
        topomesh = vertex_topomesh(positions)
        topomesh.update_wisp_property(signal_name,0,signal_ratios)
        topomesh.update_wisp_property(identity_name,0,identity_values)
        
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,[signal_name,identity_name]),(1,[]),(2,[]),(3,[])]),color_faces=False)
            

    # if not topomesh.has_wisp_property(identity_name+"_ratio",0):
    #     identity_ratios = compute_fluorescence_ratios(reference_img,identity_img,positions)     
    #     topomesh.update_wisp_property(identity_name+"_ratio",0,identity_ratios)  
    #     save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,[signal_name,identity_name,identity_name+"_ratio",'layer']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
    if not (topomesh.has_wisp_property('layer',0) and topomesh.has_wisp_property("mean_curvature",0)):
    #if True:
    
        import openalea.cellcomplex.property_topomesh.property_topomesh_analysis
        reload(openalea.cellcomplex.property_topomesh.property_topomesh_analysis)

        import vplants.sam4dmaps.nuclei_mesh_tools
        reload(vplants.sam4dmaps.nuclei_mesh_tools)
        from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer, nuclei_curvature 

        cell_layer, triangulation_topomesh, surface_topomesh = nuclei_layer(positions,size,resolution,maximal_distance=10,return_topomesh=True,display=False)
        
        world.add(triangulation_topomesh,"L1_topomesh")
        world['L1_topomesh_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
        world['L1_topomesh_cells'].set_attribute('intensity_range',(-1,0))
        world['L1_topomesh'].set_attribute('coef_3',0.98)
        
        world.add(surface_topomesh,'surface')
        
        topomesh.update_wisp_property('layer',0,cell_layer)

        cell_curvature = nuclei_curvature(positions,cell_layer,size,resolution,surface_topomesh)
        topomesh.update_wisp_property('mean_curvature',0,cell_curvature)
        
        import openalea.cellcomplex.property_topomesh.property_topomesh_io
        reload(openalea.cellcomplex.property_topomesh.property_topomesh_io)
        from openalea.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh
        save_ply_property_topomesh(topomesh,topomesh_file,properties_to_save=dict([(0,[signal_name,identity_name,identity_name+"_ratio",'layer','mean_curvature']),(1,[]),(2,[]),(3,[])]),color_faces=False) 
    
        world.add(topomesh,'detected_nuclei')
        world['detected_nuclei'].set_attribute('property_name_0','layer')
    
    nuclei_signals[filename] = topomesh
    
    # cell_layer = topomesh.wisp_property('layer',0)
    # epidermis_cells = cell_layer.keys()[cell_layer.values()==1]
    # L1_topomesh = vertex_topomesh(array_dict(array_dict(positions).values(epidermis_cells),epidermis_cells))
    # L1_topomesh.update_wisp_property('mean_curvature',0,array_dict(cell_curvature.values(epidermis_cells),epidermis_cells))
    # world.add(L1_topomesh,'L1_nuclei')
    # raw_input()
    
    
    # world.add(topomesh,'detected_nuclei')
    # world['detected_nuclei'].set_attribute('property_name_0','layer')
    # world['detected_nuclei_vertices'].set_attribute('point_radius',3)
    #world['detected_nuclei_vertices'].set_attribute('intensity_range',(0,1))

    meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
     
    import vplants.sam4dmaps.parametric_shape
    reload(vplants.sam4dmaps.parametric_shape)
    
    import vplants.sam4dmaps.sam_model
    reload(vplants.sam4dmaps.sam_model)
    from vplants.sam4dmaps.sam_model import estimate_meristem_model, read_meristem_model
    
    try:
        meristem_model = read_meristem_model(meristem_model_file)
    except:
        meristem_model,_,_ = estimate_meristem_model(positions,size,resolution,microscope_orientation=microscope_orientation,display=False)
        pickle.dump(meristem_model.parameters,open(meristem_model_file,'wb'))
    #world.add(meristem_model,'meristem_model',_repr_vtk_=meristem_model.drawing_function,colormap='leaf',alpha=0.1,z_slice=(90,100))
    meristem_models[filename] = meristem_model


import vplants.sam4dmaps.sam_model_tools
reload(vplants.sam4dmaps.sam_model_tools)
from vplants.sam4dmaps.sam_model_tools import plot_meristem_model

from vplants.sam4dmaps.sam_model import reference_meristem_model
from vplants.sam4dmaps.sam_model_registration import meristem_model_cylindrical_coordinates, meristem_model_organ_gap
from vplants.sam4dmaps.sam_map_construction import meristem_2d_cylindrical_map, draw_signal_map

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function
from openalea.mesh.property_topomesh_creation import triangle_topomesh
from openalea.mesh import TriangularMesh
            
import matplotlib.pyplot as plt
from openalea.oalab.colormap.colormap_def import load_colormaps

from scipy.stats import gamma

def local_extrema(signal, abscissa=None, scale=1, threshold=None):
    if abscissa is None:
        abscissa = np.arange(len(signal))
    
    distances = np.power(np.power(abscissa[np.newaxis] - abscissa[:,np.newaxis],2),0.5)
    
    maxima = np.ones_like(signal,bool)
    signal_neighborhood_max = np.array([np.max(signal[(distances[p]<=scale)&(distances[p]>0)]) for p in xrange(len(signal))])
    maxima = maxima & (signal > signal_neighborhood_max)
    maxima[0] = False
    maxima[-1] = False
    maximal_points = np.transpose([abscissa[maxima],signal[maxima]])
                     
    minima = np.ones_like(signal,bool)
    signal_neighborhood_min = np.array([np.min(signal[(distances[p]<=scale)&(distances[p]>0)]) for p in xrange(len(signal))])
    minima = minima & (signal < signal_neighborhood_min)
    minima[0] = False
    minima[-1] = False
    minimal_points = np.transpose([abscissa[minima],signal[minima]])
    
    return maximal_points, minimal_points
            

for colormap_name in ['curvature']:
    color_dict = load_colormaps()[colormap_name]._color_points
    mpl_color_dict = {}
    for k,col in enumerate(['red','green','blue']):
        mpl_color_dict[col] = ()
    for p in np.sort(color_dict.keys()):
        for k,col in enumerate(['red','green','blue']):
            mpl_color_dict[col] += ((p,color_dict[p][k],color_dict[p][k]),)
    plt.register_cmap(name=colormap_name, data=mpl_color_dict)


figure = plt.figure(0)

signal_ranges = {}
signal_ranges['DIIV'] = (0.0,1.0)
signal_ranges['Auxin'] = (0.2,1.0)
signal_ranges['CLV3'] = (0.0,1.0)
signal_ranges['mean_curvature'] = (-0.05,0.05)
#signal_ranges['CLV3_ratio'] = (0.0,1.0)

primordia_colors = {}
primordia_colors[-3] = "#333333"
primordia_colors[-2] = "#0ce838"
primordia_colors[-1] = "#0065ff"
primordia_colors[0] = "#64bca4"
primordia_colors[1] = "#e30d00"
primordia_colors[2] = "#ffa200"
primordia_colors[3] = "#cccccc"
primordia_colors[4] = "#dddddd"
primordia_colors[5] = "#eeeeee"

all_samples_data = {}
all_samples_data['filename'] = []
all_samples_data['growth_condition'] = []

r_max = 70

reference_dome_apex = np.zeros(3)
n_organs = 8

#world.clear()

reference_model = reference_meristem_model(reference_dome_apex,n_primordia=n_organs,developmental_time=0)
orientation = reference_model.parameters['orientation']
golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
golden_angle = 180.*golden_angle/np.pi    

auxin_extrema = {}
for field in ['filename','growth_condition','extremum_type','extremum_x','extremum_y','extremum_r','extremum_surface_distance','extremum_theta','auxin']:
    auxin_extrema[field] = []
    
meristem_data = {}
for field in ['filename','growth_condition','auxin_minimum_r','auxin_minimum_surface_distance','clv3_radius','clv3_center_error','mean_curvature']:
    meristem_data[field] = []

circle_data = {}
for field in ['filename','growth_condition','theta','aligned_theta','DIIV','normalized_DIIV']:
    circle_data[field] = []
    
slice_data = {}
for field in ['filename','r','DIIV','normalized_DIIV','primordium']:
    slice_data[field] = []
    
primordia_data = {}
for field in ['filename','growth_condition','distance','theta','aligned_theta','normalized_DIIV','circle_normalized_DIIV','primordium']:
    primordia_data[field] = []
    
draw_maps = False
mesh_landscape = False

clv3_data = {}
for field in ['filename','growth_condition','clv3_threshold','clv3_center_x','clv3_center_y','clv3_radius']:
    clv3_data[field] = []

circle_var_data = {}
for field in ['filename','growth_condition','radius_perturbation','MSE','error_percent','theta_min_error','n_minima','n_maxima']:
    circle_var_data[field] = []

for filename in filenames:
    topomesh = nuclei_signals[filename]
    positions = topomesh.wisp_property('barycenter',0)
    
    meristem_model = meristem_models[filename]

    organ_gap = meristem_model_organ_gap(reference_model,meristem_models[filename],same_individual=False)
    coords, aligned_meristem_model = meristem_model_cylindrical_coordinates(meristem_model,positions,-organ_gap,orientation)
    
    data = topomesh_to_dataframe(topomesh,0)
    data['model_coords_theta'] = coords.values(data.index)[:,0]
    data['model_coords_r'] = coords.values(data.index)[:,1]
    data['model_coords_z'] = coords.values(data.index)[:,2]
    data['model_coords_x'] = data['model_coords_r'] * np.cos(np.pi*data['model_coords_theta']/180.)
    data['model_coords_y'] = data['model_coords_r'] * np.sin(np.pi*data['model_coords_theta']/180.)
    data['Auxin'] = 1 - data['DIIV']
    data = data[data['layer']==1]
    for signal in ['CLV3','CLV3_ratio']:
        signal_data  = data[signal][data['model_coords_r']<2.*r_max/3.]
        #gamma_k, gamma_loc, gamma_theta = gamma.fit(data[signal])
        #print filename,signal," :  (",gamma_k,",",gamma_theta,")  -> ",gamma_loc,"   [",data[signal].mean(),",",data[signal].std(),"]"
        #data['Normalized_'+signal] = data[signal] - gamma_loc
        #data['Normalized_'+signal] = np.power((data[signal] - gamma_loc)/np.power(gamma_theta,1),1)
        #gamma_k, gamma_loc, gamma_theta = gamma.fit(data["Normalized_"+signal])
        #data['Normalized_'+signal] =  data['Normalized_'+signal]/(gamma_k*gamma_theta)
        #data['Normalized_'+signal] =  0.5 + 0.2*(data['Normalized_'+signal]-data['Normalized_'+signal].mean())/(data['Normalized_'+signal].std())
        #gamma_k, gamma_loc, gamma_theta = gamma.fit(data["Normalized_"+signal])
        #print filename,'Normalized_'+signal," :  (",gamma_k,",",gamma_theta,")  -> ",gamma_loc,"   [",data["Normalized_"+signal].mean(),",",data["Normalized_"+signal].std(),"]"
        #data['Normalized_'+signal] = 0.3 + 0.3*(data[signal]-np.percentile(data[signal],50))/(np.percentile(data[signal],90)-np.percentile(data[signal],10)) 
        #data['Normalized_'+signal] = data[signal]/data[signal].mean()
        #proba = np.linspace(0,100,1001)
        #proba_values = np.array([np.nanpercentile(signal_data,p) for p in proba])
        #data['Normalized_'+signal] = proba[vq(data[signal].values,proba_values)[0]]/100.
        
        #data['Normalized_'+signal] = 0.2 + 0.3*(data[signal]-signal_data.mean())/(signal_data.std()) 
        #data['Normalized_'+signal] = data[signal]/signal_data.max()
        #data['Normalized_'+signal] = data[signal]/np.percentile(signal_data,95)
        data['Normalized_'+signal] = data[signal]/12000000

        
    for signal in ['DIIV','Auxin']:
        #signal_data  = data[signal][data['model_coords_r']<2.*r_max/3.]
        signal_data  = data[signal]
        #data['Normalized_'+signal] = (data[signal]-data[signal].min())/(data[signal].max()-data[signal].min()) 
        data['Normalized_'+signal] = 0.5 + 0.2*(data[signal]-signal_data.mean())/(signal_data.std()) 
        #data['Normalized_'+signal] = 0.3 + 0.3*(data[signal]-np.percentile(data[signal],50))/(np.percentile(data[signal],90)-np.percentile(data[signal],10)) 
    
    for signal in ['mean_curvature']:
        signal_data  = data[signal]
        data['Normalized_'+signal] = data[signal]
    
    all_samples_data['filename'] += [filename for x in xrange(len(data))]
    all_samples_data['growth_condition'] += ['SD' if 'SD' in filename else 'LD' for x in xrange(len(data))]
    for field in data.keys():
        if not all_samples_data.has_key(field):
            all_samples_data[field] = []
        all_samples_data[field] += list(data[field].values)

    world.clear()
    world.add(data,'topomesh_data')
    
    import openalea.cellcomplex.mesh_oalab.widget.cute_plot 
    reload(openalea.cellcomplex.mesh_oalab.widget.cute_plot )
    from openalea.cellcomplex.mesh_oalab.widget.cute_plot import map_plot
    
    import scipy.ndimage as nd
    from scipy.cluster.vq import vq
    import matplotlib.patches as patch
    import time
     
    primordia_thetas = {}
    primordia_distances = {}
    
    for signal in ['CLV3','DIIV','Auxin','mean_curvature']:
        
        if draw_maps:    
            world['topomesh_data'].set_attribute('plot','scatter')
            world['topomesh_data'].set_attribute('label_variable','Normalized_'+signal)
            world['topomesh_data'].set_attribute('X_variable','model_coords_x')
            world['topomesh_data'].set_attribute('Y_variable','model_coords_y')
            world['topomesh_data'].set_attribute('label_colormap',load_colormaps()[signal_colors[signal]])
            world['topomesh_data'].set_attribute('legend','top_right')
            label_range = tuple([100.*(s-data['Normalized_'+signal].min())/(data['Normalized_'+signal].max()-data['Normalized_'+signal].min()) for s in signal_ranges[signal]])
            world['topomesh_data'].set_attribute('label_range',label_range)
            world['topomesh_data'].set_attribute('smooth_factor',0.1)
            world['topomesh_data'].set_attribute('plot','map')
    
        if signal in ['CLV3','CLV3_ratio']:
            map_figure = plt.figure(1)
            map_figure.clf()
            X = data['model_coords_x'].values
            Y = data['model_coords_y'].values
            data_range = [[-200,200],[-200,200]]
            xx, yy, zz = map_plot(map_figure,X,Y,data['Normalized_'+signal].values,XY_range=data_range,smooth_factor=1,n_points=100)

            resolution = np.array([xx[0,1]-xx[0,0],yy[1,0]-yy[0,0]])
            
            #clv3_threshold = 0.2
            #for clv3_threshold in np.linspace(0.05,0.5,46):
            for clv3_threshold in [0.4]:

                zz[np.isnan(zz)] = 0
                #clv3_gradient = nd.gaussian_gradient_magnitude(zz,1.0)
                
                clv3_regions = nd.label((zz>clv3_threshold).astype(int))[0]
                components = np.unique(clv3_regions)[1:]
                component_centers = np.transpose([nd.sum(xx,clv3_regions,index=components),nd.sum(yy,clv3_regions,index=components)])/nd.sum(np.ones_like(xx),clv3_regions,index=components)[:,np.newaxis]
                
                if len(component_centers)>0:
                    component_matching = vq(np.array([[0,0]]),component_centers)
                    
                    clv3_center = component_centers[component_matching[0][0]]
                    clv3_area = (clv3_regions==component_matching[0]+1).sum() * np.prod(resolution)
                    clv3_radius = np.sqrt(clv3_area/np.pi)
                    
                    clv3_data['filename'] += [filename]
                    clv3_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
                    clv3_data['clv3_threshold'] += [clv3_threshold]
                    clv3_data['clv3_center_x'] += [clv3_center[0]]
                    clv3_data['clv3_center_y'] += [clv3_center[1]]
                    clv3_data['clv3_radius'] += [clv3_radius]

# clv3_df = pd.DataFrame.from_dict(clv3_data)
# clv3_df['clv3_center_r'] = np.linalg.norm([clv3_df['clv3_center_x'],clv3_df['clv3_center_y']],axis=0)
# world.add(clv3_df,'CLV3_circle')
# world['CLV3_circle'].set_attribute('figure',1)
# world['CLV3_circle'].set_attribute('X_variable','clv3_threshold')
# world['CLV3_circle'].set_attribute('Y_variable','clv3_radius')
# world['CLV3_circle'].set_attribute('Y_range',dataframe_relative_range(clv3_df,'clv3_radius',(0,80)))
# world['CLV3_circle'].set_attribute('class_variable','growth_condition')
# world['CLV3_circle'].set_attribute('label_colormap',load_colormaps()['vegetation'])
# world['CLV3_circle'].set_attribute('plot','boxplot')

                map_figure.gca().pcolormesh(xx,yy,(clv3_regions==component_matching[0]+1).astype(int),cmap='Purples',antialiased=True,shading='gouraud',vmin=0,vmax=1)
                c = patch.Circle(xy=clv3_center,radius=clv3_radius,ec="#c94389",fc='None',lw=3,alpha=1)
                map_figure.gca().add_artist(c)
                
                plot_meristem_model(map_figure,aligned_meristem_model,r_max=r_max)
    
        elif signal in ['DIIV']:
            
            signal_data = data[signal].values
            normalized_signal_data = data['Normalized_'+signal].values
            #signal_data = data['Normalized_'+signal].values
            
            inhibition_scale = 100.
            
            X = data['model_coords_x'].values
            Y = data['model_coords_y'].values
            projected_positions = dict(zip(data.index,np.transpose([X,Y,np.zeros_like(X)])))
            
            circle_thetas = np.linspace(-180,180,361)
            circle_x = clv3_center[0] + clv3_radius*np.cos(np.pi*circle_thetas/180.)
            circle_y = clv3_center[1] + clv3_radius*np.sin(np.pi*circle_thetas/180.)
            circle_z = np.zeros_like(circle_x)
            
            cell_radius=5.0
            density_k = 0.15
            
            circle_potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=cell_radius,k=density_k)(circle_x,circle_y,circle_z) for p in data.index])
            circle_potential = np.transpose(circle_potential)
            circle_density = np.sum(circle_potential,axis=1)
            circle_membership = circle_potential/circle_density[...,np.newaxis]

            circle_signal = np.sum(circle_membership*signal_data[np.newaxis,:],axis=1)
            circle_normalized_signal = np.sum(circle_membership*normalized_signal_data[np.newaxis,:],axis=1)

            theta_min = circle_thetas[np.argmin(circle_signal)]
            print "P0 : ",theta_min
            
            aligned_thetas = ((circle_thetas-theta_min+180)%360 - 180)
            aligned_signal = circle_signal
            max_points, min_points = local_extrema(aligned_signal,abscissa=aligned_thetas)
            
            for var in np.linspace(0.4,1.6,31):
                var_radius = var*clv3_radius
                var_x = clv3_center[0] + var_radius*np.cos(np.pi*circle_thetas/180.)
                var_y = clv3_center[1] + var_radius*np.sin(np.pi*circle_thetas/180.)
                var_z = np.zeros_like(var_x)
                
                var_potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=cell_radius,k=density_k)(var_x,var_y,var_z) for p in data.index])
                var_potential = np.transpose(var_potential)
                var_density = np.sum(var_potential,axis=1)
                var_membership = var_potential/var_density[...,np.newaxis]
    
                var_signal = np.sum(var_membership*signal_data[np.newaxis,:],axis=1)
                var_normalized_signal = np.sum(var_membership*normalized_signal_data[np.newaxis,:],axis=1)
                
                var_theta_min = circle_thetas[np.argmin(var_signal)]
                var_aligned_thetas = ((circle_thetas-var_theta_min+180)%360 - 180)
                var_max_points, var_min_points = local_extrema(var_signal,abscissa=var_aligned_thetas)
            
                circle_var_data['filename'] += [filename]
                circle_var_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
                circle_var_data['radius_perturbation'] += [100.*(var - 1.)]
                circle_var_data['MSE'] += [np.power(var_normalized_signal-circle_normalized_signal,2).mean()]
                circle_var_data['error_percent'] += [100.*(np.abs(var_normalized_signal-circle_normalized_signal).mean()/np.abs(circle_normalized_signal).mean())]
                circle_var_data['theta_min_error'] += [np.abs(var_theta_min-theta_min)%360]
                circle_var_data['n_minima'] += [len(var_min_points) - len(min_points)]
                circle_var_data['n_maxima'] += [len(var_max_points) - len(max_points)]
                
circle_var_df = pd.DataFrame.from_dict(circle_var_data)
world.add(circle_var_df,'CLV3_circle_variations')


            if mesh_landscape:      
                circle_points = np.transpose([circle_x,circle_y,inhibition_scale*circle_signal])
                circle_points = dict(zip(range(len(circle_points)),circle_points))
                circle_edges = np.transpose([np.arange(len(circle_thetas))[:-1],np.arange(len(circle_thetas))[1:]])
                circle_edges = dict(zip(range(len(circle_edges)),circle_edges))
                
                circle_mesh = TriangularMesh()
                circle_mesh.points = circle_points
                circle_mesh.edges = circle_edges
                world.add(circle_mesh,'initiation_circle',linewidth=3,colormap='purple',intensity_range=(-1,1),display_colorbar=False)
            
            

            #figure.gca().scatter(max_points[:,0],max_points[:,1],color='k')
            #figure.gca().scatter(min_points[:,0],min_points[:,1],color='k')
            
            primordium = 2
            primordium_theta = (primordium*golden_angle + 180)%360 - 180
            extremum_points = np.concatenate([min_points,max_points])
            extremum_types = np.concatenate([['min' for p in min_points],['max' for p in max_points]])
            extremum_match = vq(np.array([primordium_theta]),extremum_points[:,0])
            extremum_theta = extremum_points[:,0][extremum_match[0][0]]
            extremum_type = extremum_types[extremum_match[0][0]]
            if extremum_type == 'min':
                meristem_orientation = -1
            else:
                meristem_orientation = 1
            
            if meristem_orientation == -1:
                print filename," : Wrong orientation detected!!"
            
            circle_thetas = meristem_orientation*circle_thetas
            #theta_min =  meristem_orientation*theta_min
            #aligned_thetas =  ((circle_thetas-theta_min+180)%360 - 180)
            aligned_thetas = meristem_orientation*aligned_thetas
            #aligned_signal = aligned_signal[::meristem_orientation]
            aligned_normalized_signal = circle_normalized_signal
            
            circle_data['theta'] += list(circle_thetas)
            circle_data['aligned_theta'] += list(aligned_thetas)
            circle_data[signal] += list(aligned_signal)
            circle_data["normalized_"+signal] += list(aligned_normalized_signal)
            circle_data['filename'] += [filename for t in circle_thetas]
            circle_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for t in circle_thetas]
            
            circle_df = pd.DataFrame().from_dict(circle_data)
            world.add(circle_df[circle_df['filename']==filename],'initiation_circle')
            world['initiation_circle'].set_attribute('X_variable','aligned_theta')
            world['initiation_circle'].set_attribute('Y_variable','normalized_DIIV')
            world['initiation_circle'].set_attribute('plot','line')
            #raw_input()
            
            aligned_signal = circle_df[circle_df['filename']==filename]['normalized_DIIV'].values
            aligned_thetas = circle_df[circle_df['filename']==filename]['aligned_theta'].values
            aligned_signal = aligned_signal[np.argsort(aligned_thetas)]
            aligned_thetas = aligned_thetas[np.argsort(aligned_thetas)]
            
            max_points, min_points = local_extrema(aligned_signal,abscissa=aligned_thetas)
            figure.gca().scatter(max_points[:,0],max_points[:,1],color='k')
            figure.gca().scatter(min_points[:,0],min_points[:,1],color='k')
                
            for primordium in [-3,-2,-1,0,1,2,3,5]:
            #for primordium in [-2,-1,0,1,2]:
                #primordium_theta = (theta_min + primordium*golden_angle + 180)%360 - 180
                primordium_theta = (primordium*golden_angle + 180)%360 - 180
            
                if primordium > 2:
                    extremum_points = max_points
                elif primordium > 0:
                    extremum_points = np.concatenate([min_points,max_points])
                else:
                    extremum_points = min_points
                extremum_match = vq(np.array([primordium_theta]),extremum_points[:,0])
                extremum_theta = extremum_points[:,0][extremum_match[0][0]]
                extremum_error = (extremum_theta-primordium_theta + 180)%360 - 180
                print "P",primordium," : ",extremum_theta,"(",primordium_theta,") : ",np.abs(extremum_error)
                
                if np.abs(extremum_error)<np.abs(golden_angle/4.):
                    figure.gca().plot([primordium_theta,primordium_theta],[0,1],linewidth=1,color=primordia_colors[primordium],alpha=0.5/(abs(primordium)+1))
                    figure.gca().plot([extremum_theta,extremum_theta],[0,1],linewidth=1,color=primordia_colors[primordium],alpha=1./(abs(primordium)+1))
                    figure.gca().annotate('P'+str(primordium),(extremum_theta,np.min(circle_df[circle_df['filename']==filename]['normalized_DIIV'].values)),color=primordia_colors[primordium])
            
                    primordia_thetas[primordium] = (theta_min + meristem_orientation*extremum_theta + 180)%360 - 180
            
                    slice_distances = np.linspace(0,80,161)
                    slice_x = clv3_center[0] + slice_distances*np.cos(np.pi*primordia_thetas[primordium]/180.)
                    slice_y = clv3_center[1] + slice_distances*np.sin(np.pi*primordia_thetas[primordium]/180.)
                    slice_z = np.zeros_like(slice_x)
                    
                    slice_potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=cell_radius,k=density_k)(slice_x,slice_y,slice_z) for p in data.index])
                    slice_potential = np.transpose(slice_potential)
                    slice_density = np.sum(slice_potential,axis=1)
                    slice_membership = slice_potential/slice_density[...,np.newaxis]
                    
                    slice_signal = np.sum(slice_membership*signal_data[np.newaxis,:],axis=1)
                    slice_normalized_signal = np.sum(slice_membership*normalized_signal_data[np.newaxis,:],axis=1)
                    
                    slice_data['r'] += list(slice_distances)
                    slice_data[signal] += list(slice_signal)
                    slice_data['normalized_'+signal] += list(slice_normalized_signal)
                    slice_data['filename'] += [filename for r in slice_distances]
                    slice_data['primordium'] += [primordium for r in slice_distances]
            
                    if mesh_landscape:
                        slice_points = np.transpose([slice_x,slice_y,inhibition_scale*slice_signal])
                        slice_points = dict(zip(range(len(slice_points)),slice_points))
                        slice_point_data = dict(zip(range(len(slice_points)),primordium*np.ones_like(slice_distances)))
                        slice_edges = np.transpose([np.arange(len(slice_distances))[:-1],np.arange(len(slice_distances))[1:]])
                        slice_edges = dict(zip(range(len(slice_edges)),slice_edges))
                    
                        slice_mesh = TriangularMesh()
                        slice_mesh.points = slice_points
                        slice_mesh.point_data = slice_point_data
                        slice_mesh.edges = slice_edges
                        #world.add(slice_mesh,'primordium_'+str(primordium)+'_slice',linewidth=4-abs(primordium),colormap='curvature',intensity_range=(-3,3),display_colorbar=False)
                        world.add(slice_mesh,'primordium_'+str(primordium)+'_slice',linewidth=4-abs(primordium),colormap='Tmt_jet',intensity_range=(-3,3),display_colorbar=False)
                    
            figure.set_size_inches(18,10)     
            figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_DIIV_aligned_initiation_circle.png")
            
            slice_df = pd.DataFrame().from_dict(slice_data)
            # world.add(slice_df[slice_df['filename']==filename],'primordium_slice')
            # world['primordium_slice'].set_attribute('X_variable','r')
            # world['primordium_slice'].set_attribute('Y_variable','normalized_DIIV')
            # world['primordium_slice'].set_attribute('class_variable','primordium')
            # world['primordium_slice'].set_attribute('plot','line')
            # world['primordium_slice'].set_attribute('linewidth',3)
            # world['primordium_slice'].set_attribute('legend','top_left')
            # world['primordium_slice'].set_attribute('label_colormap',load_colormaps()['Tmt_jet'])
            # label_range = tuple([100.*(s-np.min(primordia_thetas.keys()))/(np.max(primordia_thetas.keys())-np.min(primordia_thetas.keys())) for s in (-3,3)])
            # world['primordium_slice'].set_attribute('label_range',label_range)
            figure.clf()
            for p in np.sort(primordia_thetas.keys()):
                file_slice_df = slice_df[slice_df['filename']==filename]
                slice_distances = file_slice_df[file_slice_df['primordium']==p]['r'].values
                slice_signal = file_slice_df[file_slice_df['primordium']==p]['normalized_DIIV'].values
                figure.gca().plot(slice_distances,slice_signal,linewidth=3,color=primordia_colors[p],label=p)
            
                max_points, min_points = local_extrema(slice_signal,abscissa=slice_distances,scale=5)
                
                if p>-2:
                    r_threshold = 0.8*clv3_radius if p<=1 else clv3_radius
                    r_ceiling = 2.*clv3_radius if p<=1 else 2.*r_max
                else:
                    r_threshold = 0.5*clv3_radius
                    r_ceiling = clv3_radius
                if ((min_points[:,0]>r_threshold)&(min_points[:,0]<r_ceiling)).sum()>0:
                    min_points = min_points[min_points[:,0]>r_threshold]
                    min_points = min_points[np.argsort(min_points[:,0])]
                    primordia_distances[p] = min_points[0,0]
                    figure.gca().scatter(min_points[0,0],min_points[0,1],s=50,linewidth=0,c=primordia_colors[p])
                    figure.gca().annotate('P'+str(p),(min_points[0,0],min_points[0,1]+0.1),color=primordia_colors[p])
                
                    primordia_data['filename'] += [filename]
                    primordia_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
                    primordia_data['primordium'] += [p]
                    primordia_data['distance'] += [primordia_distances[p]]
                    primordia_data['theta'] += [primordia_thetas[p]]
                    primordium_aligned_theta =  meristem_orientation*((primordia_thetas[p] - theta_min + 180)%360 - 180)
                    primordia_data['aligned_theta'] +=  [primordium_aligned_theta]
                    primordia_data['normalized_DIIV'] += [min_points[0,1]]
                    primordia_data['circle_normalized_DIIV'] += [aligned_signal[primordium_aligned_theta+180]]
                elif p<0:
                    min_points = np.array([[0.8*clv3_radius,slice_signal[int(2*0.66*clv3_radius)]]])
                    primordia_distances[p] = min_points[0,0]
                    figure.gca().scatter(min_points[0,0],min_points[0,1],s=50,linewidth=0,c=primordia_colors[p])
                    figure.gca().annotate('P'+str(p),(min_points[0,0],min_points[0,1]+0.1),color=primordia_colors[p])
                    
                    primordia_data['filename'] += [filename]
                    primordia_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
                    primordia_data['primordium'] += [p]
                    primordia_data['distance'] += [primordia_distances[p]]
                    primordia_data['theta'] += [primordia_thetas[p]]
                    primordium_aligned_theta = meristem_orientation*((primordia_thetas[p] - theta_min + 180)%360 - 180)
                    primordia_data['aligned_theta'] +=  [primordium_aligned_theta]
                    primordia_data['normalized_DIIV'] += [min_points[0,1]]
                    primordia_data['circle_normalized_DIIV'] += [aligned_signal[primordium_aligned_theta+180]]
            figure.gca().plot([clv3_radius,clv3_radius],[0,1],linewidth=1,color="#c94389",alpha=0.5)
            plt.legend()
            plt.draw()
            figure.gca().set_xlim(0,80)
            figure.gca().set_ylim(*signal_ranges[signal])
            
            figure.set_size_inches(18,10)     
            figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_DIIV_extrema_primordia_profiles.png")


            if mesh_landscape:
                circle_thetas = np.linspace(-180,180,91)
                slice_distances = np.linspace(0,40,21)
                    
                rr, tt = np.meshgrid(slice_distances,circle_thetas)
                xx = clv3_center[0] + rr*np.cos(np.pi*tt/180.)
                yy = clv3_center[1] + rr*np.sin(np.pi*tt/180.)
                zz = np.zeros_like(xx)
                
                map_potential = np.array([nuclei_density_function(dict([(p,projected_positions[p])]),cell_radius=cell_radius,k=density_k)(xx,yy,zz) for p in data.index])
                map_potential = np.transpose(map_potential,(1,2,0))
                map_density = np.sum(map_potential,axis=2)
                map_membership = map_potential/map_density[...,np.newaxis]
                map_signal = np.sum(map_membership*signal_data[np.newaxis,:],axis=2)
                
                mesh_points = np.transpose([np.concatenate(xx.transpose()),np.concatenate(yy.transpose()),inhibition_scale*np.concatenate(map_signal.transpose())])
                mesh_points = mesh_points[xx.shape[0]-1:]
                mesh_points = array_dict(dict(zip(range(len(mesh_points)),mesh_points)))
                
                mesh_point_data = np.concatenate(map_signal.transpose())[xx.shape[0]-1:]
                mesh_point_data = array_dict(dict(zip(range(len(mesh_points)),mesh_point_data)))
                
                #mesh_density = np.concatenate(epidermis_model_density.transpose())[T.shape[0]-1:]
                #mesh_density = array_dict(dict(zip(range(len(mesh_points)),mesh_density)))
                
                mesh_triangles = [[0,i+1,(i+1)%xx.shape[0]+1] for i in xrange(xx.shape[0])]
                for r in xrange(yy.shape[1]-2):
                    mesh_triangles += [[r*xx.shape[0] + i+1,r*xx.shape[0] + (i+1)%xx.shape[0]+1, (r+1)*xx.shape[0] + i+1] for i in xrange(xx.shape[0])]
                    mesh_triangles += [[r*xx.shape[0] + (i+1)%xx.shape[0]+1, (r+1)*xx.shape[0] + i+1, (r+1)*xx.shape[0] + (i+1)%xx.shape[0]+1] for i in xrange(xx.shape[0])]
                
                #mesh_triangle_density = mesh_density.values(mesh_triangles).max(axis=1)
                #mesh_triangles = np.array(mesh_triangles)[mesh_triangle_density>0.5]
                mesh_triangles = array_dict(dict(zip(range(len(mesh_triangles)),mesh_triangles)))
                
                map_mesh = TriangularMesh()
                map_mesh.points = mesh_points
                map_mesh.point_data = mesh_point_data
                map_mesh.triangles = mesh_triangles
                world.add(map_mesh,'inhibition_map',intensity_range=signal_ranges[signal],colormap=signal_colors[signal],alpha=0.2,display_colorbar=False)
                
                viewer.save_screenshot(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_inhibition_landscape.png")

            world['topomesh_data'].set_attribute('plot','scatter')
            world['topomesh_data'].set_attribute('plot','map')
                
        elif signal in ['Auxin']:
            map_figure = plt.figure(1)
            map_figure.clf()
            X = data['model_coords_x'].values
            Y = data['model_coords_y'].values
            data_range = [[-200,200],[-200,200]]
            xx, yy, zz = map_plot(map_figure,X,Y,data['Normalized_'+signal].values,XY_range=data_range,smooth_factor=1,n_points=100)
            #xx, yy, zz = map_plot(map_figure,X,Y,data[signal].values,XY_range=data_range,smooth_factor=1,n_points=100)

            zz[np.isnan(zz)] = 0
            
            from vplants.morpheme.vt_exec.linearfilter import linearfilter
            from vplants.morpheme.vt_exec.regionalext import regionalext
            from vplants.morpheme.vt_exec.connexe import connexe
            from openalea.image.spatial_image import SpatialImage
        
            signal_map = 1+zz
            map_image = SpatialImage(255.*signal_map[:,:,np.newaxis],resolution=[1,1,1]).astype(np.uint16)
            map_image = linearfilter(map_image,param_str_2="-smoothing -sigma 1.0")
            
            h_min = 2
            h_max = 5
            local_max = regionalext(map_image,param_str_2="-maxima -binary -connectivity 26 -h 100 -hmin "+str(h_min)+" -hmax "+str(h_max))

            map_figure.clf()
            map_figure.gca().pcolormesh(xx,yy,local_max[:,:,0],cmap='inferno',antialiased=True,shading='gouraud')
            plot_meristem_model(map_figure,aligned_meristem_model,r_max=r_max)
            plt.draw()
            #raw_input()

            h_min = 2
            h_max = 2
            #max_regions = nd.label(local_max>50)[0]
            max_regions = connexe(local_max, param_str_2="-low-threshold "+str(h_min)+" -high-threshold "+str(h_max)+" -labels -connectivity 26")
            #max_regions = connexe(local_max, param_str_2="-low-threshold "+str(h_max)+" -high-threshold "+str(h_min)+" -labels -connectivity 26")
            # print len(np.unique(max_regions))-1
        
            max_regions = max_regions[:,:,0]
            local_max = local_max[:,:,0]
            max_points = np.array(nd.measurements.maximum_position(zz,max_regions,index=np.unique(max_regions)[1:])).astype(int)
            if len(max_points>0):
                #max_points = np.array([p for p in max_points if model_density_map[tuple(p)]>0.5])
                max_points = np.transpose([xx[0,max_points[:,0]],yy[max_points[:,1],0]])
            auxin_maxima = deepcopy(max_points)[:,::-1]
            
            map_image = map_image[:,:,0]
            
            map_figure.gca().scatter(max_points[:,1],max_points[:,0],s=200.0,c='w',alpha=0.9)
            plot_meristem_model(map_figure,aligned_meristem_model,r_max=r_max)
            
            plt.figure(world['topomesh_data']['figure']).gca().scatter(max_points[:,1],max_points[:,0],s=800.0,c='#64ec4e',alpha=0.9)
        
            signal_map = 1.-zz
            map_image = SpatialImage(255.*signal_map[:,:,np.newaxis],resolution=[1,1,1]).astype(np.uint16)
            map_image = linearfilter(map_image,param_str_2="-smoothing -sigma 1.0")
            
            h_min = 40
            h_max = 80
            
            h_min = 2
            h_max = 5
            local_max = regionalext(map_image,param_str_2="-maxima -binary -connectivity 26 -h 100 -hmin "+str(h_min)+" -hmax "+str(h_max))

            map_figure.clf()
            map_figure.gca().pcolormesh(xx,yy,local_max[:,:,0],cmap='inferno',antialiased=True,shading='gouraud')
            plot_meristem_model(map_figure,aligned_meristem_model,r_max=r_max)
            plt.draw()
            #raw_input()

            h_min = 40
            h_max = 40
            
            h_min = 2
            h_max = 2
            #max_regions = nd.label(local_max>50)[0]
            max_regions = connexe(local_max, param_str_2="-low-threshold "+str(h_min)+" -high-threshold "+str(h_max)+" -labels -connectivity 26")
            #max_regions = connexe(local_max, param_str_2="-low-threshold "+str(h_max)+" -high-threshold "+str(h_min)+" -labels -connectivity 26")
            # print len(np.unique(max_regions))-1
        
            max_regions = max_regions[:,:,0]
            local_max = local_max[:,:,0]
            max_points = np.array(nd.measurements.maximum_position(-zz,max_regions,index=np.unique(max_regions)[1:])).astype(int)
            if len(max_points>0):
                #max_points = np.array([p for p in max_points if model_density_map[tuple(p)]>0.5])
                max_points = np.transpose([xx[0,max_points[:,0]],yy[max_points[:,1],0]])
            auxin_minima = deepcopy(max_points)[:,::-1]
            
            map_image = map_image[:,:,0]
            
            map_figure.gca().scatter(max_points[:,1],max_points[:,0],s=200.0,c='w',alpha=0.9)
            plot_meristem_model(map_figure,aligned_meristem_model,r_max=r_max)
            
            plt.figure(world['topomesh_data']['figure']).gca().scatter(max_points[:,1],max_points[:,0],s=800.0,c='#6b42d4',alpha=0.9)
        
        elif signal in ['mean_curvature']:
            X = data['model_coords_x'].values
            Y = data['model_coords_y'].values
            
            data['clv3_distance'] = np.linalg.norm(np.transpose([X,Y]) - clv3_center[np.newaxis,:],axis=1)
                        
            p = np.polyfit(np.power(data[data['clv3_distance']<40]['clv3_distance'],2),data[data['clv3_distance']<40]['model_coords_z'],1)
            dome_curvature = -2*p[0]/np.power(1+2*p[0]*5,1.5)
            
            #print filename, ' <--> ',dome_curvature
            #raw_input()
            
        if draw_maps:
            plt.figure(world['topomesh_data']['figure']).gca().scatter(clv3_center[0],clv3_center[1],s=800.0,c='#92278f',alpha=0.9)  
            c = patch.Circle(xy=clv3_center,radius=clv3_radius,ec="#c94389",fc='None',lw=5,ls='dashed',alpha=1)
            plt.figure(world['topomesh_data']['figure']).gca().add_artist(c)
            
            for primordium in primordia_thetas.keys():
                if primordia_distances.has_key(primordium):
                    primordium_theta = np.pi*primordia_thetas[primordium]/180.
                    primordium_distance = primordia_distances[primordium]
                    primordium_center = clv3_center + primordium_distance*np.array([np.cos(primordium_theta),np.sin(primordium_theta)])
                    plt.figure(world['topomesh_data']['figure']).gca().plot([clv3_center[0],primordium_center[0]],[clv3_center[1],primordium_center[1]],linewidth=1,color=primordia_colors[primordium],alpha=0.5)
                    plt.figure(world['topomesh_data']['figure']).gca().scatter([primordium_center[0]],[primordium_center[1]],s=800,c=primordia_colors[primordium])
                    plt.figure(world['topomesh_data']['figure']).gca().annotate('P'+str(primordium),primordium_center+np.array([2,0]),color=primordia_colors[primordium])
            
            plot_meristem_model(plt.figure(world['topomesh_data']['figure']),aligned_meristem_model,r_max=r_max)
            plot_meristem_model(plt.figure(world['topomesh_data']['figure']),reference_model,r_max=r_max,color='g',linewidth=1,alpha=0.05)
            figure.set_size_inches(2*r_max/7.75,2*r_max/7.75)
            figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_L1_"+signal+"_map.png")
        


    auxin_maxima_distances = np.linalg.norm(auxin_maxima - clv3_center[np.newaxis,:],axis=1)
    auxin_maxima = auxin_maxima[np.argsort(auxin_maxima_distances)]
    auxin_maxima_distances = auxin_maxima_distances[np.argsort(auxin_maxima_distances)]
    auxin_maxima_angles = 180.*np.sign((auxin_maxima - clv3_center[np.newaxis,:])[:,1])*np.arccos((auxin_maxima - clv3_center[np.newaxis,:])[:,0] / auxin_maxima_distances)/np.pi

    for m, r, t in zip(auxin_maxima,auxin_maxima_distances,auxin_maxima_angles):
        if r<r_max:
            auxin_extrema['filename'] += [filename]
            auxin_extrema['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
            auxin_extrema['extremum_type'] += ['max']
            auxin_extrema['extremum_x'] += [m[0]]
            auxin_extrema['extremum_y'] += [m[1]]
            auxin_extrema['extremum_r'] += [r]
            auxin_extrema['extremum_surface_distance'] += [np.arcsin(dome_curvature*r)/dome_curvature]
            auxin_extrema['extremum_theta'] += [t]
            auxin_extrema['auxin'] += [zz[vq(np.array([m[1]]),yy[:,0])[0][0],vq(np.array([m[0]]),xx[0])[0][0]]]

    auxin_minima_distances = np.linalg.norm(auxin_minima - clv3_center[np.newaxis,:],axis=1)
    auxin_minima = auxin_minima[np.argsort(auxin_minima_distances)]
    auxin_minima_distances = auxin_minima_distances[np.argsort(auxin_minima_distances)]
    auxin_minima_angles = 180.*np.sign((auxin_minima - clv3_center[np.newaxis,:])[:,1])*np.arccos((auxin_minima - clv3_center[np.newaxis,:])[:,0] / auxin_minima_distances)/np.pi

    for m, r, t in zip(auxin_minima,auxin_minima_distances,auxin_minima_angles):
        if (r<r_max) and (r>auxin_maxima_distances.min()):
            auxin_extrema['filename'] += [filename]
            auxin_extrema['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
            auxin_extrema['extremum_type'] += ['min']
            auxin_extrema['extremum_x'] += [m[0]-clv3_center[0]]
            auxin_extrema['extremum_y'] += [m[1]-clv3_center[1]]
            auxin_extrema['extremum_r'] += [r]
            auxin_extrema['extremum_surface_distance'] += [np.arcsin(dome_curvature*r)/dome_curvature]
            auxin_extrema['extremum_theta'] += [t]
            auxin_extrema['auxin'] += [zz[vq(np.array([m[1]]),yy[:,0])[0][0],vq(np.array([m[0]]),xx[0])[0][0]]]

    meristem_data['filename'] += [filename]
    meristem_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD']
    r_min = auxin_minima_distances[auxin_minima_distances>auxin_maxima_distances.min()].min()
    meristem_data['auxin_minimum_r'] += [r_min]
    meristem_data['auxin_minimum_surface_distance'] += [np.arcsin(dome_curvature*r_min)/dome_curvature]
    meristem_data['clv3_radius'] += [clv3_radius]
    meristem_data['clv3_center_error'] += [np.linalg.norm(clv3_center)]
    meristem_data['mean_curvature'] += [dome_curvature]
    
import pandas as pd
from openalea.cellcomplex.property_topomesh.utils.pandas_tools import dataframe_relative_range

world.clear()
primordia_df = pd.DataFrame().from_dict(primordia_data)

for filename in filenames:
    file_primordia_df = primordia_df[primordia_df['filename']==filename]
    world.add(file_primordia_df,'primordia')
    world['primordia'].set_attribute('figure',1)
    world['primordia'].set_attribute('X_variable','aligned_theta')
    world['primordia'].set_attribute('X_range',dataframe_relative_range(file_primordia_df,'aligned_theta',(-180,180)))
    world['primordia'].set_attribute('Y_variable','circle_normalized_DIIV')
    world['primordia'].set_attribute('Y_range',dataframe_relative_range(file_primordia_df,'circle_normalized_DIIV',(0,1)))
    world['primordia'].set_attribute('label_variable','primordium')
    world['primordia'].set_attribute('label_range',dataframe_relative_range(file_primordia_df,'primordium',(-3,3)))
    
    
    world['primordia'].set_attribute('n_points',10)
    #world['primordia'].set_attribute('legend','top_left')
    world['primordia'].set_attribute('label_colormap',load_colormaps()['Tmt_jet'])
    for primordium in file_primordia_df['primordium'].values:
        primordium_theta = (primordium*golden_angle + 180)%360 - 180
        plt.figure(world['primordia']['figure']).gca().plot([primordium_theta,primordium_theta],[0,100],linewidth=1,color=primordia_colors[primordium],alpha=1)
        plt.figure(world['primordia']['figure']).gca().annotate('P'+str(primordium),(primordium_theta,5),color=primordia_colors[primordium])
    print filename
    raw_input()

primordia_df['log_distance'] = np.log(primordia_df['distance'])

world.add(primordia_df,'primordia')

world.add(primordia_df[(primordia_df['primordium']>=0-1) & (primordia_df['primordium']<5)],'primordia')


auxin_extrema['rank'] = []

map_figure.clf()
for filename in filenames:
    file_extrema = np.array(auxin_extrema['filename']) == filename


    extrema_x = np.array(auxin_extrema['extremum_x'])[file_extrema]
    extrema_y = np.array(auxin_extrema['extremum_y'])[file_extrema]
    
    extrema_theta = np.array(auxin_extrema['extremum_theta'])[file_extrema]
    extrema_r = np.array(auxin_extrema['extremum_r'])[file_extrema]
    extrema_type = np.array(auxin_extrema['extremum_type'])[file_extrema]
    
    extrema_rank = np.nan*np.ones_like(extrema_theta,int)
    
    map_figure.clf()
    map_figure.gca().scatter([0],[0],color="#c94389",edgecolor='k',s=200,alpha=0.5)
    map_figure.gca().scatter(extrema_x[extrema_type=='max'],extrema_y[extrema_type=='max'],color='#64ec4e',edgecolor='k',s=200,alpha=0.5)
    map_figure.gca().scatter(extrema_x[extrema_type=='min'],extrema_y[extrema_type=='min'],color='#6b42d4',edgecolor='k',s=200,alpha=0.5)
    map_figure.gca().set_xlim(-r_max,r_max)
    map_figure.gca().set_ylim(-r_max,r_max)
    plt.draw()
    raw_input()
    
    angle_threshold = 10.
    for primordium in [-2,-1,0,1,2,3,4,5,6]:
        primordium_angular_gap = np.abs((extrema_theta-primordium*golden_angle + 180)%360-180)
        
        p_x = extrema_x[(primordium_angular_gap<angle_threshold)]
        p_y = extrema_y[(primordium_angular_gap<angle_threshold)]
        p_type = extrema_type[(primordium_angular_gap<angle_threshold)]
        
        map_figure.clf()
        map_figure.gca().scatter([0],[0],color="#c94389",edgecolor='k',s=200,alpha=0.5)
        map_figure.gca().scatter(extrema_x[extrema_type=='max'],extrema_y[extrema_type=='max'],color='#64ec4e',edgecolor='k',s=200,alpha=0.5)
        map_figure.gca().scatter(extrema_x[extrema_type=='min'],extrema_y[extrema_type=='min'],color='#6b42d4',edgecolor='k',s=200,alpha=0.5)
        map_figure.gca().set_xlim(-r_max,r_max)
        map_figure.gca().set_ylim(-r_max,r_max)
        map_figure.gca().scatter(p_x[p_type=='max'],p_y[p_type=='max'],color='#64ec4e',edgecolor='k',s=200,alpha=1)
        map_figure.gca().scatter(p_x[p_type=='min'],p_y[p_type=='min'],color='#6b42d4',edgecolor='k',s=200,alpha=1)
        plt.draw()
        raw_input()
        
        extrema_rank[(primordium_angular_gap<angle_threshold) & (extrema_type == 'min')] = primordium
    print extrema_rank
    
    auxin_extrema['rank'] += list(extrema_rank)

world.clear()
extrema_data = pd.DataFrame().from_dict(auxin_extrema)
world.add(extrema_data,'auxin_extrema')

meristem_data['clv3_surface_radius'] = [np.arcsin(c*r)/c for c,r in zip(meristem_data['mean_curvature'],meristem_data['clv3_radius'])]

world.clear()
meristem_df = pd.DataFrame().from_dict(meristem_data)
world.add(meristem_df,'meristems')


world.clear()
circle_df = pd.DataFrame().from_dict(circle_data)

circle_data['aligned_theta'] = []
circle_data['growth_condition'] = []
circle_data['scaled_DIIV'] = []
for filename in filenames:
    circle_thetas = circle_df[circle_df['filename']==filename]['theta'].values
    circle_signal = circle_df[circle_df['filename']==filename]['DIIV'].values
    theta_min = circle_thetas[np.argmin(circle_signal)]
    circle_data['aligned_theta'] += list((circle_thetas-theta_min+180)%360 - 180)
    circle_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for t in circle_thetas]
    circle_data['scaled_DIIV'] += list((circle_signal-circle_signal.min())/(circle_signal.max()-circle_signal.min()))
    

world.clear()
circle_df = pd.DataFrame().from_dict(circle_data)
world.add(circle_df,'initiation_circle')

circle_df.to_csv("/Users/gcerutti/Desktop/initiation_circle.csv")

for primordium in primordia_colors.keys():
    primordium_theta = (primordium*golden_angle + 180)%360 - 180
    figure = plt.figure(0)
    figure.gca().plot([primordium_theta,primordium_theta],[0,1],linewidth=1,color=primordia_colors[primordium],alpha=1./(abs(primordium)+1))
    figure.gca().annotate('P'+str(primordium),(primordium_theta,0.22),color=primordia_colors[primordium])
     

for filename in filenames:
    world.add(circle_df[circle_df['filename']==filename],'initiation_circle')
    world['initiation_circle'].set_attribute('X_variable','theta')
    world['initiation_circle'].set_attribute('Y_variable','DIIV')
    world['initiation_circle'].set_attribute('plot','line')
    
    circle_thetas = circle_df[circle_df['filename']==filename]['theta'].values
    theta_min = circle_thetas[np.argmin(circle_df[circle_df['filename']==filename]['DIIV'].values)]
    
    for primordium in primordia_colors.keys():
        primordium_theta = (theta_min + primordium*golden_angle + 180)%360 - 180
        print primordium,primordium_theta
        figure = plt.figure(0)
        figure.gca().plot([primordium_theta,primordium_theta],[0,1],linewidth=1,color=primordia_colors[primordium],alpha=1./(abs(primordium)+1))
        figure.gca().annotate('P'+str(primordium),(primordium_theta,np.min(circle_df[circle_df['filename']==filename]['DIIV'].values)),color=primordia_colors[primordium])
    figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_DIIV_initiation_circle.png")
        


world.clear()
slice_df = pd.DataFrame().from_dict(slice_data)

world.add(slice_df,'primordioum_slice')

slice_df['actual_primordium'] = -slice_df['primordium']
for filename in filenames:
    world.add(slice_df[slice_df['filename']==filename],'primordium_slice')
    world['primordium_slice'].set_attribute('X_variable','r')
    world['primordium_slice'].set_attribute('Y_variable','DIIV')
    world['primordium_slice'].set_attribute('class_variable','primordium')
    world['primordium_slice'].set_attribute('plot','line')
    world['primordium_slice'].set_attribute('linewidth',3)
    world['primordium_slice'].set_attribute('legend','top_left')
    world['primordium_slice'].set_attribute('label_colormap',load_colormaps()['Tmt_jet'])

    clv3_radius =float(meristem_df[meristem_df.index==filename]['clv3_radius'])
    figure = plt.figure(0)
    figure.gca().plot([clv3_radius,clv3_radius],[0,1],linewidth=1,color="#c94389",alpha=0.5)
    plt.draw()
    figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_DIIV_primordia_profiles.png")


all_data = pd.DataFrame().from_dict(all_samples_data)
world.add(all_data,'all_data')




epidermis_cells = coords.keys()[cell_layer.values() == 1]
    
clv3_map, model_density, cell_density, T, R  = meristem_2d_cylindrical_map(coords,meristem_model,coords.keys(),array_dict(identity_values))



map_figure = plt.figure(0)
map_figure.clf()
map_figure.patch.set_facecolor('white')
ax = plt.subplot(111, polar=True)
            
draw_signal_map(map_figure,clv3_map,T,R,cell_density,colormap='Purples', n_levels=20, ratio_min=0, ratio_max=1)

from vplants.meshing.cute_plot import simple_plot

figure = plt.figure(1)
figure.clf()
simple_plot(figure,coords.values(epidermis_cells)[:,1],array_dict(identity_values).values(epidermis_cells)-array_dict(identity_values).values().min(),np.array([0.2,0.7,0.1]),xlabel="Distance",ylabel="CLV3",linked=False,marker_size=20,alpha=0.2)
figure.gca().set_xlim(0,60)
figure.gca().set_xticklabels(figure.gca().get_xticks())



