import numpy as np
import pandas as pd

from vplants.sam4dmaps.nuclei_detection import detect_nuclei, compute_fluorescence_ratios
from vplants.sam4dmaps.nuclei_segmentation import nuclei_active_region_segmentation, nuclei_positions_from_segmented_image

import vplants.sam4dmaps.parametric_shape
reload(vplants.sam4dmaps.parametric_shape)

import vplants.sam4dmaps.sam_model
reload(vplants.sam4dmaps.sam_model)
from vplants.sam4dmaps.sam_model import estimate_meristem_model, read_meristem_model


from openalea.container import array_dict
from openalea.mesh import TriangularMesh
from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh
from openalea.mesh.utils.pandas_tools import topomesh_to_dataframe

from openalea.image.serial.all import imread, imsave

import pickle
from copy import deepcopy

from vplants.sam4dmaps.sam_model import reference_meristem_model

import vplants.sam4dmaps.sam_model_registration
reload(vplants.sam4dmaps.sam_model_registration)
from vplants.sam4dmaps.sam_model_registration import meristem_model_cylindrical_coordinates, meristem_model_organ_gap
from vplants.sam4dmaps.sam_map_construction import meristem_2d_cylindrical_map, draw_signal_map

import vplants.sam4dmaps.sam_model_tools
reload(vplants.sam4dmaps.sam_model_tools)
from vplants.sam4dmaps.sam_model_tools import plot_meristem_model, meristem_model_topomesh

import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib import cm
from matplotlib.colors import Normalize
from openalea.oalab.colormap.colormap_def import load_colormaps

from scipy.cluster.vq import vq

sam_orientations = {}
sam_orientations[1] = 1
sam_orientations[2] = 1
sam_orientations[3] = -1
sam_orientations[5] = 1
sam_orientations[6] = -1
sam_orientations[7] = 1
sam_orientations[9] = 1
sam_orientations[10] = -1
sam_orientations[13] = -1
sam_orientations[14] = 1
sam_orientations[15] = 1
sam_orientations[16] = 1
sam_orientations[18] = 1


complete_circle_data = {}
for field in ['filename','sam_id','hour_time','growth_condition','theta','aligned_theta','DIIV','normalized_DIIV','altitude','CLV3_radius']:
    complete_circle_data[field] = []
    
complete_radial_data = {}
for field in ['filename','sam_id','hour_time','growth_condition','primordium','aligned_theta','radial_distance','DIIV','normalized_DIIV','altitude','CLV3_radius']:
    complete_radial_data[field] = []
    
primordia_angle_data = {}
for field in ['filename','sam_id','hour_time','growth_condition','primordium','aligned_theta','DIIV','normalized_DIIV','altitude']:
    primordia_angle_data[field] = []

r_max = 80

# for sam_id in [2,3,4,10,12,15,16,18,20]:
# # #for sam_id in [2]:
#     filenames = [] 
#     for time in [0,6]:
#         filenames += ["qDII-CLV3-SD_11c_160628_sam"+str(sam_id).zfill(2)+"_t"+str(time).zfill(2)]
        

#for sam_id in [1, 3, 5, 6, 7, 9, 10, 13, 14, 15, 16, 18]:
for sam_id in [1]:
    filenames = [] 
    #for time in [0,6,10]:
    for time in [0]:
        filenames += ["qDII-CLV3-LD_11c_160906_sam"+str(sam_id).zfill(2)+"_no_organs"+"_t"+str(time).zfill(2)]
        #filenames += ["qDII-CLV3-LD_11c_160906_sam"+str(sam_id).zfill(2)+"_t"+str(time).zfill(2)]
    
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
    
    reference_images = {}
    signal_images = {}
    identity_images = {}
    
    
    for i_file,filename in enumerate(filenames):
        signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
        #signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_DR5.inr.gz"
        signal_img = imread(signal_file)
        signal_images[filename]=signal_img
        
        identity_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+identity_name+".inr.gz"
        identity_img = imread(identity_file)
        identity_images[filename]=identity_img
        
        reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+reference_name+".inr.gz"
        reference_img = imread(reference_file)
        reference_images[filename]=reference_img
        
        size = np.array(reference_img.shape)
        resolution = microscope_orientation*np.array(reference_img.resolution)
        
        figure = plt.figure(1)
        figure.clf()
        xx, yy = np.mgrid[0:size[0]*resolution[0]:resolution[0],0:size[1]*resolution[1]:resolution[1]]
        depth = np.power(np.tile(np.tile(np.arange(size[2]),(size[1],1)),(size[0],1,1))/float(size[2]),2)
        #depth = np.zeros_like(reference_img).astype(float)
        extent = yy.min(),yy.max(),xx.min(),xx.max()
        
        ref = cm.ScalarMappable(cmap='gray',norm=Normalize(vmin=4000,vmax=np.percentile(reference_img,98))).to_rgba(np.transpose((reference_img*(1-depth)).max(axis=2))[:,::-1])
        sig = cm.ScalarMappable(cmap='viridis',norm=Normalize(vmin=4000,vmax=12000)).to_rgba(np.transpose((signal_img*(1-depth)).max(axis=2))[:,::-1])
        ide = cm.ScalarMappable(cmap='inferno',norm=Normalize(vmin=4000,vmax=0.4*identity_img.max())).to_rgba(np.transpose((identity_img*(1-depth)).max(axis=2))[:,::-1])
        blend = ref*(0.1 + 0.9*sig)
        figure.gca().imshow(blend,extent=extent)
        
        #figure.gca().imshow(np.transpose((reference_img*(1-depth)).max(axis=2))[:,::-1],extent=extent,cmap='gray',alpha=1)
        #figure.gca().imshow(np.transpose((signal_img*(1-depth)).max(axis=2))[:,::-1],extent=extent,cmap='viridis',vmin=4000,vmax=12000,alpha=0.5)
        #figure.gca().pcolormesh(xx,yy,reference_img.max(axis=2),cmap='gray',alpha=0.5)
        #figure.gca().pcolormesh(xx,yy,(signal_img*(1-depth)).max(axis=2),cmap='viridis',vmin=4000,vmax=10000,alpha=0.6,linewidth=0)
        #figure.gca().pcolormesh(xx,yy,identity_img.max(axis=2),cmap='inferno',alpha=0.1,linewidth=0)
        #figure.gca().axis('equal')
        
        #world.add(identity_img,'identity_image',resolution=resolution,colormap='Purples')
        #world.add(reference_img,'nuclei_image',resolution=resolution,colormap='Blues')
        #world.add(signal_img,'signal_image',resolution=resolution,colormap=signal_colors[signal_name])
        
        #topomesh_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_nuclei_signal_curvature_topomesh.ply"
        topomesh_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_nuclei_signal_curvature_topomesh.ply"
        topomesh = read_ply_property_topomesh(topomesh_file)
        positions = topomesh.wisp_property('barycenter',0)
        nuclei_signals[filename] = topomesh
        
        meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
        try:
            meristem_model = read_meristem_model(meristem_model_file)
        except:
            if i_file == 0:
                previous_parameters = None
            else:
                previous_parameters =  meristem_models[filenames[i_file-1]].parameters
            meristem_model,_,_ = estimate_meristem_model(positions,size,resolution,microscope_orientation=microscope_orientation,meristem_model_parameters=previous_parameters)
            pickle.dump(meristem_model.parameters,open(meristem_model_file,'wb'))
        meristem_models[filename] = meristem_model
        
        figure.gca().axis('off')        
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_DIIV_image.jpg")
        
        plot_meristem_model(figure,meristem_model,alpha=0,color='g',r_max=r_max,center=False)
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_DIIV_image_model.jpg")
        
        blend = ref*np.max([(0.1 + 0.9*sig), ide],axis=0)
        figure.gca().imshow(blend,extent=extent)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_image_model.jpg")
        
        
        
    
    signal_ranges = {}
    signal_ranges['DIIV'] = (0.0,1.0)
    #signal_ranges['DIIV'] = (0.0,0.4)
    signal_ranges['Auxin'] = (0.2,1.0)
    signal_ranges['CLV3'] = (0.0,1.0)
    signal_ranges['mean_curvature'] = (-0.05,0.05)
    
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
    
    reference_dome_apex = np.zeros(3)
    n_organs = 8
    reference_model = reference_meristem_model(reference_dome_apex,n_primordia=n_organs,developmental_time=0)
    orientation = reference_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi  
    
    previous_offset = 0
    previous_reference_model = reference_model
        
        
    world.clear()
    
    sup_figure = plt.figure(9)
    sup_figure.clf()
    
    circle_data = {}
    for field in ['filename','hour_time','growth_condition','theta','aligned_theta','DIIV','normalized_DIIV','altitude']:
        circle_data[field] = []
    
    clv3_radii = {}
    file_data = {}
    file_primordia_angles = {}
    file_aligned_primordia_angles = {}
    
    file_centered_positions = {}
    file_aligned_positions = {}
    
    #max_clv3 = 12000000
    max_clv3 = 8000000
    
    previous_theta = 0
    meristem_orientation = sam_orientations.get(sam_id,None)
    previous_orientation = 1
    model_orientations = {}
    
    for i_file, filename in enumerate(filenames):
        
        topomesh = nuclei_signals[filename]
        positions = topomesh.wisp_property('barycenter',0)
        
        meristem_model = meristem_models[filename]
        
        #world.add(topomesh,filename[-3:]+"_nuclei")
        #world[filename[-3:]+"_nuclei"].set_attribute('property_name_0','layer')
        #world.add(meristem_model,filename[-3:]+"_meristem_model",_repr_vtk_=meristem_model.drawing_function,colormap='Greens',alpha=0.1,z_slice=(95,100),display_colorbar=False)
    
        # if i_file == 0:
        #     organ_gap = meristem_model_organ_gap(reference_model,meristem_models[filename],same_individual=False)
        # else:
        #     hour_gap = float(int(filename[-2:]) - int(filenames[i_file-1][-2:]))
        #     organ_gap = meristem_model_organ_gap(previous_reference_model,meristem_models[filename],same_individual=True,hour_gap=hour_gap)
        # organ_gap += previous_offset
        # previous_offset = organ_gap
        # previous_reference_model = meristem_models[filename]
        if i_file == 0:
            previous_orientation = meristem_models[filename].parameters['orientation']
        model_orientation = meristem_models[filename].parameters['orientation']*previous_orientation
        model_orientations[filename] = model_orientation
        
        # coords, aligned_meristem_model = meristem_model_cylindrical_coordinates(meristem_model,positions,-organ_gap,orientation)
        
        data = topomesh_to_dataframe(topomesh,0)
        # data['model_coords_theta'] = coords.values(data.index)[:,0]
        # data['model_coords_r'] = coords.values(data.index)[:,1]
        # data['model_coords_z'] = coords.values(data.index)[:,2]
        # data['model_coords_x'] = data['model_coords_r'] * np.cos(np.pi*data['model_coords_theta']/180.)
        # data['model_coords_y'] = data['model_coords_r'] * np.sin(np.pi*data['model_coords_theta']/180.)
        data['Auxin'] = 1 - data['DIIV']
        data = data[data['layer']==1]
        for signal in ['CLV3']:
            #signal_data  = data[signal][data['model_coords_r']<2.*r_max/3.]
            signal_data  = data[signal]
            data['Normalized_'+signal] = data[signal]/signal_data.mean()
            clv3_threshold = 1.2
            #data['Normalized_'+signal] = data[signal]/max_clv3
    
        for signal in ['DIIV','Auxin']:
            signal_data  = data[signal]
            data['Normalized_'+signal] = 0.5 + 0.2*(data[signal]-signal_data.mean())/(signal_data.std()) 
            
        # for signal in ['mean_curvature']:
        #     signal_data  = data[signal]
        #     data['Normalized_'+signal] = data[signal]
    
        world.add(data,filename[-3:]+'_data')
        fig_number = 2*int(filename[-2:])/3+1
        world[filename[-3:]+'_data'].set_attribute('figure',fig_number)
        world[filename[-3:]+'_data'].set_attribute('X_variable','center_x')
        world[filename[-3:]+'_data'].set_attribute('Y_variable','center_y')
        
        signal = 'DIIV'
        world[filename[-3:]+'_data'].set_attribute('label_variable','Normalized_'+signal)
        #world[filename[-3:]+'_data'].set_attribute('label_variable',signal)
        world[filename[-3:]+'_data'].set_attribute('label_colormap',load_colormaps()[signal_colors[signal]])
        
        label_range = tuple([100.*(s-data['Normalized_'+signal].min())/(data['Normalized_'+signal].max()-data['Normalized_'+signal].min()) for s in signal_ranges[signal]])
        #label_range = tuple([100.*(s-data[signal].min())/(data[signal].max()-data[signal].min()) for s in signal_ranges[signal]])
        world[filename[-3:]+'_data'].set_attribute('label_range',label_range)
        
        #world[filename[-3:]+'_data'].set_attribute('legend','top_right')
        world[filename[-3:]+'_data'].set_attribute('markersize',200.0)
        world[filename[-3:]+'_data'].set_attribute('linewidth',0.0)
        world[filename[-3:]+'_data'].set_attribute('n_points',20.0)
        world[filename[-3:]+'_data'].set_attribute('smooth_factor',0.0)
        
        world[filename[-3:]+'_data'].set_attribute('plot','line')
        world[filename[-3:]+'_data'].set_attribute('plot','scatter')
        
        figure = plt.figure(fig_number)
        plot_meristem_model(figure,meristem_model,alpha=0,color='g',r_max=r_max,center=False)
        
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_nuclei_"+signal+".jpg")
        
        
        world[filename[-3:]+'_data'].set_attribute('plot','map')

        figure = plt.figure(fig_number)
        plot_meristem_model(figure,meristem_model,alpha=1,color='g',r_max=r_max,center=False)
        #plot_meristem_model(figure,aligned_meristem_model,alpha=1,color='g',r_max=r_max,center=False)
        
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_nuclei_"+signal+"_map.jpg")
        
        
        import vplants.sam4dmaps.signal_map_interpretation
        reload(vplants.sam4dmaps.signal_map_interpretation)
        from vplants.sam4dmaps.signal_map_interpretation import local_extrema, extract_clv3_circle, compute_circle_signal, compute_radial_signal, compute_local_2d_signal, aligned_circle_signal, circle_primordia_angles
                
        X = data['center_x'].values
        Y = data['center_y'].values
        
        #X = data['model_coords_x'].values
        #Y = data['model_coords_y'].values
        
        clv3 = data['Normalized_CLV3'].values
        dIIv = data['DIIV'].values
        normalized_dIIv = data['Normalized_DIIV'].values
        
        # circle_figure = plt.figure(7)
        # circle_figure.clf()
        # for t in [1.5,1.6,1.7,1.8,1.9,2.0]:
        #     clv3_center, clv3_radius = extract_clv3_circle(np.transpose([X,Y]),clv3,clv3_threshold=t)
        #     #clv3_center, clv3_radius = extract_clv3_circle(np.transpose([X,Y]),clv3,clv3_threshold=clv3_threshold)
        #     #clv3_radius = 24.
           
        #     circle_dIIv, circle_thetas = compute_circle_signal(np.transpose([X,Y]),clv3_center,clv3_radius,dIIv)
        
        #     circle_figure.gca().plot(circle_thetas,circle_dIIv,label=str(t))
        
        #     c = patch.Circle(xy=clv3_center,radius=clv3_radius,ec="#c94389",fc='None',lw=3,alpha=1)
        #     figure.gca().add_artist(c)
        #     figure.gca().annotate('t='+str(t),(clv3_center[0],clv3_center[1]+clv3_radius),color="#c94389")
        
        # figure.set_size_inches(12,12)
        # figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_circle_map.jpg")
        
        # signal = 'DIIV'
        # world[filename[-3:]+'_data'].set_attribute('label_variable','Normalized_'+signal)
        # #world[filename[-3:]+'_data'].set_attribute('label_variable',signal)
        # world[filename[-3:]+'_data'].set_attribute('label_colormap',load_colormaps()[signal_colors[signal]])
        
        # label_range = tuple([100.*(s-data['Normalized_'+signal].min())/(data['Normalized_'+signal].max()-data['Normalized_'+signal].min()) for s in signal_ranges[signal]])
        # #label_range = tuple([100.*(s-data[signal].min())/(data[signal].max()-data[signal].min()) for s in signal_ranges[signal]])
        # world[filename[-3:]+'_data'].set_attribute('label_range',label_range)
        
        
        clv3_center, clv3_radius = extract_clv3_circle(np.transpose([X,Y]),clv3,clv3_threshold=clv3_threshold)
        clv3_radii[filename[-3:]] = clv3_radius 
        
        if i_file == 0:
            file_centered_positions[filename[-3:]] = np.transpose([X,Y])-clv3_center
        else:
            previous_positions =  file_centered_positions[filenames[i_file-1][-3:]]
            theta_errors = []
            for theta in xrange(-180,180):
                centered_positions = np.transpose([X,Y])-clv3_center
                theta_matrix = np.array([[np.cos(theta*np.pi/180.),-np.sin(theta*np.pi/180.)],[np.sin(theta*np.pi/180.),np.cos(theta*np.pi/180.)]])
                centered_rotated_positions = np.einsum('...ij,...i->...j',theta_matrix,centered_positions)
                theta_error = np.mean(vq(centered_rotated_positions,previous_positions)[1])
                theta_error += np.power(theta*np.pi/180.,2)
                theta_errors += [theta_error]
            rotation_theta = np.arange(-180,180)[np.argmin(theta_errors)]
            print rotation_theta
            theta_matrix = np.array([[np.cos(rotation_theta*np.pi/180.),-np.sin(rotation_theta*np.pi/180.)],[np.sin(rotation_theta*np.pi/180.),np.cos(rotation_theta*np.pi/180.)]])
            centered_rotated_positions = np.einsum('...ij,...i->...j',theta_matrix,centered_positions)
            file_centered_positions[filename[-3:]] = centered_rotated_positions
               
            sup_figure.clf()
            sup_figure.gca().scatter(previous_positions[:,0],previous_positions[:,1],color='b')
            sup_figure.gca().scatter(centered_rotated_positions[:,0],centered_rotated_positions[:,1],color='r')
            #sup_figure.gca().plot(range(-180,180), theta_errors, color='k')
        
        c = patch.Circle(xy=clv3_center,radius=clv3_radius,ec="#c94389",fc='None',lw=3,alpha=1)
        figure.gca().add_artist(c)
        
        sup_figure.clf()
        xx, yy = np.mgrid[0:size[0]*resolution[0]:resolution[0],0:size[1]*resolution[1]:resolution[1]]
        depth = np.power(np.tile(np.tile(np.arange(size[2]),(size[1],1)),(size[0],1,1))/float(size[2]),2)
        #depth = np.zeros_like(reference_img).astype(float)
        extent = yy.min(),yy.max(),xx.min(),xx.max()
        ref = cm.ScalarMappable(cmap='gray',norm=Normalize(vmin=4000,vmax=0.33*reference_img.max())).to_rgba(np.transpose((reference_img*(1-depth)).max(axis=2))[:,::-1])
        ide = cm.ScalarMappable(cmap='inferno',norm=Normalize(vmin=4000,vmax=0.4*identity_img.max())).to_rgba(np.transpose((identity_img*(1-depth)).max(axis=2))[:,::-1])
        blend = ref*(0.5 + 0.5*ide)
        sup_figure.gca().imshow(blend,extent=extent)
        #sup_figure.gca().imshow(np.transpose((reference_img*(1-depth)).max(axis=2))[:,::-1],extent=extent,cmap='gray',alpha=1)
        #sup_figure.gca().imshow(np.transpose((identity_img*(1-depth)).max(axis=2))[:,::-1]/data['CLV3'].mean(),extent=extent,cmap='inferno',alpha=0.5,vmin=0,vmax=0.05)
        #figure.gca().imshow(np.transpose((signal_img*(1-depth)).max(axis=2))[:,::-1],extent=extent,cmap='viridis',vmin=4000,vmax=10000,alpha=0.5)
        #figure.gca().pcolormesh(xx,yy,reference_img.max(axis=2),cmap='gray',alpha=0.5)
        #figure.gca().pcolormesh(xx,yy,(signal_img*(1-depth)).max(axis=2),cmap='viridis',vmin=4000,vmax=10000,alpha=0.6,linewidth=0)
        #figure.gca().pcolormesh(xx,yy,identity_img.max(axis=2),cmap='inferno',alpha=0.1,linewidth=0)
        #figure.gca().axis('equal')
        
        sup_figure.gca().axis('off') 
        sup_figure.set_size_inches(12,12)
        sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_CLV3_image.jpg")
        
        plot_meristem_model(sup_figure,meristem_model,alpha=0,color='g',r_max=r_max,center=False)
        sup_figure.set_size_inches(12,12)
        sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_CLV3_image_model.jpg")
        
        c = patch.Circle(xy=clv3_center,radius=clv3_radius,ec="#c94389",fc='None',lw=3,alpha=1)
        sup_figure.gca().add_artist(c)
        sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_raw_CLV3_image_model_circle.jpg")
        
        inhibition_scale = 100.
        
        circle_dIIv, circle_thetas = compute_circle_signal(np.transpose([X,Y]),clv3_center,clv3_radius,dIIv)
        circle_normalized_dIIv, circle_thetas = compute_circle_signal(np.transpose([X,Y]),clv3_center,clv3_radius,normalized_dIIv)
        
        if meristem_orientation is None:
            max_points, min_points = local_extrema(circle_dIIv, abscissa=circle_thetas)
            primordium = 2
            primordium_theta = (circle_thetas[np.argmin(circle_dIIv)]+primordium*golden_angle + 180)%360 - 180
            extremum_points = np.concatenate([min_points,max_points])
            extremum_types = np.concatenate([['min' for p in min_points],['max' for p in max_points]])
            extremum_match = vq(np.array([primordium_theta]),extremum_points[:,0])
            extremum_theta = extremum_points[:,0][extremum_match[0][0]]
            extremum_type = extremum_types[extremum_match[0][0]]
            if extremum_type == 'min':
                meristem_orientation = -1
            else:
                meristem_orientation = 1
            print "Meristem Orientation : ",meristem_orientation
            #circle_thetas = meristem_orientation*circle_thetas
        circle_dIIv = circle_dIIv[::meristem_orientation]
        circle_normalized_dIIv = circle_normalized_dIIv[::meristem_orientation]
        
        #circle_figure = plt.figure(7)
        #circle_figure.clf()
        #circle_figure.gca().plot(circle_thetas,circle_normalized_dIIv)
        
    
        primordia_angles = circle_primordia_angles(circle_dIIv,circle_thetas,angular_error=0.25)
        
        if i_file == 0:
            file_primordia_angles[filename[-3:]] = primordia_angles
            theta_min = circle_thetas[np.argmin(circle_dIIv)] 
            previous_theta = theta_min
        else:
            gap_scores = []
            previous_angles = file_primordia_angles[filenames[i_file-1][-3:]]
            for gap in xrange(-1,3):
                gap_score = np.mean([abs(primordia_angles[p] - previous_angles[p-gap]) for p in primordia_angles.keys() if previous_angles.has_key(p-gap)])
                gap_scores += [gap_score]
                print "Gap ",gap,' : ',gap_score
            primordium_gap = np.arange(-1,3)[np.argmin(gap_scores)]
            primordia_angles = dict(zip([p-primordium_gap for p in primordia_angles.keys()],[primordia_angles[p] for p in primordia_angles.keys()]))
            file_primordia_angles[filename[-3:]] = primordia_angles
            
            theta_min = previous_theta
            # if primordia_angles.has_key(0):
            #     theta_min = primordia_angles[0]
            # else:
            #     theta_min = (circle_thetas[np.argmin(circle_dIIv)] + primordium_gap*golden_angle + 180)%360 - 180
            #theta_gap = np.mean([(primordia_angles[p] - previous_angles[p] + 180)%360 - 180 for p in primordia_angles.keys() if previous_angles.has_key(p)])
            #theta_gap = model_orientation*meristem_orientation*rotation_theta
            theta_gap = meristem_orientation*rotation_theta
            theta_min = (theta_min + theta_gap + 180)  % 360 -180
        
        aligned_thetas = ((circle_thetas-theta_min+180)%360 - 180)
        aligned_dIIv = circle_dIIv
        aligned_normalized_dIIv = circle_normalized_dIIv
        
        aligned_primordia_angles = dict(zip(primordia_angles.keys(),[(primordia_angles[p]-theta_min+180)%360 - 180 for p in primordia_angles.keys()]))
        file_aligned_primordia_angles[filename[-3:]] = aligned_primordia_angles
        
        theta_min = theta_min*np.pi/180.
        
        for primordium in primordia_angles.keys():
            if primordium in xrange(-3,6):
                #extremum_theta = model_orientation*meristem_orientation*primordia_angles[primordium]
                extremum_theta = meristem_orientation*primordia_angles[primordium]
                primordium_point = clv3_center+clv3_radius*np.array([np.cos(extremum_theta*np.pi/180.),np.sin(extremum_theta*np.pi/180.)])
                c = patch.Circle(xy=primordium_point,radius=2,ec="None",fc=primordia_colors[primordium],alpha=1)
                figure.gca().add_artist(c)
                #circle_figure.gca().scatter([extremum_theta], aligned_normalized_dIIv[extremum_theta+180],color=primordia_colors[primordium])
        
        #circle_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_primordia_circle.jpg")
        
        zero_primordium = clv3_center+clv3_radius*np.array([np.cos(meristem_orientation*theta_min),np.sin(meristem_orientation*theta_min)])
        c = patch.Circle(xy=zero_primordium,radius=3,ec="#c94389",fc='None',alpha=1)
        figure.gca().add_artist(c)
        
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal+"_primordia_map.jpg")
        
        
        theta_matrix = np.array([[np.cos(theta_min),-np.sin(theta_min)],[np.sin(theta_min),np.cos(theta_min)]])
        orientation_matrix = np.array([[1,0],[0,meristem_orientation]])
        theta_positions = np.einsum('...ij,...i->...j', theta_matrix,np.einsum('...ij,...i->...j',orientation_matrix,np.transpose([X,Y]) - clv3_center))
        theta_positions = np.einsum('...ij,...i->...j', theta_matrix,np.einsum('...ij,...i->...j',orientation_matrix,np.transpose([X,Y]) - clv3_center))
        
        data['aligned_x'] = theta_positions[:,0]
        data['aligned_y'] = theta_positions[:,1]
        
        center_altitude = compute_local_2d_signal(theta_positions,np.array([[0,0]]),data['center_z'].values)[0]
        
        data['aligned_z'] = data['center_z'] - center_altitude 
        
        file_data[filename[-3:]] = deepcopy(data)
        
        file_aligned_positions[filename[-3:]] = theta_positions
        
        sup_figure.clf()
        if i_file > 0:
            previous_positions = file_aligned_positions[filenames[i_file-1][-3:]]
            sup_figure.gca().scatter(previous_positions[:,0],previous_positions[:,1],color='b')
        sup_figure.gca().scatter(theta_positions[:,0],theta_positions[:,1],color='r')
        
    
    sup_figure.clf()
    world.clear()
    clv3_radius = np.mean(clv3_radii.values())
    fixed_circle = False
    
    for i_file, filename in enumerate(filenames):
        data = file_data[filename[-3:]]
        model_orientation = model_orientations[filename]
        
        world.add(data,filename[-3:]+'_data')
        fig_number = 2*int(filename[-2:])/3+1
        world[filename[-3:]+'_data'].set_attribute('figure',fig_number)
        world[filename[-3:]+'_data'].set_attribute('X_variable','aligned_x')
        world[filename[-3:]+'_data'].set_attribute('Y_variable','aligned_y')
        #world[filename[-3:]+'_data'].set_attribute('X_variable','center_x')
        #world[filename[-3:]+'_data'].set_attribute('Y_variable','center_y')
        
        signal = 'DIIV'
        world[filename[-3:]+'_data'].set_attribute('label_variable','Normalized_'+signal)
        #world[filename[-3:]+'_data'].set_attribute('label_variable',signal)
        world[filename[-3:]+'_data'].set_attribute('label_colormap',load_colormaps()[signal_colors[signal]])
        
        label_range = tuple([100.*(s-data['Normalized_'+signal].min())/(data['Normalized_'+signal].max()-data['Normalized_'+signal].min()) for s in signal_ranges[signal]])
        #label_range = tuple([100.*(s-data[signal].min())/(data[signal].max()-data[signal].min()) for s in signal_ranges[signal]])
        world[filename[-3:]+'_data'].set_attribute('label_range',label_range)
        
        world[filename[-3:]+'_data'].set_attribute('legend','top_right')
        world[filename[-3:]+'_data'].set_attribute('smooth_factor',0.0)
        world[filename[-3:]+'_data'].set_attribute('n_points',20.0)
        
        world[filename[-3:]+'_data'].set_attribute('plot','line')
        world[filename[-3:]+'_data'].set_attribute('plot','scatter')
        
        figure = plt.figure(fig_number)
        figure.gca().set_xlim(-r_max,r_max)
        figure.gca().set_ylim(-r_max,r_max)
        figure.gca().set_xticklabels(figure.gca().get_xticks())
        figure.gca().set_yticklabels(figure.gca().get_yticks())
        
        
        aligned_meristem_model = deepcopy(meristem_model)
        aligned_meristem_model.parameters['dome_apex_x'] -= clv3_center[0]
        aligned_meristem_model.parameters['dome_apex_y'] -= clv3_center[1]
        aligned_meristem_model.parameters['dome_apex_z'] = 0
        aligned_meristem_model.parameters['orientation'] = 1
        aligned_meristem_model.parameters['initial_angle'] -= theta_min*180./np.pi
        plot_meristem_model(figure,aligned_meristem_model,alpha=1,color='g',r_max=1.5*r_max,center=False)
        #plot_meristem_model(figure,meristem_model,alpha=1,color='g',r_max=1.5*r_max,center=False)
       
        
        
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_aligned_DIIV.jpg")
        
        world[filename[-3:]+'_data'].set_attribute('plot','map')
        
        if not fixed_circle:
            clv3_radius = clv3_radii[filename[-3:]]
        
        c = patch.Circle(xy=[0,0],radius=clv3_radius,ec="#c94389",fc='None',lw=3,alpha=1)
        figure.gca().add_artist(c)
        
        primordia_angles = file_aligned_primordia_angles[filename[-3:]]

        for primordium in primordia_angles.keys():
            if primordium in xrange(-3,6):
                extremum_theta = primordia_angles[primordium]
                primordium_point = clv3_radius*np.array([np.cos(extremum_theta*np.pi/180.),np.sin(extremum_theta*np.pi/180.)])
                c = patch.Circle(xy=primordium_point,radius=2,ec="None",fc=primordia_colors[primordium],alpha=1)
                figure.gca().add_artist(c)
                
        figure.gca().set_xlim(-r_max,r_max)
        figure.gca().set_xticklabels(figure.gca().get_xticks())
        figure.gca().set_ylim(-r_max,r_max)
        figure.gca().set_yticklabels(figure.gca().get_xticks())
        
        figure.set_size_inches(12,12)
        figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename+"_aligned_DIIV_primordia_map.jpg")
        
        X = data['aligned_x'].values
        Y = data['aligned_y'].values
        altitude = data['aligned_z'].values
        clv3 = data['Normalized_CLV3'].values
        dIIv = data['DIIV'].values
        normalized_dIIv = data['Normalized_DIIV'].values
        
        aligned_dIIv, aligned_thetas = compute_circle_signal(np.transpose([X,Y]),np.array([0,0]),clv3_radius,dIIv)
        aligned_normalized_dIIv, circle_thetas = compute_circle_signal(np.transpose([X,Y]),np.array([0,0]),clv3_radius,normalized_dIIv)
        aligned_altitude, circle_thetas = compute_circle_signal(np.transpose([X,Y]),np.array([0,0]),clv3_radius,altitude)
        
        primordia_altitudes = []
        for primordium in primordia_angles.keys():
            if primordium in xrange(-3,6):
                extremum_theta = primordia_angles[primordium]
                primordium_point = clv3_radius*np.array([np.cos(extremum_theta*np.pi/180.),np.sin(extremum_theta*np.pi/180.)])
                primordia_altitudes += list(compute_local_2d_signal(np.transpose([X,Y]),primordium_point,altitude))
               
        
        circle_data['theta'] += list(circle_thetas)
        circle_data['aligned_theta'] += list(aligned_thetas)
        circle_data[signal] += list(aligned_dIIv)
        circle_data["normalized_"+signal] += list(aligned_normalized_dIIv)
        circle_data["altitude"] += list(aligned_altitude)
        circle_data['filename'] += [filename for t in circle_thetas]
        circle_data['hour_time'] += [filename[-3:] for t in circle_thetas]
        circle_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for t in circle_thetas]
       
        primordia_angle_data['primordium'] += [primordium for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data['aligned_theta'] += [primordia_angles[primordium] for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data[signal] += [aligned_dIIv[180+primordia_angles[primordium]] for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data["normalized_"+signal] += [aligned_normalized_dIIv[180+primordia_angles[primordium]] for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data["altitude"] += primordia_altitudes
        primordia_angle_data['filename'] += [filename for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data['hour_time'] += [filename[-3:] for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        primordia_angle_data['sam_id'] += [sam_id for primordium in primordia_angles.keys() if primordium in xrange(-3,6)]
        
        complete_circle_data['theta'] += list(circle_thetas)
        complete_circle_data['aligned_theta'] += list(aligned_thetas)
        complete_circle_data[signal] += list(aligned_dIIv)
        complete_circle_data["normalized_"+signal] += list(aligned_normalized_dIIv)
        complete_circle_data["altitude"] += list(aligned_altitude)
        complete_circle_data['filename'] += [filename for t in circle_thetas]
        complete_circle_data['hour_time'] += [filename[-3:] for t in circle_thetas]
        complete_circle_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for t in circle_thetas]
        complete_circle_data['sam_id'] += [sam_id for t in circle_thetas]
        complete_circle_data['CLV3_radius'] += [clv3_radius for t in circle_thetas]
        
        sup_figure = plt.figure(7)
        circle_df = pd.DataFrame().from_dict(circle_data)
        #world.add(circle_df[circle_df['filename']==filename],'initiation_circle')
        world.add(circle_df,'initiation_circle')
        world['initiation_circle'].set_attribute('figure',7)
        world['initiation_circle'].set_attribute('X_variable','aligned_theta')
        #world['initiation_circle'].set_attribute('Y_variable','DIIV')
        world['initiation_circle'].set_attribute('Y_variable','normalized_DIIV')
        world['initiation_circle'].set_attribute('class_variable','hour_time')
        world['initiation_circle'].set_attribute('plot','line')
        world['initiation_circle'].set_attribute('label_colormap',load_colormaps()['inferno'])
        world['initiation_circle'].set_attribute('legend','top_left')
        world['initiation_circle'].set_attribute('linewidth',3)
        
        for primordium in aligned_primordia_angles.keys():
            if primordium in xrange(-3,6):
                extremum_theta = aligned_primordia_angles[primordium]
                #sup_figure.gca().plot([extremum_theta,extremum_theta],[0,1],linewidth=2,color=primordia_colors[primordium],alpha=1./(abs(primordium)+1))
                #sup_figure.gca().annotate('P'+str(primordium),(extremum_theta,np.min(circle_df[circle_df['filename']==filename]['DIIV'].values)),color=primordia_colors[primordium])
    
        if i_file>=0:
            previous_aligned_angles = file_aligned_primordia_angles[filenames[i_file-1][-3:]]
            for primordium in previous_aligned_angles.keys():
                if primordium in xrange(-3,6):
                    extremum_theta = previous_aligned_angles[primordium]
                    #sup_figure.gca().plot([extremum_theta,extremum_theta],[0,1],linewidth=2,color=primordia_colors[primordium],alpha=0.5/(abs(primordium)+1))
                    #sup_figure.gca().annotate('P'+str(primordium),(extremum_theta,np.min(circle_df[circle_df['filename']==filename]['DIIV'].values)),color=primordia_colors[primordium])
    
    
            #sup_figure.gca().set_ylim(0,0.4)
            sup_figure.gca().set_ylim(0,1)
            sup_figure.gca().set_yticklabels(sup_figure.gca().get_yticks())
            sup_figure.set_size_inches(20,12)
            #sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_DIIV_circle_t00_"+filename[-3:]+".jpg")
            sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_Normalized_DIIV_circle_t00_"+filename[-3:]+".jpg")


    sup_figure.clf()
    for i_file,filename in enumerate(filenames):
        
        data = file_data[filename[-3:]]
        data.to_csv(dirname+"/nuclei_images/"+filename+"/"+filename+"_aligned_L1_nuclei.csv")

        X = data['aligned_x'].values
        Y = data['aligned_y'].values
        
        sup_figure.gca().scatter(X,Y,s=50,edgecolor='None',c='r' if i_file==0 else 'b' if i_file==1 else 'g')
        
    sup_figure.gca().set_xlim(-1.5*r_max,1.5*r_max)
    sup_figure.gca().set_xticklabels(sup_figure.gca().get_xticks())
    sup_figure.gca().set_ylim(-1.5*r_max,1.5*r_max)
    sup_figure.gca().set_yticklabels(sup_figure.gca().get_yticks())
    sup_figure.set_size_inches(12,12)
    sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_aligned_nuclei_"+str(model_orientation*meristem_orientation)+".jpg")

    aligned_primordia_directions = {}
    for primordium in xrange(-2,5):
        angles = np.array(primordia_angle_data['aligned_theta'])[(np.array(primordia_angle_data['sam_id'])==sam_id)&(np.array(primordia_angle_data['primordium'])==primordium)]
        if len(angles)>0:
            aligned_primordia_directions[primordium] = angles.mean()

    for filename in filenames:
        data = file_data[filename[-3:]]
        
        X = data['aligned_x'].values
        Y = data['aligned_y'].values
        altitude = data['aligned_z'].values
        clv3 = data['Normalized_CLV3'].values
        dIIv = data['DIIV'].values
        normalized_dIIv = data['Normalized_DIIV'].values
        
        clv3_radius = clv3_radii[filename[-3:]]
        
        cmap = load_colormaps()['green']
        color_dict = dict(red=[],green=[],blue=[])
        for p in np.sort(cmap._color_points.keys()):
            for k,c in enumerate(['red','green','blue']):
                color_dict[c] += [(p,cmap._color_points[p][k],cmap._color_points[p][k])]
        for c in ['red','green','blue']:
            color_dict[c] = tuple(color_dict[c])
        # print color_dict
        import matplotlib as mpl
        mpl_cmap = mpl.colors.LinearSegmentedColormap(cmap.name, color_dict)
        
        cmap = load_colormaps()['curvature']
        color_dict = dict(red=[],green=[],blue=[])
        for p in np.sort(cmap._color_points.keys()):
            for k,c in enumerate(['red','green','blue']):
                color_dict[c] += [(p,cmap._color_points[p][k],cmap._color_points[p][k])]
        for c in ['red','green','blue']:
            color_dict[c] = tuple(color_dict[c])
        curvature_cmap = mpl.colors.LinearSegmentedColormap(cmap.name, color_dict)
        
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=0, altdeg=45)
        
        from vplants.sam4dmaps.sam_map_construction import extract_signal_map_maxima

        sup_figure.clf()
        angle_opening = 60
        profile_radius = 60
        
        profile_cell_radius = 5.0
        profile_density_k = 0.5
        
        theta_resolution = 1
        r_resolution = 1
        
        radial_profile_data = {}
        for field in ['filename','sam_id','hour_time','growth_condition','primordium','primordium_theta','radial_distance','radial_theta','radial_x','radial_y','clv3_radius','DIIV','normalized_DIIV','altitude']:
            radial_profile_data[field] = []
            
        for p,primordium_theta in aligned_primordia_directions.items():
            radial_thetas = np.linspace(-angle_opening/2.,angle_opening/2.,angle_opening/theta_resolution+1)
            
            radial_profile = []
            for t in radial_thetas:
                radial_signal, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta+t,profile_radius,normalized_dIIv,resolution=r_resolution,cell_radius=profile_cell_radius,density_k=profile_density_k)
                radial_profile += [radial_signal]
            radial_profile = np.transpose(radial_profile)
            
            radial_raw_profile = []
            for t in radial_thetas:
                radial_raw_signal, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta+t,profile_radius,dIIv,resolution=r_resolution,cell_radius=profile_cell_radius,density_k=profile_density_k)
                radial_raw_profile += [radial_raw_signal]
            radial_raw_profile = np.transpose(radial_raw_profile)
            
            radial_elevation = []
            for t in radial_thetas:
                radial_altitude, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta+t,profile_radius,altitude,resolution=r_resolution,cell_radius=profile_cell_radius,density_k=profile_density_k)
                radial_elevation += [radial_altitude]
            radial_elevation = np.transpose(radial_elevation)
            
            radial_thetas = np.pi*radial_thetas/180.
            
            # radial_extrema = {}
            # for i,t in enumerate(radial_thetas):
            #     radial_extrema[t] = local_extrema(radial_profile[:,i],abscissa=radial_distances,threshold=0.5)
            # tangential_extrema={}
            # for i,r in enumerate(radial_distances):
            #     tangential_extrema[r] = local_extrema(radial_profile[i],abscissa=radial_thetas,threshold=-0.5)
            
            # max_points_x = np.concatenate([radial_extrema[t][0][:,0]*np.cos(t) for t in radial_thetas]+[r*np.cos(tangential_extrema[r][0][:,0]) for r in radial_distances])
            # max_points_y = np.concatenate([radial_extrema[t][0][:,0]*np.sin(t) for t in radial_thetas]+[r*np.sin(tangential_extrema[r][0][:,0]) for r in radial_distances])
            
            # min_points_x = np.concatenate([radial_extrema[t][1][:,0]*np.cos(t) for t in radial_thetas]+[r*np.cos(tangential_extrema[r][1][:,0]) for r in radial_distances])
            # min_points_y = np.concatenate([radial_extrema[t][1][:,0]*np.sin(t) for t in radial_thetas]+[r*np.sin(tangential_extrema[r][1][:,0]) for r in radial_distances])
            
            T,R = np.meshgrid(radial_thetas,radial_distances)
            xx = R*np.cos(T)
            yy = R*np.sin(T)
            
            # vert_exag = 1.
            # mesh_zero_points = np.transpose([np.concatenate(xx),np.concatenate(yy),np.concatenate(np.zeros_like(xx))])
            # mesh_zero_points = mesh_zero_points[xx.shape[0]-1:]
            # mesh_points = np.transpose([np.concatenate(xx),np.concatenate(yy),vert_exag*np.concatenate(radial_profile)])
            # mesh_points = mesh_points[xx.shape[0]-1:]
            
            # from openalea.cellcomplex.property_topomesh.utils.delaunay_tools import delaunay_triangulation
            # from openalea.cellcomplex.property_topomesh.property_topomesh_creation import triangle_topomesh
            
            # import openalea.cellcomplex.property_topomesh.property_topomesh_analysis
            # reload(openalea.cellcomplex.property_topomesh.property_topomesh_analysis)
            # from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
            
            # mesh_triangles = delaunay_triangulation(mesh_zero_points)
            # radial_topomesh = triangle_topomesh(mesh_triangles,dict(zip(range(len(mesh_points)),mesh_points)))
            
            # curvature_names =  ['mean_curvature','gaussian_curvature','principal_curvature_min','principal_curvature_max']
            # compute_topomesh_property(radial_topomesh,'normal',2,normal_method='orientation')
            # #compute_topomesh_vertex_property_from_faces(radial_topomesh,'normal',neighborhood=4,adjacency_sigma=2)
            # compute_topomesh_vertex_property_from_faces(radial_topomesh,'normal',neighborhood=1)
            # compute_topomesh_property(radial_topomesh,'gaussian_curvature',2)
            # for property_name in curvature_names:
            #     #compute_topomesh_vertex_property_from_faces(radial_topomesh,property_name,neighborhood=4,adjacency_sigma=2)
            #     compute_topomesh_vertex_property_from_faces(radial_topomesh,property_name,neighborhood=1)
            
            # radial_curvatures = {} 
            # for property_name in curvature_names:
            #     radial_curvatures[property_name] = np.concatenate([radial_topomesh.wisp_property(property_name,0)[0]*np.ones(xx.shape[0]),radial_topomesh.wisp_property(property_name,0).values(range(len(mesh_points)))[1:]]).reshape(xx.shape[0],xx.shape[1])
            # #ax.contourf(xx,yy,radial_curvatures['mean_curvature'],40,cmap=curvature_cmap,alpha=1.0,antialiased=True,vmin=-0.005,vmax=0.005)
            # #ax.contourf(xx,yy,radial_curvatures['gaussian_curvature'],20,cmap=curvature_cmap,alpha=1.0,antialiased=True,vmin=-0.0001,vmax=0.0001)
            
            
            from vplants.sam4dmaps.sam_model_tools import nuclei_density_function
            min_region_size = 10
            extremum_density_k = 2.0
            extremum_radius = 0.5
            
            # dT = T[1,1]-T[0,0]
            # dR = R[1,1]-R[0,0]
            # radial_areas = R*dR*dT
            
            # projected_max_points = dict(zip(range(len(max_points_x)),np.transpose([max_points_x,max_points_y,np.zeros_like(max_points_x)])))
            # max_density = nuclei_density_function(projected_max_points,cell_radius=extremum_radius,k=extremum_density_k)(xx,yy,0*xx) 
            # max_regions = nd.label(max_density>0.5)[0]
            # max_components = np.unique(max_regions)[1:]
            # max_region_sizes = dict(zip(max_components,nd.sum(radial_areas,max_regions,index=max_components)))
            # maxima = {}
            # for c in max_components:
            #     if max_region_sizes[c] > min_region_size:
            #         max_region_coords = np.where(max_regions==c)
            #         max_signal_point = np.argmax(radial_profile[max_region_coords])
            #         max_coords = [coords[max_signal_point] for coords in max_region_coords]
            #         max_point = [xx[tuple(max_coords)],yy[tuple(max_coords)]]
            #         max_r = np.linalg.norm(max_point)
            #         if max_r > 0.8*clv3_radius:
            #             maxima[c] = max_point
            
            # projected_min_points = dict(zip(range(len(min_points_x)),np.transpose([min_points_x,min_points_y,np.zeros_like(min_points_x)])))
            # min_density = nuclei_density_function(projected_min_points,cell_radius=extremum_radius,k=extremum_density_k)(xx,yy,0*xx) 
            # min_regions = nd.label(min_density>0.5)[0]
            # min_components = np.unique(min_regions)[1:]
            # min_region_sizes = dict(zip(min_components,nd.sum(radial_areas,min_regions,index=min_components)))
            # minima = {}
            # for c in min_components:
            #     if min_region_sizes[c] > min_region_size:
            #         min_region_coords = np.where(min_regions==c)
            #         min_signal_point = np.argmin(radial_profile[min_region_coords])
            #         min_coords = [coords[min_signal_point] for coords in min_region_coords]
            #         min_point = [xx[tuple(min_coords)],yy[tuple(min_coords)]]
            #         min_r = np.linalg.norm(min_point)
            #         if min_r > 0.8*clv3_radius:
            #             minima[c] = min_point
            
            figure = plt.figure(1)
            figure.patch.set_facecolor('white')
            figure.patch.set_visible(True)
            figure.clf()
            ax = plt.subplot(111)
            
            #profile_elevation = ls.shade(radial_elevation,cmap=plt.cm.gray,vert_exag=0.1,blend_mode='soft')[:,:,0]
            levels = [-1]+list(np.linspace(0,1,31))+[2]
            
            ax.contourf(xx,yy,radial_profile,levels,cmap=mpl_cmap,alpha=1.0,antialiased=True,vmin=0,vmax=1)
            #ax.pcolormesh(xx,yy,radial_profile,cmap=mpl_cmap,antialiased=True,shading='gouraud',alpha=0.33,vmin=0,vmax=1)
            #ax.pcolormesh(xx,yy, profile_elevation*radial_profile,cmap=mpl_cmap,vmin=0,vmax=1,linewidth=1e-5)
            #ax.pcolormesh(xx,yy,profile_elevation,cmap='gray',antialiased=True,shading='gouraud',alpha=0.2)
            #ax.pcolormesh(xx,yy,radial_elevation*ls.hillshade(radial_elevation,vert_exag=0.1),cmap='gray',linewidth=1e-5)
            
            #ax.pcolormesh(xx,yy,min_regions!=0,cmap='Blues',antialiased=True,shading='gouraud',alpha=0.1)
            #ax.pcolormesh(xx,yy,max_regions!=0,cmap='Reds',antialiased=True,shading='gouraud',alpha=0.1)
            #ax.scatter(max_points_x,max_points_y,s=30,color='r',alpha=0.2)
            #ax.scatter(min_points_x,min_points_y,s=30,color='b',alpha=0.2)
            
            #ax.scatter(np.array(maxima.values())[:,0],np.array(maxima.values())[:,1],s=80,color='r',edgecolor='k')
            #ax.scatter(np.array(minima.values())[:,0],np.array(minima.values())[:,1],s=80,color='b',edgecolor='k')
            
            ax.plot(clv3_radius*np.cos(radial_thetas),clv3_radius*np.sin(radial_thetas),c="#c94389",linewidth=3)
            
            ax.plot([0,R.max()*np.cos(T.min())],[0,R.max()*np.sin(T.min())],c='k',linewidth=1)
            ax.plot([0,R.max()*np.cos(T.max())],[0,R.max()*np.sin(T.max())],c='k',linewidth=1)
            ax.plot(R.max()*np.cos(radial_thetas),R.max()*np.sin(radial_thetas),c='k',linewidth=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('equal')
            ax.axis('off')
            ax.set_title("P"+str(int(p)),size=24)
            plt.draw()
            
            figure.set_size_inches(12,9)
            figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_P"+str(int(p))+"_normalized_DIIV_radial_profile.png")
                        
            radial_profile_data['filename'] += [filename for i in xx.ravel()]
            radial_profile_data['sam_id'] += [sam_id for i in xx.ravel()]
            radial_profile_data['hour_time'] += [filename[-3:] for i in xx.ravel()]
            radial_profile_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for i in xx.ravel()]
            radial_profile_data['primordium'] += [p for i in xx.ravel()]
            radial_profile_data['primordium_theta'] += [primordium_theta for i in xx.ravel()]
            radial_profile_data['radial_distance'] += list(R.ravel())
            radial_profile_data['radial_theta'] += list(180.*T.ravel()/np.pi)
            radial_profile_data['radial_x'] += list(xx.ravel())
            radial_profile_data['radial_y'] += list(yy.ravel())
            radial_profile_data['clv3_radius'] += [clv3_radius for i in xx.ravel()]
            radial_profile_data['DIIV'] += list(radial_raw_profile.ravel())
            radial_profile_data['normalized_DIIV'] += list(radial_profile.ravel())
            radial_profile_data['altitude'] += list(radial_elevation.ravel())
            
            
            
            # ax.contourf(xx,yy,radial_curvatures['principal_curvature_max'],40,cmap=curvature_cmap,alpha=1.0,antialiased=True,vmin=-0.005,vmax=0.005)
            # figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_P"+str(int(p))+"_normalized_DIIV_max_curvature_profile.png")
            # ax.contourf(xx,yy,radial_curvatures['principal_curvature_min'],40,cmap=curvature_cmap,alpha=1.0,antialiased=True,vmin=-0.005,vmax=0.005)
            # figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_P"+str(int(p))+"_normalized_DIIV_min_curvature_profile.png")
            
            
            radial_dIIv, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta,80,dIIv)
            radial_normalized_dIIv, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta,80,normalized_dIIv)
            radial_altitude, radial_distances = compute_radial_signal(np.transpose([X,Y]),np.array([0,0]),primordium_theta,80,altitude)
            
            complete_radial_data['radial_distance'] += list(radial_distances)
            complete_radial_data[signal] += list(radial_dIIv)
            complete_radial_data["normalized_"+signal] += list(radial_normalized_dIIv)
            complete_radial_data["altitude"] += list(radial_altitude)
            complete_radial_data['primordium'] += [p for r in radial_distances]
            complete_radial_data['aligned_theta'] += [aligned_primordia_directions[p] for r in radial_distances]
            complete_radial_data['filename'] += [filename for r in radial_distances]
            complete_radial_data['hour_time'] += [filename[-3:] for r in radial_distances]
            complete_radial_data['growth_condition'] += ['LD' if 'LD' in filename else 'SD' for r in radial_distances]
            complete_radial_data['sam_id'] += [sam_id for r in radial_distances]
            complete_radial_data['CLV3_radius'] += [clv3_radius for r in radial_distances]
        
            radial_df = pd.DataFrame().from_dict(complete_radial_data)
            world.add(radial_df[radial_df['filename']==filename],'initiation_radial')
            world['initiation_radial'].set_attribute('figure',7)
            world['initiation_radial'].set_attribute('X_variable','radial_distance')
            world['initiation_radial'].set_attribute('Y_variable','normalized_DIIV')
            world['initiation_radial'].set_attribute('class_variable','primordium')
            world['initiation_radial'].set_attribute('plot','line')
            world['initiation_radial'].set_attribute('n_points',15)
            world['initiation_radial'].set_attribute('label_colormap',load_colormaps()['Tmt_jet'])
            world['initiation_radial'].set_attribute('legend','top_left')
            world['initiation_radial'].set_attribute('linewidth',3)
            
        pd.DataFrame().from_dict(radial_profile_data).to_csv(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_primordium_radial_data.csv")
        
        sup_figure.set_size_inches(20,12)
        #sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_DIIV_circle_t00_"+filename[-3:]+".jpg")
        sup_figure.savefig(dirname+"/nuclei_images/"+filename+"/"+filename[:-4]+"_normalized_DIIV_radial_"+filename[-3:]+".jpg")
            

for primordium in xrange(-2,5):
    world.clear()
    radial_df = pd.DataFrame().from_dict(complete_radial_data)
    world.add(radial_df[radial_df['primordium']==primordium],'initiation_radial')            
    world['initiation_radial'].set_attribute('figure',7)
    world['initiation_radial'].set_attribute('X_variable','radial_distance')
    world['initiation_radial'].set_attribute('Y_variable','normalized_DIIV')
    world['initiation_radial'].set_attribute('class_variable','hour_time')
    world['initiation_radial'].set_attribute('plot','line')
    world['initiation_radial'].set_attribute('plot','scatter')
    world['initiation_radial'].set_attribute('markersize',1)
    world['initiation_radial'].set_attribute('label_colormap',load_colormaps()['inferno'])
    world['initiation_radial'].set_attribute('legend','top_left')
    world['initiation_radial'].set_attribute('linewidth',0)
    world['initiation_radial'].set_attribute('regression','density')
    sup_figure.gca().set_ylim(0,1)
    sup_figure.gca().set_yticklabels(sup_figure.gca().get_yticks())
    sup_figure.savefig("/Users/gcerutti/Desktop/LD_radial_DIIV_P"+str(primordium)+".jpg")
            
    
    

save_data = False
if save_data:
    circle_df = pd.DataFrame().from_dict(complete_circle_data)
    filename = "qDII-CLV3-LD_11c_160906"
    #filename = "qDII-CLV3-SD_11c_160828"
    #circle_df.to_csv("/Users/gcerutti/Desktop/"+filename+"_DIIV_circle.csv")
    circle_df.to_csv("/Users/gcerutti/Desktop/"+filename+"_DIIV_no_organ_circle.csv")

    primordia_df = pd.DataFrame().from_dict(primordia_angle_data)
    primordia_df.to_csv("/Users/gcerutti/Desktop/"+filename+"_primordia_angles.csv")
    
    radial_df = pd.DataFrame().from_dict(complete_radial_data)
    radial_df.to_csv("/Users/gcerutti/Desktop/"+filename+"_DIIV_radial.csv")




#     if i_file == 0:
#         theta_min = theta_min*np.pi/180.
#         previous_theta = theta_min
#     else:
#         gap_scores = []
#         for i in range(-3,4):
#             gap_score = (previous_theta*180./np.pi - (theta_min+i*golden_angle) + 180)  % 360 -180
#             print i," : ",theta_min+i*golden_angle," -> ",previous_theta*180./np.pi,"   [",gap_score,']'
#             gap_scores += [gap_score]
#         theta_gap =  np.arange(-3,4)[np.argmin(np.abs(gap_scores))]
#         theta_min = (theta_min + theta_gap*golden_angle + 180)  % 360 -180
#         theta_min = theta_min*np.pi/180.
#         #previous_theta = theta_min
        
        
        
#     zero_primordium = clv3_center+clv3_radius*np.array([np.cos(theta_min),np.sin(theta_min)])
    
#     c = patch.Circle(xy=zero_primordium,radius=2,ec="None",fc='#c94389',alpha=1)
#     figure.gca().add_artist(c)
    
    
#     theta_matrix = np.array([[np.cos(theta_min),-np.sin(theta_min)],[np.sin(theta_min),np.cos(theta_min)]])
#     theta_positions = np.einsum('...ij,...i->...j',theta_matrix,np.transpose([X,Y]) - clv3_center)
    
#     data['aligned_x'] = theta_positions[:,0]
#     data['aligned_y'] = theta_positions[:,1]
    
#     file_data[filename[-3:]] = deepcopy(data)
    
#     sup_figure.gca().scatter(theta_positions[:,0],theta_positions[:,1],color='r' if i_file==0 else 'b')
    

# clv3_radius = np.mean(clv3_radii.values())


# sup_figure.clf()

# circle_signals = []

# for i_file, filename in enumerate(filenames):
#     data = file_data[filename[-3:]]
    
#     X = data['aligned_x'].values
#     Y = data['aligned_y'].values
#     clv3 = data['Normalized_CLV3'].values
#     dIIv = data['DIIV'].values
    
#     circle_dIIv, circle_thetas = compute_circle_signal(np.transpose([X,Y]),np.array([0,0]),clv3_radius,dIIv)
#     circle_signals += [circle_dIIv]
    
#     sup_figure.gca().plot(circle_thetas,circle_dIIv,color='r' if i_file==0 else 'b')
    

# for t in np.linspace(0,100,101):
#     hour_time = t*6/100.
#     sup_figure.clf()
#     circle_dIIv = (100-t)/100.*circle_signals[0] + t/100.*circle_signals[1] 
#     sup_figure.gca().plot(circle_thetas,circle_dIIv,color='k',lw=3)
#     sup_figure.gca().set_xlim(-180,180)
#     sup_figure.gca().set_ylim(0,0.3)
#     sup_figure.gca().set_ylabel("DIIV")
    
#     for primordium in range(-3,3):
#         primordium_theta = (primordium*golden_angle+180) % 360 - 180
#         sup_figure.gca().plot([primordium_theta,primordium_theta],[0,1],linewidth=1,color=primordia_colors[primordium],alpha=1)
#         sup_figure.gca().annotate('P'+str(primordium),(primordium_theta,0.02),color=primordia_colors[primordium])
            
#     plt.draw()
#     hour_string = str(int(hour_time)).zfill(1)
#     minute_string = str(int((hour_time-int(hour_time))*60)).zfill(2)
#     sup_figure.savefig("/Users/gcerutti/Desktop/CircleDynamics/"+filename[:-3]+"DIIV_circle_t0"+hour_string+"-"+minute_string+".jpg")

    
    

            

            
    
    
