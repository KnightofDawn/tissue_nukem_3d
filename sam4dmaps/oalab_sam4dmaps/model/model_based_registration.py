import numpy as np
from scipy import ndimage as nd

from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imread, imsave

from openalea.deploy.shared_data import shared_data
from openalea.oalab.colormap.colormap_def import load_colormaps

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface
from vplants.sam4dmaps.nuclei_detection import read_nuclei_points, compute_fluorescence_ratios
from vplants.sam4dmaps.sam_model import read_meristem_model, reference_meristem_model
from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer
from vplants.sam4dmaps.sam_map_construction import meristem_2d_polar_map, draw_signal_map, extract_signal_map_maxima

from openalea.mesh.property_topomesh_creation import vertex_topomesh
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh

from openalea.container import array_dict, PropertyTopomesh

from time import time
from copy import deepcopy

world.clear()

filenames = []
signal_names = []
microscope_orientations = []
# filenames += ["r2DII_1.2_141202_sam03_t04"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam03_t08"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam03_t28"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam03_t32"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam01_t00"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam01_t04"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam01_t08"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam01_t24"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam01_t28"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam07_t00"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam07_t04"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_2.2_141204_sam07_t08"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam06_t00"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam06_t04"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam06_t08"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam06_t24"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
# filenames += ["r2DII_1.2_141202_sam06_t28"]; signal_names += ['DIIV'];  microscope_orientations += [-1]
filenames += ["r2DII_1.2_141202_sam06_t32"]; signal_names += ['DIIV'];  microscope_orientations += [-1]

#filenames += ["DR5N_5.2_150415_sam01_t00"]; signal_names += ['DR5'];  microscope_orientations += [-1]
#filenames += ["DR5N_7.1_150415_sam03_t00"]; signal_names += ['DR5'];  microscope_orientations += [-1]
#filenames += ["DR5N_7.1_150415_sam04_t00"]; signal_names += ['DR5'];  microscope_orientations += [-1]
#filenames += ["DR5N_7.1_150415_sam07_t00"]; signal_names += ['DR5'];  microscope_orientations += [-1]

signal_colors = {}
signal_colors['DIIV'] = 'viridis'
signal_colors['DR5'] = 'Blues'

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)

reference_images = {}
signal_images = {}
for i_file,filename in enumerate(filenames):
    filetime = filename[-4:]
    sequence_name = filename[:-4]
    signal_name = signal_names[i_file]    
    
    signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
    signal_images[filename] = imread(signal_file)
    reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
    reference_images[filename] = imread(reference_file)
    
    size = np.array(reference_images[filename].shape)
    resolution = np.array(reference_images[filename].resolution)
    
    #world.add(reference_images[filename],filename+"_reference_image",resolution=-resolution,colormap='invert_grey')
    #world.add(signal_images[filename],filename+"_signal_image",resolution=-resolution,colormap=signal_colors[signal_name])

import vplants.sam4dmaps.nuclei_mesh_tools
reload(vplants.sam4dmaps.nuclei_mesh_tools)
from vplants.sam4dmaps.nuclei_mesh_tools import nuclei_layer

nuclei_positions = {}
nuclei_signals = {}
for i_file,filename in enumerate(filenames):
    topomesh_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_nuclei_signal_topomesh.ply"
    
    try:
        nuclei_signals[filename] = read_ply_property_topomesh(topomesh_file)
        nuclei_positions[filename] = nuclei_signals[filename].wisp_property('barycenter',0)
    except:
        nuclei_file = dirname+"/nuclei_images/"+filename+"/cells.csv"
        positions = read_nuclei_points(nuclei_file,return_data=False)
        nuclei_positions[filename] = positions
        
        size = np.array(reference_images[filename].shape)
        resolution = np.array(reference_images[filename].resolution)
        
        image_positions = array_dict(np.array(positions.values())*microscope_orientations[i_file],positions.keys())
        fluorescence_ratios = compute_fluorescence_ratios(reference_images[filename],signal_images[filename],image_positions,negative=(signal_names[i_file] in ['DIIV']))
        cell_layer = nuclei_layer(positions,size,-resolution)
        
        nuclei_points = vertex_topomesh(positions)
        nuclei_points.update_wisp_property('signal',0,fluorescence_ratios)
        nuclei_points.update_wisp_property('layer',0,cell_layer)
        
        nuclei_signals[filename] = nuclei_points
        save_ply_property_topomesh(nuclei_signals[filename],topomesh_file,properties_to_save=dict([(0,['signal','layer']),(1,[]),(2,[]),(3,[])]),color_faces=False)

    #world.add(nuclei_signals[filename],filename+"_nuclei")
    #world[filename+"_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()[signal_colors[signal_name]])
    #world[filename+"_nuclei"].set_attribute("property_name_0",'layer')
    #world[filename+"_nuclei"].set_attribute("property_name_0",'signal')
    #world[filename+"_nuclei_vertices"].set_attribute("intensity_range",(0.0,0.8))

    
meristem_models = {}
for i_file,filename in enumerate(filenames):
    meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
    meristem_models[filename] = read_meristem_model(meristem_model_file)
    
    #world.add(meristem_models[filename],filename+"_meristem_model",_repr_vtk_=meristem_models[filename].drawing_function,colormap='leaf',alpha=0.1,z_slice=(95,100))

import vplants.sam4dmaps.sam_model_registration
reload(vplants.sam4dmaps.sam_model_registration)
from vplants.sam4dmaps.sam_model_registration import meristem_model_registration, meristem_model_organ_gap, meristem_model_alignement, meristem_model_cylindrical_coordinates

import vplants.sam4dmaps.sam_map_construction
reload(vplants.sam4dmaps.sam_map_construction)
from vplants.sam4dmaps.sam_map_construction import meristem_2d_polar_map, meristem_2d_cylindrical_map, meristem_2dt_cylindrical_map, draw_signal_map, extract_signal_map_maxima

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pylab
import matplotlib.patches as patch
from vplants.meshing.cute_plot import simple_plot

reference_dome_apex = np.array([(size*resolution/2.)[0],(size*resolution/2.)[0],0])
n_organs = 8

world.clear()

reference_model = reference_meristem_model(reference_dome_apex,n_primordia=n_organs,developmental_time=0)
orientation = reference_model.parameters['orientation']

golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
golden_angle = 180.*golden_angle/np.pi    

time_cylindrical_coords = {}
time_epidermis_cells = {}
time_signal_ratios = {}

time_radius = 0.0
time_density_k = 2.0

previous_offset = 0
previous_reference_model = reference_model
for  i_file,filename in enumerate(filenames):
#for  i_file,filename in enumerate(['r2DII_1.2_141202_sam03_t08','r2DII_1.2_141202_sam03_t28','r2DII_1.2_141202_sam06_t08','r2DII_2.2_141204_sam07_t00']):
   
    meristem_model = meristem_models[filename]
    positions = nuclei_positions[filename]
    reference_image = reference_images[filename]
    signal_image = signal_images[filename]
    
    organ_gap = meristem_model_organ_gap(reference_model,meristem_models[filename],same_individual=False)
    print filename," : ",organ_gap," [",meristem_models[filename].parameters['primordium_offset'],"->",meristem_models[filename].parameters['orientation']*meristem_models[filename].parameters['primordium_1_angle'] - golden_angle,"] (",meristem_models[filename].parameters['orientation'],")",
    
    # if i_file == 0:
    #     plastochron_gap = meristem_model_organ_gap(reference_model,meristem_models[filename],same_individual=False)
    # else:
    #     #plastochron_gap = meristem_model_organ_gap(reference_model,meristem_models[filename],same_individual=True)
    #     plastochron_gap = meristem_model_organ_gap(previous_reference_model,meristem_models[filename],same_individual=True)
    # plastochron_gap += previous_offset
    # previous_offset = plastochron_gap
    # previous_reference_model = meristem_models[filename]
    
    #plastochron_gap = 0
    
    epidermis_cells = np.array(list(nuclei_signals[filename].wisps(0)))[nuclei_signals[filename].wisp_property('layer',0).values() == 1]
    signal_ratios = nuclei_signals[filename].wisp_property('signal',0)
        
    #for plastochron_gap in [-4,-3,-2,-1,0,1,2,3,4]:
    for plastochron_gap in [-1,0,1,2]:
        print filename,plastochron_gap,"( t =",plastochron_gap + (meristem_models[filename].parameters['developmental_time']%9.)/9.,")   [",meristem_models[filename].parameters['developmental_time']/9.,"]"
        
        developmental_time = plastochron_gap + (meristem_models[filename].parameters['developmental_time']%9.)/9.
    
        #coords, aligned_meristem_model = meristem_model_cylindrical_coordinates(meristem_model,positions,plastochron_gap,orientation)
        coords, aligned_meristem_model = meristem_model_cylindrical_coordinates(meristem_model,positions,-plastochron_gap - organ_gap,orientation)
    
        time_cylindrical_coords[developmental_time] = coords
    
        time_epidermis_cells[developmental_time] = epidermis_cells
        time_signal_ratios[developmental_time] = signal_ratios
      
        epidermis_nuclei_ratio, epidermis_model_density, epidermis_positions, T, R = meristem_2d_cylindrical_map(coords, aligned_meristem_model, epidermis_cells, signal_ratios)
            
        map_figure = plt.figure(1)
        map_figure.clf()
        map_figure.patch.set_facecolor('white')
        ax = plt.subplot(111, polar=True)

        draw_signal_map(map_figure,epidermis_nuclei_ratio,T,R,epidermis_model_density,colormap='viridis', n_levels=0, ratio_min=0.0, ratio_max=1.0)
        plt.show(block=False)
        map_figure.savefig("/Users/gcerutti/Desktop/2Dt_SignalMaps/map_"+filename+"_"+str(plastochron_gap)+".jpg")
            
        figure = plt.figure(2)
        figure.clf()
        figure.patch.set_facecolor('white')
        ax = plt.subplot(111, polar=True)
        
        for p in xrange(n_organs):
            organ_radius = reference_model.parameters['primordium_'+str(p+1)+'_radius']
            organ_distance = reference_model.parameters['primordium_'+str(p+1)+'_distance']
            organ_angle = reference_model.parameters['primordium_'+str(p+1)+'_angle']*np.pi/180.
            organ_center = [organ_distance*np.cos(organ_angle),organ_distance*np.sin(organ_angle)]
            organ_circle = pylab.Circle(xy=organ_center, radius=organ_radius,transform=ax.transProjectionAffine + ax.transAxes ,ec='g',fc='None',lw=3)
            ax.add_artist(organ_circle)
    
        for p in xrange(n_organs):
            organ_radius = aligned_meristem_model.parameters['primordium_'+str(p+1)+'_radius']
            #organ_distance = meristem_model.parameters['primordium_'+str(p+1)+'_distance']
            #organ_angle = (meristem_model.parameters['primordium_'+str(p+1)+'_angle'] + (plastochron_gap + meristem_model.parameters['primordium_offset'])*golden_angle)*np.pi/180.
            #organ_center = [organ_distance*np.cos(organ_angle),organ_distance*np.sin(organ_angle)]
            organ_center = aligned_meristem_model.shape_model['primordia_centers'][p][:2]
            organ_circle = pylab.Circle(xy=organ_center, radius=organ_radius,transform=ax.transProjectionAffine + ax.transAxes ,ec='k',fc='None',lw=1,alpha=0.2)
            ax.add_artist(organ_circle)
            
        ax.scatter(coords.values(epidermis_cells)[:,0]*np.pi/180.,coords.values(epidermis_cells)[:,1],s=15.0,c='w',alpha=0.2)
                  
        ax.set_rmax(160)
        ax.set_rmin(0)
        ax.grid(True)
        ax.set_yticklabels([])
        plt.show(block=False)
        figure.savefig("/Users/gcerutti/Desktop/2Dt_SignalMaps/nuclei_"+filename+"_"+str(plastochron_gap)+".jpg")
            
    import vplants.sam4dmaps.sam_model_tools
    reload(vplants.sam4dmaps.sam_model_tools)
    from vplants.sam4dmaps.sam_model_tools import meristem_model_topomesh
    
    model_topomesh = meristem_model_topomesh(reference_model.shape_model,grid_resolution = np.array([4,4,4]))
    world.add(model_topomesh,"model_topomesh")



    
    # epidermis_cells = np.array(list(nuclei_signals[filename].wisps(0)))[nuclei_signals[filename].wisp_property('layer',0).values() == 1]
    # ax.scatter(coords.values(epidermis_cells)[:,0]*np.pi/180.,coords.values(epidermis_cells)[:,1],s=15.0,c='w',alpha=0.2)
        
    # ax.set_rmax(160)
    # ax.set_rmin(0)
    # ax.grid(True)
    # ax.set_yticklabels([])
    # plt.show(block=False)
    # raw_input()

developmental_times = np.arange(19.)

for d_time in developmental_times:
    
    reference_model = reference_meristem_model(np.array([0,0,0]),n_primordia=n_organs,developmental_time=d_time)

    epidermis_nuclei_ratio, epidermis_model_density, epidermis_positions, T, R = meristem_2dt_cylindrical_map(time_cylindrical_coords, d_time, reference_model, time_epidermis_cells, time_signal_ratios, time_radius=0.0, time_density_k=2.0)

    # dome_radius = reference_model.parameters['dome_radius']
    # r_max = 160.
    
    # cell_radius = 2.39
    # density_k = 0.25
    
    # time_radius = 0.0
    # time_density_k = 5.0
    
    # #r = np.linspace(0,160,81)
    # r = np.linspace(0,r_max,r_max/2+1)
    # #t = np.linspace(0,2*np.pi,360)
    # t = np.linspace(0,2*np.pi,180)
    # R,T = np.meshgrid(r,t)

    # e_x = R*np.cos(T)
    # e_y = R*np.sin(T)
    # e_z = R*0
    
    # from vplants.sam4dmaps.sam_model_tools import nuclei_density_function
    
    # epidermis_nuclei_time_potentials = {}
    # epidermis_nuclei_time_signals = {}
    
    # ratio_min = 0
    # ratio_max = 0
    # time_density = 0
    
    # for i_file,filename in enumerate(filenames):
    #     epidermis_cells = np.array(list(nuclei_signals[filename].wisps(0)))[nuclei_signals[filename].wisp_property('layer',0).values() == 1]
    #     signal_ratios = nuclei_signals[filename].wisp_property('signal',0)
        
    #     epidermis_cell_distances = model_referential_coords[filename].values(epidermis_cells)[:,1]
    #     epidermis_cell_surface_distances = (2.*dome_radius)*np.arcsin(np.minimum(epidermis_cell_distances/(2.*dome_radius),1.0))
    #     epidermis_cell_angles = np.pi*model_referential_coords[filename].values(epidermis_cells)[:,0]/180.
        
    #     epidermis_cell_x = epidermis_cell_surface_distances*np.cos(epidermis_cell_angles)
    #     epidermis_cell_y = epidermis_cell_surface_distances*np.sin(epidermis_cell_angles)
    #     epidermis_cell_z = epidermis_cell_distances*0
    #     projected_epidermis_points = dict(zip(epidermis_cells,np.transpose([epidermis_cell_x,epidermis_cell_y,epidermis_cell_z])))
        
    #     epidermis_nuclei_potential = np.array([nuclei_density_function(dict([(p,projected_epidermis_points[p])]),cell_radius=cell_radius,k=density_k)(e_x,e_y,e_z) for p in epidermis_cells])
    #     epidermis_nuclei_potential = np.transpose(epidermis_nuclei_potential,(1,2,0))
    
    #     time_distance = np.abs(d_time/9. - model_developmental_times[filename])
    #     time_potential = 1./2. * (1. - np.tanh(time_density_k*(time_distance - time_radius)))
        
    #     ratio_min += time_potential*(np.mean(signal_ratios.values()) - np.sqrt(2.)*np.std(signal_ratios.values()))
    #     ratio_max += time_potential*(np.mean(signal_ratios.values()) + np.sqrt(2.)*np.std(signal_ratios.values()))
    #     time_density += time_potential
        
    #     epidermis_nuclei_time_potentials[filename] = time_potential*epidermis_nuclei_potential
    #     epidermis_nuclei_time_signals[filename] = signal_ratios.values(epidermis_cells)
        
    # epidermis_nuclei_time_potential = np.concatenate(epidermis_nuclei_time_potentials.values(),axis=2)
    # epidermis_nuclei_time_signal = np.concatenate(epidermis_nuclei_time_signals.values(),axis=0)
    
    # epidermis_nuclei_time_density = np.sum(epidermis_nuclei_time_potential,axis=2)
    # epidermis_nuclei_time_membership = epidermis_nuclei_time_potential/epidermis_nuclei_time_density[...,np.newaxis]
    
    # ratio_min /= time_density
    # ratio_max /= time_density
            
    # epidermis_nuclei_ratio = np.sum(epidermis_nuclei_time_membership*epidermis_nuclei_time_signal[np.newaxis,np.newaxis,:],axis=2)
    # epidermis_nuclei_ratio = (epidermis_nuclei_ratio-ratio_min)/(ratio_max-ratio_min)
        
    # e_x = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.cos(T)
    # e_y = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.sin(T)
    # e_z = R*0.
    
    # primordia_centers = reference_model.shape_model['primordia_centers']
    # primordia_centers[:,2] = 0
    # primordia_radiuses = reference_model.shape_model['primordia_radiuses']
    # epidermis_model_density = R*0.
    
    # dome_density = nuclei_density_function(dict([(p,np.array([0,0,0]))]),dome_radius,k=1)(e_x,e_y,e_z)
    # epidermis_model_density += dome_density   
    # for p in xrange(n_organs):
    #     organ_density = nuclei_density_function(dict([(p,np.array(primordia_centers[p]))]),primordia_radiuses[p],k=1)(e_x,e_y,e_z)
    #     epidermis_model_density += organ_density
    
    # figure = plt.figure(2)
    # figure.clf()
    # figure.patch.set_facecolor('white')
    # ax = plt.subplot(111, polar=True)

    # ax.pcolormesh(T,R,epidermis_model_density,shading='gouraud',cmap="Greys")

    # ax.set_rmax(160)
    # ax.set_rmin(0)
    # ax.grid(True)
    # ax.set_yticklabels([])
    # plt.show(block=False)
    
    #epidermis_model_density = np.array([[[ reference_model.shape_model_density_function()(e_x[i,j,np.newaxis][:,np.newaxis,np.newaxis]+reference_dome_apex[0],e_y[i,j,np.newaxis][np.newaxis,:,np.newaxis]+reference_dome_apex[1],e_z[i,j,np.newaxis][np.newaxis,np.newaxis,:]+reference_dome_apex[2]-(20.*(k-3)))[0,0,0] for k in np.arange(0,6)] for j in xrange(e_x.shape[1])] for i in xrange(e_x.shape[0])]).max(axis=2)
    
    figure = plt.figure(0)
    figure.clf()
    draw_signal_map(figure,epidermis_nuclei_ratio,T,R,epidermis_model_density,colormap='viridis', n_levels=0, ratio_min=0.0, ratio_max=1.0)
    plt.show(block=False)
    
    figure.savefig("/Users/gcerutti/Desktop/2Dt_SignalMaps/map_2dt_"+str(100*d_time)+".jpg")
    
    





average_signal_maps = {}
average_model_maps = {}
        
    
        aligned_meristem_model, aligned_positions, aligned_reference_image, aligned_signal_image = meristem_model_alignement(meristem_model,positions,reference_dome_apex,reference_image,signal_image,plastochron_gap,orientation)
        world.add(aligned_meristem_model,filename+"_aligned_meristem_model",_repr_vtk_=meristem_models[filename].drawing_function,colormap='leaf',alpha=0.1,z_slice=(95,100),display_colorbar=False)
    
        aligned_nuclei_points = vertex_topomesh(aligned_positions)
        for property_name in ['signal','layer']:
            aligned_nuclei_points.update_wisp_property(property_name,0,nuclei_signals[filename].wisp_property(property_name,0))
        
        world.add(aligned_nuclei_points,filename+"_aligned_nuclei")
        world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()[signal_colors[signal_name]])
        world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'signal')
        world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.4,1.0))
        world[filename+"_aligned_nuclei_vertices"].set_attribute("point_radius",3)
        world[filename+"_aligned_nuclei_vertices"].set_attribute("display_colorbar",False)
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()['jet'])
        #world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'layer')
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.0,2.0))
    

    aligned_meristem_models, aligned_nuclei_positions, aligned_nuclei_images, aligned_signal_images = meristem_model_registration(meristem_models,nuclei_positions,reference_dome_apex,reference_images,signal_images,reference_model,same_individual=True)
    
    world.add(reference_model,"reference_meristem_model",_repr_vtk_=reference_model.drawing_function,colormap='Greens',alpha=0.1,z_slice=(95,100),display_colorbar=False)
    
    aligned_nuclei_signals = {}
    for  i_file,filename in enumerate(filenames):
        world.add(aligned_meristem_models[filename],filename+"_aligned_meristem_model",_repr_vtk_=meristem_models[filename].drawing_function,colormap='leaf',alpha=0.1,z_slice=(95,100),display_colorbar=False)
    
        aligned_nuclei_points = vertex_topomesh(aligned_nuclei_positions[filename])
        for property_name in ['signal','layer']:
            aligned_nuclei_points.update_wisp_property(property_name,0,nuclei_signals[filename].wisp_property(property_name,0))
        
        world.add(aligned_nuclei_points,filename+"_aligned_nuclei")
        world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()[signal_colors[signal_name]])
        world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'signal')
        world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.4,1.0))
        world[filename+"_aligned_nuclei_vertices"].set_attribute("point_radius",3)
        world[filename+"_aligned_nuclei_vertices"].set_attribute("display_colorbar",False)
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()['jet'])
        #world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'layer')
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.0,2.0))
        
        aligned_nuclei_signals[filename] = aligned_nuclei_points
    raw_input()  
    
    

developmental_times = np.arange(252.)/4. - 8.5
developmental_times = np.arange(10.)
#cam.SetFocalPoint(reference_dome_apex)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patch
from vplants.meshing.cute_plot import simple_plot

distances = []
heights = []

organ_centers = dict()
n_organs = 8
for p in xrange(n_organs):
    organ_centers[p] = []
    
import vtk

world.clear()

for d_time in developmental_times:

    reference_model = reference_meristem_model(reference_dome_apex,n_primordia=n_organs,developmental_time=d_time)

    radius = reference_model.shape_model['dome_radius']
    scales = reference_model.shape_model['dome_scales']
    
    n_primordia = reference_model.parameters['n_primordia']
    
    dome_circle = vtk.vtkRegularPolygonSource()
    dome_circle.SetNumberOfSides(50)
    dome_circle.SetRadius(radius)
    dome_circle.SetCenter([0,0,d_time*50./9.])
    dome_circle.Update()
    world.add(dome_circle.GetOutput(),"dome_"+str(d_time),colormap='Greens',alpha=0.1,display_colorbar=False)
           
    for p in xrange(n_primordia):
        organ_radius = reference_model.parameters['primordium_'+str(p+1)+'_radius']
        organ_distance = reference_model.parameters['primordium_'+str(p+1)+'_distance']
        organ_angle = reference_model.parameters['primordium_'+str(p+1)+'_angle']*np.pi/180.
        organ_center = [organ_distance*np.cos(organ_angle),organ_distance*np.sin(organ_angle)]
        
        organ_circle = vtk.vtkRegularPolygonSource()
        organ_circle.SetNumberOfSides(50)
        organ_circle.SetRadius(organ_radius)
        organ_circle.SetCenter(organ_center+[d_time*50./9.])
        organ_circle.Update()
        
        world.add(organ_circle.GetOutput(),"organ_"+str(p+1)+"_"+str(d_time),colormap='Greens',alpha=0.1,display_colorbar=False)
        
    
    
    color = np.array([0.4,0.9,0.1])
    
    figure = plt.figure(1)
    figure.clf()
    plt.axis('equal')
    
    dome_circle = patch.Circle(xy=[0,0], radius=radius,ec=color,fc='None',lw=3)
    figure.gca().add_patch(dome_circle)
    
    
    for p in xrange(n_primordia):

        organ_radius = reference_model.parameters['primordium_'+str(p+1)+'_radius']
        organ_distance = reference_model.parameters['primordium_'+str(p+1)+'_distance']
        organ_angle = reference_model.parameters['primordium_'+str(p+1)+'_angle']*np.pi/180.
        organ_center = [organ_distance*np.cos(organ_angle),organ_distance*np.sin(organ_angle)]
        organ_circle = patch.Circle(xy=organ_center, radius=organ_radius,ec=color,fc='None',lw=3)
        
        organ_centers[p] += [organ_center]
        
        #if len(organ_centers[p])>1:
        #    simple_plot(figure,np.array(organ_centers[p])[:,0],np.array(organ_centers[p])[:,1],color,marker_size=0,alpha=0.2)
        
        figure.gca().add_patch(organ_circle)
    
    figure.gca().set_ylim(-120,120)
    figure.gca().set_xlim(-120,120)
    
    plt.axis('off')
    figure.patch.set_facecolor('white')
    plt.show(block=False)
    
    figure_model_file = "/Users/gcerutti/Desktop/SAM_Model/meristem_model_2D_"+str(100*(d_time+8.5))+".jpg"
    figure.savefig(figure_model_file,facecolor=figure.get_facecolor())
    
    
    figure = plt.figure(0)
    figure.clf()
    plt.axis('equal')
    D = np.linspace(0,160,321)
    dome = radius*scales[2]*np.sqrt(1 - np.power(D/(radius*scales[0]),2)) - radius*scales[2]

    simple_plot(figure,D,dome,color,marker_size=0)
    simple_plot(figure,-D,dome,color,marker_size=0)
    
    organ_number = np.maximum(int(d_time-0.25)/9,1)

    organ_distance = reference_model.parameters['primordium_'+str(organ_number)+'_distance']
    organ_height = reference_model.parameters['primordium_'+str(organ_number)+'_height']
    organ_radius = reference_model.parameters['primordium_'+str(organ_number)+'_radius']
    organ_center = [organ_distance,organ_height]
    
    distances += [organ_distance]
    heights += [organ_height]
    
    if len(distances)>1:
        simple_plot(figure,np.array(distances),np.array(heights),color,marker_size=0,alpha=0.2)
        
    figure.gca().scatter(np.array([organ_distance]),np.array([organ_height]),c=color,s=20)
    
    circle = patch.Circle(xy=organ_center, radius=organ_radius,ec=color,fc='None',lw=3)
    figure.gca().add_patch(circle)
    
    figure.gca().set_ylim(-60,20)
    figure.gca().set_xlim(-60,160)
    plt.axis('off')
    figure.patch.set_facecolor('white')
    plt.show(block=False)
    
    figure_model_file = "/Users/gcerutti/Desktop/SAM_Model/meristem_model_1D_"+str(100*(d_time+8.5))+".jpg"
    figure.savefig(figure_model_file,facecolor=figure.get_facecolor())
    
    
    
    world.add(reference_model,"reference_meristem_model",_repr_vtk_=reference_model.drawing_function,colormap='Greens',alpha=1.0,z_slice=(95,100))
    world["reference_meristem_model"].set_attribute("display_colorbar",False)
    
    
    
    
    cam = viewer.ren.GetActiveCamera()
    cam.OrthogonalizeViewUp()	
    
    scene_center = np.array(cam.GetFocalPoint())
    camera_position = np.array(cam.GetPosition())
    camera_orientation = np.array(cam.GetOrientation())*np.pi/180.

    camera_viewup = cam.GetViewUp()
    
    distance = np.linalg.norm(camera_position - scene_center)
    elevation = np.arcsin((camera_position-scene_center)[2]/distance)
    azimut = np.sign((camera_position-scene_center)[1])*np.arccos((camera_position-scene_center)[0]/(distance*np.cos(elevation)))
    roll = cam.GetRoll()*np.pi/180.
    
    azimut += 0.02
    
    new_camera_position = scene_center+np.array([np.cos(azimut)*np.cos(elevation),np.sin(azimut)*np.cos(elevation),np.sin(elevation)])*distance
    
    cam.SetPosition(new_camera_position)    
    cam.SetFocalPoint(scene_center)
    cam.SetViewUp((0,0,1))
    
    reference_model_file = "/Users/gcerutti/Desktop/SAM_Model/meristem_model_"+str(100*d_time)+".jpg"
    viewer.save_screenshot(reference_model_file)
    
    import vplants.sam4dmaps.sam_model_registration
    reload(vplants.sam4dmaps.sam_model_registration)
    from vplants.sam4dmaps.sam_model_registration import meristem_model_registration
    
    aligned_meristem_models, aligned_nuclei_positions, aligned_nuclei_images, aligned_signal_images = meristem_model_registration(meristem_models,nuclei_positions,reference_dome_apex,reference_images,signal_images,reference_model,same_individual=False)
    
    aligned_nuclei_signals = {}
    for  i_file,filename in enumerate(filenames):
        #world.add(aligned_meristem_models[filename],filename+"_aligned_meristem_model",_repr_vtk_=meristem_models[filename].drawing_function,colormap='leaf',alpha=0.1,z_slice=(95,100))
    
        aligned_nuclei_points = vertex_topomesh(aligned_nuclei_positions[filename])
        for property_name in ['signal','layer']:
            aligned_nuclei_points.update_wisp_property(property_name,0,nuclei_signals[filename].wisp_property(property_name,0))
        
        #world.add(aligned_nuclei_points,filename+"_aligned_nuclei")
        # world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()[signal_colors[signal_name]])
        # world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'signal')
        # world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.4,1.0))
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("polydata_colormap",load_colormaps()['jet'])
        #world[filename+"_aligned_nuclei"].set_attribute("property_name_0",'layer')
        #world[filename+"_aligned_nuclei_vertices"].set_attribute("intensity_range",(0.0,2.0))
        
        aligned_nuclei_signals[filename] = aligned_nuclei_points
        
    cell_radius = 2.39
    density_k = 0.33
    r_max = 140
    
    signal_maps = {}
    model_maps = {}
    for i_file,filename in enumerate(filenames):
        epidermis_cells = np.array(list(aligned_nuclei_signals[filename].wisps(0)))[aligned_nuclei_signals[filename].wisp_property('layer',0).values() == 1]
        signal_ratios = aligned_nuclei_signals[filename].wisp_property('signal',0)
        signal_map, model_density_map, _, T, R = meristem_2d_polar_map(aligned_nuclei_positions[filename], aligned_meristem_models[filename], epidermis_cells, signal_ratios, reference_dome_apex, r_max=r_max, cell_radius=cell_radius, density_k=density_k, normalize=False)
    
        signal_maps[filename] = signal_map
        model_maps[filename] = model_density_map
        
    
    import matplotlib.pyplot as plt
    figure = plt.figure(0)
    
    # for i_file,filename in enumerate(filenames):
    #     max_points = extract_signal_map_maxima(signal_map, T, R, model_density_map)
    
    #     figure.clf()   
    #     #draw_signal_map(figure, signal_maps[filename], T, R, model_maps[filename], normalize=False, signal_ratios=signal_ratios, max_points=max_points, colormap=signal_colors[signal_names[0]])
    #     draw_signal_map(figure, signal_maps[filename], T, R, model_maps[filename], normalize=False, signal_ratios=signal_ratios, colormap=signal_colors[signal_names[0]])
    #     plt.show(block=False)
    #     raw_input()
    
    average_signal_map = np.nanmean(np.concatenate(np.array(signal_maps.values())[np.newaxis,:,:],axis=0),axis=0)
    average_model_map = np.nanmean(np.concatenate(np.array(model_maps.values())[np.newaxis,:,:],axis=0),axis=0)
    average_max_points = extract_signal_map_maxima(average_signal_map, T, R, average_model_map)
   
    average_signal_maps[d_time] = average_signal_map 
    average_model_maps[d_time] = average_model_map
    
average_maps_file = "/Users/gcerutti/Desktop/signal_maps.pkl"
pickle.dump((average_signal_maps,average_model_maps),open(average_maps_file,"w"))

signal_ratios = np.concatenate([aligned_nuclei_signals[filename].wisp_property('signal',0).values() for filename in filenames])
signal_ratios = dict(zip(range(len(signal_ratios)),signal_ratios))

average_signal_maps,average_model_maps = pickle.load(open(average_maps_file,'r'))


developmental_times = np.arange(18.)/2.

for plastochrone in xrange(10):
    for d_time in developmental_times:
        signal_map = average_signal_maps[d_time+9.0*plastochrone]
        average_signal_maps[d_time+9.0*(plastochrone+1)] = 0.5*np.concatenate([signal_map[69:],signal_map[:69]]) + 0.5*np.concatenate([signal_map[68:],signal_map[:68]]) 
        model_map = average_model_maps[d_time+9.0*plastochrone]
        average_model_maps[d_time+9.0*(plastochrone+1)] = 0.5*np.concatenate([model_map[69:],model_map[:69]]) + 0.5*np.concatenate([model_map[68:],model_map[:68]])
developmental_times = np.sort(average_signal_maps.keys())
    
import vplants.sam4dmaps.sam_map_construction
reload(vplants.sam4dmaps.sam_map_construction)
from vplants.sam4dmaps.sam_map_construction import draw_signal_map

n_points = (len(developmental_times)-1)*5.+1
times_smooth = np.linspace(developmental_times.min(),developmental_times.max(),n_points)

from scipy.interpolate import interp1d, spline, splrep, splev

spline_degree = 1
smooth_factor = 0

transposed_signal_maps = np.array([average_signal_maps[d_time] for d_time in developmental_times]).transpose((1,2,0))
transposed_signal_maps = np.concatenate(transposed_signal_maps)

if spline_degree > 1:
    signal_map_interpolator = [splrep(developmental_times,transposed_map,s=smooth_factor,k=spline_degree) for transposed_map in transposed_signal_maps]
    interpolated_signal_maps = np.array([splev(times_smooth,interpolator,der=0) for interpolator in signal_map_interpolator])
else:
    signal_map_interpolator = [interp1d(developmental_times,transposed_map) for transposed_map in transposed_signal_maps]
    interpolated_signal_maps = np.array([[interpolator(t) for t in times_smooth] for interpolator in signal_map_interpolator])

interpolated_signal_maps = interpolated_signal_maps.reshape(T.shape+(len(times_smooth),)).transpose((2,0,1))
interpolated_signal_maps = dict(zip(times_smooth,interpolated_signal_maps))

transposed_model_maps = np.array([average_model_maps[d_time] for d_time in developmental_times]).transpose((1,2,0))
transposed_model_maps = np.concatenate(transposed_model_maps)

if spline_degree > 1:
    model_map_interpolator = [splrep(developmental_times,transposed_map,s=smooth_factor,k=spline_degree) for transposed_map in transposed_model_maps]
    interpolated_model_maps = np.array([splev(times_smooth,interpolator,der=0) for interpolator in model_map_interpolator])
else:
    model_map_interpolator = [interp1d(developmental_times,transposed_map) for transposed_map in transposed_model_maps]
    interpolated_model_maps = np.array([[interpolator(t) for t in times_smooth] for interpolator in model_map_interpolator])

interpolated_model_maps = interpolated_model_maps.reshape(T.shape+(len(times_smooth),)).transpose((2,0,1))
interpolated_model_maps = dict(zip(times_smooth,interpolated_model_maps))



for d_time in times_smooth:
    figure.clf() 
    average_signal_map = interpolated_signal_maps[d_time]
    average_model_map = interpolated_model_maps[d_time] 
    draw_signal_map(figure, average_signal_map, T, R, average_model_map, ratio_min = 0.2, ratio_max=1.0, n_levels=30, colormap=signal_colors[signal_names[0]])
    plt.show(block=False)
    figure.savefig("/Users/gcerutti/Desktop/SignalMaps/map_"+str(100*d_time)+".jpg")






