import numpy as np
import scipy.ndimage as nd

from openalea.container import array_dict

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function

from vplants.sam4dmaps.sam_model_tools import spherical_parametric_meristem_model, phyllotaxis_based_parametric_meristem_model
from vplants.sam4dmaps.sam_model_tools import meristem_model_density_function, draw_meristem_model_vtk

from copy import deepcopy


def meristem_2d_polar_map(aligned_positions, aligned_meristem_model, epidermis_cells, signal_ratios, reference_dome_apex, r_max=180, cell_radius=5.0, density_k=0.15, normalize=False):
    
    dome_radius = 80.

    epidermis_cell_vectors = aligned_positions.values(epidermis_cells) - reference_dome_apex
    epidermis_cell_distances = np.linalg.norm(epidermis_cell_vectors[:,:2],axis=1)
    epidermis_cell_surface_distances = (2.*dome_radius)*np.arcsin(np.minimum(epidermis_cell_distances/(2.*dome_radius),1.0))
    epidermis_cell_cosines = epidermis_cell_vectors[:,0]/epidermis_cell_distances
    epidermis_cell_sinuses = epidermis_cell_vectors[:,1]/epidermis_cell_distances
    epidermis_cell_angles = np.arctan(epidermis_cell_sinuses/epidermis_cell_cosines)
    epidermis_cell_angles[epidermis_cell_cosines<0] = np.pi + epidermis_cell_angles[epidermis_cell_cosines<0]

    epidermis_cell_x = epidermis_cell_surface_distances*np.cos(epidermis_cell_angles)
    epidermis_cell_y = epidermis_cell_surface_distances*np.sin(epidermis_cell_angles)
    epidermis_cell_z = epidermis_cell_distances*0
    projected_epidermis_points = dict(zip(epidermis_cells,np.transpose([epidermis_cell_x,epidermis_cell_y,epidermis_cell_z])))

    #r = np.linspace(0,160,81)
    r = np.linspace(0,r_max,r_max/2+1)
    #t = np.linspace(0,2*np.pi,360)
    t = np.linspace(0,2*np.pi,180)
    R,T = np.meshgrid(r,t)

    e_x = R*np.cos(T)
    e_y = R*np.sin(T)
    e_z = R*0


    epidermis_nuclei_potential = np.array([nuclei_density_function(dict([(p,projected_epidermis_points[p])]),cell_radius=cell_radius,k=density_k)(e_x,e_y,e_z) for p in epidermis_cells])
    epidermis_nuclei_potential = np.transpose(epidermis_nuclei_potential,(1,2,0))
    epidermis_nuclei_density = np.sum(epidermis_nuclei_potential,axis=2)
    epidermis_nuclei_membership = epidermis_nuclei_potential/epidermis_nuclei_density[...,np.newaxis]

    signal_ratios = array_dict(deepcopy(signal_ratios))
    epidermis_nuclei_ratio = np.sum(epidermis_nuclei_membership*signal_ratios.values(epidermis_cells)[np.newaxis,np.newaxis,:],axis=2)

    if normalize:
        #ratio_min = np.mean(reference_values.values()) - np.sqrt(2.)*np.std(reference_values.values())
        #ratio_min = 0.
        #ratio_max = np.mean(reference_values.values()) + np.sqrt(2.)*np.std(reference_values.values())
        ratio_min = np.mean(signal_ratios.values()) - np.sqrt(4.)*np.std(signal_ratios.values())
        ratio_max = np.mean(signal_ratios.values()) + np.sqrt(4.)*np.std(signal_ratios.values())
        #ratio_step = (ratio_max - ratio_min)/20.
    else:
        ratio_min = 0.
        ratio_max = 1.

    epidermis_nuclei_ratio = (epidermis_nuclei_ratio-ratio_min)/(ratio_max-ratio_min)

    e_x = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.cos(T)
    e_y = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.sin(T)
    epidermis_model_density = np.array([[[ aligned_meristem_model.shape_model_density_function()(e_x[i,j,np.newaxis][:,np.newaxis,np.newaxis]+reference_dome_apex[0],e_y[i,j,np.newaxis][np.newaxis,:,np.newaxis]+reference_dome_apex[1],e_z[i,j,np.newaxis][np.newaxis,np.newaxis,:]+reference_dome_apex[2]-(20.*(k-3)))[0,0,0] for k in np.arange(0,6)] for j in xrange(e_x.shape[1])] for i in xrange(e_x.shape[0])]).max(axis=2)

    epidermis_positions = array_dict(np.transpose([epidermis_cell_angles,epidermis_cell_distances]),epidermis_cells)

    return epidermis_nuclei_ratio, epidermis_model_density, epidermis_positions, T, R 



    # figure = plt.figure(1)
    # figure.clf()    
    # figure.patch.set_facecolor('white')
    # ax = plt.subplot(111, polar=True)

    # #levels = np.arange(0,1.1,0.2)
    # ratio_min = cell_fluorescence_ratios[filename].point_data.values().mean() - np.sqrt(2.)*cell_fluorescence_ratios[filename].point_data.values().std()
    # ratio_max = cell_fluorescence_ratios[filename].point_data.values().mean() + np.sqrt(2.)*cell_fluorescence_ratios[filename].point_data.values().std()
    # ratio_step = (ratio_max - ratio_min)/20.
    # levels = np.arange(ratio_min-ratio_step,ratio_max+ratio_step,ratio_step)
    # levels[0] = 0
    # levels[-1] = 2

    # levels = [-1,0.5,0.8,2]

    # ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap=cmaps[signal_name],alpha=1.0,antialiased=True,vmin=ratio_min,vmax=ratio_max)  
    # #ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap='YWGn',alpha=1.0/len(filenames),antialiased=True,vmin=0.5,vmax=1.0)    
    # levels = [-1.0,0.5]
    # #levels = np.arange(0,5.1,0.05)
    # #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0,antialiased=True)
    # #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0/len(filenames),antialiased=True)
    # #ax.contour(T,R,epidermis_model_density,levels,cmap='RdBu',alpha=1.0,antialiased=True)
    # #ax.contour(T,R,epidermis_model_density,levels,colors='k',alpha=1.0,antialiased=True)
    # ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=0.8,antialiased=True)
    # #ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=2.0/len(filenames),antialiased=True)
    # #ax.scatter(epidermis_cell_angles,epidermis_cell_distances,s=15.0,c=[0.2,0.7,0.1],alpha=0.5)
    # #ax.scatter(epidermis_cell_angles,epidermis_cell_surface_distances,s=15.0,c='w',alpha=0.2)
    # levels = [0.0,1.0]
    # ax.contour(T,R,epidermis_nuclei_density,levels,colors='w',alpha=1,antialiased=True)
    # #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=0.8,antialiased=True)
    # ax.set_rmax(r.max())    
    # ax.set_rmin(0)
    # ax.grid(True)
    # ax.set_yticklabels([])
    # plt.show(block=False)
    # raw_input()

def extract_signal_map_maxima(signal_map, T, R, model_density_map=None):
    from vplants.morpheme.vt_exec.linearfilter import linearfilter
    from vplants.morpheme.vt_exec.regionalext import regionalext
    from vplants.morpheme.vt_exec.connexe import connexe
    from openalea.image.spatial_image import SpatialImage

    map_image = SpatialImage(255.*signal_map[:,:,np.newaxis],resolution=[1,1,1]).astype(np.uint16)
    map_image -= (linearfilter(map_image,param_str_2="-smoothing -sigma 4.0") - 128)
    
    #local_max = regionalext(map_image,param_str_2="-minima -binary -connectivity 26 -h 200")
    h_min = 20
    h_max = 10
    local_max = regionalext(map_image,param_str_2="-maxima -binary -connectivity 26 -h 100 -hmin "+str(h_min)+" -hmax "+str(h_max))
    #local_max = regionalext(map_image,param_str_2="-maxima -binary -connectivity 26 -h 100 -hmax "+str(h_max))
    #local_max = regionalext(map_image,param_str_2="-maxima -binary -connectivity 26 -h 100")
    # print local_max.max()
    
    #h_thr = (h_min+h_max)/2
    #l_thr = 45
    #h_thr = 60
    #max_regions = nd.label(local_max>50)[0]
    max_regions = connexe(local_max, param_str_2="-low-threshold "+str(h_min)+" -high-threshold "+str(h_max)+" -labels -connectivity 26")
    # print len(np.unique(max_regions))-1
    
    if model_density_map is None:
        model_density_map = np.ones_like(signal_map)

    max_regions = max_regions[:,:,0]
    local_max = local_max[:,:,0]
    max_points = np.array(nd.measurements.maximum_position(signal_map,max_regions,index=np.unique(max_regions)[1:])).astype(int)
    if len(max_points>0):
        max_points = np.array([p for p in max_points if model_density_map[tuple(p)]>0.5])
        max_points = np.transpose([T[max_points[:,0],0],R[0,max_points[:,1]]])
    
    map_image = map_image[:,:,0]

    return max_points


def draw_signal_map(figure, signal_map, T, R, model_density_map=None, map_positions=None, max_points=None, signal_ratios=None, normalize=False, ratio_min=None, ratio_max=None, n_levels=20, colormap='Greys'):
        import matplotlib.pyplot as plt

        figure.patch.set_facecolor('white')
        ax = plt.subplot(111, polar=True)
            
        if ratio_min is None:
            if normalize:
                if signal_ratios is not None:
                    ratio_min = np.mean(signal_ratios.values()) - np.sqrt(2.)*np.std(signal_ratios.values())
                else:
                    ratio_min = np.mean(signal_map) - np.sqrt(2.)*np.std(signal_map)
            else:
                ratio_min = 0.0
        if ratio_max is None:
            if normalize:
                if signal_ratios is not None:
                    ratio_max = np.mean(signal_ratios.values()) + np.sqrt(2.)*np.std(signal_ratios.values())
                else:
                    ratio_max = np.mean(signal_map) + np.sqrt(2.)*np.std(signal_map)
            else:
                ratio_max = 1.0

        if n_levels > 0:
            ratio_step = (ratio_max - ratio_min)/float(n_levels)
            ratio_levels = np.arange(ratio_min-ratio_step,ratio_max+2*ratio_step,ratio_step)
            ratio_levels[0] = -0.5
            ratio_levels[-1] = 1.5
            #ratio_levels[0] = 0.
            #ratio_levels[-1] = np.max(reference_values.values())
        density_levels = [-1.0,0.25,0.5]
        
        if n_levels > 0:
            c_map = ax.contourf(T,R,signal_map,ratio_levels,cmap=colormap,alpha=1.0,antialiased=True,vmin=ratio_min,vmax=ratio_max)
            ax.pcolormesh(T,R,signal_map,cmap=colormap,alpha=0.33,antialiased=True,shading='gouraud',vmin=ratio_min,vmax=ratio_max)
        else:
            c_map = ax.pcolormesh(T,R,signal_map,cmap=colormap,alpha=1.0,antialiased=True,shading='gouraud',vmin=ratio_min,vmax=ratio_max)  
        figure.colorbar(c_map)
        if model_density_map is not None: 
            ax.contourf(T,R,model_density_map,density_levels,colors='k',alpha=0.6,antialiased=True)
        if map_positions is not None:
            ax.scatter(map_positions.values()[:,0],map_positions.values()[:,1],s=15.0,c='w',alpha=0.2)
        if max_points is not None and len(max_points)>0:
            ax.scatter(max_points[:,0],max_points[:,1],s=50.0,c='w',alpha=0.9)

        ax.set_rmax(R.max())
        ax.set_rmin(0)
        ax.grid(True)
        ax.set_yticklabels([])



