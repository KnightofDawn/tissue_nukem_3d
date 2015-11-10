import numpy as np
from scipy import ndimage as nd

from openalea.image.spatial_image           import SpatialImage
from openalea.image.serial.all              import imread, imsave

from scipy.ndimage.filters import gaussian_filter

from openalea.deploy.shared_data import shared_data
import vplants.meshing_data

from vplants.meshing.property_topomesh_analysis     import *
from vplants.meshing.intersection_tools     import inside_triangle, intersecting_segment, intersecting_triangle
from vplants.meshing.evaluation_tools       import jaccard_index
from vplants.meshing.tetrahedrization_tools import tetrahedra_dual_topomesh, tetrahedra_from_triangulation, tetra_geometric_features, triangle_geometric_features, triangulated_interface_topomesh
from vplants.meshing.topomesh_from_image    import *

from vplants.meshing.triangular_mesh import TriangularMesh, _points_repr_geom_, _points_repr_vtk_

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface
from vplants.sam4dmaps.sam_model_tools import nuclei_density_function, meristem_model_density_function, draw_meristem_model_vtk

from openalea.container.array_dict             import array_dict
from openalea.container.property_topomesh      import PropertyTopomesh

from time                                   import time, sleep
import csv
import pickle
from copy import deepcopy

import matplotlib
import matplotlib.font_manager as fm
matplotlib.use( "MacOSX" )
import matplotlib.pyplot as plt
from vplants.meshing.cute_plot                              import simple_plot, density_plot, smooth_plot, histo_plot, bar_plot, violin_plot, spider_plot

world.clear()

filenames = []
signal_names = []
#filenames += ["r2DII_1.2_141202_sam03_t04"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam03_t08"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam03_t28"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam03_t32"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam01_t00"]; signal_names += ['DIIV']
filenames += ["r2DII_2.2_141204_sam01_t04"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam01_t08"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam01_t24"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam01_t28"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam07_t00"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam07_t04"]; signal_names += ['DIIV']
#filenames += ["r2DII_2.2_141204_sam07_t08"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t00"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t04"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t08"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t24"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t28"]; signal_names += ['DIIV']
#filenames += ["r2DII_1.2_141202_sam06_t32"]; signal_names += ['DIIV']

#filenames += ["DR5N_5.2_150415_sam01_t00"]; signal_names += ['DR5']
#filenames += ["DR5N_7.1_150415_sam03_t00"]; signal_names += ['DR5']
#filenames += ["DR5N_7.1_150415_sam04_t00"]; signal_names += ['DR5']
#filenames += ["DR5N_7.1_150415_sam07_t00"]; signal_names += ['DR5']

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)

signal_colors = {}
signal_colors['DIIV'] = 'Greens'
signal_colors['DR5'] = 'Blues'

nuclei_images = {}
signal_images = {}

cell_positions = {}
cell_fluorescence_ratios = {}
meristem_models = {}
meristem_surface_ratios = {}

for i_file,filename in enumerate(filenames):
    filetime = filename[-4:]
    sequence_name = filename[:-4]
    signal_name = signal_names[i_file]    
    
    signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
    signal_img = imread(signal_file)
    tag_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
    tag_img = imread(tag_file)
    
    nuclei_images[filename] = deepcopy(tag_img)
    signal_images[filename] = deepcopy(signal_img)
    
    #world.add(signal_img,'signal_image'+filetime,position=np.array([4*i_file-1,1,1])*np.array(tag_img.shape)/2.,resolution=np.array(signal_img.resolution)*np.array([-1.,-1.,-1.]),colormap=signal_colors[signal_name])
    #world.add(tag_img,'nuclei_image'+filetime,position=np.array([4*i_file-1,1,1])*np.array(tag_img.shape)/2.,resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey')
    
    inputfile = dirname+"/nuclei_images/"+filename+"/cells.csv"

    nuclei_data = csv.reader(open(inputfile,"rU"),delimiter=';')
    column_names = np.array(nuclei_data.next())

    nuclei_cells = []
    for data in nuclei_data:
        nuclei_cells.append([float(d) for d in data])
    nuclei_cells = np.array(nuclei_cells)

    points = np.array(nuclei_cells[:,0],int)
    n_points = points.shape[0]  

    points_coordinates = nuclei_cells[:,1:4]

    resolution = np.array(tag_img.resolution)*np.array([-1.,-1.,-1.])
    size = np.array(tag_img.shape)
    
    filtered_signal_img = gaussian_filter(signal_img,sigma=1.5/np.array(tag_img.resolution))
    filtered_tag_img = gaussian_filter(tag_img,sigma=1.5/np.array(tag_img.resolution))

    coords = np.array(points_coordinates/resolution,int)

    points_signal = filtered_signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
    points_tag = filtered_tag_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]

    if signal_name == "DIIV":
        cell_ratio = array_dict(1.0-np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)
    else:
        cell_ratio = array_dict(0.5*(points_signal+0.001)/(points_tag+0.001),points)
    
    positions = array_dict(points_coordinates,points)
    
    detected_cells = TriangularMesh()
    detected_cells.points = positions
    detected_cells.point_data = cell_ratio
    
    cell_fluorescence_ratios[filename] = deepcopy(detected_cells)
    
    meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
    meristem_model_parameters =  pickle.load(open(meristem_model_file,'rb'))
    
    from vplants.sam4dmaps.sam_model_tools import spherical_parametric_meristem_model, phyllotaxis_based_parametric_meristem_model

    meristem_model = ParametricShapeModel()
    meristem_model.parameters = deepcopy(meristem_model_parameters)
    meristem_model.parametric_function = spherical_parametric_meristem_model
    meristem_model.update_shape_model()
    meristem_model.density_function = meristem_model_density_function
    meristem_model.drawing_function = draw_meristem_model_vtk

    meristem_models[filename] = deepcopy(meristem_model)


offset_gaps = [0]

orientation_0 = meristem_models[filenames[0]].parameters['orientation']
orientation = 1.0
golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
golden_angle = 180.*golden_angle/np.pi
    
for i_file,filename in enumerate(filenames[1:]):
    gap_score = {}
    gap_range = np.arange(8)-3 
    for gap in gap_range:
        angle_gap = {}
        distance_gap = {}
        radius_gap = {}
        height_gap = {}
        
        if gap>=0:
            matching_primordia = (np.arange(8-gap)+1)
        else:
            matching_primordia = (np.arange(8-abs(gap))+1-gap)
        
        for p in matching_primordia:
            angle_0 = orientation_0*(meristem_models[filenames[0]].parameters["primordium_"+str(p)+"_angle"]) + meristem_models[filenames[0]].parameters['primordium_offset']*golden_angle
            angle = meristem_models[filename].parameters['orientation']*(meristem_models[filename].parameters["primordium_"+str(p+gap)+"_angle"]) + meristem_models[filename].parameters['primordium_offset']*golden_angle - gap*golden_angle
            angle_gap[p] = np.cos(np.pi*(angle-angle_0)/180.)
            distance_gap[p]  = meristem_models[filename].parameters["primordium_"+str(p+gap)+"_distance"] - meristem_models[filenames[0]].parameters["primordium_"+str(p)+"_distance"]
            radius_gap[p] = meristem_models[filename].parameters["primordium_"+str(p+gap)+"_radius"] - meristem_models[filenames[0]].parameters["primordium_"+str(p)+"_radius"]
            height_gap[p] = meristem_models[filename].parameters["primordium_"+str(p+gap)+"_height"] - meristem_models[filenames[0]].parameters["primordium_"+str(p)+"_height"]
        rotation_0 = (meristem_models[filenames[0]].parameters['initial_angle'] - meristem_models[filenames[0]].parameters['primordium_offset']*golden_angle) %360
        rotation_1 = (meristem_models[filename].parameters['initial_angle'] - meristem_models[filename].parameters['primordium_offset']*golden_angle + gap*golden_angle) %360
        rotation_gap = np.cos(np.pi*(rotation_1 - rotation_0)/180.)
       
        gap_score[gap] = 10.0*np.mean(angle_gap.values())*np.exp(-np.power(np.mean(np.array(distance_gap.values())/6.),2.0))
            
        #print "Gap = ",gap," : r -> ",np.mean(distance_gap.values()),", A -> ",np.mean(angle_gap.values())," (",rotation_0,"->",rotation_1,":",rotation_gap,") [",gap_score[gap],"]"
    offset_gaps.append(gap_range[np.argmax([gap_score[gap] for gap in gap_range])])
print offset_gaps
raw_input()

#offset_gaps[-1] = 2

figure = plt.figure(0)
figure.clf()    
figure.patch.set_facecolor('white')
size_coef = 1.0
for i_file,filename in enumerate(filenames):
    ax = plt.subplot(111, polar=True)
    dome_radius = meristem_models[filename].parameters['dome_radius']
    print dome_radius
    print meristem_models[filename].parameters['primordium_offset']
    plt.scatter([0],[0],s=2.*np.pi*np.power(dome_radius,2.0),facecolors='none', edgecolors=[0.2,0.7,0.1],alpha=1.0/len(filenames),linewidths=5)
    primordia_distances = np.array([meristem_models[filename].parameters["primordium_"+str(p)+'_distance'] for p in np.arange(8)+1])
    primordia_angles = meristem_models[filename].parameters['orientation']*(np.array([meristem_models[filename].parameters["primordium_"+str(p)+'_angle'] for p in np.arange(8)+1]) + (meristem_models[filename].parameters['primordium_offset'] - offset_gaps[i_file])*golden_angle)
    primordia_radiuses = np.array([meristem_models[filename].parameters["primordium_"+str(p)+'_radius'] for p in np.arange(8)+1])
    primordia_colors = np.array([[0.2+0.05*p,0.7-0.025*p,0.1+0.075*p] for p in np.arange(8)+1])
    plt.scatter(np.pi*primordia_angles/180.,primordia_distances,s=size_coef*np.pi*np.power(primordia_radiuses,2.0),c=primordia_colors,alpha=1.0/len(filenames))
    ax.set_rmax(160)
    ax.grid(True)
    ax.set_yticklabels([])
    plt.show(block=False)
raw_input()
    

aligned_positions = {}
aligned_meristem_models = {}
aligned_nuclei_images = {}
aligned_signal_images = {}
    
for i_file,filename in enumerate(filenames):
    resolution = np.array(nuclei_images[filename].resolution)*np.array([-1.,-1.,-1.])
    size = np.array(nuclei_images[filename].shape)
    reference_dome_apex = np.array([(size*resolution/2.)[0],(size*resolution/2.)[0],0])
    
    orientation = meristem_models[filename].parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi

    cell_points = array_dict(cell_fluorescence_ratios[filename].points)
    dome_apex = np.array([meristem_models[filename].parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_phi = np.pi*meristem_models[filename].parameters['dome_phi']/180.
    dome_psi = np.pi*meristem_models[filename].parameters['dome_psi']/180.
    initial_angle = meristem_models[filename].parameters['initial_angle']
    initial_angle -= meristem_models[filename].parameters['primordium_offset']*golden_angle
    initial_angle += offset_gaps[i_file]*golden_angle 
    #initial_angle *= orientation
    dome_theta = np.pi*initial_angle/180.
    
    world.add(nuclei_images[filename],'nuclei_image',position=np.array([-1,1,1])*size/2.,resolution=resolution,colormap='invert_grey')

    #aligned_nuclei_image = np.zeros(tuple(3*size),nuclei_images[filename].dtype)
    #dome_apex_coordinates = np.array(dome_apex/resolution,int)
    #aligned_origin = np.array(aligned_nuclei_image.shape)/2.-dome_apex_coordinates
    #aligned_nuclei_image[aligned_origin[0]:aligned_origin[0]+size[0],aligned_origin[1]:aligned_origin[1]+size[1],aligned_origin[2]:aligned_origin[2]+size[2]] = deepcopy(nuclei_images[filename])
    aligned_nuclei_image = deepcopy(nuclei_images[filename])
    aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_phi/np.pi,axes=[0,2],reshape=False)
    aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_psi/np.pi,axes=[1,2],reshape=False)
    aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_theta/np.pi,axes=[0,1],reshape=False)
    aligned_nuclei_images[filename] = deepcopy(aligned_nuclei_image)
    
    aligned_signal_image = deepcopy(signal_images[filename])
    aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_phi/np.pi,axes=[0,2],reshape=False)
    aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_psi/np.pi,axes=[1,2],reshape=False)
    aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_theta/np.pi,axes=[0,1],reshape=False)
    aligned_signal_images[filename] = deepcopy(aligned_signal_image)
    
    rotation_phi = np.array([[np.cos(dome_phi),0,np.sin(dome_phi)],[0,1,0],[-np.sin(dome_phi),0,np.cos(dome_phi)]])
    rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),np.sin(dome_psi)],[0,-np.sin(dome_psi),np.cos(dome_psi)]])
    rotation_theta = np.array([[np.cos(dome_theta),np.sin(dome_theta),0],[-np.sin(dome_theta),np.cos(dome_theta),0],[0,0,1]])
    print "Rotation Theta : ",(initial_angle%360)
    
    relative_points = (cell_points.values()-dome_apex[np.newaxis,:])
    relative_points = np.einsum('...ij,...j->...i',rotation_phi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_psi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_theta,relative_points)
    relative_points = relative_points * np.array([1,orientation,1])[np.newaxis,:]
    
    aligned_cell_points = array_dict(reference_dome_apex + relative_points,cell_points.keys())
    aligned_positions[filename] = deepcopy(aligned_cell_points)
    
    golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi
    
    aligned_meristem_model = ParametricShapeModel()
    aligned_meristem_model.parameters = deepcopy(meristem_models[filename].parameters)
    aligned_meristem_model.parameters['orientation'] = 1.0
    aligned_meristem_model.parameters['dome_apex_x'] = reference_dome_apex[0]
    aligned_meristem_model.parameters['dome_apex_y'] = reference_dome_apex[1]
    aligned_meristem_model.parameters['dome_apex_z'] = reference_dome_apex[2]
    aligned_meristem_model.parameters['dome_phi'] = 0
    aligned_meristem_model.parameters['dome_psi'] = 0
    aligned_meristem_model.parameters['initial_angle'] = 0
    #aligned_meristem_model.parameters['initial_angle'] += meristem_models[filename].parameters['primordium_offset']*golden_angle
    #aligned_meristem_model.parameters['initial_angle'] -= offset_gaps[i_file]*golden_angle
    for p in aligned_meristem_model.parameters.keys():
        if ('primordium' in p) and ('angle' in p):
             aligned_meristem_model.parameters[p] += meristem_models[filename].parameters['primordium_offset']*golden_angle
             aligned_meristem_model.parameters[p] -= offset_gaps[i_file]*golden_angle
             aligned_meristem_model.parameters[p] *= meristem_models[filename].parameters['orientation']
    aligned_meristem_model.parametric_function = spherical_parametric_meristem_model
    aligned_meristem_model.update_shape_model()
    aligned_meristem_model.density_function = meristem_model_density_function
    aligned_meristem_model.drawing_function = draw_meristem_model_vtk
    aligned_meristem_models[filename] = deepcopy(aligned_meristem_model)
    
    detected_cells = TriangularMesh()
    detected_cells.points = aligned_positions[filename]
    world.add(detected_cells,"detected_cells",position=resolution*size/2.,colormap='grey',intensity_range=(-1,0))
    
    world.add(aligned_meristem_models[filename],'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
    #raw_input()
    
import vplants.sam4dmaps.parametric_shape
reload(vplants.sam4dmaps.parametric_shape)
from vplants.sam4dmaps.parametric_shape import implicit_surface

epidermis_cells = {}

for i_file,filename in enumerate(filenames):
    resolution = np.array(nuclei_images[filename].resolution)*np.array([-1.,-1.,-1.])
    size = np.array(nuclei_images[filename].shape)
    
    signal_name = signal_names[i_file] 
    
    grid_resolution = resolution*[4,4,2]
    x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:2*grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:2*grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:2*grid_resolution[2]]
    grid_size = 1.5*size

    cell_radius = 5.0
    density_k = 2.0
    nuclei_potential = np.array([nuclei_density_function(dict([(p,aligned_positions[filename][p])]),cell_radius=cell_radius,k=density_k)(x,y,z) for p in aligned_positions[filename].keys()])
    nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    nuclei_density = np.sum(nuclei_potential,axis=3)
    
    model_density = aligned_meristem_models[filename].shape_model_density_function()(x,y,z)
    
    surface_points,surface_triangles = implicit_surface(nuclei_density,grid_size,resolution)
    #surface_points,surface_triangles = implicit_surface(model_density,grid_size,resolution)
    
    surface_mesh = TriangularMesh()
    surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
    surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
    #surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    world.add(surface_mesh,'nuclei_implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=0.5)
    
    surface_vertex_cell_membership = array_dict(np.transpose([nuclei_density_function(dict([(p,aligned_positions[filename][p]- size*resolution/2.)]),cell_radius=cell_radius,k=density_k)(surface_points[:,0],
                                                                                                                                    surface_points[:,1],
                                                                                                                                    surface_points[:,2]) for p in aligned_positions[filename].keys()]),keys=np.arange(len(surface_points)))
    surface_vertex_cell = array_dict([aligned_positions[filename].keys()[np.argmax(surface_vertex_cell_membership[p])] for p in np.arange(len(surface_points))],np.arange(len(surface_points)))
    
    epidermis_cells[filename] = np.unique(surface_vertex_cell.values()[surface_points[:,2] > (2.*size*resolution/5.)[2]])
    #epidermis_cells[filename] = np.unique(surface_vertex_cell.values()[surface_points[:,2] > (size*resolution/2.)[2]])
    
    epidermis_detected_cells = TriangularMesh()
    epidermis_detected_cells.points = dict(zip(epidermis_cells[filename],aligned_positions[filename].values(epidermis_cells[filename])))
    epidermis_detected_cells.point_data = dict(zip(epidermis_cells[filename],cell_fluorescence_ratios[filename].point_data.values(epidermis_cells[filename])))
    world.add(epidermis_detected_cells,"epidermis_detected_cells",position=resolution*size/2.,colormap=signal_colors[signal_name],point_radius=2,intensity_range=(0,1))
    
    detected_cells = TriangularMesh()
    detected_cells.points = aligned_positions[filename]
    world.add(detected_cells,"detected_cells",position=resolution*size/2.,colormap='grey',intensity_range=(-1,0))
    
    world.add(aligned_meristem_models[filename],'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))

    raw_input()

figure = plt.figure(1)
figure.clf()    
figure.patch.set_facecolor('white')

from matplotlib.colors import LinearSegmentedColormap
cdict = {'red':    ((0.0, 1.0, 1.0),
                    (0.3, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),
         'green':  ((0.0, 0.9, 0.9),
                    (0.3, 1.0, 1.0),
                    (1.0, 0.6, 0.6)),
         'blue':   ((0.0,  0.7, 0.7),
                    (0.3,  1.0, 1.0),
                    (1.0,  0.0, 0.0))}
plt.register_cmap(name='YWGn', data=cdict)

cmaps = {}
cmaps['DIIV'] = 'RdYlGn'
cmaps['DR5'] = 'Blues'

for i_file,filename in enumerate(filenames):
    resolution = np.array(nuclei_images[filename].resolution)*np.array([-1.,-1.,-1.])
    size = np.array(nuclei_images[filename].shape)
    dome_radius = 80.
    
    epidermis_cell_vectors = aligned_positions[filename].values(epidermis_cells[filename]) - reference_dome_apex
    epidermis_cell_distances = np.linalg.norm(epidermis_cell_vectors[:,:2],axis=1)
    epidermis_cell_surface_distances = (2.*dome_radius)*np.arcsin(np.minimum(epidermis_cell_distances/(2.*dome_radius),1.0))
    epidermis_cell_cosines = epidermis_cell_vectors[:,0]/epidermis_cell_distances
    epidermis_cell_sinuses = epidermis_cell_vectors[:,1]/epidermis_cell_distances
    epidermis_cell_angles = np.arctan(epidermis_cell_sinuses/epidermis_cell_cosines)
    epidermis_cell_angles[epidermis_cell_cosines<0] = np.pi + epidermis_cell_angles[epidermis_cell_cosines<0]
    
    #epidermis_cell_x = epidermis_cell_distances*np.cos(epidermis_cell_angles)
    #epidermis_cell_y = epidermis_cell_distances*np.sin(epidermis_cell_angles)
    epidermis_cell_x = epidermis_cell_surface_distances*np.cos(epidermis_cell_angles)
    epidermis_cell_y = epidermis_cell_surface_distances*np.sin(epidermis_cell_angles)
    epidermis_cell_z = epidermis_cell_distances*0
    projected_epidermis_points = dict(zip(epidermis_cells[filename],np.transpose([epidermis_cell_x,epidermis_cell_y,epidermis_cell_z])))
    
    #r = np.linspace(0,160,81)
    r = np.linspace(0,180,91)
    #t = np.linspace(0,2*np.pi,360)
    t = np.linspace(0,2*np.pi,180)
    R,T = np.meshgrid(r,t)
    
    e_x = R*np.cos(T)
    e_y = R*np.sin(T)
    e_z = R*0
    
    cell_radius = 5.0
    density_k = 0.15
    #density_k = 0.33
    
    epidermis_nuclei_potential = np.array([nuclei_density_function(dict([(p,projected_epidermis_points[p])]),cell_radius=cell_radius,k=density_k)(e_x,e_y,e_z) for p in epidermis_cells[filename]])
    epidermis_nuclei_potential = np.transpose(epidermis_nuclei_potential,(1,2,0))
    epidermis_nuclei_density = np.sum(epidermis_nuclei_potential,axis=2)
    
    epidermis_nuclei_membership = epidermis_nuclei_potential/epidermis_nuclei_density[...,np.newaxis]
    epidermis_nuclei_ratio = np.sum(epidermis_nuclei_membership*cell_fluorescence_ratios[filename].point_data.values(epidermis_cells[filename])[np.newaxis,np.newaxis,:],axis=2)
    
    e_x = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.cos(T)
    e_y = (2.*dome_radius)*np.sin(R/(2.*dome_radius))*np.sin(T)
    epidermis_model_density = np.array([[[ aligned_meristem_models[filename].shape_model_density_function()(e_x[i,j,np.newaxis][:,np.newaxis,np.newaxis]+reference_dome_apex[0],e_y[i,j,np.newaxis][np.newaxis,:,np.newaxis]+reference_dome_apex[1],e_z[i,j,np.newaxis][np.newaxis,np.newaxis,:]+(k*size*resolution/2.)[2])[0,0,0] for k in np.arange(1,4)] for j in xrange(e_x.shape[1])] for i in xrange(e_x.shape[0])]).max(axis=2)
    
    signal_name = signal_names[i_file]   
    
    figure = plt.figure(1)
    figure.clf()    
    figure.patch.set_facecolor('white')
    ax = plt.subplot(111, polar=True)
    
    #levels = np.arange(0,1.1,0.2)
    ratio_min = cell_fluorescence_ratios[filename].point_data.values().mean() - np.sqrt(2.)*cell_fluorescence_ratios[filename].point_data.values().std()
    ratio_max = cell_fluorescence_ratios[filename].point_data.values().mean() + np.sqrt(2.)*cell_fluorescence_ratios[filename].point_data.values().std()
    ratio_step = (ratio_max - ratio_min)/20.
    levels = np.arange(ratio_min-ratio_step,ratio_max+ratio_step,ratio_step)
    levels[0] = 0
    levels[-1] = 2
    
    levels = [-1,0.5,0.8,2]
    
    ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap=cmaps[signal_name],alpha=1.0,antialiased=True,vmin=ratio_min,vmax=ratio_max)  
    #ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap='YWGn',alpha=1.0/len(filenames),antialiased=True,vmin=0.5,vmax=1.0)    
    levels = [-1.0,0.5]
    #levels = np.arange(0,5.1,0.05)
    #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0,antialiased=True)
    #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0/len(filenames),antialiased=True)
    #ax.contour(T,R,epidermis_model_density,levels,cmap='RdBu',alpha=1.0,antialiased=True)
    #ax.contour(T,R,epidermis_model_density,levels,colors='k',alpha=1.0,antialiased=True)
    ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=0.8,antialiased=True)
    #ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=2.0/len(filenames),antialiased=True)
    #ax.scatter(epidermis_cell_angles,epidermis_cell_distances,s=15.0,c=[0.2,0.7,0.1],alpha=0.5)
    #ax.scatter(epidermis_cell_angles,epidermis_cell_surface_distances,s=15.0,c='w',alpha=0.2)
    levels = [0.0,1.0]
    ax.contour(T,R,epidermis_nuclei_density,levels,colors='w',alpha=1,antialiased=True)
    #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=0.8,antialiased=True)
    ax.set_rmax(r.max())    
    ax.set_rmin(0)
    ax.grid(True)
    ax.set_yticklabels([])
    plt.show(block=False)
    
    world.add(aligned_nuclei_images[filename],'nuclei_image',position=np.array(tag_img.shape)/2.,resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey')   
    world.add(aligned_signal_images[filename],'signal_image',position=np.array(tag_img.shape)/2.,resolution=np.array(signal_img.resolution)*np.array([-1.,-1.,-1.]),colormap=signal_colors[signal_name])

    epidermis_detected_cells = TriangularMesh()
    epidermis_detected_cells.points = dict(zip(epidermis_cells[filename],aligned_positions[filename].values(epidermis_cells[filename])))
    epidermis_detected_cells.point_data = dict(zip(epidermis_cells[filename],cell_fluorescence_ratios[filename].point_data.values(epidermis_cells[filename])))
    world.add(epidermis_detected_cells,"epidermis_detected_cells",position=np.array([3,1,1])*resolution*size/2.,colormap=signal_colors[signal_name],point_radius=2,intensity_range=(0,1))
    
    detected_cells = TriangularMesh()
    detected_cells.points = aligned_positions[filename]
    world.add(detected_cells,"detected_cells",position=np.array([3,1,1])*resolution*size/2.,colormap='grey',intensity_range=(-1,0))
    
    world.add(aligned_meristem_models[filename],'meristem_model',position=np.array([3,1,1])*resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))

    raw_input()
    

figure = plt.figure(2)
figure.clf()    
figure.patch.set_facecolor('white')

for i_file,filename in enumerate(filenames):
    cell_vectors = aligned_positions[filename].values() - reference_dome_apex
    cell_distances = np.linalg.norm(cell_vectors[:,:2],axis=1) 
    print cell_distances.min()
    cell_cosines = cell_vectors[:,0]/cell_distances
    cell_sinuses = cell_vectors[:,1]/cell_distances
    cell_angles = np.arctan(cell_sinuses/cell_cosines)
    cell_angles[cell_cosines<0] = np.pi + cell_angles[cell_cosines<0]
    
    cell_ratios = cell_fluorescence_ratios[filename].point_data.values()
    cell_colors = cell_ratios[:,np.newaxis]*np.array([0.2,0.7,0.1]) + (1.0-cell_ratios)[:,np.newaxis]*np.array([1.0,0.9,0.8])
    
    ax = plt.subplot(111, polar=True)
    plt.scatter(cell_angles,cell_distances,s=20.0,c=cell_ratios,edgecolor='none',alpha=0.5,cmap='RdYlGn')
    ax.set_rmax(160)
    ax.grid(True)
    ax.set_yticklabels([])
    plt.show(block=False)
    
raw_input()



