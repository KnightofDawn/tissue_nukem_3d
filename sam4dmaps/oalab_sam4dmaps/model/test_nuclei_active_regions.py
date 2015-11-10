import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq                       import kmeans, vq

from openalea.image.spatial_image           import SpatialImage
from openalea.image.serial.all              import imread, imsave

from openalea.container import PropertyTopomesh
from vplants.meshing.property_topomesh_analysis import *

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
from openalea.deploy.shared_data import shared_data

from vplants.meshing.triangular_mesh import TriangularMesh, topomesh_to_triangular_mesh
from vplants.meshing.array_tools import array_unique

from openalea.container.array_dict             import array_dict

from sys                                    import argv
from time                                   import time, sleep
import csv

import pickle
from copy import deepcopy

#filename = "r2DII_3.2_141127_sam08_t24"
#filename = "DR5N_5.2_150415_sam01_t00"
#filename = "r2DII_1.2_141202_sam03_t32"
#filename = "r2DII_1.2_141202_sam03_t28"
filename = "r2DII_1.2_141202_sam06_t04"
#previous_filename = "r2DII_2.2_141204_sam01_t28"
#previous_filename = "r2DII_1.2_141202_sam03_t28"
#previous_filename = "r2DII_2.2_141204_sam07_t04"
#previous_filename = "r2DII_3.2_141127_sam08_t00"

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)
signal_name = 'DIIV'
#signal_name = 'DR5'

signal_colors = {}
signal_colors['DIIV'] = 'RdYlGn'
signal_colors['DR5'] = 'Blues'

reference_img = None
signal_img = None

world.clear()
from openalea.core.service.plugin import plugin_instance
viewer = plugin_instance('oalab.applet','TissueViewer').vtk

omega_energies = {}
omega_energies['intensity'] = 1.0
omega_energies['gradient'] = 100.0

signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
signal_img = imread(signal_file)
    
reference_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
reference_img = imread(reference_file)

signal_tif = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".tif"
imsave(signal_tif,signal_img)
reference_tif = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.tif"
imsave(reference_tif,reference_img)

size = np.array(reference_img.shape)
resolution = np.array(reference_img.resolution)*np.array([-1.,-1.,-1.])
    
world.add(reference_img,'reference_image',position=size/2.,resolution=resolution,colormap='invert_grey')
world.add(signal_img,signal_name+"_image",position=size/2.,resolution=resolution,colormap=signal_colors[signal_name],intensity_range=(4000,40000))

if omega_energies.has_key('gradient'):
    from scipy.ndimage.filters import gaussian_gradient_magnitude
    gradient = gaussian_gradient_magnitude(np.array(reference_img,np.float64),sigma=0.5/np.array(reference_img.resolution))
    gradient_img = SpatialImage(np.array(gradient,np.uint16),resolution=reference_img.resolution)
    world.add(gradient_img,"gradient_image",position=size/2.,resolution=resolution,colormap='grey',cut_planes=True,volume=False)

inputfile = dirname+"/nuclei_images/"+filename+"/cells.csv"

nuclei_data = csv.reader(open(inputfile,"rU"),delimiter=';')
column_names = np.array(nuclei_data.next())

nuclei_cells = []
# while True:
for data in nuclei_data:
	# print data
	nuclei_cells.append([float(d) for d in data])
nuclei_cells = np.array(nuclei_cells)

points = np.array(nuclei_cells[:,0],int)+2
n_points = points.shape[0]  

points_coordinates = nuclei_cells[:,1:4]
positions = array_dict(points_coordinates,points)

from scipy.ndimage.filters import gaussian_filter
filtered_signal_img = gaussian_filter(signal_img,sigma=1.5/np.array(reference_img.resolution))
filtered_reference_img = gaussian_filter(reference_img,sigma=1.5/np.array(reference_img.resolution))

coords = np.array(points_coordinates/resolution,int)

points_signal = filtered_signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
points_tag = filtered_reference_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]

if signal_name == 'DIIV':
    cell_ratio = array_dict(1.0-np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)
else:
    cell_ratio = array_dict(np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)

detected_cells = TriangularMesh()
detected_cells.points = positions
#world.add(detected_cells,'detected_cells',position=size*resolution/2.,colormap='grey',intensity_range=(-1,0))
#raw_input()


seed_img = np.ones_like(reference_img,np.uint16)
for p in positions.keys():
    point_radius = 1.0
    image_neighborhood = np.array(np.ceil(point_radius/np.array(reference_img.resolution)),int)
    neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
    neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + np.array(positions[p]/resolution,int)
    neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),np.array(reference_img.shape)-1)
    neighborhood_coords = array_unique(neighborhood_coords)
    
    neighborhood_distance = np.linalg.norm(neighborhood_coords*resolution - positions[p],axis=1)
    neighborhood_coords = neighborhood_coords[neighborhood_distance<=point_radius]
    neighborhood_coords = tuple(np.transpose(neighborhood_coords))
    
    #coords = tuple(np.array(positions[p]/resolution,int))
    seed_img[neighborhood_coords] = p
    
#world.add(seed_img,'active_regions_seeds',colormap='invert_grey',resolution=resolution,position=size/2.,intensity_range=(1,2))
world.add(seed_img,'active_regions_seeds',colormap='glasbey',resolution=resolution,position=size/2.,alphamap='constant',volume=False,cut_planes=True)
print np.sum(seed_img>1)
raw_input()

def inside_image(points,img):
    inside = np.ones(len(points),bool)
    inside = inside & (np.array(points)[:,0] >= 0)
    inside = inside & (np.array(points)[:,1] >= 0)
    inside = inside & (np.array(points)[:,2] >= 0)
    inside = inside & (np.array(points)[:,0] < img.shape[0])
    inside = inside & (np.array(points)[:,1] < img.shape[1])
    inside = inside & (np.array(points)[:,2] < img.shape[2])
    return inside


neighborhood_6 = [[0,0,-1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]]
neighborhood_18 = [[0,0,-1],[1,0,-1],[0,1,-1],[-1,0,-1],[0,-1,-1],[1,0,0],[1,1,0],[0,1,0],[-1,1,0],[-1,0,0],[-1,-1,0],[0,-1,0],[1,-1,0],[0,0,1],[1,0,1],[0,1,1],[-1,0,1],[0,-1,1],]

i_min = 20000.

probability_coef = 1.0

omega_energies = {}
omega_energies['intensity'] = 1.0
omega_energies['gradient'] = 1.5
omega_energies['smoothness'] = 10000.0

for iteration in xrange(10):
    
    candidate_points = []
    candidate_labels = []
    candidate_inside = []
    candidate_current_labels = []
    candidate_potential_labels = []
    
    outer_margin = nd.binary_dilation(seed_img>1) - (seed_img>1)
    outer_margin_points = list(np.transpose(np.where(outer_margin)))
    
    neighbor_labels = []
    for n in neighborhood_18:
        neighbor_points = np.array(outer_margin_points) + np.array(n)
        neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),np.array(reference_img.shape)-1)
        neighbor_labels += [seed_img[tuple(np.transpose(neighbor_points))]]
    
    neighbor_labels = np.transpose(neighbor_labels)
    neighbor_labels = [ list(set(l).difference({1})) for l in neighbor_labels]
    outer_margin_labels = [l[0] for l in neighbor_labels]
    outer_margin_current_labels = [1 for l in neighbor_labels]
    outer_margin_potential_labels = [l[0] for l in neighbor_labels]
    
    candidate_points += outer_margin_points
    candidate_labels += outer_margin_labels
    candidate_inside += [False for p in outer_margin_labels]
    candidate_current_labels += outer_margin_current_labels
    candidate_potential_labels += outer_margin_potential_labels
    
    inner_margin = (seed_img>1) - nd.binary_erosion(seed_img>1) 
    inner_margin_points = list(np.transpose(np.where(inner_margin)))
    inner_margin_labels = list(seed_img[inner_margin])
    inner_margin_current_labels = list(seed_img[inner_margin])
    inner_margin_potential_labels = [1 for l in inner_margin_current_labels]
    
    candidate_points += inner_margin_points
    candidate_labels += inner_margin_labels
    candidate_inside += [True for p in inner_margin_labels]
    candidate_current_labels += inner_margin_current_labels
    candidate_potential_labels += inner_margin_potential_labels
    
    region_points = list(np.transpose(np.where(seed_img>1)))
    
    neighbor_labels = []
    for n in neighborhood_18:
        neighbor_points = np.array(region_points) + np.array(n)
        neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),np.array(reference_img.shape)-1)
        neighbor_labels += [seed_img[tuple(np.transpose(neighbor_points))]]
    neighbor_labels = np.transpose(neighbor_labels)
    neighbor_labels = [ list(set(l).difference({1})) for l in neighbor_labels]
    
    neighbor_number = np.array(map(len,neighbor_labels))
    frontier_points = list(np.array(region_points)[neighbor_number>1])
    
    if len(frontier_points)>0:
        frontier_neighbor_labels = np.array(neighbor_labels)[neighbor_number>1]
        frontier_labels =  [list(set(l).difference({c}))[0] for l,c in zip(frontier_neighbor_labels,seed_img[tuple(np.transpose(frontier_points))])]
        frontier_current_labels = list(seed_img[tuple(np.transpose(frontier_points))])
        frontier_potential_labels = frontier_labels
        
        candidate_points += frontier_points
        candidate_labels += frontier_labels
        candidate_inside += [True for p in frontier_labels]
        candidate_current_labels += frontier_current_labels
        candidate_potential_labels += frontier_potential_labels
    
    # region_points = list(np.transpose(np.where(seed_img>1)))
    # region_labels = list(seed_img[np.where(seed_img>1)])
    # region_inside = [True for p in region_labels]
    
    # candidate_points = deepcopy(region_points)
    # candidate_labels = deepcopy(region_labels)
    # candidate_inside = deepcopy(region_inside)
    
    # for n in neighborhood_18:
    #     neighbor_points = np.array(region_points) + np.array(n)
    #     neighbor_labels = seed_img[np.where(seed_img>1)]
    #     outside_points = True - inside_image(neighbor_points,reference_img)
    #     neighbor_labels = np.delete(neighbor_labels,np.where(outside_points)[0],0)
    #     neighbor_points = np.delete(neighbor_points,np.where(outside_points)[0],0)
    #     candidate_points += list(neighbor_points)
    #     candidate_labels += list(neighbor_labels)
    #     candidate_inside += [False for p in neighbor_labels]
    
    candidate_points = np.array(candidate_points)
    candidate_labels = np.array(candidate_labels)
    candidate_inside = np.array(candidate_inside)
    candidate_current_labels = np.array(candidate_current_labels)
    candidate_potential_labels = np.array(candidate_potential_labels)
    
    speeds = np.zeros(len(candidate_labels),float)
    
    if omega_energies.has_key('intensity'):
        #intensity_speeds = i_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float)
        intensity_speeds = np.zeros(len(candidate_points)) 
        intensity_speeds -= (candidate_current_labels>1)*(i_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float))
        intensity_speeds += (candidate_potential_labels>1)*(i_min - np.array(reference_img[tuple(np.transpose(candidate_points))],float))
        speeds += omega_energies['intensity']*intensity_speeds
    
    if omega_energies.has_key('gradient'):
        #gradient_speeds = -np.array(gradient[tuple(np.transpose(candidate_points))],float)
        gradient_speeds = np.zeros(len(candidate_points)) 
        gradient_speeds -= (candidate_current_labels>1)*(-np.array(gradient[tuple(np.transpose(candidate_points))],float))
        gradient_speeds += (candidate_potential_labels>1)*(-np.array(gradient[tuple(np.transpose(candidate_points))],float))
        speeds += omega_energies['gradient']*gradient_speeds
        
    if omega_energies.has_key('smoothness'):
        neighbor_labels = []
        for n in neighborhood_6:
            neighbor_points = np.array(candidate_points) + np.array(n)
            neighbor_points = np.minimum(np.maximum(neighbor_points,np.array([0,0,0])),np.array(reference_img.shape)-1)
            neighbor_labels += [seed_img[tuple(np.transpose(neighbor_points))]]
        neighbor_labels = np.transpose(neighbor_labels)
        smoothness_speeds = 1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_labels)])
        
        smoothness_speeds = np.zeros(len(candidate_points)) 
        smoothness_speeds -= (candidate_current_labels>1)*(1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_current_labels)]))
        smoothness_speeds += (candidate_potential_labels>1)*(1.0 - 2.0*np.array([np.sum(l == r)/float(len(l)) for l,r in zip(neighbor_labels,candidate_potential_labels)]))
        
        speeds += omega_energies['smoothness']*smoothness_speeds
    
    #voxels_to_remove = tuple(np.transpose(candidate_points[(candidate_inside) & (speeds>0)]))
    #seed_img[voxels_to_remove] = 1
    
    #probabilities = np.exp(np.minimum(-speeds/probability_coef-1,0))
    #add_decision = (np.random.rand(len(speeds))<=probabilities)
    #voxels_to_add = tuple(np.transpose(candidate_points[(True-candidate_inside) & (speeds<0) & (add_decision)]))
    #labels_to_add = candidate_labels[(True-candidate_inside) & (speeds<0) & (add_decision)]
    #seed_img[voxels_to_add] = labels_to_add
    
    change_probabilities = np.exp(np.minimum(-speeds/probability_coef-1,0))
    change_decision = (np.random.rand(len(speeds))<=change_probabilities)
    
    voxels_to_change = tuple(np.transpose(candidate_points[change_decision]))
    labels_to_change = candidate_potential_labels[change_decision]
    seed_img[voxels_to_change] = labels_to_change
    
    world.add(seed_img,'active_regions',colormap='invert_grey',resolution=resolution,position=size/2.,intensity_range=(1,2))
    print np.sum(seed_img>1)
    
world.add(seed_img,'active_regions',colormap='glasbey',resolution=resolution,position=size/2.,alphamap='constant')
raw_input()

segmented_nuclei_img = SpatialImage(seed_img,resolution=reference_img.resolution)

img_graph = graph_from_image(segmented_nuclei_img, spatio_temporal_properties=['volume','barycenter'],background=1,ignore_cells_at_stack_margins=False,property_as_real=True)

segmented_positions = array_dict([-img_graph.vertex_property('barycenter')[p] for p in img_graph.vertices()],list(img_graph.vertices()))


detected_cells.points = array_dict([-positions[c] for c in positions.keys()],positions.keys())
detected_cells.point_data = cell_ratio
world.add(detected_cells,'fluorescence_ratios',position=-np.array([3,1,1])*size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0,1),point_radius=2)
raw_input()

#world.add(detected_cells,'detected_cells',position=-size*resolution/2.,colormap='glasbey')
#world.add(updated_cells,'upddted_cells',position=-size*resolution/2.,colormap='glasbey')
#raw_input()

segmented_cell_reference = nd.sum(reference_img,segmented_nuclei_img,index=segmented_positions.keys())
segmented_cell_signal = nd.sum(signal_img,segmented_nuclei_img,index=segmented_positions.keys())
segmented_cell_ratios = array_dict(1.0-np.minimum((segmented_cell_signal)/(segmented_cell_reference),1.0),segmented_positions.keys())

updated_cells = TriangularMesh()
updated_cells.points = segmented_positions
updated_cells.point_data = segmented_cell_ratios
world.add(updated_cells,'segmented_fluorescence_ratios',position=size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0,1),point_radius=2)
raw_input()

print "Ratio mean difference : ",np.power(np.power(np.abs(cell_ratio.values(segmented_cell_ratios.keys()) - segmented_cell_ratios.values(segmented_cell_ratios.keys())),2.0).mean(),0.5)
print "Position mean difference : ",np.power(np.power(np.linalg.norm(positions.values(segmented_cell_ratios.keys()) - segmented_positions.values(segmented_cell_ratios.keys()),axis=1),2.0).mean(),0.5)

segmented_cell_ratios[1] = 0.01
cell_ratio_image = SpatialImage(np.array(100.*segmented_cell_ratios.values(segmented_nuclei_img),np.uint8),resolution=reference_img.resolution)
world.add(cell_ratio_image,'segmented_nuclei_ratios',colormap=signal_colors[signal_name],resolution=resolution,position=size/2.,intensity_range=(0,100),alphamap='constant')
del segmented_cell_ratios[1]

#image_ratio_image = np.array(100.*(1.0 - np.minimum(np.array(signal_img,float)/np.array(reference_img,float),1.0)),np.uint8)
#image_ratio_image[segmented_nuclei_img==1] = 1 
#world.add(image_ratio_image,'segmented_image_ratios',colormap=signal_colors[signal_name],resolution=reference_img.resolution,position=np.array([-1,1,1])*size/2.,alphamap='constant')

#image_cell_ratios = array_dict(1.0-np.minimum(nd.mean(np.array(signal_img,float)/np.array(reference_img,float),segmented_nuclei_img,index=segmented_positions.keys()),1.0),segmented_positions.keys())
#del segmented_cell_ratios[1]
#np.power(np.power(np.abs(cell_ratio.values() - image_cell_ratios.values()),2.0).mean(),0.5)
#np.power(np.power(np.abs(cell_ratio.values() - segmented_cell_ratios.values()),2.0).mean(),0.5)
#np.power(np.power(np.abs(segmented_cell_ratios.values() - image_cell_ratios.values()),1.0).mean(),1.0)
#raw_input()


grid_resolution = resolution*[8,8,4]
x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:grid_resolution[2]]
grid_size = 1.5*size

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function

nuclei_potential = np.array([nuclei_density_function(dict([(p,segmented_positions[p])]),cell_radius=5,k=1.0)(x,y,z) for p in segmented_positions.keys()])
nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
nuclei_density = np.sum(nuclei_potential,axis=3)

import vplants.sam4dmaps.parametric_shape
reload(vplants.sam4dmaps.parametric_shape)
from vplants.sam4dmaps.parametric_shape import implicit_surface, implicit_surface_topomesh

surface_topomesh = implicit_surface_topomesh(nuclei_density,grid_size,resolution)

surface_topomesh.update_wisp_property('barycenter',0,surface_topomesh.wisp_property('barycenter',0).values()+size*resolution/2.,keys=list(surface_topomesh.wisps(0)))

mesh,_,_ = topomesh_to_triangular_mesh(surface_topomesh,degree=2,coef=1,mesh_center=[0,0,0],property_name='eccentricity')
world.add(mesh,'surface',colormap='invert_grey',position=size*resolution/2.,alpha=1.0,intensity_range=(0,1))

import vplants.meshing.tetrahedrization_tools
reload(vplants.meshing.tetrahedrization_tools)
from vplants.meshing.tetrahedrization_tools import *
    
triangulation_topomesh = delaunay_tetrahedrization_topomesh(segmented_positions,image_cell_vertex=None,clean_surface=False)

triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0])
world.add(triangulation_mesh,'delaunay_triangulation',position=size*resolution/2.,colormap='grey',intensity_range=(-1,0))

def point_nuclei_density(nuclei_positions,points,cell_radius=5,k=1.0):
    def nuclei_density_function(nuclei_positions,cell_radius,k=0.1):
        import numpy as np
        
        def density_func(x,y,z):
            density = np.zeros_like(x+y+z,float)
            max_radius = cell_radius
            # max_radius = 0.
    
            for p in nuclei_positions.keys():
                cell_distances = np.power(np.power(x-nuclei_positions[p][0],2) + np.power(y-nuclei_positions[p][1],2) + np.power(z-nuclei_positions[p][2],2),0.5)
                density += 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))
            return density
        return density_func
    return nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=k)(points[:,0],points[:,1],points[:,2])
    

compute_topomesh_property(triangulation_topomesh,'normal',2)
compute_topomesh_property(triangulation_topomesh,'barycenter',2)
triangulation_triangle_exterior_density = point_nuclei_density(positions,triangulation_topomesh.wisp_property('barycenter',2).values()+triangulation_topomesh.wisp_property('normal',2).values(),cell_radius=10.,k=0.5)
triangulation_triangle_interior_density = point_nuclei_density(positions,triangulation_topomesh.wisp_property('barycenter',2).values()-triangulation_topomesh.wisp_property('normal',2).values(),cell_radius=10,k=0.5)
normal_orientation = 2*(triangulation_triangle_exterior_density<triangulation_triangle_interior_density)-1
triangulation_topomesh.update_wisp_property('normal',2,normal_orientation[...,np.newaxis]*triangulation_topomesh.wisp_property('normal',2).values(),list(triangulation_topomesh.wisps(2)))

compute_topomesh_property(triangulation_topomesh,'cells',2)
compute_topomesh_property(triangulation_topomesh,'epidermis',2)

compute_topomesh_property(triangulation_topomesh,'normal',0)

compute_topomesh_property(triangulation_topomesh,'vertices',3)

tetra_features = tetra_geometric_features(triangulation_topomesh.wisp_property('vertices',3).values(),positions,features=['max_distance','eccentricity','min_dihedral_angle','max_dihedral_angle'])

compute_topomesh_property(triangulation_topomesh,'triangles',1)
compute_topomesh_property(triangulation_topomesh,'triangles',0)
compute_topomesh_property(triangulation_topomesh,'epidermis',1)
compute_topomesh_property(triangulation_topomesh,'epidermis',0)

compute_topomesh_property(triangulation_topomesh,'vertices',1)
compute_topomesh_property(triangulation_topomesh,'regions',1)
compute_topomesh_property(triangulation_topomesh,'cells',1)
compute_topomesh_property(triangulation_topomesh,'vertices',2)
compute_topomesh_property(triangulation_topomesh,'regions',2)
compute_topomesh_property(triangulation_topomesh,'cells',2)
compute_topomesh_property(triangulation_topomesh,'vertices',3)
compute_topomesh_property(triangulation_topomesh,'edges',3)
compute_topomesh_property(triangulation_topomesh,'triangles',3)
compute_topomesh_property(triangulation_topomesh,'epidermis',1)
compute_topomesh_property(triangulation_topomesh,'epidermis',0)

compute_topomesh_property(triangulation_topomesh,'regions',2)
compute_topomesh_property(triangulation_topomesh,'border_neighbors',degree=2)
compute_topomesh_property(triangulation_topomesh,'barycenter',degree=3)

compute_topomesh_property(triangulation_topomesh,'length',degree=1)
compute_topomesh_property(triangulation_topomesh,'area',degree=2)
compute_topomesh_property(triangulation_topomesh,'volume',degree=3)

triangulation_tetrahedra_triangles = np.array([list(triangulation_topomesh.borders(3,t)) for t in triangulation_topomesh.wisps(3)])
triangulation_tetrahedra_area = np.sum(triangulation_topomesh.wisp_property('area',2).values(triangulation_tetrahedra_triangles),axis=1)
triangulation_tetrahedra_volume = triangulation_topomesh.wisp_property('volume',3).values()

triangulation_tetrahedra_eccentricities = array_dict(1.0 - 216.*np.sqrt(3.)*np.power(triangulation_tetrahedra_volume,2.0)/np.power(triangulation_tetrahedra_area,3.0),list(triangulation_topomesh.wisps(3)))
triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    

compute_topomesh_property(triangulation_topomesh,'cells',degree=2)
triangulation_triangle_sliver = array_dict(map(np.mean,triangulation_tetrahedra_eccentricities.values(triangulation_topomesh.wisp_property('cells',2).values())),list(triangulation_topomesh.wisps(2)))

triangulation_triangle_edges = triangulation_topomesh.wisp_property('borders',2).values()
triangulation_edge_points = triangulation_topomesh.wisp_property('barycenter',0).values(triangulation_topomesh.wisp_property('vertices',1).values())

surface_triangle_points = surface_topomesh.wisp_property('barycenter',0).values(surface_topomesh.wisp_property('vertices',2).values())

#triangulation_points = triangulation_topomesh.wisp_property('barycenter',0).values()
#triangulation_point_coords = tuple(np.array(np.round(triangulation_points/resolution+size/2),np.uint16).transpose())
#exterior_point = array_dict(True-np.array(original_binary_img[triangulation_point_coords],bool),list(triangulation_topomesh.wisps(0)))
#triangulation_exterior_triangles = array_dict(np.any(exterior_point.values(triangulation_topomesh.wisp_property('vertices',2).values()),axis=1),list(triangulation_topomesh.wisps(2)))

triangulation_edge_surface_intersection =  array_dict([intersecting_triangle(e,surface_triangle_points).any() for e in triangulation_edge_points],list(triangulation_topomesh.wisps(1)))
triangulation_triangle_surface_intersection = array_dict(np.any(triangulation_edge_surface_intersection.values(triangulation_triangle_edges),axis=1),list(triangulation_topomesh.wisps(2)))

triangulation_triangle_max_length = array_dict(np.max(triangulation_topomesh.wisp_property('length',1).values(triangulation_triangle_edges),axis=1),list(triangulation_topomesh.wisps(2)))

maximal_distance = 15.0
maximal_eccentricity = 0.8

triangulation_topomesh_triangle_to_delete = np.zeros_like(list(triangulation_topomesh.wisps(2)),bool)
triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_triangle_surface_intersection.values(list(triangulation_topomesh.wisps(2)))
#triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_exterior_triangles.values(list(triangulation_topomesh.wisps(2)))
triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_max_length.values(list(triangulation_topomesh.wisps(2))) > maximal_distance)
triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_sliver.values(list(triangulation_topomesh.wisps(2))) > maximal_eccentricity)

triangulation_topomesh.update_wisp_property('to_delete',2,triangulation_topomesh_triangle_to_delete,list(triangulation_topomesh.wisps(2)))
triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    

triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='to_delete',property_degree=2)
#triangulation_topomesh.update_wisp_property('surface_intersection',2,triangulation_triangle_surface_intersection.values(list(triangulation_topomesh.wisps(2))),list(triangulation_topomesh.wisps(2)))    
#triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='surface_intersection',property_degree=2)
#triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='eccentricity')
world.add(triangulation_mesh,'delaunay_triangulation',position=size*resolution/2.,colormap='Oranges',intensity_range=(0,1),x_slice=(0,75))
raw_input()


n_triangles_0 = triangulation_topomesh.nb_wisps(2)
n_triangles_initial = triangulation_topomesh.nb_wisps(2)+1
n_triangles = triangulation_topomesh.nb_wisps(2)
print n_triangles,"Triangles"
iteration = 0
  
while n_triangles < n_triangles_initial:
    n_triangles_initial = n_triangles
    iteration = iteration+1

    exterior_triangles = [t for t in triangulation_topomesh.wisps(2) if len(list(triangulation_topomesh.regions(2,t))) < 2]

    compute_topomesh_property(triangulation_topomesh,'cells',degree=2)
    compute_topomesh_property(triangulation_topomesh,'triangles',degree=0)
    compute_topomesh_property(triangulation_topomesh,'regions',degree=2)
    compute_topomesh_property(triangulation_topomesh,'borders',degree=2)
    compute_topomesh_property(triangulation_topomesh,'epidermis',degree=2)
    compute_topomesh_property(triangulation_topomesh,'vertices',degree=2)
    compute_topomesh_property(triangulation_topomesh,'epidermis',degree=0)
    compute_topomesh_property(triangulation_topomesh,'triangles',degree=1)
    
    triangulation_triangle_sliver = array_dict(map(np.mean,triangulation_tetrahedra_eccentricities.values(triangulation_topomesh.wisp_property('cells',2).values())),list(triangulation_topomesh.wisps(2)))

    triangulation_triangle_n_cells = array_dict(map(len,triangulation_topomesh.wisp_property('cells',2).values()),list(triangulation_topomesh.wisps(2)))
    triangulation_edge_n_triangles = array_dict(map(len,[[t for t in triangulation_topomesh.regions(1,e) if triangulation_topomesh.wisp_property('epidermis',2)[t]] for e in triangulation_topomesh.wisps(1)]),list(triangulation_topomesh.wisps(1)))
    triangulation_triangle_edge_n_triangles = array_dict(triangulation_edge_n_triangles.values(triangulation_topomesh.wisp_property('borders',2).values(list(triangulation_topomesh.wisps(2)))).max(axis=1),list(triangulation_topomesh.wisps(2)))

    triangles_to_delete = []

    for t in triangulation_topomesh.wisps(2):
        if len(list(triangulation_topomesh.regions(2,t)))==1:
            #if triangulation_exterior_triangles[t]:
            #    triangles_to_delete.append(t)
            if triangulation_triangle_surface_intersection[t]:
                triangles_to_delete.append(t)
            elif triangulation_triangle_max_length[t] > maximal_distance:
                triangles_to_delete.append(t)
            elif triangulation_triangle_sliver[t] > maximal_eccentricity:
                triangles_to_delete.append(t)
            elif triangulation_triangle_edge_n_triangles[t]>2:
                triangles_to_delete.append(t)
        elif len(list(triangulation_topomesh.regions(2,t)))==0:
                triangles_to_delete.append(t)
    
    for t in triangles_to_delete:
        for c in triangulation_topomesh.regions(2,t):
            triangulation_topomesh.remove_wisp(3,c)
        triangulation_topomesh.remove_wisp(2,t)
    
    lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,triangulation_topomesh.wisp_property('triangles',1).values(list(triangulation_topomesh.wisps(1)))))==0)[0]]
    for e in lonely_edges:
        triangulation_topomesh.remove_wisp(1,e)

    triangulation_topomesh_triangle_to_delete = np.zeros_like(list(triangulation_topomesh.wisps(2)),bool)
    triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_triangle_surface_intersection.values(list(triangulation_topomesh.wisps(2)))
    #triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_exterior_triangles.values(list(triangulation_topomesh.wisps(2)))
    triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_max_length.values(list(triangulation_topomesh.wisps(2))) > maximal_distance)
    triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_sliver.values(list(triangulation_topomesh.wisps(2))) > maximal_eccentricity)
    
    triangulation_topomesh.update_wisp_property('to_delete',2,triangulation_topomesh_triangle_to_delete,list(triangulation_topomesh.wisps(2)))
    triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    

    triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='to_delete',property_degree=2)
    #triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='eccentricity')
    world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap='Oranges',intensity_range=(0,1))

    n_triangles = triangulation_topomesh.nb_wisps(2)
    print n_triangles,"Triangles"
    
discarded_cells = np.array(list(triangulation_topomesh.wisps(0)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(0,v,2)) for v in triangulation_topomesh.wisps(0)]))==0)[0]]

for v in discarded_cells:
    triangulation_topomesh.remove_wisp(0,v)

triangulation_topomesh.update_wisp_property('ratios',0,segmented_cell_ratios.values(list(triangulation_topomesh.wisps(0))),list(triangulation_topomesh.wisps(0)))
triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.99,mesh_center=[0,0,0],property_name='ratios',property_degree=0)
world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0,1))


surface_triangulation_topomesh = epidermis_topomesh(triangulation_topomesh)
surface_triangulation_topomesh.add_wisp(3,1)
for t in surface_triangulation_topomesh.wisps(2):
    for c in surface_triangulation_topomesh.regions(2,t):
        surface_triangulation_topomesh.unlink(3,c,t)
    surface_triangulation_topomesh.link(3,1,t)
cells_to_remove = set(surface_triangulation_topomesh.wisps(3)).difference({1})
for c in cells_to_remove:
    surface_triangulation_topomesh.remove_wisp(3,c)
        
triangulation_mesh,_,_ = topomesh_to_triangular_mesh(surface_triangulation_topomesh,degree=3,coef=0.99,mesh_center=[0,0,0])
world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap='grey',intensity_range=(0,1))

import vplants.meshing.property_topomesh_analysis
reload(vplants.meshing.property_topomesh_analysis)
from vplants.meshing.property_topomesh_analysis import *

compute_topomesh_property(surface_triangulation_topomesh,'oriented_borders',2)
compute_topomesh_property(surface_triangulation_topomesh,'oriented_borders',3)

surface_components = array_dict(*tuple(surface_triangulation_topomesh.wisp_property('oriented_border_components',3).values()[0][::-1]))
surface_triangulation_topomesh.update_wisp_property('component',2,surface_components.values(list(surface_triangulation_topomesh.wisps(2))),list(surface_triangulation_topomesh.wisps(2)))
                                                   

triangulation_mesh,_,_ = topomesh_to_triangular_mesh(surface_triangulation_topomesh,degree=3,coef=1,mesh_center=[0,0,0],property_name='component',property_degree=2)
world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap='glasbey')

triangle_orientations = array_dict(surface_triangulation_topomesh.wisp_property('oriented_borders',3)[1][1],surface_triangulation_topomesh.wisp_property('oriented_borders',3)[1][0])
surface_triangulation_topomesh.update_wisp_property('orientation',2,triangle_orientations.values(list(surface_triangulation_topomesh.wisps(2))),list(surface_triangulation_topomesh.wisps(2)))


compute_topomesh_property(surface_triangulation_topomesh,'barycenter',2)
compute_topomesh_property(surface_triangulation_topomesh,'oriented_vertices',2)

vertices_positions = surface_triangulation_topomesh.wisp_property('barycenter',0).values(surface_triangulation_topomesh.wisp_property('oriented_vertices',2).values())
normal_vectors = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])
normal_norms = np.linalg.norm(normal_vectors,axis=1)
normal_orientations = surface_triangulation_topomesh.wisp_property('orientation',2).values()
surface_triangulation_topomesh.update_wisp_property('normal',2,normal_orientations[:,np.newaxis]*normal_vectors/normal_norms[:,np.newaxis],list(topomesh.wisps(2)))
#surface_triangulation_topomesh.update_wisp_property('normal',2,normal_vectors/normal_norms[:,np.newaxis],list(surface_triangulation_topomesh.wisps(2)))
#compute_topomesh_property(surface_triangulation_topomesh,'normal',degree=0)
#compute_topomesh_property(surface_triangulation_topomesh,'normal',2)

normal_mesh = TriangularMesh()
#normal_points = np.concatenate([np.array(list(topomesh.wisps(0))),np.array(list(topomesh.wisps(0)))+np.max(list(topomesh.wisps(0)))])
#point_normals = topomesh.wisp_property('normal',0).values()
#normal_point_positions = np.concatenate([topomesh.wisp_property('barycenter',0).values(),topomesh.wisp_property('barycenter',0).values()+0.05*point_normals])
maximal_triangle_id = np.max(list(surface_triangulation_topomesh.wisps(2)))
normal_points = np.concatenate([np.array(list(surface_triangulation_topomesh.wisps(2))),np.array(list(surface_triangulation_topomesh.wisps(2)))+maximal_triangle_id])
point_normals = surface_triangulation_topomesh.wisp_property('normal',2).values()
normal_point_positions = np.concatenate([surface_triangulation_topomesh.wisp_property('barycenter',2).values()-size*resolution/2.,surface_triangulation_topomesh.wisp_property('barycenter',2).values()-size*resolution/2.+point_normals])
normal_mesh.points = array_dict(normal_point_positions,normal_points)
#normal_mesh.edges = array_dict(np.array([(c,c+np.max(list(topomesh.wisps(0)))) for c in topomesh.wisps(0)]))
normal_mesh.edges = array_dict(np.array([(c,c+maximal_triangle_id) for c in surface_triangulation_topomesh.wisps(2)]))
world.add(normal_mesh,'triangulation_normals',colormap='invert_grey',linewidth=3)

normal_dot_product = np.einsum('ij,ij->i',surface_triangulation_topomesh.wisp_property('normal',2).values(),np.array([np.array([0,0,1]) for t in surface_triangulation_topomesh.wisps(2)]))
surface_triangulation_topomesh.update_wisp_property('normal_direction',2,normal_dot_product,list(surface_triangulation_topomesh.wisps(2)))
triangulation_mesh,_,_ = topomesh_to_triangular_mesh(surface_triangulation_topomesh,degree=3,coef=1,mesh_center=[0,0,0],property_name='normal_direction',property_degree=2)
world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap='curvature',intensity_range=(-1,1))

triangles_to_delete = np.array(list(surface_triangulation_topomesh.wisps(2)))[normal_dot_product<0]
for t in triangles_to_delete:
    surface_triangulation_topomesh.remove_wisp(2,t)

edges_to_delete = [e for e in surface_triangulation_topomesh.wisps(1) if surface_triangulation_topomesh.nb_regions(1,e)==0]
for e in edges_to_delete:
    surface_triangulation_topomesh.remove_wisp(1,e)
    
vertices_to_delete = [v for v in surface_triangulation_topomesh.wisps(0) if surface_triangulation_topomesh.nb_regions(0,v)==0]
for v in vertices_to_delete:
    surface_triangulation_topomesh.remove_wisp(0,v)


surface_triangulation_topomesh.update_wisp_property('ratios',0,segmented_cell_ratios.values(list(surface_triangulation_topomesh.wisps(0))),list(surface_triangulation_topomesh.wisps(0)))

triangulation_mesh,_,_ = topomesh_to_triangular_mesh(surface_triangulation_topomesh,degree=3,coef=1,mesh_center=[0,0,0],property_name='ratios',property_degree=0)
world.add(triangulation_mesh,'optimized_delaunay_triangulation',position=size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0,1))




