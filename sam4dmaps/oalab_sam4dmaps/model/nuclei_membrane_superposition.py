import numpy as np
from scipy import ndimage as nd
from scipy.cluster.vq                       import kmeans, vq

from openalea.image.spatial_image   import SpatialImage
from openalea.image.serial.all      import imread, imsave

from openalea.deploy.shared_data import shared_data
import vplants.meshing_data

from vplants.segmentation.tool import images_path
from vplants.segmentation.filter import cellfilter, connexe, filters, regionalmax
from vplants.segmentation.watershed import watershed

from vplants.meshing.image_viewing_tools import *
from vplants.meshing.topomesh_from_image    import *

from openalea.container.array_dict             import array_dict

from vplants.nuclei_segmentation.scale_space_detection import basic_scale_space, detect_peaks_3D_scale_space, frange

from sys import argv
import os

world.clear()

filename = "L1PMN_1.1_150129_sam06_t00"
dirname = shared_data(vplants.meshing_data)

membrane_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_PMCit.inr.gz"
membrane_img = imread(membrane_file)
resolution = np.array(membrane_img.resolution)*np.array([-1.,-1.,-1.])
size = np.array(membrane_img.shape)
world.add(membrane_img,'membrane_image',colormap='invert_grey',resolution=resolution)


h_min = 2

segmentation_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_seg_hmin_"+str(h_min)+".inr.gz"
try:
    segmented_img = imread(segmentation_file)
except:
    
    img = np.array(membrane_img/256,np.uint8)
    img = 20. + 80.*(img - img.mean())/img.std()
    img = np.array(np.maximum(np.minimum(img,255),0),np.uint8)
    imsave(dirname+'/tmp/img.inr',SpatialImage(img,resolution=membrane_img.resolution))
    smooth_img = filters(dirname+'/tmp/img.inr',dirname+'/tmp/smooth.inr', sigma=1.0, sigma_z=0.5)
    
    levelset_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_levelset.inr.gz"
    try:
        background = imread(levelset_file)
    except:
        def levelset(image_in, image_out,  threshold_up, threshold_down, b_curvature, sigma):
            import subprocess
            bin_levelset = "/Users/gcerutti/Developpement/openalea/vplants_branches/MARS_ALT_HD/segmentation/src/cpp/lsm_detect_contour"
            arguments = " " + image_in + " "+ image_out+" "+ str(threshold_up) + " " + str(threshold_down) + " " + str(b_curvature) + " " + str(sigma)
            subprocess.call(bin_levelset + arguments, shell=True)
        
        levelset_img = levelset(dirname+'/tmp/smooth.inr',dirname+'/tmp/levelset.inr',threshold_up = 8, threshold_down = 8, b_curvature=0, sigma=0.0 )
        background = 2-imread(dirname+'/tmp/levelset.inr')
        imsave(levelset_file,background)
       
    #world.add(background,name="background_image",colormap='morocco',alpha=0.5,resolution=resolution)
    
    smooth = imread(dirname+'/tmp/smooth.inr')
    smooth_background = np.copy(smooth)
    smooth_background[np.where(background==1)] = 0
    
    imsave(dirname+'/tmp/smooth_background.inr',SpatialImage(smooth_background,resolution=membrane_img.resolution))
    
    #world.add(smooth_background,name="filtered_membrane_image",colormap='purple',alpha=0.5,resolution=resolution)
    
    regmax_img = regionalmax(dirname+'/tmp/smooth_background.inr', dirname+'/tmp/tmp.inr', dirname+'/tmp/regmax.inr', h_min=h_min)
    
    conn_img = connexe(dirname+'/tmp/regmax.inr', dirname+'/tmp/conn.inr', low_th=1, high_th=h_min)
    
    wat_img = watershed(dirname+'/tmp/conn.inr', dirname+'/tmp/smooth_background.inr', dirname+'/tmp/wat.inr')
    segmented_img = imread(dirname+'/tmp/wat.inr')
    
    imsave(segmentation_file,SpatialImage(segmented_img,resolution=membrane_img.resolution))
    segmentation_file = dirname+"/share/nuclei_images/"+filename+"/"+filename+"_seg_hmin_"+str(h_min)+".tif"
    imsave(segmentation_file,SpatialImage(segmented_img,resolution=membrane_img.resolution))

import vplants.meshing.vtk_tools
reload(vplants.meshing.vtk_tools)
from vplants.meshing.vtk_tools import image_to_triangular_mesh
segmented_img_mesh = image_to_triangular_mesh(segmented_img,cell_coef=1.2,smooth=0.8,mesh_fineness=3.0)
segmented_img_mesh.points = array_dict(segmented_img_mesh.points.values()*np.array([-1,-1,-1]),segmented_img_mesh.points.keys()).to_dict()
world.add(segmented_img,name="segmented_image",colormap='glasbey',alpha=1.0,alphamap='constant',erosion=False,shade=False,bg_id=1,resolution=resolution)
#world.add(segmented_img_mesh,name="segmented_image",colormap='glasbey')
raw_input()    

from openalea.image.algo.graph_from_image   import graph_from_image
img_graph = graph_from_image(segmented_img,spatio_temporal_properties=['barycenter'],ignore_cells_at_stack_margins = False,property_as_real=False)
img_cell_centers = array_dict(img_graph.vertex_property('barycenter').values()*resolution,img_graph.vertex_property('barycenter').keys())


nuclei_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_H2BTFP.inr.gz"
nuclei_img = imread(nuclei_file)
world.add(nuclei_img,'nuclei_image',colormap='green',resolution=resolution)

inputfile = dirname+"/nuclei_images/"+filename+"/cells.csv"

try:
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

    resolution = np.array(nuclei_img.resolution)*np.array([-1.,-1.,-1.])
    size = np.array(nuclei_img.shape)
except:
    from vplants.nuclei_segmentation.scale_space_detection import basic_scale_space, detect_peaks_3D_scale_space, frange
    import SimpleITK as sitk

    step = 0.1
    start = 0.4
    end = 0.7
    sigma = frange(start, end, step)

    scale_space = basic_scale_space(sitk.GetImageFromArray(nuclei_img.transpose((2,0,1))),sigma)

    scale_space_DoG = []
    scale_space_sigmas = []
    for i in xrange(len(sigma)):
        if i>1:
            print "_______________________"
            print ""
            print "Sigma = ",np.exp(sigma[i])," - ",np.exp(sigma[i-1])," -> ",np.power(np.exp(sigma[i-1]),2)
            print "(k-1)*sigma^2 : ",(np.exp(step)-1)*np.power(np.exp(sigma[i-1]),2)

            DoG_image = np.power(np.exp(sigma[i-1]),2)*scale_space[i] - np.power(np.exp(sigma[i]),2)*scale_space[i-1]
            scale_space_DoG.append(DoG_image)
            scale_space_sigmas.append(np.exp(sigma[i-1]))

    scale_space_DoG = np.array(scale_space_DoG)
    scale_space_sigmas = np.array(scale_space_sigmas)

    print "Scale Space Size : ",scale_space_DoG.shape

    peaks = detect_peaks_3D_scale_space(scale_space_DoG,scale_space_sigmas,threshold=3000,resolution=np.array(nuclei_img.resolution))
    print peaks.shape[0]," Detected points"

    resolution = np.array(nuclei_img.resolution)*np.array([-1.,-1.,-1.])
    size = np.array(nuclei_img.shape)

    peak_scales = array_dict(scale_space_sigmas[peaks[:,0]],np.arange(peaks.shape[0]))
    peak_positions = array_dict(peaks[:,1:]*resolution,np.arange(peaks.shape[0]))

    points = peak_positions.keys()
    points_coordinates = peak_positions.values()

    nuclei_file =  open(inputfile,'w+',1)
    nuclei_file.write("Cell id;x;y;z\n")
    for p in peak_positions.keys():
        nuclei_file.write(str(p)+";"+str(peak_positions[p][0])+";"+str(peak_positions[p][1])+";"+str(peak_positions[p][2])+"\n")
    nuclei_file.flush()
    nuclei_file.close()
    
positions = array_dict(points_coordinates,points)

from vplants.meshing.triangular_mesh import TriangularMesh

cell_centers = TriangularMesh()
cell_centers.points = img_cell_centers.to_dict()
cell_centers.point_radius = 1
world.add(cell_centers,'segmented_cell_centers',colormap='glasbey')

nuclei_cells = {}
nuc_cells = positions.keys()
img_cells = img_cell_centers.keys()
for c in xrange(len(positions.keys())):
    cell_matching = vq(positions.values(nuc_cells),img_cell_centers.values(img_cells))
    nuc_match = np.argmin(cell_matching[1])
    img_match = cell_matching[0][np.argmin(cell_matching[1])]
    nuclei_cells[nuc_cells[nuc_match]] = img_cells[img_match]
    nuc_cells = np.delete(nuc_cells,nuc_match,0)
    img_cells = np.delete(img_cells,img_match,0)
nuclei_cells = array_dict(nuclei_cells)

from vplants.meshing.triangular_mesh import TriangularMesh
detected_cells = TriangularMesh()
detected_cells.points = positions.to_dict()
detected_cells.point_radius = 1
#detected_cells.point_data = nuclei_cells.to_dict()
world.add(detected_cells,'detected_cells',colormap='glasbey')
raw_input()

from vplants.nuclei_segmentation.scale_space_detection import array_unique
from vplants.meshing.tetrahedrization_tools import tetrahedra_dual_topomesh, tetrahedra_from_triangulation, triangle_geometric_features
from openalea.container.property_topomesh import PropertyTopomesh
tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

l1_triangulation = delaunay_triangulation(np.concatenate([peak_positions.values()[:,:-1],np.zeros_like(peak_positions.keys())[:,np.newaxis]],axis=1))
triangulation_triangles = peak_positions.keys()[np.sort(np.array(l1_triangulation))]

triangulation_triangle_features = triangle_geometric_features(triangulation_triangles,peak_positions,features=['area','max_distance'])
triangulation_triangles = np.delete(triangulation_triangles,np.where((triangulation_triangle_features[:,0]>60)|(triangulation_triangle_features[:,1]>25)),0)

triangulation_edges = array_unique(np.concatenate(triangulation_triangles[:,triangle_edge_list ],axis=0))

start_time = time()
print "--> Generating triangulation topomesh"
triangle_edges = np.concatenate(triangulation_triangles[:,triangle_edge_list],axis=0)
triangle_edge_matching = vq(triangle_edges,triangulation_edges)[0]

triangulation_tetrahedra = np.sort([np.concatenate([[1],t]) for t in triangulation_triangles])
tetrahedra_triangles = np.concatenate(triangulation_tetrahedra[:,tetra_triangle_list])
tetrahedra_triangle_matching = vq(tetrahedra_triangles,triangulation_triangles)[0]

triangulation_topomesh = PropertyTopomesh(3)
for c in np.unique(triangulation_triangles):
    triangulation_topomesh.add_wisp(0,c)
for e in triangulation_edges:
    eid = triangulation_topomesh.add_wisp(1)
    for pid in e:
        triangulation_topomesh.link(1,eid,pid)
for t in triangulation_triangles:
    fid = triangulation_topomesh.add_wisp(2)
    for eid in triangle_edge_matching[3*fid:3*fid+3]:
        triangulation_topomesh.link(2,fid,eid)
for t in triangulation_tetrahedra:
    cid = triangulation_topomesh.add_wisp(3)
    for fid in tetrahedra_triangle_matching[4*cid:4*cid+4]:
        triangulation_topomesh.link(3,cid,fid)
triangulation_topomesh.update_wisp_property('barycenter',0,peak_positions.values(),keys=peak_positions.keys())
end_time = time()
print "<-- Generating triangulation topomesh [",end_time-start_time,"s]"

compute_topomesh_property(triangulation_topomesh,'vertices',1)
triangulation_edges = np.sort([nuclei_cells.values(e) for e in triangulation_topomesh.wisp_property('vertices',1).values()])
image_edges = np.sort([img_graph.edge_vertices(e) for e in img_graph.edges()])
corresponding_edges = array_dict(1-np.array(vq(triangulation_edges,image_edges)[1] == 0,int),list(triangulation_topomesh.wisps(1)))

from vplants.meshing.triangular_mesh import topomesh_to_triangular_mesh
triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=1,mesh_center=[0,0,0])
triangulation_mesh.edge_data = corresponding_edges.to_dict()

world.add(triangulation_mesh,"delaunay_triangulation",colormap="jet",linewidth=3,intensity_range=(0,1))

compute_topomesh_property(triangulation_topomesh,'vertices',2)
triangulation_tetrahedra = np.sort([np.concatenate([[1],triangulation_topomesh.wisp_property('vertices',2)[t]]) for t in triangulation_topomesh.wisps(2)])

voronoi_topomesh = tetrahedra_dual_topomesh(triangulation_tetrahedra,peak_positions,voronoi=True,exterior=False)
voronoi_topomesh.update_wisp_property('barycenter',0,voronoi_topomesh.wisp_property('barycenter',0).values(),voronoi_topomesh.wisp_property('barycenter',0).keys())
compute_topomesh_property(voronoi_topomesh,'cells',2)

import vplants.meshing.triangular_mesh
reload(vplants.meshing.triangular_mesh)
from vplants.meshing.triangular_mesh import TriangularMesh, topomesh_to_triangular_mesh

voronoi_mesh,_,_ = topomesh_to_triangular_mesh(voronoi_topomesh,degree=1,mesh_center=[0,0,0])
world.add(voronoi_mesh,'voronoi_diagram',colormap='quercus',linewidth=5)
raw_input()


omega_energies = {}
omega_energies['area'] = 0.35
omega_energies['eccentricity'] = 0.85
omega_energies['length'] = 0.5
omega_energies['neighborhood'] = 0.15

compute_topomesh_property(triangulation_topomesh,'epidermis',2)
compute_topomesh_property(triangulation_topomesh,'epidermis',0)
compute_topomesh_property(triangulation_topomesh,'cells',0)

import vplants.meshing.optimization_tools
reload(vplants.meshing.optimization_tools)
from vplants.meshing.optimization_tools import property_topomesh_edge_flip_optimization
from copy import deepcopy
optimized_triangulation_topomesh = deepcopy(triangulation_topomesh)
minimal_temperature = 0.05
lambda_temperature = 0.9
iterations = 60
property_topomesh_edge_flip_optimization(optimized_triangulation_topomesh,omega_energies=omega_energies,nested_mesh=False,simulated_annealing=True,minimal_temperature=minimal_temperature,lamba_temperature=lambda_temperature,iterations=iterations,display=False)

compute_topomesh_property(optimized_triangulation_topomesh,'vertices',1)
optimized_triangulation_edges = np.sort([nuclei_cells.values(e) for e in optimized_triangulation_topomesh.wisp_property('vertices',1).values()])
optimized_corresponding_edges = array_dict(1-np.array(vq(optimized_triangulation_edges,image_edges)[1] == 0,int),list(optimized_triangulation_topomesh.wisps(1)))

from vplants.meshing.triangular_mesh import topomesh_to_triangular_mesh
optimized_triangulation_mesh,_,_ = topomesh_to_triangular_mesh(optimized_triangulation_topomesh,degree=1,mesh_center=[0,0,0])
optimized_triangulation_mesh.edge_data = optimized_corresponding_edges.to_dict()

world.add(optimized_triangulation_mesh,"optimized_triangulation",colormap="curvature",linewidth=3,intensity_range=(0,1))

compute_topomesh_property(optimized_triangulation_topomesh,'vertices',2)
triangulation_tetrahedra = np.sort([np.concatenate([[1],optimized_triangulation_topomesh.wisp_property('vertices',2)[t]]) for t in optimized_triangulation_topomesh.wisps(2)])

dual_topomesh = tetrahedra_dual_topomesh(triangulation_tetrahedra,peak_positions,voronoi=True,exterior=False)
dual_topomesh.update_wisp_property('barycenter',0,dual_topomesh.wisp_property('barycenter',0).values(),dual_topomesh.wisp_property('barycenter',0).keys())

dual_mesh,_,_ = topomesh_to_triangular_mesh(dual_topomesh,degree=1,mesh_center=[0,0,0])
world.add(dual_mesh,'dual_cell_geometry',colormap='altitude',linewidth=5)

print 1-float(corresponding_edges.values().sum())/len(corresponding_edges)," --> ", 1-float(optimized_corresponding_edges.values().sum())/len(optimized_corresponding_edges)
raw_input()

compute_topomesh_property(dual_topomesh,'regions',2)
compute_topomesh_property(dual_topomesh,'borders',2)
from vplants.meshing.tetrahedrization_tools import star_interface_topomesh
star_topomesh = star_interface_topomesh(dual_topomesh)

star_mesh,_,_ = topomesh_to_triangular_mesh(star_topomesh,degree=2,coef=0.9,mesh_center=[0,0,0],property_name='area')
world.add(star_mesh,"star_dual_topomesh",colormap='curvature',intensity_range=(0,1))