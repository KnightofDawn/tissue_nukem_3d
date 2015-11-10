import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq                       import kmeans, vq

from openalea.image.spatial_image           import SpatialImage
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from openalea.image.serial.all              import imread, imsave
#from vplants.mars_alt.mars.segmentation     import filtering
from vplants.tissue_analysis.temporal_graph_from_image   import graph_from_image

from scipy.ndimage.filters import gaussian_filter

from openalea.deploy.shared_data import shared_data

from vplants.meshing.property_topomesh_analysis     import *
from vplants.meshing.intersection_tools     import inside_triangle, intersecting_segment, intersecting_triangle
from vplants.meshing.evaluation_tools       import jaccard_index
from vplants.meshing.tetrahedrization_tools import tetrahedra_dual_topomesh, tetrahedra_from_triangulation, tetra_geometric_features, triangle_geometric_features, triangulated_interface_topomesh
from vplants.meshing.topomesh_from_image    import *

from vplants.meshing.triangular_mesh import TriangularMesh

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface

from openalea.container.array_dict             import array_dict
from openalea.container.property_topomesh      import PropertyTopomesh

from sys                                    import argv
from time                                   import time, sleep
import csv


import matplotlib
matplotlib.use( "MacOSX" )
import matplotlib.pyplot as plt
from vplants.meshing.cute_plot                              import simple_plot, density_plot, smooth_plot, histo_plot, bar_plot, violin_plot, spider_plot

import pickle
from copy import deepcopy


tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

#filename = "r2DII_3.2_141127_sam08_t24"
#filename = "DR5N_5.2_150415_sam01_t00"
filename = "r2DII_1.2_141202_sam06_t04"
#filename = "r2DII_1.2_141202_sam03_t32"
#previous_filename = "r2DII_2.2_141204_sam01_t28"
#previous_filename = "r2DII_1.2_141202_sam03_t28"
#previous_filename = "r2DII_2.2_141204_sam07_t04"
#previous_filename = "r2DII_3.2_141127_sam08_t00"
#filename = str(argv[1])

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)
signal_name = 'DIIV'
#signal_name = 'DR5'

signal_colors = {}
signal_colors['DIIV'] = 'RdYlGn'
signal_colors['DR5'] = 'Blues'

tag_img = None
signal_img = None
img = None

image_cell_vertex = None
world.clear()

from openalea.core.service.plugin import plugin_instance
viewer = plugin_instance('oalab.applet','TissueViewer').vtk

try:
    signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_"+signal_name+".inr.gz"
    #signal_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_DR5.inr.gz"
    signal_img = imread(signal_file)
    
    tag_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
    tag_img = imread(tag_file)
    
    size = np.array(tag_img.shape)
    resolution = np.array(tag_img.resolution)*np.array([-1.,-1.,-1.])
    
    world.add(tag_img,'nuclei_image',position=size/2.,resolution=resolution,colormap='invert_grey')
    world.add(signal_img,signal_name+"_image",position=size/2.,resolution=resolution,colormap=signal_colors[signal_name])

except:
    image_file = dirname+"/segmented_images/"+filename+".inr.gz"
    image_cell_vertex = pickle.load(open(dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict","rb"))
    img = imread(image_file)
    world.add(img,'segmented_image',resolution=img.resolution,colormap='glasbey')
    
    img_graph = graph_from_image(img, spatio_temporal_properties=['volume','barycenter'],background=0,ignore_cells_at_stack_margins=False,property_as_real=True,min_contact_surface=0.5)

    img_labels = np.array(list(img_graph.vertices()))
    img_volumes = array_dict([img_graph.vertex_property('volume')[v] for v in img_labels],img_labels)

    img_center = np.nanmean(img_graph.vertex_property('barycenter').values(),axis=0)

viewer.ren.SetBackground(1,1,1)
# plt.imshow(tag_img[:,:,2],cmap='gray')
# plt.show(block=False)

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

    if tag_img is None:
        points = np.array(nuclei_cells[:,0],int)
    else:
        points = np.array(nuclei_cells[:,0],int)+2
    n_points = points.shape[0]  

    points_coordinates = nuclei_cells[:,1:4]

    if tag_img != None:
        resolution = np.array(tag_img.resolution)*np.array([-1.,-1.,-1.])
        # resolution = [0.25,0.25,0.28]
        # points_coordinates = (points_coordinates/resolution)[:,[1,0,2]]
        size = np.array(tag_img.shape)
    else:
        resolution = np.array(img.resolution)
        # resolution = [0.25,0.25,0.28]
        points_coordinates = (points_coordinates*resolution)*np.array([1.,1.,-1.])
        size = np.array(img.shape)
except:
    if tag_img is not None:
        from vplants.nuclei_segmentation.scale_space_detection import basic_scale_space, detect_peaks_3D_scale_space, frange
        import SimpleITK as sitk

        step = 0.1
        start = 0.4
        end = 0.7
        threshold = 3000.
        sigma = frange(start, end, step)

        scale_space = basic_scale_space(sitk.GetImageFromArray(tag_img.transpose((2,0,1))),sigma)

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

        peaks = detect_peaks_3D_scale_space(scale_space_DoG,scale_space_sigmas,threshold=threshold,resolution=np.array(tag_img.resolution))
        print peaks.shape[0]," Detected points"

        resolution = np.array(tag_img.resolution)*np.array([-1.,-1.,-1.])
        size = np.array(tag_img.shape)

        peak_scales = array_dict(scale_space_sigmas[peaks[:,0]],np.arange(peaks.shape[0]))
        # peak_positions = array_dict((peaks[:,1:] - np.array(tag_img.shape)/2.)*np.array(tag_img.resolution),np.arange(peaks.shape[0]))

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

detected_cells = TriangularMesh()
detected_cells.points = positions
#world.add(detected_cells,'detected_cells',position=size*resolution/2.,colormap=signal_colors[signal_name])
#raw_input()
    

if tag_img is not None:        
    
    filtered_signal_img = gaussian_filter(signal_img,sigma=1.5/np.array(tag_img.resolution))
    filtered_tag_img = gaussian_filter(tag_img,sigma=1.5/np.array(tag_img.resolution))

    #filtered_signal_img = filtering(signal_img,"gaussian",3.0)
    #filtered_tag_img = filtering(tag_img,"gaussian",3.0)

    coords = np.array(points_coordinates/resolution,int)

    points_signal = filtered_signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
    points_tag = filtered_tag_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]

    if signal_name == 'DIIV':
        cell_ratio = array_dict(1.0-np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)
    else:
        cell_ratio = array_dict(np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)


detected_cells = TriangularMesh()
detected_cells.points = positions
detected_cells.point_data = cell_ratio
world.add(detected_cells,'fluorescence_ratios',position=size*resolution/2.,colormap=signal_colors[signal_name],point_radius=3.0)
raw_input()

world['nuclei_image'].set_attribute('volume',False)
world[signal_name+'_image'].set_attribute('volume',False)

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

grid_resolution = resolution*[8,8,4]

if tag_img is not None:
    # x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
    # x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:2*grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:2*grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:2*grid_resolution[2]]
    x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:2*grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:2*grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:2*grid_resolution[2]]
    grid_size = 1.5*size
else:
    x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]] - size*resolution/2.
    grid_size = size

nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(x,y,z) for p in positions.keys()])
nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
nuclei_density = np.sum(nuclei_potential,axis=3)

surface_points,surface_triangles = implicit_surface(nuclei_density,grid_size,resolution)

surface_mesh = TriangularMesh()
surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
#surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()

#world.add(surface_mesh,'nuclei_implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=0.5)

import vplants.sam4dmaps.parametric_shape
reload(vplants.sam4dmaps.parametric_shape)
from vplants.sam4dmaps.sam_model_tools import spherical_parametric_meristem_model, phyllotaxis_based_parametric_meristem_model
#from vplants.sam4dmaps.parametric_shape import meristem_model_density_function, draw_meristem_model_pgl, meristem_model_energy

def meristem_model_density_function(model):
    import numpy as np
    dome_center = model['dome_center']
    dome_radius = model['dome_radius']
    primordia_centers = model['primordia_centers']
    primordia_radiuses = model['primordia_radiuses']
    dome_axes = model['dome_axes']
    dome_scales = model['dome_scales']
    k=1
    R_dome=1.0
    R_primordium=1.0

    def density_func(x,y,z):
        mahalanobis_matrix = np.einsum('...i,...j->...ij',dome_axes[0],dome_axes[0])/np.power(dome_scales[0],2.) + np.einsum('...i,...j->...ij',dome_axes[1],dome_axes[1])/np.power(dome_scales[1],2.) + np.einsum('...i,...j->...ij',dome_axes[2],dome_axes[2])/np.power(dome_scales[2],2.)
        
        dome_vectors = np.zeros((x.shape[0],y.shape[1],z.shape[2],3))
        dome_vectors[:,:,:,0] = x-dome_center[0]
        dome_vectors[:,:,:,1] = y-dome_center[1]
        dome_vectors[:,:,:,2] = z-dome_center[2]

        # dome_distance = np.power(np.power(x-dome_center[0],2) + np.power(y-dome_center[1],2) + np.power(z-dome_center[2],2),0.5)
        # dome_distance = np.power(np.power(x-dome_center[0],2)/np.power(dome_scales[0],2) + np.power(y-dome_center[1],2)/np.power(dome_scales[1],2) + np.power(z-dome_center[2],2)/np.power(dome_scales[2],2),0.5)
        dome_distance = np.power(np.einsum('...ij,...ij->...i',dome_vectors,np.einsum('...ij,...j->...i',mahalanobis_matrix,dome_vectors)),0.5)

        max_radius = R_dome*dome_radius
        density = 1./2. * (1. - np.tanh(k*(dome_distance - (dome_radius+max_radius)/2.)))
        for p in xrange(len(primordia_radiuses)):
            primordium_distance = np.power(np.power(x-primordia_centers[p][0],2) + np.power(y-primordia_centers[p][1],2) + np.power(z-primordia_centers[p][2],2),0.5)
            max_radius = R_primordium*primordia_radiuses[p]
            density +=  1./2. * (1. - np.tanh(k*(primordium_distance - (primordia_radiuses[p]+max_radius)/2.)))
        return density
    return density_func


def meristem_model_energy(parameters,density_function,x=x,y=y,z=z,nuclei_density=nuclei_density,minimum_density=0.5):
    import numpy as np
    meristem_density = density_function(x,y,z)
    # density_error = np.power(np.abs(np.minimum(nuclei_density,1)-np.minimum(meristem_density,1)),0.5)  
    # return density_error.sum()
    # return density_error.max()
    external_energy = ((minimum_density*np.ones_like(nuclei_density) - nuclei_density)[np.where(meristem_density>0.5)]).sum()
    
    internal_energy = 0.
    internal_energy += 10.*np.linalg.norm([parameters['dome_phi'],parameters['dome_psi']],1)

    return internal_energy + external_energy

def draw_meristem_model_vtk(meristem_model):
    import vtk
    from time import time
    
    model_polydata = vtk.vtkPolyData()
    model_points = vtk.vtkPoints()
    model_triangles = vtk.vtkCellArray()
    model_data = vtk.vtkLongArray()

    start_time = time()
    print "--> Creating VTK PolyData"
    
    dome_sphere = vtk.vtkSphereSource()
    #dome_sphere.SetCenter(meristem_model.shape_model['dome_center'])
    dome_sphere.SetRadius(meristem_model.shape_model['dome_radius'])
    dome_sphere.SetThetaResolution(32)
    dome_sphere.SetPhiResolution(32)
    dome_sphere.Update()
    ellipsoid_transform = vtk.vtkTransform()
    axes_transform = vtk.vtkLandmarkTransform()
    source_points = vtk.vtkPoints()
    source_points.InsertNextPoint([1,0,0])
    source_points.InsertNextPoint([0,1,0])
    source_points.InsertNextPoint([0,0,1])
    target_points = vtk.vtkPoints()
    target_points.InsertNextPoint(meristem_model.shape_model['dome_axes'][0])
    target_points.InsertNextPoint(meristem_model.shape_model['dome_axes'][1])
    target_points.InsertNextPoint(meristem_model.shape_model['dome_axes'][2])
    axes_transform.SetSourceLandmarks(source_points)
    axes_transform.SetTargetLandmarks(target_points)
    axes_transform.SetModeToRigidBody()
    axes_transform.Update()
    ellipsoid_transform.SetMatrix(axes_transform.GetMatrix())
    ellipsoid_transform.Scale(meristem_model.shape_model['dome_scales'][0],
                              meristem_model.shape_model['dome_scales'][1],
                              meristem_model.shape_model['dome_scales'][2])
    center_transform = vtk.vtkTransform()
    center_transform.Translate(meristem_model.shape_model['dome_center'][0],
                                  meristem_model.shape_model['dome_center'][1],
                                  meristem_model.shape_model['dome_center'][2])
    center_transform.Concatenate(ellipsoid_transform)
    dome_ellipsoid = vtk.vtkTransformPolyDataFilter()
    dome_ellipsoid.SetInput(dome_sphere.GetOutput())
    dome_ellipsoid.SetTransform(center_transform)
    dome_ellipsoid.Update()
    sphere_points = {}
    for p in xrange(dome_ellipsoid.GetOutput().GetPoints().GetNumberOfPoints()):
        pid = model_points.InsertNextPoint(dome_ellipsoid.GetOutput().GetPoints().GetPoint(p))
        sphere_points[p] = pid
    for t in xrange(dome_ellipsoid.GetOutput().GetNumberOfCells()):
        tid = model_triangles.InsertNextCell(3)
        for i in xrange(3):
            model_triangles.InsertCellPoint(sphere_points[dome_ellipsoid.GetOutput().GetCell(t).GetPointIds().GetId(i)])
        model_data.InsertValue(tid,1)
    print  model_triangles.GetNumberOfCells(), "(",dome_ellipsoid.GetOutput().GetNumberOfCells(),")"
    
    for primordium in xrange(len(meristem_model.shape_model['primordia_centers'])):
        primordium_sphere = vtk.vtkSphereSource()
        #primordium_sphere.SetCenter(meristem_model.shape_model['primordia_centers'][primordium])
        primordium_sphere.SetRadius(meristem_model.shape_model['primordia_radiuses'][primordium])
        primordium_sphere.SetThetaResolution(16)
        primordium_sphere.SetPhiResolution(16)
        primordium_sphere.Update()
        center_transform = vtk.vtkTransform()
        center_transform.Translate(meristem_model.shape_model['primordia_centers'][primordium][0],
                                  meristem_model.shape_model['primordia_centers'][primordium][1],
                                  meristem_model.shape_model['primordia_centers'][primordium][2])
        center_transform.Concatenate(axes_transform)
        primordium_rotated_sphere = vtk.vtkTransformPolyDataFilter()
        primordium_rotated_sphere.SetInput(primordium_sphere.GetOutput())
        primordium_rotated_sphere.SetTransform(center_transform)
        primordium_rotated_sphere.Update()
        sphere_points = {}
        for p in xrange(primordium_rotated_sphere.GetOutput().GetPoints().GetNumberOfPoints()):
            pid = model_points.InsertNextPoint(primordium_rotated_sphere.GetOutput().GetPoints().GetPoint(p))
            sphere_points[p] = pid
        for t in xrange(primordium_rotated_sphere.GetOutput().GetNumberOfCells()):
            tid = model_triangles.InsertNextCell(3)
            for i in xrange(3):
                model_triangles.InsertCellPoint(sphere_points[primordium_rotated_sphere.GetOutput().GetCell(t).GetPointIds().GetId(i)]) 
            model_data.InsertValue(tid,1)
        print  model_triangles.GetNumberOfCells(), "(",primordium_rotated_sphere.GetOutput().GetNumberOfCells(),")"
    
    model_polydata.SetPoints(model_points)
    model_polydata.SetPolys(model_triangles)
    model_polydata.GetCellData().SetScalars(model_data)

    end_time = time()
    print "<-- Creating VTK PolyData      [",end_time-start_time,"s]"
    return model_polydata


try:
    meristem_model_file = dirname+"/nuclei_images/"+previous_filename+"/"+previous_filename+"_meristem_model.prm"
    meristem_model_parameters =  pickle.load(open(meristem_model_file,'rb'))
except:
    initial_parameters = {}
    initial_parameters['dome_apex_x'] = size[0]*resolution[0]/2.
    initial_parameters['dome_apex_y'] = size[1]*resolution[1]/2.
    initial_parameters['dome_apex_z'] = 0
    initial_parameters['dome_radius'] = 60
    initial_parameters['initial_angle'] = 0.
    initial_parameters['dome_psi'] = 0.
    initial_parameters['dome_phi'] = 0.
    initial_parameters['n_primordia'] = 8
    initial_parameters['developmental_time'] = 6.
    
    initial_temperature = 10.
    minimal_temperature = 0.05
    lambda_temperature = 0.9
else:
    initial_parameters = meristem_model_parameters
    
    initial_temperature = 1.
    minimal_temperature = 0.2
    lambda_temperature = 0.98

# clockwise_energies = {}
# counterclockwise_energies = {}
# thetas = np.linspace(0,360,51)
# for theta in thetas:
#     clockwise_meristem_model = ParametricShapeModel()
#     counterclockwise_meristem_model = ParametricShapeModel()
    
#     clockwise_meristem_model.parameters = deepcopy(initial_parameters)
#     clockwise_meristem_model.parameters['orientation'] = 1.
#     clockwise_meristem_model.parameters['initial_angle'] = theta
#     clockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
#     clockwise_meristem_model.update_shape_model()
#     clockwise_meristem_model.density_function = meristem_model_density_function
#     clockwise_meristem_model.drawing_function = draw_meristem_model_vtk
    
#     counterclockwise_meristem_model.parameters = deepcopy(initial_parameters)
#     counterclockwise_meristem_model.parameters['orientation'] = -1.
#     counterclockwise_meristem_model.parameters['initial_angle'] = theta
#     counterclockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
#     counterclockwise_meristem_model.update_shape_model()
#     counterclockwise_meristem_model.density_function = meristem_model_density_function
#     counterclockwise_meristem_model.drawing_function = draw_meristem_model_vtk
    
#     world.add(detected_cells,'detected_cells',position=size*resolution/2.,intensity_range=(-1,0))
#     world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=0.1,z_slice=(92,100))
#     world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='vegetation',alpha=0.1,z_slice=(92,100))
#     raw_input()
    
    
#     # iteration = 0
    
#     n_cycles = 2
    
#     optimization_parameters = ['dome_apex_x','dome_apex_y','dome_apex_z','dome_radius','dome_phi','dome_psi','initial_angle','developmental_time']
#     #optimization_parameters = ['dome_apex_x','dome_apex_y','dome_apex_z','dome_radius','dome_phi','dome_psi']
    
#     for cycle in xrange(n_cycles):
#         temperature = initial_temperature
#         clockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
#         counterclockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
#         # iteration = iteration+1
        
#         while temperature>minimal_temperature:
#             temperature *= lambda_temperature
#             # iteration = iteration+1
#             # print temperature
#             clockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
#             counterclockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
#             #world.clear()
#             #world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=0.1,z_slice=(95,100))
#             #world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='vegetation',alpha=0.1,z_slice=(95,100))
    
#     world['counterclockwise_meristem_model'].set_attribute('display_polydata',False)
#     world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='morocco',alpha=0.4,z_slice=(92,100))
#     #screenshot_file = dirname+"/nuclei_images/"+filename+"/optimized_models/"+filename+"_clockwise_model_"+str(theta)+".jpg"
#     #viewer.save_screenshot(screenshot_file)
    
#     world['clockwise_meristem_model'].set_attribute('display_polydata',False)
#     world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='morocco',alpha=0.4,z_slice=(92,100))
#     #screenshot_file = dirname+"/nuclei_images/"+filename+"/optimized_models/"+filename+"_counterclockwise_model_"+str(theta)+".jpg"
#     #viewer.save_screenshot(screenshot_file)
#     raw_input()
  
#     clockwise_energy = meristem_model_energy(clockwise_meristem_model.parameters,clockwise_meristem_model.shape_model_density_function())
#     print "Clockwise Model Energy : ",clockwise_energy
#     clockwise_energies[theta] = clockwise_energy
    
#     counterclockwise_energy = meristem_model_energy(counterclockwise_meristem_model.parameters,counterclockwise_meristem_model.shape_model_density_function())
#     print "Counter-Clockwise Model Energy : ",counterclockwise_energy
#     counterclockwise_energies[theta] = counterclockwise_energy
     
# figure = plt.figure(0)
# figure.clf()
# smooth_plot(figure,thetas,np.array([clockwise_energies[theta] for theta in thetas]),color1=np.array([0.11,0.44,0.11]),color2=np.array([0.11,0.44,0.11]),smooth_factor=1000)
# smooth_plot(figure,thetas,np.array([counterclockwise_energies[theta] for theta in thetas]),color1=np.array([0.58,0.31,0.44]),color2=np.array([0.58,0.31,0.44]),smooth_factor=1000)
# #plt.plot(thetas,[clockwise_energies[theta] for theta in thetas],color='r')
# #plt.plot(thetas,[counterclockwise_energies[theta] for theta in thetas],color='b')
# plt.show(block=False)
# raw_input()

clockwise_meristem_model = ParametricShapeModel()
counterclockwise_meristem_model = ParametricShapeModel()

clockwise_meristem_model.parameters = deepcopy(initial_parameters)
clockwise_meristem_model.parameters['orientation'] = 1.
clockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
clockwise_meristem_model.update_shape_model()
clockwise_meristem_model.density_function = meristem_model_density_function
clockwise_meristem_model.drawing_function = draw_meristem_model_vtk

counterclockwise_meristem_model.parameters = deepcopy(initial_parameters)
counterclockwise_meristem_model.parameters['orientation'] = -1.
counterclockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
counterclockwise_meristem_model.update_shape_model()
counterclockwise_meristem_model.density_function = meristem_model_density_function
counterclockwise_meristem_model.drawing_function = draw_meristem_model_vtk

world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=0.1,z_slice=(92,100))
world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='vegetation',alpha=0.1,z_slice=(92,100))

n_cycles = 2
    
optimization_parameters = ['dome_apex_x','dome_apex_y','dome_apex_z','dome_radius','dome_phi','dome_psi','initial_angle','developmental_time']

for cycle in xrange(n_cycles):
    temperature = initial_temperature
    clockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
    counterclockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
    # iteration = iteration+1
    
    while temperature>minimal_temperature:
        temperature *= lambda_temperature
        # iteration = iteration+1
        # print temperature
        clockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
        counterclockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
        #world.clear()
        #world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=0.1,z_slice=(95,100))
        #world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='vegetation',alpha=0.1,z_slice=(95,100))

world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='morocco',alpha=0.4,z_slice=(92,100))
world.add(counterclockwise_meristem_model,'counterclockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='morocco',alpha=0.4,z_slice=(92,100))
  
clockwise_energy = meristem_model_energy(clockwise_meristem_model.parameters,clockwise_meristem_model.shape_model_density_function())
print "Clockwise Model Energy : ",clockwise_energy
#clockwise_energies[theta] = clockwise_energy

counterclockwise_energy = meristem_model_energy(counterclockwise_meristem_model.parameters,counterclockwise_meristem_model.shape_model_density_function())
print "Counter-Clockwise Model Energy : ",counterclockwise_energy
#counterclockwise_energies[theta] = counterclockwise_energy
     
raw_input()

#world.clear()
#world.add(surface_mesh,'implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)       

meristem_flexible_model = ParametricShapeModel()
if clockwise_energy < counterclockwise_energy:
    reference_parameters = deepcopy(clockwise_meristem_model.parameters)
else:
    reference_parameters = deepcopy(counterclockwise_meristem_model.parameters)
meristem_flexible_model.parameters = deepcopy(reference_parameters)
meristem_flexible_model.parametric_function = spherical_parametric_meristem_model
meristem_flexible_model.update_shape_model()
meristem_flexible_model.density_function = meristem_model_density_function
meristem_flexible_model.drawing_function = draw_meristem_model_vtk

#world.clear()
#world.add(nuclei_density,'nuclei_density',resolution=grid_resolution,position=resolution*size/2.)
#world.add(positions,'detected_cells',position=resolution*size/2.,_repr_vtk_=_points_repr_vtk_)
world.add(meristem_flexible_model,'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
world['clockwise_meristem_model'].set_attribute('display_polydata',False)
world['counterclockwise_meristem_model'].set_attribute('display_polydata',False)

def meristem_flexible_model_energy(parameters,density_function,x=x,y=y,z=z,nuclei_density=nuclei_density,minimum_density=0.5,reference_parameters=reference_parameters):
    import numpy as np
    meristem_density = density_function(x,y,z)
    # density_error = np.power(np.abs(np.minimum(nuclei_density,1)-np.minimum(meristem_density,1)),0.5)  
    # return density_error.sum()
    # return density_error.max()
    external_energy = ((minimum_density*np.ones_like(nuclei_density) - nuclei_density)[np.where(meristem_density>0.5)]).sum()
    
    internal_energy = 0.
    internal_energy += 10.*np.linalg.norm([parameters['dome_phi'],parameters['dome_psi']],1)
    internal_energy += 5.*np.linalg.norm([parameters[p]-reference_parameters[p] for p in parameters.keys() if 'primordium' in p],2)
    # internal_energy += 1*np.linalg.norm([parameters[p]-reference_parameters[p] for p in parameters.keys() if 'primordium' in p],2)
    #for primordium in xrange(parameters['n_primordia']-1):
        #internal_energy += -2.*np.minimum(parameters['primordium_'+str(primordium+1)+"_radius"] + 5 - parameters['primordium_'+str(primordium+2)+"_radius"],-10)
        #internal_energy += -10.*np.minimum(parameters['primordium_'+str(primordium+1)+"_distance"] - parameters['primordium_'+str(primordium+2)+"_distance"],-10)

    return internal_energy + external_energy


initial_temperature = 1.5
minimal_temperature = 0.2
lambda_temperature = 0.96

n_cycles = 2

optimization_parameters = []
for primordium in xrange(meristem_flexible_model.parameters['n_primordia']) :
  optimization_parameters += ['primordium_'+str(primordium+1)+'_distance','primordium_'+str(primordium+1)+'_angle','primordium_'+str(primordium+1)+"_height",'primordium_'+str(primordium+1)+"_radius"]

for cycle in xrange(n_cycles):
    temperature = initial_temperature
    meristem_flexible_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
    # iteration = iteration+1    
    #world.add(meristem_flexible_model,'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
    
    while temperature>minimal_temperature:
        temperature *= lambda_temperature
        # iteration = iteration+1
        meristem_flexible_model.parameter_optimization_annealing(meristem_flexible_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
        #world.add(meristem_flexible_model,'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
    
world.add(meristem_flexible_model,'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
    
meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
pickle.dump(meristem_flexible_model.parameters,open(meristem_model_file,'wb'))

mesh_surface = False

if mesh_surface:
    grid_resolution = resolution*[2,2,1]
    
    if tag_img is not None:
        # x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
        # x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:2*grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:2*grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:2*grid_resolution[2]]
        x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:2*grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:2*grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:2*grid_resolution[2]]
        grid_size = 2.*size
    
    surface_points,surface_triangles = implicit_surface(meristem_flexible_model.shape_model_density_function()(x,y,z),grid_size,resolution,iso=0.5)
    
    meristem_model_surface_mesh = TriangularMesh()
    meristem_model_surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
    meristem_model_surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
    #surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    meristem_model_surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    
    #world.clear()
    #world.add(meristem_flexible_model,'meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1)
    #world.add(meristem_model_surface_mesh,'meristem_model_implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)
    
    start_time = time()
    print "--> Generating Surface Topomesh"
    surface_topomesh = PropertyTopomesh(3)
    
    topomesh_start_time = time()
    print "  --> Creating points"
    for p in surface_points:
        pid = surface_topomesh.add_wisp(0)
    
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
    surface_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    _,unique_edges = np.unique(np.ascontiguousarray(surface_edges).view(np.dtype((np.void,surface_edges.dtype.itemsize * surface_edges.shape[1]))),return_index=True)
    surface_edges = surface_edges[unique_edges]
    topomesh_end_time = time()
    print "  <-- Creating points              [",topomesh_end_time - topomesh_start_time,"s]"
    
    topomesh_start_time = time()
    print "  --> Creating edges"
    for e in surface_edges:
        eid = surface_topomesh.add_wisp(1)
        for pid in e:
            surface_topomesh.link(1,eid,pid)
    topomesh_end_time = time()
    print "  <-- Creating edges               [",topomesh_end_time - topomesh_start_time,"s]"
    
    topomesh_start_time = time()
    print "  --> Creating faces"
    surface_triangle_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    surface_triangle_edge_matching = vq(surface_triangle_edges,surface_edges)[0].reshape(surface_triangles.shape[0],3)
    
    for t in surface_triangles:
        fid = surface_topomesh.add_wisp(2)
        for eid in surface_triangle_edge_matching[fid]:
            surface_topomesh.link(2,fid,eid)
    topomesh_end_time = time()
    print "  <-- Creating faces               [",topomesh_end_time - topomesh_start_time,"s]"
    
    cid = surface_topomesh.add_wisp(3)
    for fid in surface_topomesh.wisps(2):
        surface_topomesh.link(3,cid,fid)
    end_time=time()
    print "<-- Generating Surface Topomesh    [",end_time-start_time,"s]"
    compute_topomesh_property(surface_topomesh,'barycenter',0,positions=array_dict(surface_points,keys=list(surface_topomesh.wisps(0))))
    
    from vplants.meshing.optimization_tools    import optimize_topomesh
    surface_topomesh = optimize_topomesh(surface_topomesh,omega_forces=dict([('taubin_smoothing',1.0)]),iterations=30)
    
    from vplants.meshing.triangular_mesh import topomesh_to_triangular_mesh
    meristem_model_mesh,_,_ = topomesh_to_triangular_mesh(surface_topomesh,mesh_center=np.array([0,0,0]))
    
    #world.clear()
    world.add(meristem_model_mesh,'meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)
    #world.add(positions,'detected_cells',position=resolution*size/2.,_repr_vtk_=_points_repr_vtk_)
    
    surface_vertex_cell_membership = array_dict(np.transpose([nuclei_density_function(dict([(p,positions[p]- size*resolution/2.)]),cell_radius=5,k=0.5)(surface_topomesh.wisp_property('barycenter',0).values()[:,0],
                                                                                                                                    surface_topomesh.wisp_property('barycenter',0).values()[:,1],                                                                                                                              surface_topomesh.wisp_property('barycenter',0).values()[:,2]) for p in positions.keys()]),keys=list(surface_topomesh.wisps(0)))
    #surface_vertex_cell = array_dict([positions.keys()[np.argmax(surface_vertex_cell_membership[p])] for p in surface_topomesh.wisps(0)],list(surface_topomesh.wisps(0)))
    
    #coords = np.array(points_coordinates/resolution,int)
    #points_signal = signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
    #points_tag = tag_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
    #cell_ratio = array_dict(np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)
    
    surface_vertex_ratio = array_dict((surface_vertex_cell_membership.values(list(surface_topomesh.wisps(0)))*cell_ratio.values()).sum(axis=1)/surface_vertex_cell_membership.values(list(surface_topomesh.wisps(0))).sum(axis=1),keys=list(surface_topomesh.wisps(0)))
    
    
    #meristem_model_mesh.triangle_data = array_dict(meristem_model_mesh.triangle_data.keys(),meristem_model_mesh.triangle_data.keys()).to_dict()
    meristem_model_mesh.triangle_data = {}
    meristem_model_mesh.point_data = surface_vertex_ratio.to_dict()
    
    #world.clear()
    world.add(meristem_model_mesh,'meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap=signal_colors[signal_name],alpha=0.5)
    #world.add(signal_img,'DII_signal',resolution=np.array(signal_img.resolution)*np.array([-1.,-1.,-1.]),colormap='green')
    #world.add(tag_img,'nuclei_image',resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey')
    #world.add(meristem_model_surface_mesh,'meristem_model_implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)
    #world.add(positions,'detected_cells',position=resolution*size/2.,_repr_vtk_=_points_repr_vtk_,colormap='glasbey')



