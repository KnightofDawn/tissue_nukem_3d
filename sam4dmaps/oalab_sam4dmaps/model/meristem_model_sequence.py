import numpy as np
from scipy import ndimage as nd

from openalea.image.spatial_image           import SpatialImage
from openalea.image.serial.all              import imread, imsave

from scipy.ndimage.filters import gaussian_filter

from openalea.deploy.shared_data import shared_data
import vplants.meshing

from vplants.meshing.property_topomesh_analysis     import *
from vplants.meshing.intersection_tools     import inside_triangle, intersecting_segment, intersecting_triangle
from vplants.meshing.evaluation_tools       import jaccard_index
from vplants.meshing.tetrahedrization_tools import tetrahedra_dual_topomesh, tetrahedra_from_triangulation, tetra_geometric_features, triangle_geometric_features, triangulated_interface_topomesh
from vplants.meshing.topomesh_from_image    import *

from vplants.meshing.triangular_mesh import TriangularMesh, _points_repr_geom_, _points_repr_vtk_

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface

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



#filenames = ["r2DII_1.2_141202_sam03_t08","r2DII_1.2_141202_sam03_t28"]
#signal_names = ['DIIV','DIIV']
#previous_offset = 0

#filenames = ["r2DII_1.2_141202_sam03_t28","r2DII_1.2_141202_sam03_t32"]
#signal_names = ['DIIV','DIIV']
#previous_offset = 0
#previous_offset = -1
#previous_offset = 6

#filenames = ["r2DII_1.2_141202_sam03_t28","DR5N_7.1_150415_sam04_t00"]
#signal_names = ['DIIV','DR5']
#previous_offset = 0

#filenames = ["r2DII_2.2_141204_sam01_t24", "r2DII_2.2_141204_sam01_t28"]
#signal_names = ['DIIV','DIIV']
#previous_offset = 0

#filenames = ["r2DII_1.2_141202_sam06_t00","r2DII_1.2_141202_sam06_t04"] 
filenames = ["r2DII_1.2_141202_sam06_t04","r2DII_1.2_141202_sam06_t08"] 
#filenames = ["r2DII_1.2_141202_sam06_t00","r2DII_1.2_141202_sam06_t08"] 
signal_names = ['DIIV','DIIV']
previous_offset = 0

import vplants.meshing
import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)

from openalea.core.service.plugin import plugin_instance
viewer = plugin_instance('oalab.applet','TissueViewer').vtk

signal_colors = {}
signal_colors['DIIV'] = 'RdYlGn'
signal_colors['DR5'] = 'Blues'

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
        primordium_sphere.SetCenter(meristem_model.shape_model['primordia_centers'][primordium])
        primordium_sphere.SetRadius(meristem_model.shape_model['primordia_radiuses'][primordium])
        primordium_sphere.SetThetaResolution(16)
        primordium_sphere.SetPhiResolution(16)
        primordium_sphere.Update()
        sphere_points = {}
        for p in xrange(primordium_sphere.GetOutput().GetPoints().GetNumberOfPoints()):
            pid = model_points.InsertNextPoint(primordium_sphere.GetOutput().GetPoints().GetPoint(p))
            sphere_points[p] = pid
        for t in xrange(primordium_sphere.GetOutput().GetNumberOfCells()):
            tid = model_triangles.InsertNextCell(3)
            for i in xrange(3):
                model_triangles.InsertCellPoint(sphere_points[primordium_sphere.GetOutput().GetCell(t).GetPointIds().GetId(i)]) 
            model_data.InsertValue(tid,1)
        print  model_triangles.GetNumberOfCells(), "(",primordium_sphere.GetOutput().GetNumberOfCells(),")"
    
    model_polydata.SetPoints(model_points)
    model_polydata.SetPolys(model_triangles)
    model_polydata.GetCellData().SetScalars(model_data)

    end_time = time()
    print "<-- Creating VTK PolyData      [",end_time-start_time,"s]"
    return model_polydata


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
    
    world.add(tag_img,'nuclei_image'+filetime,position=np.array([4*i_file-1,1,1])*np.array(tag_img.shape)/2.,resolution=np.array(tag_img.resolution)*np.array([-1.,-1.,-1.]),colormap='invert_grey')
    world.add(signal_img,'signal_image'+filetime,position=np.array([4*i_file-1,1,1])*np.array(tag_img.shape)/2.,resolution=np.array(signal_img.resolution)*np.array([-1.,-1.,-1.]),colormap=signal_colors[signal_name])
    
    viewer.ren.SetBackground(1,1,1)
    
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
        cell_ratio = array_dict(np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)
    
    positions = array_dict(points_coordinates,points)
    
    detected_cells = TriangularMesh()
    detected_cells.points = positions
    detected_cells.point_data = cell_ratio
    world.add(detected_cells,'detected_cells_'+filetime,position=np.array([4*i_file-1,1,1])*size*resolution/2.,colormap='grey',intensity_range=(-1.0,0.0),point_radius=2.0)
    
    detected_cells.point_data = cell_ratio
    world.add(detected_cells,'fluorescence_ratios_'+filetime,position=np.array([4*i_file-1,1,1])*size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0.0,1.0),point_radius=2.0)
    
    raw_input()
    
    cell_fluorescence_ratios[filetime[1:]] = deepcopy(detected_cells)
    
    meristem_model_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_meristem_model.prm"
    meristem_model_parameters =  pickle.load(open(meristem_model_file,'rb'))
    
    from vplants.sam4dmaps.sam_model_tools import spherical_parametric_meristem_model, phyllotaxis_based_parametric_meristem_model

    meristem_model = ParametricShapeModel()
    meristem_model.parameters = deepcopy(meristem_model_parameters)
    # for p in meristem_model.parameters.keys():
    #     if 'angle' in p:
    #         meristem_model.parameters[p] = meristem_model.parameters[p]%360
    meristem_model.parametric_function = spherical_parametric_meristem_model
    meristem_model.update_shape_model()
    meristem_model.density_function = meristem_model_density_function
    meristem_model.drawing_function = draw_meristem_model_vtk
    world.add(meristem_model,'meristem_model'+filetime,position=np.array([4*i_file-1,1,1])*resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))

    meristem_models[filetime[1:]] = deepcopy(meristem_model)

    # surface_points,surface_triangles = implicit_surface(meristem_model.shape_model_density_function()(x,y,z),grid_size,resolution,iso=0.5)

    # meristem_model_surface_mesh = TriangularMesh()
    # meristem_model_surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
    # meristem_model_surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
    # #surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    # meristem_model_surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()

    # #world.add(meristem_model_surface_mesh,'meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)

    # start_time = time()
    # print "--> Generating Surface Topomesh"
    # surface_topomesh = PropertyTopomesh(3)
    
    # topomesh_start_time = time()
    # print "  --> Creating points"
    # for p in surface_points:
    #     pid = surface_topomesh.add_wisp(0)
    
    # triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
    # surface_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    # _,unique_edges = np.unique(np.ascontiguousarray(surface_edges).view(np.dtype((np.void,surface_edges.dtype.itemsize * surface_edges.shape[1]))),return_index=True)
    # surface_edges = surface_edges[unique_edges]
    # topomesh_end_time = time()
    # print "  <-- Creating points              [",topomesh_end_time - topomesh_start_time,"s]"
    
    # topomesh_start_time = time()
    # print "  --> Creating edges"
    # for e in surface_edges:
    #     eid = surface_topomesh.add_wisp(1)
    #     for pid in e:
    #         surface_topomesh.link(1,eid,pid)
    # topomesh_end_time = time()
    # print "  <-- Creating edges               [",topomesh_end_time - topomesh_start_time,"s]"
    
    # topomesh_start_time = time()
    # print "  --> Creating faces"
    # surface_triangle_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    # surface_triangle_edge_matching = vq(surface_triangle_edges,surface_edges)[0].reshape(surface_triangles.shape[0],3)
    
    # for t in surface_triangles:
    #     fid = surface_topomesh.add_wisp(2)
    #     for eid in surface_triangle_edge_matching[fid]:
    #         surface_topomesh.link(2,fid,eid)
    # topomesh_end_time = time()
    # print "  <-- Creating faces               [",topomesh_end_time - topomesh_start_time,"s]"
    
    # cid = surface_topomesh.add_wisp(3)
    # for fid in surface_topomesh.wisps(2):
    #     surface_topomesh.link(3,cid,fid)
    # end_time=time()
    # print "<-- Generating Surface Topomesh    [",end_time-start_time,"s]"
    # compute_topomesh_property(surface_topomesh,'barycenter',0,positions=array_dict(surface_points,keys=list(surface_topomesh.wisps(0))))
    
    # from vplants.meshing.optimization_tools    import optimize_topomesh
    # surface_topomesh = optimize_topomesh(surface_topomesh,omega_forces=dict([('taubin_smoothing',1.0)]),iterations=30)
    
    # from vplants.meshing.triangular_mesh import topomesh_to_triangular_mesh
    # meristem_model_mesh,_,_ = topomesh_to_triangular_mesh(surface_topomesh,mesh_center=np.array([0,0,0]))
    
    # #world.add(meristem_model_mesh,'meristem_model_mesh'+filetime,_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)

    # surface_vertex_cell_membership = array_dict(np.transpose([nuclei_density_function(dict([(p,positions[p]- size*resolution/2.)]),cell_radius=5,k=0.5)(surface_topomesh.wisp_property('barycenter',0).values()[:,0],
    #                                                                                                                                                     surface_topomesh.wisp_property('barycenter',0).values()[:,1], 
    #                                                                                                                                                     surface_topomesh.wisp_property('barycenter',0).values()[:,2]) for p in positions.keys()]),keys=list(surface_topomesh.wisps(0)))
    # #surface_vertex_cell = array_dict([positions.keys()[np.argmax(surface_vertex_cell_membership[p])] for p in surface_topomesh.wisps(0)],list(surface_topomesh.wisps(0)))
    
    # surface_vertex_ratio = array_dict((surface_vertex_cell_membership.values(list(surface_topomesh.wisps(0)))*cell_ratio.values()).sum(axis=1)/surface_vertex_cell_membership.values(list(surface_topomesh.wisps(0))).sum(axis=1),keys=list(surface_topomesh.wisps(0)))
    
    # #meristem_model_mesh.triangle_data = array_dict(meristem_model_mesh.triangle_data.keys(),meristem_model_mesh.triangle_data.keys()).to_dict()
    # meristem_model_mesh.triangle_data = {}
    # meristem_model_mesh.point_data = surface_vertex_ratio.to_dict()
    
    # world.add(meristem_model_mesh,'meristem_model_mesh'+filetime,_repr_vtk_=TriangularMesh._repr_vtk_,colormap='vegetation',alpha=0.5)
    # meristem_surface_ratios[filetime[1:]] = deepcopy(meristem_model_mesh)
raw_input()


grid_resolution = resolution*[8,8,2]
#grid_resolution = resolution*[4,4,1]
#grid_resolution = resolution*[2,2,1]
x,y,z = np.ogrid[-0.5*size[0]*resolution[0]:1.5*size[0]*resolution[0]:2*grid_resolution[0],-0.5*size[1]*resolution[1]:1.5*size[1]*resolution[1]:2*grid_resolution[1],-0.5*size[2]*resolution[2]:1.5*size[2]*resolution[2]:2*grid_resolution[2]]
grid_size = 2.*size

time_steps = [filename[-3:] for filename in filenames]
sequence_names = [filename[:-4] for filename in filenames]
time_hours = np.array([float(t[1:]) for t in time_steps])

reference_dome_apex = np.array([(size*resolution/2.)[0],(size*resolution/2.)[0],0])
aligned_cell_fluorescence_ratios = {}

orientation_0 = meristem_models[time_steps[0]].parameters['orientation']
orientation_1 = meristem_models[time_steps[1]].parameters['orientation']
orientation = 1.0
golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
golden_angle = 180.*golden_angle/np.pi

gap_score = {}
gap_range = np.arange(10)-3 
for gap in gap_range:
    angle_gap = {}
    distance_gap = {}
    radius_gap = {}
    height_gap = {}
    
    if gap>=0:
        matching_primordia = (np.arange(8-gap)+1)
    else:
        matching_primordia = (np.arange(8-abs(gap))+1-gap)
    print matching_primordia
    
    for p in matching_primordia:
        #angle_0 = (meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_angle"] + meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle) % 360
        #angle_1 = (meristem_models[time_steps[1]].parameters["primordium_"+str(p+gap)+"_angle"] + meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle + gap*golden_angle) % 360
        angle_0 = orientation_0*(meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_angle"]) + meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle
        angle_1 = orientation_1*(meristem_models[time_steps[1]].parameters["primordium_"+str(p+gap)+"_angle"]) + meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle - gap*golden_angle
        angle_gap[p] = np.cos(np.pi*(angle_1-angle_0)/180.)
        distance_gap[p]  = meristem_models[time_steps[1]].parameters["primordium_"+str(p+gap)+"_distance"] - meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_distance"]
        radius_gap[p] = meristem_models[time_steps[1]].parameters["primordium_"+str(p+gap)+"_radius"] - meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_radius"]
        height_gap[p] = meristem_models[time_steps[1]].parameters["primordium_"+str(p+gap)+"_height"] - meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_height"]
    rotation_0 = (meristem_models[time_steps[0]].parameters['initial_angle'] - meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle) %360
    rotation_1 = (meristem_models[time_steps[1]].parameters['initial_angle'] - meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle + gap*golden_angle) %360
    rotation_gap = np.cos(np.pi*(rotation_1 - rotation_0)/180.)
    gap_penalty = np.exp(-np.power(gap - (time_hours[1]-time_hours[0])/6.,2.0)/np.power(6.,2.0))

    if sequence_names[1]==sequence_names[0]:
        #gap_score.append(10*rotation_gap + np.mean(distance_gap))
        #gap_score[gap] = 10*rotation_gap*np.sign(np.mean(distance_gap.values()))
        gap_score[gap] = 10.0*np.mean(angle_gap.values())*np.exp(rotation_gap)*np.sign(np.mean(distance_gap.values()))*gap_penalty
        #gap_score[gap] = np.mean(angle_gap.values())
    else:
        gap_score[gap] = 10.0*np.mean(angle_gap.values())*np.exp(-np.power(np.mean(np.array(distance_gap.values())/6.),2.0))
        
    print "Gap = ",gap,"[",gap_penalty,"] : r -> ",np.mean(distance_gap.values()),", A -> ",np.mean(angle_gap.values())," (",rotation_0,"->",rotation_1,":",rotation_gap,") [",gap_score[gap],"]"


offset_gap = gap_range[np.argmax([gap_score[gap] for gap in gap_range])]
#raw_input()

#offset_gap = -1
#offset_gap = 6
offset_gaps = {}
offset_gaps[time_steps[0]] = previous_offset
offset_gaps[time_steps[1]] = previous_offset + offset_gap

figure = plt.figure(0)
figure.clf()    
figure.patch.set_facecolor('white')
size_coef = 1.0
for t in time_steps:
    ax = plt.subplot(111, polar=True)
    dome_radius = meristem_models[t].parameters['dome_radius']
    plt.scatter([0],[0],s=2.*np.pi*np.power(dome_radius,2.0),facecolors='none', edgecolors=[0.2,0.7,0.1],alpha=1.0/len(filenames),linewidths=5)
    primordia_distances = np.array([meristem_models[t].parameters["primordium_"+str(p)+'_distance'] for p in np.arange(8)+1])
    primordia_angles = meristem_models[t].parameters['orientation']*(np.array([meristem_models[t].parameters["primordium_"+str(p)+'_angle'] for p in np.arange(8)+1]) + (meristem_models[t].parameters['primordium_offset'] - offset_gaps[t])*golden_angle)
    primordia_radiuses = np.array([meristem_models[t].parameters["primordium_"+str(p)+'_radius'] for p in np.arange(8)+1])
    primordia_colors = np.array([[0.2+0.05*p,0.7-0.025*p,0.1+0.075*p] for p in np.arange(8)+1])
    plt.scatter(np.pi*primordia_angles/180.,primordia_distances,s=size_coef*np.pi*np.power(primordia_radiuses,2.0),c=primordia_colors,alpha=1.0/len(filenames))
    ax.set_rmax(160)
    ax.grid(True)
    ax.set_yticklabels([])
    plt.show(block=False)
raw_input()

aligned_positions = {}
for t in time_steps:
    signal_name = np.array(signal_names)[np.array(time_steps) == t][0]
    
    orientation = meristem_models[t].parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi

    cell_points = array_dict(cell_fluorescence_ratios[t].points)
    dome_apex = np.array([meristem_models[t].parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_phi = np.pi*meristem_models[t].parameters['dome_phi']/180.
    dome_psi = np.pi*meristem_models[t].parameters['dome_psi']/180.
    initial_angle = meristem_models[t].parameters['initial_angle'] - meristem_models[t].parameters['primordium_offset']*golden_angle 
    #initial_angle = meristem_models[t].parameters['initial_angle']
    initial_angle += previous_offset*golden_angle
    if t == time_steps[1]:
        initial_angle += offset_gap*golden_angle 
    dome_theta = np.pi*initial_angle/180.
    
    rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),np.sin(dome_psi)],[0,-np.sin(dome_psi),np.cos(dome_psi)]])
    rotation_phi = np.array([[np.cos(dome_phi),0,np.sin(dome_phi)],[0,1,0],[-np.sin(dome_phi),0,np.cos(dome_phi)]])
    rotation_theta = np.array([[np.cos(dome_theta),np.sin(dome_theta),0],[-np.sin(dome_theta),np.cos(dome_theta),0],[0,0,1]])
    print "Rotation Theta : ",(initial_angle%360)
    
    relative_points = (cell_points.values()-dome_apex[np.newaxis,:])*np.array([orientation,1,1])
    relative_points = np.einsum('...ij,...j->...i',rotation_phi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_psi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_theta,relative_points)
    
    aligned_cell_points = array_dict(reference_dome_apex + relative_points,cell_points.keys())
    aligned_positions[t] = deepcopy(aligned_cell_points)
    
    aligned_cells = TriangularMesh()
    aligned_cells.points = aligned_cell_points
    aligned_cells.point_data = cell_fluorescence_ratios[t].point_data
    world.add(aligned_cells,'aligned_cells_'+t,position=size*resolution/2.,colormap=signal_colors[signal_name],intensity_range=(0.5,1.0),point_radius=2)
    
    aligned_cell_fluorescence_ratios[t] = deepcopy(aligned_cells)
raw_input()

epidermis_cells = {}
for i_time,t in enumerate(time_steps):
    signal_name = signal_names[i_time] 
    
    grid_resolution = resolution*[2,2,2]
    x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:2*grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:2*grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:2*grid_resolution[2]]
    grid_size = 1.5*size

    cell_radius = 5.0
    density_k = 2.0

    nuclei_potential = np.array([nuclei_density_function(dict([(p,aligned_positions[t][p])]),cell_radius=cell_radius,k=density_k)(x,y,z) for p in aligned_positions[t].keys()])
    nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    nuclei_density = np.sum(nuclei_potential,axis=3)
    
    #world.add(np.array(40.*nuclei_density,np.uint8),'nuclei_density_'+t,position=-resolution*size/2.,colormap='grey',resolution=-4.*resolution)
    
    surface_points,surface_triangles = implicit_surface(nuclei_density,grid_size,resolution)
    
    surface_mesh = TriangularMesh()
    surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
    surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
    #surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
    #world.add(surface_mesh,'nuclei_implicit_surface',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=0.5)
    
    
    #detected_cells = deepcopy(aligned_cell_fluorescence_ratios[t])
    #detected_cells.point_data = {}
    #world.add(detected_cells,'detected_cells_'+t,position=size*resolution/2.,colormap='glasbey',point_radius=2)
    
    
    surface_vertex_cell_membership = array_dict(np.transpose([nuclei_density_function(dict([(p,aligned_positions[t][p]- size*resolution/2.)]),cell_radius=cell_radius,k=density_k)(surface_points[:,0],
                                                                                                                                    surface_points[:,1],
                                                                                                                                    surface_points[:,2]) for p in aligned_positions[t].keys()]),keys=np.arange(len(surface_points)))
    surface_vertex_cell = array_dict([aligned_positions[t].keys()[np.argmax(surface_vertex_cell_membership[p])] for p in np.arange(len(surface_points))],np.arange(len(surface_points)))
    
    #surface_mesh.point_data = array_dict(surface_vertex_cell.values()%256,surface_vertex_cell.keys())
    #world.add(surface_mesh,'nuclei_surface_membership',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='glasbey',alpha=0.5,intensity_range=None)
    
    epidermis_cells[t] = np.unique(surface_vertex_cell.values()[surface_points[:,2] > (size*resolution/3.)[2]])
    
    epidermis_detected_cells = TriangularMesh()
    epidermis_detected_cells.points = dict(zip(epidermis_cells[t],aligned_positions[t].values(epidermis_cells[t])))
    #epidermis_detected_cells.point_data = dict(zip(epidermis_cells[t],aligned_cell_fluorescence_ratios[t].point_data.values(epidermis_cells[t])))
    epidermis_detected_cells.point_data = dict(zip(epidermis_cells[t],epidermis_cells[t]%256))
    #world.add(epidermis_detected_cells,"epidermis_aligned_cells_"+t,position=resolution*size/2.,colormap=signal_colors[signal_name],point_radius=2,intensity_range=(0,1))

    raw_input()
    

# dr5_values = np.array(aligned_cell_fluorescence_ratios['t00'].point_data.values())
# figure = plt.figure(0)
# figure.clf()
# histo_plot(figure,dr5_values,color=np.array([0,0,0]),bar=False)
# plt.show(block=False)

# n_steps = 40
# for step in xrange(n_steps+1): 
#     time_weights = [(n_steps-step)/float(n_steps), step/float(n_steps)]
#     for t,w in zip(time_steps,time_weights):
#         world.add(aligned_cell_fluorescence_ratios[t],'aligned_cells_'+t,position=size*resolution/2.,colormap='vegetation',alpha=w,_repr_vtk_=_points_repr_vtk_)
#     screenshot_file = dirname+"/nuclei_images/"+filename+"/fluorescence_cells/"+filename+"_auxin_map_"+str(step)+".jpg"
#     viewer.save_screenshot(screenshot_file)

def sphere_density_function(sphere,k=0.1,R=1.0):
    import numpy as np
    center = sphere.parameters['center']
    radius = sphere.parameters['radius']
    axes = sphere.parameters['axes']
    scales = sphere.parameters['scales']

    def density_func(x,y,z):
        mahalanobis_matrix = np.einsum('...i,...j->...ij',axes[0],axes[0])/np.power(scales[0],2.) + np.einsum('...i,...j->...ij',axes[1],axes[1])/np.power(scales[1],2.) + np.einsum('...i,...j->...ij',axes[2],axes[2])/np.power(scales[2],2.)
        if x.ndim == 3:
            vectors = np.zeros((x.shape[0],y.shape[1],z.shape[2],3))
            vectors[:,:,:,0] = x-center[0]
            vectors[:,:,:,1] = y-center[1]
            vectors[:,:,:,2] = z-center[2]
        elif x.ndim == 1:
            vectors = np.zeros((x.shape[0],3))
            vectors[:,0] = x-center[0]
            vectors[:,1] = y-center[1]
            vectors[:,2] = z-center[2]
        distance = np.power(np.einsum('...ij,...ij->...i',vectors,np.einsum('...ij,...j->...i',mahalanobis_matrix,vectors)),0.5)

        max_radius = R*radius
        density = 1./2. * (1. - np.tanh(k*(distance - (radius+max_radius)/2.)))
        return density
    return density_func

golden_angle = 180.*np.sign(meristem_models[time_steps[0]].parameters['orientation'])*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)/np.pi


# stat_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_growing_nuclei_stats.csv"
# stat = open(stat_file,'w+')

# stat.write(" ;r;k;")
# for dist in xrange(101):
#     stat.write(str(10.*dist/100.)+";")
# stat.write("\n")

#for k in [0.0001, 0.001, 0.002, 0.0033, 0.005, 0.01, 0.02, 0.033, 0.05, 0.1, 0.2, 0.33, 0.5, 0.75, 1.]:
for k in [0.004]:
#for k in [0.5]:

    growing_meristem_model = ParametricShapeModel()
    aligned_meristem_models = {}
    
    for t in time_steps:
        growing_meristem_model.parameters = deepcopy(meristem_models[t].parameters)
        growing_meristem_model.parameters['dome_apex_x'] = reference_dome_apex[0]
        growing_meristem_model.parameters['dome_apex_y'] = reference_dome_apex[1]
        growing_meristem_model.parameters['dome_apex_z'] = reference_dome_apex[2]
        growing_meristem_model.parameters['dome_phi'] = 0
        growing_meristem_model.parameters['dome_psi'] = 0
        growing_meristem_model.parameters['initial_angle'] = 0
        growing_meristem_model.parameters['initial_angle'] -= previous_offset*golden_angle
        for p in growing_meristem_model.parameters.keys():
            if ('primordium' in p) and ('angle' in p):
                if t == time_steps[0]:
                    growing_meristem_model.parameters[p] += meristem_models[t].parameters['primordium_offset']*golden_angle 
                else:
                    growing_meristem_model.parameters[p] += meristem_models[t].parameters['primordium_offset']*golden_angle - offset_gap*golden_angle
        growing_meristem_model.parametric_function = spherical_parametric_meristem_model
        growing_meristem_model.update_shape_model()
        growing_meristem_model.density_function = meristem_model_density_function
        growing_meristem_model.drawing_function = draw_meristem_model_vtk
        aligned_meristem_models[t] = deepcopy(growing_meristem_model)
        world.add(growing_meristem_model,'growing_meristem_model_'+t,position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
    
    initium_angle = np.pi/6.
    initial_distance = aligned_meristem_models[time_steps[1]].parameters['dome_radius']*np.sin(4*initium_angle/3)
    zero_distance  = aligned_meristem_models[time_steps[1]].parameters['dome_radius']*0.28
    growth_factor = 0.01
    initial_time = np.log((1/growth_factor)*zero_distance/initial_distance)
    plastochrone_time = 0.09/growth_factor
    
    developmental_time_gap = meristem_models[time_steps[1]].parameters['developmental_time']
    developmental_time_gap -= meristem_models[time_steps[0]].parameters['developmental_time']
    developmental_time_gap += plastochrone_time*offset_gap

    matched_meristem_models = {}
    for t in time_steps:
        matched_meristem_model = ParametricShapeModel()
        matched_meristem_model = deepcopy(aligned_meristem_models[t])
        considered_primordia = np.arange(meristem_models[time_steps[0]].parameters['n_primordia'])+1
        for p in considered_primordia:
            try:
                if t == time_steps[0]:
                    matched_meristem_model.parameters["primordium_"+str(p)+"_angle"] = (aligned_meristem_models[t].parameters["primordium_"+str(p-offset_gap)+"_angle"]) % 360
                    matched_meristem_model.parameters["primordium_"+str(p)+"_distance"] = aligned_meristem_models[t].parameters["primordium_"+str(p-offset_gap)+"_distance"]
                    matched_meristem_model.parameters["primordium_"+str(p)+"_height"] = aligned_meristem_models[t].parameters["primordium_"+str(p-offset_gap)+"_height"] 
                    matched_meristem_model.parameters["primordium_"+str(p)+"_radius"] = aligned_meristem_models[t].parameters["primordium_"+str(p-offset_gap)+"_radius"]
                else:
                    matched_meristem_model.parameters["primordium_"+str(p)+"_angle"] = (aligned_meristem_models[t].parameters["primordium_"+str(p)+"_angle"]) % 360
                    
            except:
                print "Matching error : ",p," -> ",p-offset_gap
                matched_meristem_model.parameters["primordium_"+str(p)+"_angle"] = (aligned_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_angle"]) % 360
                
                primordium_distance = aligned_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_distance"]
                primordium_time = initial_time + 1./growth_factor * np.log(primordium_distance/initial_distance) - developmental_time_gap
                print "  --> Primordium ",p," : ",primordium_time
                
                matched_meristem_model.parameters["primordium_"+str(p)+"_distance"] = initial_distance*np.exp(growth_factor*(primordium_time-initial_time)) 
                primordium_distance = matched_meristem_model.parameters["primordium_"+str(p)+"_distance"]
                dome_radius = matched_meristem_model.parameters['dome_radius']
                matched_meristem_model.parameters["primordium_"+str(p)+"_radius"] = (0.26*primordium_distance/dome_radius + 0.15)*dome_radius
                matched_meristem_model.parameters["primordium_"+str(p)+"_height"] = (-0.08*np.power(primordium_distance/dome_radius - 0.2,2)-0.53)*dome_radius
        matched_meristem_model.update_shape_model()
        matched_meristem_models[t] = deepcopy(matched_meristem_model)
        world.add(matched_meristem_model,'matched_meristem_model_'+t,position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))
        
    organ_spheres = {}
    organ_memberships = {}
    relative_organ_points = {}
    for t in time_steps:
        cell_points = array_dict(aligned_cell_fluorescence_ratios[t].points)
            
        # density_k = 0.15
        # density_R = 1.0
        density_k = k
        #density_k = 0.1
        density_R = 1.0
        
        dome_sphere = ParametricShapeModel()
        dome_sphere.parameters['radius'] = aligned_meristem_models[t].parameters['dome_radius'] 
        dome_sphere.parameters['center'] = aligned_meristem_models[t].shape_model['dome_center']
        dome_sphere.parameters['scales'] = aligned_meristem_models[t].shape_model['dome_scales']
        dome_sphere.parameters['axes'] = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        
        dome_density = sphere_density_function(dome_sphere,k=density_k,R=density_R)(cell_points.values()[:,0],cell_points.values()[:,1],cell_points.values()[:,2])
        
        dome_cells = TriangularMesh()
        dome_cells.points = cell_points
        dome_cells.point_data = array_dict(dome_density,cell_points.keys()).to_dict()
            
        organ_spheres[t] = {}
        if t == time_steps[0]:
            considered_primordia = np.arange(meristem_models[time_steps[0]].parameters['n_primordia'])+1
        else:
            considered_primordia = organ_spheres[time_steps[0]].keys()
        
        for p in considered_primordia:
            try:
                if t == time_steps[0]:
                    #angle = aligned_meristem_models[t].parameters["primordium_"+str(p+offset_gap)+"_angle"] % 360
                    angle = matched_meristem_models[t].parameters["primordium_"+str(p)+"_angle"] % 360
                else: 
                    #angle = aligned_meristem_models[t].parameters["primordium_"+str(p)+"_angle"] % 360
                    angle = matched_meristem_models[t].parameters["primordium_"+str(p)+"_angle"] % 360
                    
                print "Primordium ",p," : ",angle
                organ_sphere = ParametricShapeModel()
                organ_sphere.parameters['theta'] = angle
                organ_theta = organ_sphere.parameters['theta']
                
                # if t == time_steps[0]:
                #     organ_sphere.parameters['radius'] = aligned_meristem_models[t].parameters["primordium_"+str(p+offset_gap)+"_radius"]
                #     organ_sphere.parameters['center'] = aligned_meristem_models[t].shape_model['primordia_centers'][p-1+offset_gap]
                # else:
                #     organ_sphere.parameters['radius'] = aligned_meristem_models[t].parameters["primordium_"+str(p)+"_radius"]
                #     organ_sphere.parameters['center'] = aligned_meristem_models[t].shape_model['primordia_centers'][p-1]
                    
                organ_sphere.parameters['radius'] = matched_meristem_models[t].parameters["primordium_"+str(p)+"_radius"]
                organ_sphere.parameters['center'] = matched_meristem_models[t].shape_model['primordia_centers'][p-1]
                    
                organ_sphere.parameters['scales'] = np.array([1.,1.,1.])
                organ_sphere.parameters['axes'] = np.array([[np.cos(organ_theta),np.sin(organ_theta),0.],[-np.sin(organ_theta),np.cos(organ_theta),0.],[0.,0.,1.]])
                organ_spheres[t][p] = organ_sphere
            except:
                pass
            
        organ_densities = np.transpose([sphere_density_function(organ_spheres[t][p],k=density_k,R=density_R)(cell_points.values()[:,0],cell_points.values()[:,1],cell_points.values()[:,2]) for p in organ_spheres[t].keys()])
        organ_memberships[t] = organ_densities/(dome_density+organ_densities.sum(axis=1)+0.000001)[:,np.newaxis]
        
        dome_cells.point_data = array_dict(organ_memberships[t].max(axis=1),cell_points.keys()).to_dict()
        world.add(dome_cells,'dome_cells_'+t,position=size*resolution/2.,colormap='jet',point_radius=2)
            
        relative_organ_points[t] = {}
        for p in organ_spheres[t].keys():
            organ_center = organ_spheres[t][p].parameters['center']
            #organ_rotation = organ_spheres[t][p].parameters['axes']
            organ_rotation =  np.identity(3)
            relative_points = cell_points.values()-organ_center[np.newaxis,:]
            relative_points = np.einsum('...ij,...j->...i',organ_rotation,relative_points)
            relative_organ_points[t][p] = relative_points
        raw_input()
        
    figure = plt.figure(1)
    figure.clf()    
    figure.patch.set_facecolor('white')
    
    for i_time,t in enumerate(time_steps):
        epidermis_cell_vectors = aligned_positions[t].values(epidermis_cells[t]) - reference_dome_apex
        epidermis_cell_distances = np.linalg.norm(epidermis_cell_vectors[:,:2],axis=1) 
        epidermis_cell_cosines = epidermis_cell_vectors[:,0]/epidermis_cell_distances
        epidermis_cell_sinuses = epidermis_cell_vectors[:,1]/epidermis_cell_distances
        epidermis_cell_angles = np.arctan(epidermis_cell_sinuses/epidermis_cell_cosines)
        epidermis_cell_angles[epidermis_cell_cosines<0] = np.pi + epidermis_cell_angles[epidermis_cell_cosines<0]
        
        epidermis_cell_x = epidermis_cell_distances*np.cos(epidermis_cell_angles)
        epidermis_cell_y = epidermis_cell_distances*np.sin(epidermis_cell_angles)
        epidermis_cell_z = epidermis_cell_distances*0
        projected_epidermis_points = dict(zip(epidermis_cells[t],np.transpose([epidermis_cell_x,epidermis_cell_y,epidermis_cell_z])))
        
        rho = np.linspace(0,200,81)
        theta = np.linspace(0,2*np.pi,360)
        R,T = np.meshgrid(rho,theta)
        
        e_x = R*np.cos(T)
        e_y = R*np.sin(T)
        e_z = R*0
        
        cell_radius = 5.0
        density_k = 0.15
        
        epidermis_nuclei_potential = np.array([nuclei_density_function(dict([(p,projected_epidermis_points[p])]),cell_radius=cell_radius,k=density_k)(e_x,e_y,e_z) for p in epidermis_cells[t]])
        epidermis_nuclei_potential = np.transpose(epidermis_nuclei_potential,(1,2,0))
        epidermis_nuclei_density = np.sum(epidermis_nuclei_potential,axis=2)
        epidermis_nuclei_membership = epidermis_nuclei_potential/epidermis_nuclei_density[...,np.newaxis]
        epidermis_nuclei_ratio = np.sum(epidermis_nuclei_membership*cell_fluorescence_ratios[t].point_data.values(epidermis_cells[t])[np.newaxis,np.newaxis,:],axis=2)
        
        epidermis_model_density = np.array([[[aligned_meristem_models[t].shape_model_density_function()(e_x[i,j,np.newaxis][:,np.newaxis,np.newaxis]+reference_dome_apex[0],e_y[i,j,np.newaxis][np.newaxis,:,np.newaxis]+reference_dome_apex[1],e_z[i,j,np.newaxis][np.newaxis,np.newaxis,:]+(k*size*resolution/2.)[2])[0,0,0] for k in np.arange(1,4)] for j in xrange(e_x.shape[1])] for i in xrange(e_x.shape[0])]).max(axis=2)
        
        signal_name = signal_names[i_time]   
        
        figure = plt.figure(1)
        figure.clf()    
        figure.patch.set_facecolor('white')
        ax = plt.subplot(111, polar=True)
        
        #levels = np.arange(0,1.1,0.05)
        ratio_min = cell_fluorescence_ratios[t].point_data.values().mean() - np.sqrt(2.)*cell_fluorescence_ratios[t].point_data.values().std()
        ratio_max = cell_fluorescence_ratios[t].point_data.values().mean() + np.sqrt(2.)*cell_fluorescence_ratios[t].point_data.values().std()
        ratio_step = (ratio_max - ratio_min)/100.
        levels = np.arange(ratio_min-ratio_step,ratio_max+ratio_step,ratio_step)
        levels[0] = 0
        levels[-1] = 2
        
        #ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap=signal_colors[signal_name],alpha=1.0,antialiased=True,vmin=ratio_min,vmax=ratio_max)  
        #ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap='YWGn',alpha=1.0/len(filenames),antialiased=True,vmin=0.5,vmax=1.0)    
        levels = [-1.0,0.5]
        levels = np.arange(0,5.1,0.05)
        #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0,antialiased=True)
        #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0/len(filenames),antialiased=True)
        #ax.contour(T,R,epidermis_model_density,levels,cmap='RdBu',alpha=1.0,antialiased=True)
        #ax.contour(T,R,epidermis_model_density,levels,colors='k',alpha=1.0,antialiased=True)
        #ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=0.8,antialiased=True)
        #ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=2.0/len(filenames),antialiased=True)
        #ax.scatter(epidermis_cell_angles,epidermis_cell_distances,s=15.0,c=[0.2,0.7,0.1],alpha=0.5)
        #levels = [0.0,1.0]
        #ax.contour(T,R,epidermis_nuclei_density,levels,colors='w',alpha=1,antialiased=True)
        ax.contourf(T,R,epidermis_nuclei_density,levels,cmap='gray',alpha=1,antialiased=True)
        ax.set_rmax(200)    
        ax.set_rmin(0)
        ax.grid(True)
        ax.set_yticklabels([])
        plt.show(block=False)
        raw_input()
        
        
    golden_angle = 180.*np.sign(growing_meristem_model.parameters['orientation'])*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)/np.pi
    auxin_map = False
    nuclei_deformation = True
    surface_map = True
    normalized_signal = True
    
    angle_resolution = 1.0
    minute_step = 20.
    n_steps = int(60.*(time_hours[1] - time_hours[0])/minute_step)
    
    grid_resolution = resolution*[2,2,2]
    x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:2*grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:2*grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:2*grid_resolution[2]]
    grid_size = 1.5*size
    
    import matplotlib.pyplot as plt
    figure = plt.figure(2)
    figure.clf()
    
    world.clear()
    
    for step in xrange(n_steps+1):
        
        hours = time_hours[0] + step*(time_hours[1]-time_hours[0])/n_steps
        minutes = np.round(60.*(hours - int(hours)))
        hours_minutes = "%02d-%02d" % (hours, minutes)
        
        parameters = {}
        time_weights = [(n_steps-step)/float(n_steps), step/float(n_steps)]
        
        for p in growing_meristem_model.parameters.keys():
            if isinstance(meristem_models[time_steps[0]].parameters[p],float):
                parameters[p] = time_weights[0]*meristem_models[time_steps[0]].parameters[p] + time_weights[1]*meristem_models[time_steps[1]].parameters[p]
            else:
                parameters[p] = meristem_models[time_steps[0]].parameters[p]
        
        parameters['dome_apex_x'] = reference_dome_apex[0]
        parameters['dome_apex_y'] = reference_dome_apex[1]
        parameters['dome_apex_z'] = reference_dome_apex[2]
        parameters['dome_phi'] = 0
        parameters['dome_psi'] = 0
        parameters['initial_angle'] = 0
        parameters['initial_angle'] -= previous_offset*golden_angle
        
        for p in (np.arange(meristem_models[time_steps[0]].parameters['n_primordia'])+1):
            # try:
            #     # angle_0 = (meristem_models[time_steps[0]].parameters["primordium_"+str(p+offset_gap)+"_angle"] + meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle) % 360
            #     # angle_1 = (meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_angle"] + meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle - offset_gap*golden_angle) % 360
            #     # parameters["primordium_"+str(p)+"_angle"] = time_weights[0]*angle_0 + time_weights[1]*angle_1
            #     # parameters["primordium_"+str(p)+"_distance"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p+offset_gap)+"_distance"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_distance"]
            #     # parameters["primordium_"+str(p)+"_height"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p+offset_gap)+"_height"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_height"]
            #     # parameters["primordium_"+str(p)+"_radius"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p+offset_gap)+"_radius"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_radius"]
            #     angle_0 = (meristem_models[time_steps[0]].parameters["primordium_"+str(p-offset_gap)+"_angle"] + meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle) % 360
            #     angle_1 = (meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_angle"] + meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle - offset_gap*golden_angle) % 360
            #     parameters["primordium_"+str(p)+"_angle"] = time_weights[0]*angle_0 + time_weights[1]*angle_1
            #     parameters["primordium_"+str(p)+"_distance"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p-offset_gap)+"_distance"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_distance"]
            #     parameters["primordium_"+str(p)+"_height"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p-offset_gap)+"_height"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_height"]
            #     parameters["primordium_"+str(p)+"_radius"] = time_weights[0]*meristem_models[time_steps[0]].parameters["primordium_"+str(p-offset_gap)+"_radius"] + time_weights[1]*meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_radius"]
            # except:
            #      parameters["primordium_"+str(p)+"_angle"] = meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_angle"] + meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle - offset_gap*golden_angle
            #      parameters["primordium_"+str(p)+"_distance"] = meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_distance"]
            #      parameters["primordium_"+str(p)+"_height"] = meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_height"]
            #      parameters["primordium_"+str(p)+"_radius"] = meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_radius"]
            parameters["primordium_"+str(p)+"_angle"] = time_weights[0]*matched_meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_angle"] + time_weights[1]*matched_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_angle"]
            parameters["primordium_"+str(p)+"_distance"] = time_weights[0]*matched_meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_distance"] + time_weights[1]*matched_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_distance"]
            parameters["primordium_"+str(p)+"_height"] = time_weights[0]*matched_meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_height"] + time_weights[1]*matched_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_height"]
            parameters["primordium_"+str(p)+"_radius"] = time_weights[0]*matched_meristem_models[time_steps[0]].parameters["primordium_"+str(p)+"_radius"] + time_weights[1]*matched_meristem_models[time_steps[1]].parameters["primordium_"+str(p)+"_radius"]

        
        # for p in parameters.keys():
        #     if ('primordium' in p) and ('angle' in p):
        #         parameters[p] += time_weights[0]*meristem_models[time_steps[0]].parameters['primordium_offset']*golden_angle
        #         parameters[p] += time_weights[1]*meristem_models[time_steps[1]].parameters['primordium_offset']*golden_angle 
        
        growing_meristem_model.parameters = deepcopy(parameters)
        growing_meristem_model.update_shape_model()
        
        time_cell_points = {}
        if nuclei_deformation:
            for t_id,t in enumerate(time_steps):
                cell_points = array_dict(aligned_cell_fluorescence_ratios[t].points)
                dome_center = growing_meristem_model.shape_model['dome_center']
                dome_apex = np.array([growing_meristem_model.parameters['dome_apex_x'],growing_meristem_model.parameters['dome_apex_y'],growing_meristem_model.parameters['dome_apex_z']])
        
                deformed_points = cell_points.values()
                deformed_points = dome_apex + (1.-organ_memberships[t].sum(axis=1)[:,np.newaxis])*(deformed_points-dome_apex)
                #deformed_points = dome_center + (1.-organ_memberships.sum(axis=1)[:,np.newaxis])*(deformed_points-dome_center)
                for o,p in enumerate(organ_spheres[t].keys()):
                    organ_spheres[t][p].parameters['theta'] = growing_meristem_model.parameters["primordium_"+str(p)+"_angle"]
                    organ_theta = organ_spheres[t][p].parameters['theta']
                    organ_spheres[t][p].parameters['radius'] = growing_meristem_model.parameters["primordium_"+str(p)+"_radius"]
                    organ_spheres[t][p].parameters['center'] = growing_meristem_model.shape_model['primordia_centers'][p-1]
                    organ_spheres[t][p].parameters['scales'] = np.array([1.,1.,1.])
                    organ_spheres[t][p].parameters['axes'] = np.array([[np.cos(organ_theta),np.sin(organ_theta),0.],[-np.sin(organ_theta),np.cos(organ_theta),0.],[0.,0.,1.]])
                    organ_center = organ_spheres[t][p].parameters['center']
                    #organ_rotation = np.linalg.inv(organ_spheres[p].parameters['axes'])
                    organ_rotation =  np.identity(3)*(organ_spheres[t][p].parameters['radius']/aligned_meristem_models[t].parameters["primordium_"+str(p)+"_radius"])
                    deformed_points += organ_memberships[t][:,o,np.newaxis]*(organ_center-dome_apex + np.einsum('...ij,...j->...i',organ_rotation,relative_organ_points[t][p]))
                    #deformed_points += organ_memberships[t][:,o,np.newaxis]*(organ_center-dome_center + np.einsum('...ij,...j->...i',organ_rotation,relative_organ_points[t][p]))
          
                time_cell_points[t] = array_dict(deformed_points,cell_points.keys())
                dome_cells.points = time_cell_points[t].to_dict()
                #dome_cells.point_data = array_dict(organ_memberships[t].max(axis=1),cell_points.keys()).to_dict()
                dome_cells.point_data = cell_fluorescence_ratios[t].point_data
                #dome_cells.point_data = {}
                if not auxin_map and t_id == 0:
                    #world.add(dome_cells,'growing_dome_cells_'+t,position=size*resolution/2.,colormap='jet',intesity_range=(0.0,1.0),alpha=1.0,point_radius=2.0)
                    world.add(dome_cells,'growing_dome_cells_'+t,position=size*resolution/2.,colormap=signal_colors[signal_name],alpha=time_weights[t_id],point_radius=2)
                    #world.add(dome_cells,'growing_dome_cells_'+t,position=size*resolution/2.,colormap='vegetation',intensity_range=(0.75,1.0),alpha=1.0,point_radius=2)
                #world.add(dome_cells,'growing_dome_cells_'+t,position=size*resolution/2.,colormap=signal_colors[signal_name],alpha=time_weights[t_id],point_radius=2)
                    
            # if auxin_map == False:
            #     hausdorff_t0 =  vq(time_cell_points[time_steps[1]].values(),time_cell_points[time_steps[0]].values())[1]
            #     hausdorff_t1 =  vq(time_cell_points[time_steps[0]].values(),time_cell_points[time_steps[1]].values())[1]
        
            #     hausdorff_t0_values = np.array(np.minimum(np.around(100.*hausdorff_t0/10.),101),int)
            #     histo_hausdorff_t0 = np.array([nd.sum(np.ones_like(hausdorff_t0_values,float),hausdorff_t0_values,index=dist) for dist in xrange(101)])/(len(hausdorff_t0_values))
            #     hausdorff_t1_values = np.array(np.minimum(np.around(100.*hausdorff_t1/10.),101),int)
            #     histo_hausdorff_t1 = np.array([nd.sum(np.ones_like(hausdorff_t1_values,float),hausdorff_t1_values,index=dist) for dist in xrange(101)])/(len(hausdorff_t1_values))
            #     for dist in np.arange(100)+1 : 
            #         histo_hausdorff_t0[dist] += histo_hausdorff_t0[dist-1]
            #         histo_hausdorff_t1[dist] += histo_hausdorff_t1[dist-1]
        
            #     histo_plot(figure,100.*hausdorff_t0/10.,color=np.array([0.0,0,0.6]),bar=False,cumul=True,xlabel="Hausdorff distance (%)",ylabel="Cells (%)",alpha=(n_steps-step)/float(n_steps+1))
            #     histo_plot(figure,100.*hausdorff_t1/10.,color=np.array([0.8,0.0,0.0]),bar=False,cumul=True,xlabel="Hausdorff distance (%)",ylabel="Cells (%)",alpha=(step+1)/float(n_steps+1))
        
            #     if step == 0:
            #         stat.write("haussdorf_"+time_steps[1]+"_"+time_steps[0]+";")
            #         stat.write(str(density_R)+";"+str(density_k)+";")
            #         for dist in xrange(101):
            #             stat.write(str(histo_hausdorff_t1[dist])+";")
            #         stat.write("\n")
            #     if step == n_steps:
            #         stat.write("haussdorf_"+time_steps[0]+"_"+time_steps[1]+";")
            #         stat.write(str(density_R)+";"+str(density_k)+";")
            #         for dist in xrange(101):
            #             stat.write(str(histo_hausdorff_t0[dist])+";")
            #         stat.write("\n")
            
        else:
            for t_id,t in enumerate(time_steps):
                time_cell_points[t] = aligned_cell_fluorescence_ratios[t].points
        
        if surface_map:
            figure = plt.figure(1)
            figure.clf()    
            figure.patch.set_facecolor('white')
            
            epidermis_nuclei_distances = {}
            epidermis_nuclei_angles = {}
            epidermis_nuclei_ratios = {}
            
            for i_time,t in enumerate(time_steps):
                epidermis_cell_vectors = time_cell_points[t].values(epidermis_cells[t]) - reference_dome_apex
                epidermis_cell_distances = np.linalg.norm(epidermis_cell_vectors[:,:2],axis=1) 
                epidermis_cell_cosines = epidermis_cell_vectors[:,0]/epidermis_cell_distances
                epidermis_cell_sinuses = epidermis_cell_vectors[:,1]/epidermis_cell_distances
                epidermis_cell_angles = np.arctan(epidermis_cell_sinuses/epidermis_cell_cosines)
                epidermis_cell_angles[epidermis_cell_cosines<0] = np.pi + epidermis_cell_angles[epidermis_cell_cosines<0]
                epidermis_nuclei_distances[t] = deepcopy(epidermis_cell_distances)
                epidermis_nuclei_angles[t] = deepcopy(epidermis_cell_angles)
                
                epidermis_cell_x = epidermis_cell_distances*np.cos(epidermis_cell_angles)
                epidermis_cell_y = epidermis_cell_distances*np.sin(epidermis_cell_angles)
                epidermis_cell_z = epidermis_cell_distances*0
                projected_epidermis_points = dict(zip(epidermis_cells[t],np.transpose([epidermis_cell_x,epidermis_cell_y,epidermis_cell_z])))
                
                rho = np.linspace(0,160,81)
                theta = np.linspace(0,2*np.pi,360./angle_resolution)
                R,T = np.meshgrid(rho,theta)
                
                e_x = R*np.cos(T)
                e_y = R*np.sin(T)
                e_z = R*0
                
                cell_radius = 5.0
                density_k = 0.15
                
                epidermis_nuclei_potential = np.array([nuclei_density_function(dict([(p,projected_epidermis_points[p])]),cell_radius=cell_radius,k=density_k)(e_x,e_y,e_z) for p in epidermis_cells[t]])
                epidermis_nuclei_potential = np.transpose(epidermis_nuclei_potential,(1,2,0))
                epidermis_nuclei_density = np.sum(epidermis_nuclei_potential,axis=2)
                epidermis_nuclei_membership = epidermis_nuclei_potential/epidermis_nuclei_density[...,np.newaxis]
                epidermis_nuclei_ratio = np.sum(epidermis_nuclei_membership*cell_fluorescence_ratios[t].point_data.values(epidermis_cells[t])[np.newaxis,np.newaxis,:],axis=2)
                epidermis_nuclei_ratios[t] = deepcopy(epidermis_nuclei_ratio)
            
            epidermis_nuclei_ratio = time_weights[0]*epidermis_nuclei_ratios[time_steps[0]]
            epidermis_nuclei_ratio += time_weights[1]*epidermis_nuclei_ratios[time_steps[1]]
                
            epidermis_model_density = np.array([[[growing_meristem_model.shape_model_density_function()(e_x[i,j,np.newaxis][:,np.newaxis,np.newaxis]+reference_dome_apex[0],e_y[i,j,np.newaxis][np.newaxis,:,np.newaxis]+reference_dome_apex[1],e_z[i,j,np.newaxis][np.newaxis,np.newaxis,:]+(k*size*resolution/2.)[2])[0,0,0] for k in np.arange(1,4)] for j in xrange(e_x.shape[1])] for i in xrange(e_x.shape[0])]).max(axis=2)
                
            signal_name = signal_names[0]   
                
            figure = plt.figure(1)
            figure.clf()    
            figure.patch.set_facecolor('white')
            ax = plt.subplot(111, polar=True)
                
            #levels = np.arange(0,1.1,0.05)
            if normalized_signal:
                ratio_min = np.sum([w*(cell_fluorescence_ratios[t].point_data.values().mean() - np.sqrt(2.)*cell_fluorescence_ratios[t].point_data.values().std()) for t,w in zip (time_steps,time_weights)])
                ratio_max = np.sum([w*(cell_fluorescence_ratios[t].point_data.values().mean() + np.sqrt(2.)*cell_fluorescence_ratios[t].point_data.values().std()) for t,w in zip (time_steps,time_weights)])
            else:
                ratio_min = 0.0
                ratio_max = 1.0
            ratio_step = (ratio_max - ratio_min)/20.
            levels = np.arange(ratio_min-ratio_step,ratio_max+ratio_step,ratio_step)
            levels[0] = 0
            levels[-1] = 2
                
            ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap=signal_colors[signal_name],alpha=1.0,antialiased=True,vmin=ratio_min,vmax=ratio_max)  
            #ax.contourf(T,R,epidermis_nuclei_ratio,levels,cmap='YWGn',alpha=1.0/len(filenames),antialiased=True,vmin=0.5,vmax=1.0)    
            levels = [-1.0,0.5]
            #levels = np.arange(0,5.1,0.05)
            #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0,antialiased=True)
            #ax.contourf(T,R,epidermis_nuclei_density,levels,colors='k',alpha=1.0/len(filenames),antialiased=True)
            #ax.contour(T,R,epidermis_model_density,levels,cmap='RdBu',alpha=1.0,antialiased=True)
            #ax.contour(T,R,epidermis_model_density,levels,colors='k',alpha=1.0,antialiased=True)
            ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=0.8,antialiased=True)
            #ax.contourf(T,R,epidermis_model_density,levels,colors='k',alpha=2.0/len(filenames),antialiased=True)
            #for t,w in zip(time_steps,time_weights):
            #    ax.scatter(epidermis_nuclei_angles[t],epidermis_nuclei_distances[t],s=15.0,c=[0.2,0.7,0.1],alpha=w)
            #levels = [0.0,1.0]
            #ax.contour(T,R,epidermis_nuclei_density,levels,colors='w',alpha=1,antialiased=True)
            #ax.contourf(T,R,epidermis_nuclei_density,levels,cmap='jet',alpha=1,antialiased=True)
            ax.set_rmax(160)    
            ax.set_rmin(0)
            ax.grid(True)
            ax.set_yticklabels([])
            
            screenshot_file = dirname+"/nuclei_images/"+filename+"/2d_auxin_map/"+filename+"_map_"+hours_minutes+".jpg"
            figure.savefig(screenshot_file)
            
            plt.show(block=False)
            #raw_input()    
        
        if auxin_map:
            surface_points,surface_triangles = implicit_surface(growing_meristem_model.shape_model_density_function()(x,y,z),grid_size,resolution,iso=0.5)
        
            meristem_model_surface_mesh = TriangularMesh()
            meristem_model_surface_mesh.points = array_dict(surface_points,np.arange(len(surface_points))).to_dict()
            meristem_model_surface_mesh.triangles = array_dict(surface_triangles,np.arange(len(surface_triangles))).to_dict()
            #surface_mesh.triangle_data = array_dict(np.arange(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
            meristem_model_surface_mesh.triangle_data = array_dict(np.ones(len(surface_triangles)),np.arange(len(surface_triangles))).to_dict()
        
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
            
            surface_point_positions = surface_topomesh.wisp_property('barycenter',0).values()
            density_k = 2.0
            
            surface_dome_sphere = ParametricShapeModel()
            surface_dome_sphere.parameters['radius'] = growing_meristem_model.parameters['dome_radius'] 
            surface_dome_sphere.parameters['center'] = growing_meristem_model.shape_model['dome_center']-resolution*size/2.
            surface_dome_sphere.parameters['scales'] = growing_meristem_model.shape_model['dome_scales']
            surface_dome_sphere.parameters['axes'] = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            surface_dome_density = sphere_density_function(surface_dome_sphere,k=density_k,R=1)(surface_point_positions[:,0],surface_point_positions[:,1],surface_point_positions[:,2])
    
            density_k = 0.15
            surface_organ_spheres = {}
            for p in np.arange(8)+1:
                surface_organ_sphere = ParametricShapeModel()
                surface_organ_sphere.parameters['theta'] = growing_meristem_model.parameters["primordium_"+str(p)+"_angle"] % 360
                organ_theta = surface_organ_sphere.parameters['theta']

                surface_organ_sphere.parameters['radius'] = growing_meristem_model.parameters["primordium_"+str(p)+"_radius"]
                surface_organ_sphere.parameters['center'] = growing_meristem_model.shape_model['primordia_centers'][p-1]-resolution*size/2.
                    
                surface_organ_sphere.parameters['scales'] = np.array([1.,1.,1.])
                surface_organ_sphere.parameters['axes'] = np.array([[np.cos(organ_theta),np.sin(organ_theta),0.],[-np.sin(organ_theta),np.cos(organ_theta),0.],[0.,0.,1.]])
                surface_organ_spheres[p] = surface_organ_sphere

            surface_organ_densities = np.transpose([sphere_density_function(surface_organ_spheres[p],k=density_k,R=1)(surface_point_positions[:,0],surface_point_positions[:,1],surface_point_positions[:,2]) for p in surface_organ_spheres.keys()])
            surface_organ_memberships = surface_organ_densities/(surface_dome_density+surface_organ_densities.sum(axis=1)+0.000001)[:,np.newaxis]
            surface_dome_memberships = 1 - surface_organ_memberships.sum(axis=1)
        
            surface_dome_vectors = (surface_point_positions - surface_dome_sphere.parameters['center'])/surface_dome_sphere.parameters['scales']
            surface_dome_targets = float(surface_dome_sphere.parameters['radius'])*(surface_dome_vectors/np.linalg.norm(surface_dome_vectors,axis=1)[:,np.newaxis])
            surface_dome_projectors = surface_dome_sphere.parameters['scales']*(surface_dome_targets-surface_dome_vectors)*surface_dome_memberships[:,np.newaxis]
        
            surface_organ_projectors = surface_dome_projectors
            #surface_organ_projectors = np.zeros_like(surface_point_positions)
            for i,p in enumerate(surface_organ_spheres.keys()):
                surface_organ_vectors = surface_point_positions - surface_organ_spheres[p].parameters['center']
                surface_organ_targets = float(surface_organ_spheres[p].parameters['radius'])*surface_organ_vectors/np.linalg.norm(surface_organ_vectors,axis=1)[:,np.newaxis]
                surface_organ_projectors += (surface_organ_targets-surface_organ_vectors)*surface_organ_memberships[:,i][:,np.newaxis]
            
            compute_topomesh_property(surface_topomesh,'barycenter',0,positions=array_dict(surface_point_positions+surface_organ_projectors,keys=list(surface_topomesh.wisps(0))))
    
            from vplants.meshing.optimization_tools    import optimize_topomesh
            surface_topomesh = optimize_topomesh(surface_topomesh,omega_forces=dict([('taubin_smoothing',1.0)]),iterations=10)
            
            from vplants.meshing.triangular_mesh import topomesh_to_triangular_mesh
            meristem_model_mesh,_,_ = topomesh_to_triangular_mesh(surface_topomesh,mesh_center=np.array([0,0,0]))
            #world.add(meristem_model_mesh,'growing_meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='leaf',alpha=1.0)
        
        
            cell_radius = 5.0
            density_k = 0.3
            surface_vertex_cell_memberships = {}
            surface_point_positions = surface_topomesh.wisp_property('barycenter',0).values(list(surface_topomesh.wisps(0)))
            for t in time_steps:
                surface_vertex_cell_memberships[t] = array_dict(np.transpose([nuclei_density_function(dict([(p,time_cell_points[t][p]- size*resolution/2.)]),cell_radius=cell_radius,k=density_k)(surface_point_positions[:,0],
                                                                                                            surface_point_positions[:,1],           
                                                                                                            surface_point_positions[:,2]) for p in aligned_cell_fluorescence_ratios[t].points.keys()]),keys=list(surface_topomesh.wisps(0)))
            surface_vertex_ratio = np.zeros_like(list(surface_topomesh.wisps(0)),float)
            for t,w in zip(time_steps, time_weights):
                time_vertex_ratio = (surface_vertex_cell_memberships[t].values(list(surface_topomesh.wisps(0)))*aligned_cell_fluorescence_ratios[t].point_data.values()).sum(axis=1)/surface_vertex_cell_memberships[t].values(list(surface_topomesh.wisps(0))).sum(axis=1)
                time_vertex_ratio[np.where(np.isnan(time_vertex_ratio))] = 0.0
                surface_vertex_ratio += w*time_vertex_ratio
            surface_vertex_ratio = array_dict(surface_vertex_ratio,keys=list(surface_topomesh.wisps(0)))
            
            #meristem_model_mesh.triangle_data = array_dict(meristem_model_mesh.triangle_data.keys(),meristem_model_mesh.triangle_data.keys()).to_dict()
            meristem_model_mesh.triangle_data = {}
            meristem_model_mesh.point_data = surface_vertex_ratio.to_dict()
            #meristem_model_mesh.point_data={}
            
            if normalized_signal:
                ratio_min = np.sum([w*(cell_fluorescence_ratios[t].point_data.values().mean() - np.sqrt(2.)*np.maximum(cell_fluorescence_ratios[t].point_data.values().std(),0.02)) for t,w in zip (time_steps,time_weights)])
                ratio_max = np.sum([w*(cell_fluorescence_ratios[t].point_data.values().mean() + np.sqrt(2.)*np.maximum(cell_fluorescence_ratios[t].point_data.values().std(),0.02)) for t,w in zip (time_steps,time_weights)])

            else:
                ratio_min = 0.0
                ratio_max = 1.0
            
            #world.add(meristem_model_mesh,'growing_meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap='vegetation',alpha=1.0)
            world.add(meristem_model_mesh,'growing_meristem_model_mesh',_repr_vtk_=TriangularMesh._repr_vtk_,colormap=signal_colors[signal_name],intensity_range=(ratio_min,ratio_max),alpha=1.0)
            #raw_input()
                
        elif not nuclei_deformation:
            world.add(growing_meristem_model,'growing_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1,z_slice=(95,100))

        screenshot_file = dirname+"/nuclei_images/"+filename+"/fluorescence_map/"+filename+"_auxin_map_"+hours_minutes+".jpg"
        #screenshot_file = dirname+"/nuclei_images/"+filename+"/growing_nuclei/"+filename+"_growing_nuclei_"+hours_minutes+".jpg"
        viewer.save_screenshot(screenshot_file)
        #raw_input()
    # print "Hausdorff : ",hausdorff_t0.mean()," [",hausdorff_t0.max(),"]  /  ",hausdorff_t1.mean()," [",hausdorff_t1.max(),"]"
    # plt.show()
    

# stat.flush()
# stat.close()
    

        
        
    
    
