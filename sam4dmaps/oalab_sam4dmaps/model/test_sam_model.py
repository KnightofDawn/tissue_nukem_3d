import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq                       import kmeans, vq

from openalea.container.array_dict             import array_dict
from openalea.container.property_topomesh     import PropertyTopomesh

from vplants.meshing.triangular_mesh import TriangularMesh
from vplants.sam4dmaps.parametric_shape import ParametricShapeModel, implicit_surface
from vplants.sam4dmaps.parametric_shape import spherical_parametric_meristem_model
#from vplants.sam4dmaps.parametric_shape import meristem_model_density_function, draw_meristem_model_pgl, meristem_model_energy

from copy import deepcopy

world.clear()
viewer.ren.SetBackground(1,1,1)

size = np.array([400,400,400])
resolution = np.array([0.25,0.25,0.25])

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

import vplants.sam4dmaps.parametric_shape
reload(vplants.sam4dmaps.parametric_shape)
from vplants.sam4dmaps.parametric_shape import phyllotaxis_based_parametric_meristem_model


clockwise_meristem_model = ParametricShapeModel()
counterclockwise_meristem_model = ParametricShapeModel()

initial_parameters = {}
initial_parameters['dome_apex_x'] = size[0]*resolution[0]/2.
initial_parameters['dome_apex_y'] = size[1]*resolution[1]/2.
initial_parameters['dome_apex_z'] = 0
initial_parameters['dome_radius'] = 70
initial_parameters['initial_angle'] = 0.
initial_parameters['dome_psi'] = 0.
initial_parameters['dome_phi'] = 0.
initial_parameters['n_primordia'] = 8
initial_parameters['developmental_time'] = 0.

clockwise_meristem_model.parameters = deepcopy(initial_parameters)
clockwise_meristem_model.parameters['orientation'] = 1.
clockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
clockwise_meristem_model.update_shape_model()
clockwise_meristem_model.density_function = meristem_model_density_function
clockwise_meristem_model.drawing_function = draw_meristem_model_vtk

# counterclockwise_meristem_model.parameters = deepcopy(initial_parameters)
# counterclockwise_meristem_model.parameters['orientation'] = -1.
# counterclockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
# counterclockwise_meristem_model.update_shape_model()
# counterclockwise_meristem_model.density_function = meristem_model_density_function
# counterclockwise_meristem_model.drawing_function = draw_meristem_model_vtk

from vplants.sam4dmaps.parametric_shape import draw_meristem_model_pgl

world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=1.0)

#world.add(counterclockwise_meristem_model,'counterlockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='leaf',alpha=0.1)
raw_input()


for time in xrange(1000):
#for phi in xrange(300):
    clockwise_meristem_model.parameters['developmental_time'] = time-500
    #clockwise_meristem_model.parameters['initial_angle'] = time 
#    clockwise_meristem_model.parameters['dome_phi'] = phi
    clockwise_meristem_model.update_shape_model()
    world.add(clockwise_meristem_model,'clockwise_meristem_model',position=resolution*size/2.,_repr_vtk_=draw_meristem_model_vtk,colormap='green',alpha=1.0)
    
    
    


