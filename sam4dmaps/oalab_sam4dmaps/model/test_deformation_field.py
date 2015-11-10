import numpy as np
from scipy import ndimage as nd

from openalea.container import array_dict

from vplants.meshing.triangular_mesh import TriangularMesh, _points_repr_vtk_
from vplants.sam4dmaps.parametric_shape import ParametricShapeModel

world.clear()
viewer.ren.SetBackground(1,1,1)

grid_size = 30
grid_step = 2.
grid = np.ogrid[-grid_size:(grid_size+1),-grid_size:(grid_size+1)]
grid = np.transpose([grid[0] + np.zeros_like(grid[1]),grid[1] + np.zeros_like(grid[0]),np.zeros_like(grid[0]) + np.zeros_like(grid[1])])
grid = np.array(grid_step*grid.reshape(grid.shape[0]*grid.shape[1],grid.shape[2]),float)
np.random.shuffle(grid)

grid_points = TriangularMesh()
grid_points.points = array_dict(grid,np.arange(grid.shape[0])).to_dict()
#world.add(grid_points,name='grid_points',colormap='morocco')
#raw_input()

def draw_sphere_vtk(sphere):
    import vtk
    from time import time
    
    sphere_polydata = vtk.vtkPolyData()
    sphere_points = vtk.vtkPoints()
    sphere_triangles = vtk.vtkCellArray()
    sphere_data = vtk.vtkLongArray()

    start_time = time()
    print "--> Creating VTK PolyData"
    
    sphere_sphere = vtk.vtkSphereSource()
    #sphere_sphere.SetCenter(sphere.parameters['sphere_center'])
    sphere_sphere.SetRadius(sphere.parameters['radius'])
    sphere_sphere.SetThetaResolution(32)
    sphere_sphere.SetPhiResolution(32)
    sphere_sphere.Update()
    ellipsoid_transform = vtk.vtkTransform()
    axes_transform = vtk.vtkLandmarkTransform()
    source_points = vtk.vtkPoints()
    source_points.InsertNextPoint([1,0,0])
    source_points.InsertNextPoint([0,1,0])
    source_points.InsertNextPoint([0,0,1])
    target_points = vtk.vtkPoints()
    target_points.InsertNextPoint(sphere.parameters['axes'][0])
    target_points.InsertNextPoint(sphere.parameters['axes'][1])
    target_points.InsertNextPoint(sphere.parameters['axes'][2])
    axes_transform.SetSourceLandmarks(source_points)
    axes_transform.SetTargetLandmarks(target_points)
    axes_transform.SetModeToRigidBody()
    axes_transform.Update()
    ellipsoid_transform.SetMatrix(axes_transform.GetMatrix())
    ellipsoid_transform.Scale(sphere.parameters['scales'][0],
                              sphere.parameters['scales'][1],
                              sphere.parameters['scales'][2])
    center_transform = vtk.vtkTransform()
    center_transform.Translate(sphere.parameters['center'][0],
                                  sphere.parameters['center'][1],
                                  sphere.parameters['center'][2])
    center_transform.Concatenate(ellipsoid_transform)
    sphere_ellipsoid = vtk.vtkTransformPolyDataFilter()
    sphere_ellipsoid.SetInput(sphere_sphere.GetOutput())
    sphere_ellipsoid.SetTransform(center_transform)
    sphere_ellipsoid.Update()
    sphere_point_ids = {}
    for p in xrange(sphere_ellipsoid.GetOutput().GetPoints().GetNumberOfPoints()):
        pid = sphere_points.InsertNextPoint(sphere_ellipsoid.GetOutput().GetPoints().GetPoint(p))
        sphere_point_ids[p] = pid
    for t in xrange(sphere_ellipsoid.GetOutput().GetNumberOfCells()):
        tid = sphere_triangles.InsertNextCell(3)
        for i in xrange(3):
            sphere_triangles.InsertCellPoint(sphere_point_ids[sphere_ellipsoid.GetOutput().GetCell(t).GetPointIds().GetId(i)])
        sphere_data.InsertValue(tid,1)
    print  sphere_triangles.GetNumberOfCells(), "(",sphere_ellipsoid.GetOutput().GetNumberOfCells(),")"
    sphere_polydata.SetPoints(sphere_points)
    sphere_polydata.SetPolys(sphere_triangles)
    sphere_polydata.GetCellData().SetScalars(sphere_data)

    end_time = time()
    print "<-- Creating VTK PolyData      [",end_time-start_time,"s]"
    return sphere_polydata


dome_sphere = ParametricShapeModel()
dome_sphere.parameters['radius'] = grid_step*10.
dome_sphere.parameters['center'] = np.array([0.,0.,0.])
dome_sphere.parameters['scales'] = np.array([1.,1.,1.])
dome_sphere.parameters['axes'] = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
world.add(dome_sphere,'dome_sphere',_repr_vtk_=draw_sphere_vtk,colormap='green',alpha=0.4,intensity_range=(0,1))

golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)

organ_spheres = []

for o in xrange(3):
    organ_sphere = ParametricShapeModel()
    organ_sphere.parameters['theta'] = (o+1)*golden_angle
    organ_theta = organ_sphere.parameters['theta']
    organ_sphere.parameters['distance'] = np.sqrt(2+o)*grid_step*6.
    organ_distance = organ_sphere.parameters['distance']
    organ_sphere.parameters['radius'] = grid_step*3*np.sqrt(4+o)
    organ_sphere.parameters['center'] = np.array([organ_distance*np.cos(organ_theta) ,organ_distance*np.sin(organ_theta),0.])
    organ_sphere.parameters['scales'] = np.array([1.,1.,1.])
    organ_sphere.parameters['axes'] = np.array([[np.cos(organ_theta),np.sin(organ_theta),0.],[-np.sin(organ_theta),np.cos(organ_theta),0.],[0.,0.,1.]])
    organ_spheres.append(organ_sphere)
    world.add(organ_sphere,'organ_sphere_'+str(o+1),_repr_vtk_=draw_sphere_vtk,colormap='green',alpha=0.4,intensity_range=(0,1))
raw_input()

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

density_k = 0.12
density_R = 0.0
dome_density = sphere_density_function(dome_sphere,k=density_k,R=density_R)(grid[:,0],grid[:,1],grid[:,2])
organ_densities = np.transpose([sphere_density_function(organ_spheres[o],k=density_k,R=density_R)(grid[:,0],grid[:,1],grid[:,2]) for o in xrange(3)])

organ_memberships = organ_densities/(dome_density+organ_densities.sum(axis=1)+0.000001)[:,np.newaxis]
grid_points.point_data = array_dict(organ_memberships.max(axis=1),np.arange(grid.shape[0])).to_dict()
world.add(grid_points,name='grid_points',colormap='acidity',_repr_vtk_=_points_repr_vtk_,i_min=0,i_max=1)
raw_input()

relative_organ_points = []
for o in xrange(3):
    points = array_dict(grid_points.points)
    organ_center = organ_spheres[o].parameters['center']
    #organ_rotation = np.linalg.inv(organ_sphere.parameters['axes'])
    organ_rotation = organ_spheres[o].parameters['axes']
    relative_points = points.values()-organ_center[np.newaxis,:]
    relative_points = np.einsum('...ij,...j->...i',organ_rotation,relative_points)
    relative_organ_points.append(relative_points)

deformed_grid_points = TriangularMesh()
deformed_grid_points.points = array_dict(grid,np.arange(grid.shape[0])).to_dict()
deformed_grid_points.point_data = array_dict(organ_memberships.max(axis=1),np.arange(grid.shape[0])).to_dict()
for o in xrange(3):
    world.add(organ_spheres[o],'organ_sphere_'+str(o+1),_repr_vtk_=draw_sphere_vtk,colormap='green',alpha=0.4,intensity_range=(0,1))
world.add(deformed_grid_points,"deformed_grid_points",colormap='acidity',_repr_vtk_=_points_repr_vtk_,intensity_range=(0,1))
raw_input()
dirname = "/Users/gcerutti/Developpement/openalea/vplants_branches/meshing/share/model_based_deformation"
viewer.save_screenshot(dirname+"/deformation_0.jpg")

for r in xrange(100):
    
    dome_center = dome_sphere.parameters['center']
    
    deformed_points = array_dict(grid_points.points).values()
    deformed_points = dome_center + (1.-organ_memberships.sum(axis=1)[:,np.newaxis])*(deformed_points-dome_center)
    
    for o in xrange(3):
        #organ_spheres[o].parameters['theta'] += np.pi*(grid_step/5.)/180.
        organ_spheres[o].parameters['theta'] += np.pi*(grid_step/5.)/180.
        organ_spheres[o].parameters['distance'] += grid_step/5.
        organ_theta = organ_spheres[o].parameters['theta']
        organ_distance = organ_spheres[o].parameters['distance']
        organ_spheres[o].parameters['center'] = np.array([organ_distance*np.cos(organ_theta) ,organ_distance*np.sin(organ_theta),0.])
        organ_spheres[o].parameters['axes'] = np.array([[np.cos(organ_theta),np.sin(organ_theta),0.],[-np.sin(organ_theta),np.cos(organ_theta),0.],[0.,0.,1.]])
    
        # import vtk
        # points = vtk.vtkPoints()
        # points.InsertNextPoint(organ_sphere.parameters['center'])
        # points.InsertNextPoint(organ_sphere.parameters['center']+organ_sphere.parameters['radius']*organ_sphere.parameters['axes'][0])
        # points.InsertNextPoint(organ_sphere.parameters['center']+organ_sphere.parameters['radius']*organ_sphere.parameters['axes'][1])
        # points.InsertNextPoint(organ_sphere.parameters['center']+organ_sphere.parameters['radius']*organ_sphere.parameters['axes'][2])
        
        # lines = vtk.vtkCellArray()
        # line_ids = vtk.vtkIntArray()
        # for i in xrange(3):
        #     line = vtk.vtkLine()
        #     line.GetPointIds().SetId(0,0)
        #     line.GetPointIds().SetId(1,i+1)
        #     lines.InsertNextCell(line)
        #     line_ids.InsertNextValue(i)
        
        # lines_polydata = vtk.vtkPolyData()
        # lines_polydata.SetPoints(points)
        # lines_polydata.SetLines(lines)
        # lines_polydata.GetCellData().SetScalars(line_ids)
        
        # world.add(lines_polydata,'organ_axes')
    
        world.add(organ_spheres[o],'organ_sphere_'+str(o+1),_repr_vtk_=draw_sphere_vtk,colormap='green',alpha=0.4,intensity_range=(0,1))
        
        organ_center = organ_spheres[o].parameters['center']
        organ_rotation = np.linalg.inv(organ_spheres[o].parameters['axes'])
    
        deformed_points += organ_memberships[:,o,np.newaxis]*(organ_center-dome_center + np.einsum('...ij,...j->...i',organ_rotation,relative_organ_points[o]))
    deformed_grid_points.points = array_dict(deformed_points,np.arange(grid.shape[0])).to_dict()
    
    world.add(deformed_grid_points,"deformed_grid_points",colormap='acidity',_repr_vtk_=_points_repr_vtk_,intensity_range=(0,1))
    viewer.save_screenshot(dirname+"/deformation_"+str(r+1)+".jpg") 




