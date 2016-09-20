import numpy as np


def spherical_parametric_meristem_model(parameters):

    dome_apex = np.array([parameters['dome_apex_x'],parameters['dome_apex_y'],parameters['dome_apex_z']])
    dome_scales = np.array([2,2,5])
    # dome_scales = np.array([parameters['dome_scales_x'],parameters['dome_scales_y'],parameters['dome_scales_z']])
    dome_radius = parameters['dome_radius']
    initial_angle = np.pi*parameters['initial_angle']/180.

    n_primordia = parameters['n_primordia']

    primordia_angles = np.array([np.pi*parameters['primordium_'+str(primordium+1)+'_angle']/180. for primordium in xrange(n_primordia)])
    primordia_distances = np.array([parameters['primordium_'+str(primordium+1)+'_distance'] for primordium in xrange(n_primordia)])
    primordia_heights = np.array([parameters['primordium_'+str(primordium+1)+'_height'] for primordium in xrange(n_primordia)])
    primordia_radiuses = np.array([parameters['primordium_'+str(primordium+1)+'_radius'] for primordium in xrange(n_primordia)])

    primordia_centers = dome_apex[np.newaxis,:] + primordia_distances[:,np.newaxis]*np.transpose([np.cos(initial_angle + primordia_angles),np.sin(initial_angle + primordia_angles),np.zeros_like(primordia_angles)]) 
    primordia_centers += np.transpose([np.zeros_like(primordia_heights),np.zeros_like(primordia_heights),primordia_heights])

    dome_center = dome_apex-dome_scales*np.array([0,0,dome_radius])
    #dome_center = dome_apex+dome_scales*np.array([0,0,dome_radius])

    dome_psi = np.pi*parameters['dome_psi']/180.
    rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),-np.sin(dome_psi)],[0,np.sin(dome_psi),np.cos(dome_psi)]])
    #rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),np.sin(dome_psi)],[0,-np.sin(dome_psi),np.cos(dome_psi)]])
    dome_phi = np.pi*parameters['dome_phi']/180.
    rotation_phi = np.array([[np.cos(dome_phi),0,-np.sin(dome_phi)],[0,1,0],[np.sin(dome_phi),0,np.cos(dome_phi)]])
    #rotation_phi = np.array([[np.cos(dome_phi),0,np.sin(dome_phi)],[0,1,0],[-np.sin(dome_phi),0,np.cos(dome_phi)]])

    dome_axes = np.array([[1,0,0],[0,1,0],[0,0,1]])

    dome_center = dome_apex + np.einsum('...ij,...j->...i',rotation_psi,dome_center-dome_apex)
    primordia_centers = dome_apex[np.newaxis,:] + np.einsum('...ij,...j->...i',rotation_psi,primordia_centers-dome_apex[np.newaxis,:])
    dome_axes = np.einsum('...ij,...j->...i',rotation_psi,dome_axes)

    dome_center = dome_apex + np.einsum('...ij,...j->...i',rotation_phi,dome_center-dome_apex)
    primordia_centers = dome_apex[np.newaxis,:] + np.einsum('...ij,...j->...i',rotation_phi,primordia_centers-dome_apex[np.newaxis,:])
    dome_axes = np.einsum('...ij,...j->...i',rotation_phi,dome_axes)

    meristem_model = {}
    meristem_model['dome_center'] = dome_center
    meristem_model['dome_radius'] = dome_radius
    meristem_model['dome_axes'] = dome_axes
    meristem_model['dome_scales'] = dome_scales

    meristem_model['primordia_centers'] = primordia_centers
    meristem_model['primordia_radiuses'] = primordia_radiuses

    return meristem_model


def phyllotaxis_based_parametric_meristem_model(parameters):
    
    golden_angle = np.sign(parameters['orientation'])*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    initium_angle = np.pi/6.

    dome_apex = np.array([parameters['dome_apex_x'],parameters['dome_apex_y'],parameters['dome_apex_z']])
    # dome_scales = np.array([parameters['dome_scales_x'],parameters['dome_scales_y'],parameters['dome_scales_z']])
    dome_scales = np.array([2,2,5])
    dome_radius = parameters['dome_radius']

    
    initial_distance = dome_radius*np.sin(4*initium_angle/3)
    zero_distance  = dome_radius*0.28
    growth_factor = 0.01
    initial_time = np.log((1/growth_factor)*zero_distance/initial_distance)
    #plastochrone_distance = dome_radius*np.sin(4.*initium_angle/3.)
    #plastochrone_distance = 1.15*initial_distance
    plastochrone_time = 0.09/growth_factor
    max_time = (parameters['n_primordia']+1)*plastochrone_time

    primordia_times = parameters['developmental_time'] + np.arange(parameters['n_primordia'])*plastochrone_time
    
    primordium_offset = 0
    if primordia_times.min() < 0:
        primordium_offset = np.ceil(primordia_times.min()/plastochrone_time)
    elif primordia_times.max() > max_time:
        primordium_offset = np.ceil((primordia_times.max()-max_time)/plastochrone_time)
    parameters['primordium_offset'] = primordium_offset
    

    primordia_angles = golden_angle*(np.arange(parameters['n_primordia']) + 1 - primordium_offset)
    primordia_times = primordia_times - primordium_offset*plastochrone_time

    primordia_distances = initial_distance*np.exp(growth_factor*(primordia_times-initial_time)) 
    primordia_radiuses = (0.26*primordia_distances/dome_radius + 0.15)*dome_radius
    primordia_heights = (-0.08*np.power(primordia_distances/dome_radius - 0.2,2)-0.53)*dome_radius

    primordia_centers = dome_apex[np.newaxis,:] + primordia_distances[:,np.newaxis]*np.transpose([np.cos(primordia_angles),np.sin(primordia_angles),np.zeros_like(primordia_angles)]) 
    primordia_centers += np.transpose([np.zeros_like(primordia_heights),np.zeros_like(primordia_heights),primordia_heights])

    for primordium in xrange(parameters['n_primordia']):
        parameters['primordium_'+str(primordium+1)+"_distance"] = primordia_distances[primordium]
        parameters['primordium_'+str(primordium+1)+"_angle"] = primordia_angles[primordium]*180./np.pi
        parameters['primordium_'+str(primordium+1)+"_height"] = primordia_heights[primordium]
        parameters['primordium_'+str(primordium+1)+"_radius"] = primordia_radiuses[primordium]

    return spherical_parametric_meristem_model(parameters)

# def phyllotaxis_based_parametric_meristem_model(parameters):
#     import numpy as np
    
#     golden_angle = np.sign(parameters['orientation'])*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
#     initium_angle = np.pi/6.

#     dome_apex = np.array([parameters['dome_apex_x'],parameters['dome_apex_y'],parameters['dome_apex_z']])
#     # dome_scales = np.array([parameters['dome_scales_x'],parameters['dome_scales_y'],parameters['dome_scales_z']])
#     dome_scales = np.array([2,2,5])
#     dome_radius = parameters['dome_radius']

#     initial_distance = dome_radius*0.42
#     # initial_distance = dome_radius*np.sin(initium_angle)
#     growth_factor = 0.02
#     plastochrone_distance = dome_radius*np.sin(4.*initium_angle/3.)
#     plastochrone_time = 1./growth_factor * (np.log(plastochrone_distance) - np.log(initial_distance))

#     primordia_times = parameters['developmental_time'] + np.arange(parameters['n_primordia'])*plastochrone_time

#     primordia_angles = golden_angle*np.arange(parameters['n_primordia'])

#     # primordia_distances = np.array([dome_radius*np.sin(initium_angle)])
#     # primordia_heights = np.array([dome_radius*(np.cos(initium_angle)-1.)])
#     # primordia_radiuses = np.array([dome_radius/5.])

#     # primordia_centers = dome_apex[np.newaxis,:] + primordia_distances[:,np.newaxis]*np.transpose([np.cos(primordia_angles),np.sin(primordia_angles),np.zeros_like(primordia_angles)]) 
#     # primordia_centers += np.transpose([np.zeros_like(primordia_heights),np.zeros_like(primordia_heights),primordia_heights])

#     # while (len(primordia_angles)<parameters['n_primordia']):
#     #     primordia_distances += 1.0*np.ones_like(primordia_distances)
#     #     primordia_radiuses += 0.34*np.ones_like(primordia_radiuses)

#     #     if primordia_distances.min() > dome_radius*np.sin(4.*initium_angle/3.):
#     #         primordia_angles = np.concatenate([primordia_angles,np.array([primordia_angles[-1]+golden_angle])])
#     #         primordia_distances = np.concatenate([primordia_distances,np.array([dome_radius*np.sin(initium_angle)])])
#     #         primordia_radiuses = np.concatenate([primordia_radiuses,np.array([dome_radius*0.28])])
#     #         primordia_heights = np.concatenate([primordia_heights,np.array([dome_radius*(np.cos(initium_angle)-1.)])])

#     #     primordia_heights = (0.0043*np.power((primordia_distances-100.),2)/100.-0.72)*dome_radius
        
#     #     primordia_centers = dome_apex[np.newaxis,:] + primordia_distances[:,np.newaxis]*np.transpose([np.cos(primordia_angles),np.sin(primordia_angles),np.zeros_like(primordia_angles)]) 
#     #     primordia_centers += np.transpose([np.zeros_like(primordia_heights),np.zeros_like(primordia_heights),primordia_heights])

#     # primordia_distances += 0.1*parameters['developmental_time']*np.ones_like(primordia_distances)
#     # primordia_radiuses += 0.034*parameters['developmental_time']*np.ones_like(primordia_radiuses)
#     # primordia_distances *= np.exp(0.01*parameters['developmental_time'])*np.ones_like(primordia_distances)

#     primordia_distances = dome_radius*np.sin(initium_angle)*np.exp(growth_factor*primordia_times) 
#     primordia_radiuses = (0.33*primordia_distances/dome_radius + 0.11)*dome_radius
#     #primordia_radiuses = np.exp(3.17)*np.exp(0.00618*primordia_distances)
#     primordia_heights = (0.0043*np.power((primordia_distances-100.),2)/100.-0.72)*dome_radius

#     primordia_centers = dome_apex[np.newaxis,:] + primordia_distances[:,np.newaxis]*np.transpose([np.cos(primordia_angles),np.sin(primordia_angles),np.zeros_like(primordia_angles)]) 
#     primordia_centers += np.transpose([np.zeros_like(primordia_heights),np.zeros_like(primordia_heights),primordia_heights])

#     for primordium in xrange(parameters['n_primordia']):
#         parameters['primordium_'+str(primordium+1)+"_distance"] = primordia_distances[primordium]
#         parameters['primordium_'+str(primordium+1)+"_angle"] = primordia_angles[primordium]*180./np.pi
#         parameters['primordium_'+str(primordium+1)+"_height"] = primordia_heights[primordium]
#         parameters['primordium_'+str(primordium+1)+"_radius"] = primordia_radiuses[primordium]

#     print "  -->  Primordium distance : ",parameters['primordium_1_distance']

#     return spherical_parametric_meristem_model(parameters)


def nuclei_density_function(nuclei_positions,cell_radius,k=0.1):
    import numpy as np
    
    def density_func(x,y,z,return_potential=False):
        
        max_radius = cell_radius
        # max_radius = 0.

        points = np.array(nuclei_positions.values())

        if len((x+y+z).shape) == 1:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 2:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis],2),0.5)
        elif len((x+y+z).shape) == 3:
            cell_distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis,np.newaxis],2) + np.power(z[np.newaxis] - points[:,2,np.newaxis,np.newaxis,np.newaxis],2),0.5)

        density_potential = 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))
        density = density_potential.sum(axis=0)

        # density = np.zeros_like(x+y+z)
        # for p in nuclei_positions.keys():
        #     cell_distances = np.power(np.power(x-nuclei_positions[p][0],2) + np.power(y-nuclei_positions[p][1],2) + np.power(z-nuclei_positions[p][2],2),0.5)
        #     density += 1./2. * (1. - np.tanh(k*(cell_distances - (cell_radius+max_radius)/2.)))
        
        if return_potential:
            return density, density_potential
        else:
            return density
    return density_func


def sphere_density_function(sphere,k=0.1):
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

        density = 1./2. * (1. - np.tanh(k*(distance - radius)))
        return density
    return density_func


def point_nuclei_density(nuclei_positions,points,cell_radius=5,k=1.0):
    return nuclei_density_function(nuclei_positions,cell_radius=cell_radius,k=k)(points[:,0],points[:,1],points[:,2])


def meristem_model_density_function(model, density_k=1.0):

    import numpy as np
    dome_center = model['dome_center']
    dome_radius = model['dome_radius']
    primordia_centers = model['primordia_centers']
    primordia_radiuses = model['primordia_radiuses']
    dome_axes = model['dome_axes']
    dome_scales = model['dome_scales']
    # R_dome=1.0
    # R_primordium=1.0

    def density_func(x,y,z):
        mahalanobis_matrix = np.einsum('...i,...j->...ij',dome_axes[0],dome_axes[0])/np.power(dome_scales[0],2.) \
                             + np.einsum('...i,...j->...ij',dome_axes[1],dome_axes[1])/np.power(dome_scales[1],2.) \
                             + np.einsum('...i,...j->...ij',dome_axes[2],dome_axes[2])/np.power(dome_scales[2],2.)

        dome_vectors = np.zeros((x.shape[0],y.shape[1],z.shape[2],3))
        dome_vectors[:,:,:,0] = x - dome_center[0]
        dome_vectors[:,:,:,1] = y - dome_center[1]
        dome_vectors[:,:,:,2] = z - dome_center[2]

        # dome_distance = np.power(np.power(x-dome_center[0],2) + np.power(y-dome_center[1],2) + np.power(z-dome_center[2],2),0.5)
        # dome_distance = np.power(np.power(x-dome_center[0],2)/np.power(dome_scales[0],2) + np.power(y-dome_center[1],2)/np.power(dome_scales[1],2) + np.power(z-dome_center[2],2)/np.power(dome_scales[2],2),0.5)
        dome_distance = np.power(np.einsum('...ij,...ij->...i', dome_vectors,np.einsum('...ij,...j->...i',mahalanobis_matrix,dome_vectors)),0.5)  # Mahalanobis distance

        max_radius = dome_radius
        density = 0.5 * (1. - np.tanh(density_k * (dome_distance - (dome_radius+max_radius)/2.)))

        for ip, center in enumerate(primordia_centers):
            primordium_distance = \
                np.power(np.power(x - center[0],2) +
                         np.power(y - center[1],2) +
                         np.power(z - center[2],2), 0.5)

            max_radius = primordia_radiuses[ip]
            density +=  \
                0.5 * (1. - np.tanh(density_k * (primordium_distance -
                                                 (primordia_radiuses[ip] + max_radius)/2.)))

        return density

    return density_func


def meristem_model_density_function_quadric(model, density_k=1.0):
    """

    Args:
        model: structural model of the meristem (must contain a non-null 'dome_matrix' key').
        density_k: sharpness coefficient of the density function (default is 1).

    Returns: density function used to define the implicit surface describing the meristem

    """

    import numpy as np

    assert 'dome_matrix' in model
    assert 'dome_center' in model
    assert dome_matrix is not None
    assert dome_center is not None

    dome_center = model['dome_center']
    primordia_centers = model['primordia_centers']
    primordia_radiuses = model['primordia_radiuses']
    dome_matrix = model['dome_matrix']

    def density_func(x, y, z):

        dome_vectors = np.zeros((x.shape[0], y.shape[1], z.shape[2], 4))
        dome_vectors[:, :, :, 0] = x - dome_center[0]
        dome_vectors[:, :, :, 1] = y - dome_center[1]
        dome_vectors[:, :, :, 2] = z - dome_center[2]
        dome_vectors[:, :, :, 3] = 1.  # homogeneous coordinates

        # Difference of potential (not a distance in the mathematical sense)
        # The potential is defined by the quadric's equation:
        # V(X) = X^t M X = Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J
        dome_distance = np.einsum('...ij,...ij->...i', dome_vectors, np.einsum('...ij,...j->...i', dome_matrix, dome_vectors))

        # Blending function (sigmoid function)
        density = 0.5 * (1. - np.tanh(density_k * dome_distance))

        # Evaluating the density resulting from the dome and the primordia
        for ip, center in enumerate(primordia_centers):
            primordium_distance = np.sqrt(np.power(x - center[0], 2) +
                                           np.power(y - center[1], 2) +
                                           np.power(z - center[2], 2))

            max_radius = primordia_radiuses[ip]
            density +=  0.5 * (1. - np.tanh(density_k * (primordium_distance - 0.5 * (primordia_radiuses[ip] + max_radius))))

        return density

    return density_func

def meristem_model_topomesh(model, grid_resolution=None, density_k=0.33, smoothing=True, topological_optimization=False):

    from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh
    from openalea.mesh.property_topomesh_optimization import property_topomesh_vertices_deformation, property_topomesh_edge_flip_optimization

    if grid_resolution is None:

        grid_resolution = np.array([2,2,2])

    if 'bounding_box' not in model or model['bounding_box'] is None:

        primordia_centers = model['primordia_centers']
        primordia_radiuses = model['primordia_radiuses']
        dome_radius = model['dome_radius']
        dome_center = np.array(model['dome_center'])

        bounding_box = \
            np.transpose([np.floor(primordia_centers.min(axis=0) - primordia_radiuses.max() - dome_radius/2),
            np.ceil(primordia_centers.max(axis=0) + primordia_radiuses.max() + dome_radius/2)]).astype(int)

        # bounding_box[2,0] += dome_radius/2
        # bounding_box[2,1] -= dome_radius/4

    else:

        bounding_box = model['bounding_box']

    x,y,z = np.ogrid[bounding_box[0,0]:bounding_box[0,1]:grid_resolution[0], bounding_box[1,0]:bounding_box[1,1]:grid_resolution[1], bounding_box[2,0]:bounding_box[2,1]:grid_resolution[2]]
    grid_size = (x.shape[0], y.shape[1], z.shape[2])

    if 'dome_matrix' in model and model['dome_matrix'] is not None:

        # The dome surface is defined by a quadric
        model_density_field = meristem_model_density_function_quadric(model,density_k=density_k)(x,y,z)

    else:

        # The dome surface is defined by an ellipsoid (old version)
        model_density_field = meristem_model_density_function(model,density_k=density_k)(x,y,z)

    model_topomesh = implicit_surface_topomesh(model_density_field,grid_size,grid_resolution,iso=0.5,center=False)

    if smoothing or topological_optimization:
        for iterations in range(10):
            if smoothing:
                property_topomesh_vertices_deformation(model_topomesh, iterations=10, omega_forces=dict([('taubin_smoothing', 0.65)]), sigma_deformation=1.0, gaussian_sigma=10.0)
            if topological_optimization:
                property_topomesh_edge_flip_optimization(model_topomesh,omega_energies=dict([('regularization',0.15),('neighborhood',0.65)]),simulated_annealing=False,iterations=5)
    return model_topomesh
    

def meristem_model_organ_weighted_density_function(model):
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
            density +=  1./np.power((p+4),0.5) * (1. - np.tanh(k*(primordium_distance - (primordia_radiuses[p]+max_radius)/2.)))
        return density
    return density_func

def plot_meristem_model(figure, meristem_model, r_max=80, color=None, linewidth=2, alpha=0.1, center=True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patch

    if color is None:
        color = 'k'

    ax = figure.gca()

    if center:
        dome_center = [0,0]
    else:
        dome_center = [meristem_model.parameters['dome_apex_'+xy] for xy in ['x','y']]
    
    dome_radius = meristem_model.parameters['dome_radius']
    dome_circle = patch.Circle(xy=dome_center, radius=dome_radius,ec='k',fc='None',lw=1,alpha=0.1)
    figure.gca().add_patch(dome_circle)

    orientation = meristem_model.parameters['orientation']
    
    for p in xrange(meristem_model.parameters['n_primordia']):
        organ_radius = meristem_model.parameters['primordium_'+str(p+1)+'_radius']
        organ_distance = meristem_model.parameters['primordium_'+str(p+1)+'_distance']
        organ_angle = (meristem_model.parameters['initial_angle'] + meristem_model.parameters['primordium_'+str(p+1)+'_angle'])*np.pi/180.
        organ_center = [dome_center[0]+organ_distance*np.cos(organ_angle),dome_center[1]+organ_distance*np.sin(organ_angle)]
        organ_circle = patch.Circle(xy=organ_center, radius=organ_radius,ec=color,fc='None',lw=linewidth,alpha=alpha)
        ax.add_artist(organ_circle)
    ax.set_xlim(dome_center[0]-r_max,dome_center[0]+r_max)
    ax.set_xticklabels(ax.get_xticks())
    ax.set_ylim(dome_center[1]-r_max,dome_center[1]+r_max)
    ax.set_yticklabels(ax.get_yticks())


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


def draw_meristem_model_pgl(meristem_model,center=None,mat=None):
    import openalea.plantgl.all as pgl
    if mat is None:
        mat = pgl.Material((177, 204, 113))
    if center is None:
        center = np.array([0,0,0])
    scene = pgl.Scene()
    scene += pgl.Shape(pgl.Translated(meristem_model.shape_model['dome_center']-center,pgl.Oriented(meristem_model.shape_model['dome_axes'][0],meristem_model.shape_model['dome_axes'][1],pgl.Scaled(meristem_model.shape_model['dome_scales'],pgl.Sphere(meristem_model.shape_model['dome_radius'],slices=128,stacks=128)))),mat)
    for primordium in xrange(len(meristem_model.shape_model['primordia_centers'])):
        scene += pgl.Shape(pgl.Translated(meristem_model.shape_model['primordia_centers'][primordium]-center,pgl.Oriented(meristem_model.shape_model['dome_axes'][0],meristem_model.shape_model['dome_axes'][1],pgl.Sphere(meristem_model.shape_model['primordia_radiuses'][primordium],slices=32,stacks=32))),mat)
    return scene

