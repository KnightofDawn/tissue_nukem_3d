import numpy as np
from scipy.cluster.vq import vq

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function

from vplants.sam4dmaps.sam_model_tools import spherical_parametric_meristem_model, phyllotaxis_based_parametric_meristem_model
from vplants.sam4dmaps.sam_model_tools import meristem_model_density_function, meristem_model_organ_weighted_density_function, draw_meristem_model_vtk

from copy import deepcopy
import pickle


def read_meristem_model(meristem_model_filename):
    parameters = pickle.load(open(meristem_model_filename,'r'))
    return create_meristem_model(parameters)

def create_meristem_model(meristem_model_parameters):

    meristem_model = ParametricShapeModel()

    meristem_model.parameters = deepcopy(meristem_model_parameters)
    meristem_model.parametric_function = spherical_parametric_meristem_model
    meristem_model.update_shape_model()
    meristem_model.density_function = meristem_model_density_function
    # meristem_model.density_function = meristem_model_organ_weighted_density_function
    meristem_model.drawing_function = draw_meristem_model_vtk

    return meristem_model

def reference_meristem_model(dome_apex, n_primordia=8, developmental_time=0.):
    initial_parameters = {}
    initial_parameters['dome_apex_x'] = dome_apex[0]
    initial_parameters['dome_apex_y'] = dome_apex[1]
    initial_parameters['dome_apex_z'] = dome_apex[2]
    initial_parameters['dome_radius'] = 80
    initial_parameters['initial_angle'] = 0.
    initial_parameters['dome_psi'] = 0.
    initial_parameters['dome_phi'] = 0.
    initial_parameters['n_primordia'] = n_primordia
    initial_parameters['developmental_time'] = developmental_time

    meristem_model = ParametricShapeModel()
    meristem_model.parameters = deepcopy(initial_parameters)
    meristem_model.parameters['orientation'] = 1.
    meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
    meristem_model.update_shape_model()
    meristem_model.density_function = meristem_model_organ_weighted_density_function
    meristem_model.drawing_function = draw_meristem_model_vtk

    return meristem_model


def estimate_meristem_model(positions, size, resolution, meristem_model_parameters=None, n_cycles=2, microscope_orientation=1, organ_weighting=0.3, density_k=0.1, dome_organ_ratio=1.0):
    
    np.random.seed(134560)

    grid_resolution = resolution*np.array([8,8,4])
    x,y,z = np.ogrid[-0.25*size[0]*resolution[0]:1.25*size[0]*resolution[0]:2*grid_resolution[0],-0.25*size[1]*resolution[1]:1.25*size[1]*resolution[1]:2*grid_resolution[1],-0.25*size[2]*resolution[2]:1.25*size[2]*resolution[2]:2*grid_resolution[2]]
    grid_size = 1.5*size
    # x,y,z = np.ogrid[0:size[0]*resolution[0]:grid_resolution[0],0:size[1]*resolution[1]:grid_resolution[1],0:size[2]*resolution[2]:grid_resolution[2]]
    # grid_size = size

    nuclei_potential = np.array([nuclei_density_function(dict([(p,positions[p])]),cell_radius=5,k=1.0)(x,y,z) for p in positions.keys()])
    nuclei_potential = np.transpose(nuclei_potential,(1,2,3,0))
    nuclei_density = np.sum(nuclei_potential,axis=3)

    def meristem_model_organ_weighted_density_function(model):
        import numpy as np
        dome_center = model['dome_center']
        dome_radius = model['dome_radius']
        primordia_centers = model['primordia_centers']
        primordia_radiuses = model['primordia_radiuses']
        dome_axes = model['dome_axes']
        dome_scales = model['dome_scales']

        k = density_k
        R_dome = np.sqrt(dome_organ_ratio)
        R_primordium = 1./np.sqrt(dome_organ_ratio)

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
                density +=  0.5* np.power((p+1),-organ_weighting) * (1. - np.tanh(k*(primordium_distance - (primordia_radiuses[p]+max_radius)/2.)))
            return density
        return density_func

    def meristem_model_energy(parameters,density_function,x=x,y=y,z=z,nuclei_density=nuclei_density,minimum_density=0.5):
        meristem_density = density_function(x,y,z)
        external_energy = ((minimum_density*np.ones_like(nuclei_density) - nuclei_density)[np.where(meristem_density>0.5)]).sum()
        internal_energy = 0.
        internal_energy += 0.1*np.linalg.norm([parameters['dome_phi'],parameters['dome_psi']],1)
        # internal_energy += 1.*np.abs(parameters['developmental_time']-10.)
        # internal_energy += 0.1*parameters['primordium_'+str(parameters['n_primordia'])+'_distance']
        return internal_energy + external_energy

    def meristem_model_void_energy(parameters,density_function,x=x,y=y,z=z,nuclei_density=nuclei_density,minimum_density=0.5):
        meristem_density = density_function(x,y,z)
        external_energy = - (meristem_density[np.where(meristem_density>0.5)]).sum()
        internal_energy = 0.
        internal_energy += 0.1*np.linalg.norm([parameters['dome_phi'],parameters['dome_psi']],1)
        return internal_energy + external_energy

    clockwise_meristem_model = ParametricShapeModel()
    counterclockwise_meristem_model = ParametricShapeModel()

    if meristem_model_parameters is None:
        initial_parameters = {}
        initial_parameters['dome_apex_x'] = size[0]*resolution[0]/2.
        initial_parameters['dome_apex_y'] = size[1]*resolution[1]/2.
        initial_parameters['dome_apex_z'] = size[2]*resolution[2] if microscope_orientation==1 else 0.
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

    clockwise_meristem_model.parameters = deepcopy(initial_parameters)
    clockwise_meristem_model.parameters['orientation'] = 1.
    clockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
    clockwise_meristem_model.update_shape_model()
    # clockwise_meristem_model.density_function = meristem_model_density_function
    clockwise_meristem_model.density_function = meristem_model_organ_weighted_density_function
    clockwise_meristem_model.drawing_function = draw_meristem_model_vtk

    counterclockwise_meristem_model.parameters = deepcopy(initial_parameters)
    counterclockwise_meristem_model.parameters['orientation'] = -1.
    counterclockwise_meristem_model.parametric_function = phyllotaxis_based_parametric_meristem_model
    counterclockwise_meristem_model.update_shape_model()
    # counterclockwise_meristem_model.density_function = meristem_model_density_function
    counterclockwise_meristem_model.density_function = meristem_model_organ_weighted_density_function
    counterclockwise_meristem_model.drawing_function = draw_meristem_model_vtk

    optimization_parameters = ['dome_apex_x','dome_apex_y','dome_apex_z','dome_radius','dome_phi','dome_psi','initial_angle','developmental_time']

    for cycle in xrange(n_cycles):
        temperature = initial_temperature
        clockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
        counterclockwise_meristem_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)

        while temperature>minimal_temperature:
            temperature *= lambda_temperature
            clockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)
            counterclockwise_meristem_model.parameter_optimization_annealing(meristem_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)

    clockwise_energy = meristem_model_energy(clockwise_meristem_model.parameters,clockwise_meristem_model.shape_model_density_function())
    clockwise_void_energy = meristem_model_void_energy(clockwise_meristem_model.parameters,clockwise_meristem_model.shape_model_density_function())
    clockwise_energy_ratio = clockwise_energy/clockwise_void_energy
    print "Clockwise Model Energy : ",clockwise_energy," (",clockwise_void_energy,")   -->",clockwise_energy_ratio
    
    counterclockwise_energy = meristem_model_energy(counterclockwise_meristem_model.parameters,counterclockwise_meristem_model.shape_model_density_function())
    counterclockwise_void_energy = meristem_model_void_energy(counterclockwise_meristem_model.parameters,counterclockwise_meristem_model.shape_model_density_function())
    counterclockwise_energy_ratio = counterclockwise_energy/counterclockwise_void_energy
    print "Counter-Clockwise Model Energy : ",counterclockwise_energy," (",counterclockwise_void_energy,")   -->",counterclockwise_energy_ratio
    
    reference_parameters = deepcopy(clockwise_meristem_model.parameters)

    meristem_flexible_model = ParametricShapeModel()
    # if clockwise_energy < counterclockwise_energy:
    # if clockwise_energy_ratio > counterclockwise_energy_ratio:
    if clockwise_energy_ratio*clockwise_energy < counterclockwise_energy_ratio*counterclockwise_energy:
        reference_parameters = deepcopy(clockwise_meristem_model.parameters)
    else:
        reference_parameters = deepcopy(counterclockwise_meristem_model.parameters)
    meristem_flexible_model.parameters = deepcopy(reference_parameters)
    meristem_flexible_model.parametric_function = spherical_parametric_meristem_model
    meristem_flexible_model.update_shape_model()
    # meristem_flexible_model.density_function = meristem_model_density_function
    meristem_flexible_model.density_function = meristem_model_organ_weighted_density_function
    meristem_flexible_model.drawing_function = draw_meristem_model_vtk

    def meristem_flexible_model_energy(parameters,density_function,x=x,y=y,z=z,nuclei_density=nuclei_density,minimum_density=0.5,reference_parameters=reference_parameters):
        import numpy as np
        meristem_density = density_function(x,y,z)
        external_energy = ((minimum_density*np.ones_like(nuclei_density) - nuclei_density)[np.where(meristem_density>0.5)]).sum()
        internal_energy = 0.
        internal_energy += 10.*np.linalg.norm([parameters['dome_phi'],parameters['dome_psi']],1)
        internal_energy += 5.*np.linalg.norm([parameters[p]-reference_parameters[p] for p in parameters.keys() if 'primordium' in p],2)
        return internal_energy + external_energy

    initial_temperature = 1.5
    minimal_temperature = 0.2
    lambda_temperature = 0.96

    optimization_parameters = []
    for primordium in xrange(meristem_flexible_model.parameters['n_primordia']) :
      optimization_parameters += ['primordium_'+str(primordium+1)+'_distance','primordium_'+str(primordium+1)+'_angle','primordium_'+str(primordium+1)+"_height",'primordium_'+str(primordium+1)+"_radius"]

    for cycle in xrange(n_cycles):
        temperature = initial_temperature
        meristem_flexible_model.perturbate_parameters(10.-cycle,parameters_to_perturbate=optimization_parameters)
        while temperature>minimal_temperature:
            temperature *= lambda_temperature
            meristem_flexible_model.parameter_optimization_annealing(meristem_flexible_model_energy,parameters_to_optimize=optimization_parameters,temperature=temperature)

    return meristem_flexible_model, clockwise_meristem_model, counterclockwise_meristem_model



def evaluate_meristem_model_quality(meristem_model, reference_dome_center, reference_organ_centers, quality_criteria = ['Organ Center','Dome Angle','Dome Distance','Dome Center','Phyllotaxis']):

    quality_data = {}
        
    model_organ_centers = meristem_model.shape_model['primordia_centers'][:,:2]
    model_dome_center = np.array([meristem_model.parameters['dome_apex_'+k] for k in ['x','y']])[np.newaxis,:]

    n_organs = len(model_organ_centers)
    model_reference_organ_matching = vq(model_organ_centers,reference_organ_centers)
    model_reference_dome_matching = vq(model_dome_center,reference_dome_center)

    if 'Phyllotaxis' in quality_criteria:
        model_reference_phyllotaxy_error = (np.abs(model_reference_organ_matching[0][1:]-model_reference_organ_matching[0][:-1]) != 1).sum()/float(n_organs)
        quality_data['Phyllotaxis'] = 1.0 - model_reference_phyllotaxy_error
        
    reference_organ_vectors = reference_organ_centers-model_dome_center
    #reference_organ_vectors = reference_organ_centers-reference_dome_center
    reference_organ_distances = np.linalg.norm(reference_organ_vectors,axis=1)

    meristem_radius = reference_organ_distances.max()
    #meristem_radius = reference_organ_distances.mean()

    golden_angle = 360./((np.sqrt(5)+1)/2.+1)
    #spiral_error = np.min([(golden_angle*8)%360,360-(golden_angle*8)%360])
    spiral_error = np.min([(golden_angle*5)%360,360-(golden_angle*5)%360])

    if 'Organ Center' in quality_criteria:
        model_reference_organ_error = dict(zip(range(1,n_organs+1),model_reference_organ_matching[1]/(0.5*meristem_radius)))
        quality_data['Organ Center'] = 1.0 - np.mean(model_reference_organ_matching[1])/(0.5*meristem_radius)

    if 'Dome Distance' in quality_criteria:
        model_organ_distances = dict([(o,(meristem_model.parameters['primordium_'+str(o)+'_distance'])) for o in range(1,n_organs+1)])
        matched_reference_organ_distances = dict(zip(range(1,n_organs+1),reference_organ_distances[model_reference_organ_matching[0]]))
        model_reference_distance_error =  dict(zip(range(1,n_organs+1),[np.abs(model_organ_distances[o] - matched_reference_organ_distances[o])/(0.5*meristem_radius) for o in range(1,9)]))
        #model_reference_distance_error =  dict(zip(range(1,n_organs+1),[np.abs(model_organ_distances[o] - matched_reference_organ_distances[o])/matched_reference_organ_distances[o] for o in range(1,n_organs+1)]))
        quality_data['Dome Distance'] = 1.0 - np.mean(model_reference_distance_error.values())

    if 'Dome Center' in quality_criteria:
        model_reference_dome_error = model_reference_dome_matching[1][0]/(0.28*meristem_radius)
        quality_data['Dome Center'] = 1.0 - model_reference_dome_error

    if 'Dome Angle' in quality_criteria:
        reference_organ_directions = reference_organ_vectors/reference_organ_distances[:,np.newaxis]
        reference_organ_angles = 180.*np.arccos(reference_organ_directions[:,0])*np.sign(reference_organ_directions[:,1])/np.pi%360
        model_organ_angles = dict([(o,(meristem_model.parameters['initial_angle']+meristem_model.parameters['orientation']*meristem_model.parameters['primordium_'+str(o)+'_angle'])%360) for o in range(1,n_organs+1)])
        matched_reference_organ_angles = dict(zip(range(1,n_organs+1),reference_organ_angles[model_reference_organ_matching[0]]))
        
        model_reference_angle_error = dict(zip(range(1,n_organs+1),[np.abs(model_organ_angles[o] - matched_reference_organ_angles[o])/spiral_error for o in range(1,n_organs+1)]))
        quality_data['Dome Angle'] = 1.0 - np.mean(model_reference_angle_error.values()) 

    dome_organ_error = np.concatenate([[model_reference_dome_error],np.sum([model_reference_distance_error.values(),model_reference_angle_error.values()],axis=0)])

    return quality_data, dome_organ_error
        
