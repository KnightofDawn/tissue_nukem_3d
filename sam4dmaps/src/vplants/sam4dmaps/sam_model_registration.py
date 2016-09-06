import numpy as np
import scipy.ndimage as nd

from openalea.container import array_dict

from vplants.sam4dmaps.parametric_shape import ParametricShapeModel
from vplants.sam4dmaps.sam_model import create_meristem_model, reference_meristem_model

from vplants.sam4dmaps.sam_model_tools import nuclei_density_function, sphere_density_function

from copy import deepcopy
import pickle


def meristem_model_organ_gap(reference_model, meristem_model, orientation=None, same_individual=False, hour_gap=4.):
    if orientation is None:
        orientation = reference_model.parameters['orientation']

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
        
        for p in matching_primordia:
            #angle_0 = (reference_model.parameters["primordium_"+str(p)+"_angle"] + reference_model.parameters['primordium_offset']*golden_angle) % 360
            #angle_1 = (meristem_model.parameters["primordium_"+str(p+gap)+"_angle"] + meristem_model.parameters['primordium_offset']*golden_angle + gap*golden_angle) % 360
            angle_0 = reference_model.parameters['orientation']*(reference_model.parameters["primordium_"+str(p)+"_angle"]) + reference_model.parameters['primordium_offset']*golden_angle
            angle_1 = meristem_model.parameters['orientation']*(meristem_model.parameters["primordium_"+str(p+gap)+"_angle"]) + meristem_model.parameters['primordium_offset']*golden_angle - gap*golden_angle
            angle_gap[p] = np.cos(np.pi*(angle_1-angle_0)/180.)
            distance_gap[p]  = meristem_model.parameters["primordium_"+str(p+gap)+"_distance"] - reference_model.parameters["primordium_"+str(p)+"_distance"]
            radius_gap[p] = meristem_model.parameters["primordium_"+str(p+gap)+"_radius"] - reference_model.parameters["primordium_"+str(p)+"_radius"]
            height_gap[p] = meristem_model.parameters["primordium_"+str(p+gap)+"_height"] - reference_model.parameters["primordium_"+str(p)+"_height"]
        rotation_0 = (reference_model.parameters['initial_angle'] - reference_model.parameters['primordium_offset']*golden_angle) %360
        rotation_1 = (meristem_model.parameters['initial_angle'] - meristem_model.parameters['primordium_offset']*golden_angle + gap*golden_angle) %360
        rotation_gap = np.cos(np.pi*(rotation_1 - rotation_0)/180.)
        gap_penalty = np.exp(-np.power(gap - (hour_gap)/6.,2.0)/np.power(6.,2.0))

        if same_individual:
            #gap_score.append(10*rotation_gap + np.mean(distance_gap))
            #gap_score[gap] = 10*rotation_gap*np.sign(np.mean(distance_gap.values()))
            gap_score[gap] = 10.0*np.mean(angle_gap.values())*np.exp(rotation_gap)*np.sign(np.mean(distance_gap.values()))*gap_penalty
            #gap_score[gap] = np.mean(angle_gap.values())
        else:
            gap_score[gap] = 10.0*np.mean(angle_gap.values())*np.exp(-np.power(np.mean(np.array(distance_gap.values())/6.),2.0))
            
        # print "Gap = ",gap,"[",gap_penalty,"] : r -> ",np.mean(distance_gap.values()),", A -> ",np.mean(angle_gap.values())," (",rotation_0,"->",rotation_1,":",rotation_gap,") [",gap_score[gap],"]"
        print "Gap = ",gap,"[",gap_penalty,"] : r -> ",np.mean(distance_gap.values())," (",rotation_0,"->",rotation_1,":",rotation_gap,") [",gap_score[gap],"]"
    print "Best Gap  :  ",gap_range[np.argmax([gap_score[gap] for gap in gap_range])]
    return gap_range[np.argmax([gap_score[gap] for gap in gap_range])]


def meristem_model_alignement(meristem_model, positions, reference_dome_apex, nuclei_image=None, signal_image=None, organ_gap=0., orientation=None):

    if orientation is None:
        orientation = meristem_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    #golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi


    dome_apex = np.array([meristem_model.parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_phi = np.pi*meristem_model.parameters['dome_phi']/180.
    dome_psi = np.pi*meristem_model.parameters['dome_psi']/180.
    
    initial_angle = meristem_model.parameters['initial_angle']
    initial_angle -= meristem_model.parameters['primordium_offset']*golden_angle
    initial_angle += organ_gap*golden_angle 
    dome_theta = np.pi*initial_angle/180.

    if nuclei_image is not None:
        aligned_nuclei_image = deepcopy(nuclei_image)
        aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_phi/np.pi,axes=[0,2],reshape=False)
        aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_psi/np.pi,axes=[1,2],reshape=False)
        aligned_nuclei_image = nd.rotate(aligned_nuclei_image,angle=-180.*dome_theta/np.pi,axes=[0,1],reshape=False)
    else:
        aligned_nuclei_image = None
    
    if signal_image is not None:
        aligned_signal_image = deepcopy(signal_image)
        aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_phi/np.pi,axes=[0,2],reshape=False)
        aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_psi/np.pi,axes=[1,2],reshape=False)
        aligned_signal_image = nd.rotate(aligned_signal_image,angle=-180.*dome_theta/np.pi,axes=[0,1],reshape=False)
    else:
        aligned_signal_image = None
    
    rotation_phi = np.array([[np.cos(dome_phi),0,np.sin(dome_phi)],[0,1,0],[-np.sin(dome_phi),0,np.cos(dome_phi)]])
    rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),np.sin(dome_psi)],[0,-np.sin(dome_psi),np.cos(dome_psi)]])
    rotation_theta = np.array([[np.cos(dome_theta),np.sin(dome_theta),0],[-np.sin(dome_theta),np.cos(dome_theta),0],[0,0,1]])
    
    relative_points = (positions.values()-dome_apex[np.newaxis,:])
    relative_points = np.einsum('...ij,...j->...i',rotation_phi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_psi,relative_points)
    relative_points = np.einsum('...ij,...j->...i',rotation_theta,relative_points)
    relative_points = relative_points * np.array([1,orientation,1])[np.newaxis,:]
    
    aligned_positions = array_dict(reference_dome_apex + relative_points,positions.keys())
    aligned_position = deepcopy(aligned_positions)
    
    # golden_angle = (2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    # golden_angle = 180.*golden_angle/np.pi
    
    parameters = deepcopy(meristem_model.parameters)
    parameters['orientation'] = orientation
    parameters['dome_apex_x'] = reference_dome_apex[0]
    parameters['dome_apex_y'] = reference_dome_apex[1]
    parameters['dome_apex_z'] = reference_dome_apex[2]
    parameters['dome_phi'] = 0
    parameters['dome_psi'] = 0
    parameters['initial_angle'] = 0
    #parameters['initial_angle'] += meristem_model.parameters['primordium_offset']*golden_angle
    parameters['initial_angle'] -= organ_gap*golden_angle
    for p in parameters.keys():
        if ('primordium' in p) and ('angle' in p):
             parameters[p] += meristem_model.parameters['primordium_offset']*golden_angle
             parameters[p] *= meristem_model.parameters['orientation']

    aligned_meristem_model = create_meristem_model(parameters)
    
    return aligned_meristem_model, aligned_positions, aligned_nuclei_image, aligned_signal_image


def meristem_model_image_registration(meristem_model, image, microscope_orientation=1, organ_gap=0., orientation=None):
    from openalea.image.spatial_image import SpatialImage

    from scipy.ndimage.interpolation import affine_transform

    if orientation is None:
        orientation = meristem_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi 

    dome_apex = np.array([meristem_model.parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_phi = np.pi*meristem_model.parameters['dome_phi']/180.
    dome_psi = np.pi*meristem_model.parameters['dome_psi']/180.
    
    initial_angle = meristem_model.parameters['initial_angle']
    initial_angle -= meristem_model.parameters['primordium_offset']*golden_angle
    initial_angle += organ_gap*golden_angle 
    dome_theta = np.pi*initial_angle/180.

    interpolation_method = "constant"
    #interpolation_method = "nearest"

    rotation_phi = np.array([[np.cos(dome_phi),0,np.sin(dome_phi)],[0,1,0],[-np.sin(dome_phi),0,np.cos(dome_phi)]])
    rotation_psi = np.array([[1,0,0],[0,np.cos(dome_psi),np.sin(dome_psi)],[0,-np.sin(dome_psi),np.cos(dome_psi)]])
    rotation_theta = np.array([[np.cos(dome_theta),np.sin(dome_theta),0],[-np.sin(dome_theta),np.cos(dome_theta),0],[0,0,1]])
    rotation = np.dot(rotation_theta,np.dot(rotation_psi,rotation_phi))
    #rotation = rotation_theta
    print rotation

    size = np.array(image.shape)
    resolution = microscope_orientation*np.array(image.resolution)

    output_shape = 4.*size
    translation = size/2. + dome_apex/resolution
    # translation = np.zeros(3)

    # aligned_image = affine_transform(image, matrix=rotation, offset=translation, mode=interpolation_method, order=1, output_shape=output_shape, cval=0.)
    aligned_image = affine_transform(image, matrix=rotation, offset=translation, output_shape=output_shape)

    # aligned_image = deepcopy(image)
    # aligned_image = nd.rotate(aligned_image,angle=-180.*dome_phi/np.pi,axes=[0,2],reshape=False)
    # aligned_image = nd.rotate(aligned_image,angle=-180.*dome_psi/np.pi,axes=[1,2],reshape=False)
    # aligned_image = nd.rotate(aligned_image,angle=-180.*dome_theta/np.pi,axes=[0,1],reshape=False)

    aligned_image = SpatialImage(aligned_image,resolution=image.resolution)

    return aligned_image




def meristem_model_cylindrical_coordinates(meristem_model, positions, organ_gap=0., orientation=None):
    
    if orientation is None:
        orientation = meristem_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi

    n_organs = meristem_model.parameters['n_primordia']

    dome_apex = np.array([meristem_model.parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_center = meristem_model.shape_model['dome_center']

    #organ_center = meristem_model.shape_model['primordia_centers'][-1]
    organ_center = meristem_model.shape_model['primordia_centers'][0]

    angular_offset = meristem_model.parameters['primordium_1_angle'] - golden_angle

    point_vectors = (positions.values()-dome_apex[np.newaxis,:])
    model_vertical_axis = (dome_apex-dome_center)/np.linalg.norm(dome_apex-dome_center)

    model_radial_axis = (organ_center-dome_apex)
    model_radial_axis -= np.dot(model_radial_axis,model_vertical_axis)*model_vertical_axis
    model_radial_axis = model_radial_axis/np.linalg.norm(model_radial_axis)

    model_coradial_axis = orientation*np.cross(model_vertical_axis,model_radial_axis)

    model_points_vertical = np.dot(point_vectors,model_vertical_axis)

    point_normal_vectors = point_vectors - model_points_vertical[:,np.newaxis]*model_vertical_axis 

    model_points_radial = np.linalg.norm(point_normal_vectors,axis=1)

    model_points_angular = 180.*np.arccos(np.dot(point_normal_vectors,model_radial_axis)/model_points_radial)/np.pi
    model_points_angular *= np.sign(np.dot(point_normal_vectors,model_coradial_axis))
    model_points_angular += angular_offset
    #model_points_angular += (organ_gap+n_organs-1)*golden_angle 
    model_points_angular += golden_angle 
    model_points_angular += organ_gap*golden_angle 
    model_points_angular += (meristem_model.parameters['primordium_offset'])*golden_angle

    print organ_gap, meristem_model.parameters['primordium_offset'], organ_gap*golden_angle - angular_offset

    model_coordinates = array_dict(np.transpose([model_points_angular,model_points_radial,model_points_vertical]),keys=positions.keys())

    parameters = deepcopy(meristem_model.parameters)
    parameters['orientation'] = orientation
    parameters['dome_apex_x'] = 0
    parameters['dome_apex_y'] = 0
    parameters['dome_apex_z'] = 0
    parameters['dome_phi'] = 0
    parameters['dome_psi'] = 0
    parameters['initial_angle'] = 0 
    parameters['initial_angle'] += meristem_model.parameters['primordium_offset']*golden_angle
    parameters['initial_angle'] += organ_gap*golden_angle
    for p in parameters.keys():
        if ('primordium' in p) and ('angle' in p):
            #parameters[p] += meristem_model.parameters['primordium_offset']*golden_angle
            parameters[p] *= meristem_model.parameters['orientation']

    aligned_meristem_model = create_meristem_model(parameters)

    return model_coordinates, aligned_meristem_model


def meristem_model_composite_cylindrical_coordinates(meristem_model, positions, organ_gap=0., orientation=None, membership_density_k=0.33):

    if orientation is None:
        orientation = meristem_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi

    n_organs = meristem_model.parameters['n_primordia']

    dome_apex = np.array([meristem_model.parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_center = meristem_model.shape_model['dome_center']

    #organ_center = meristem_model.shape_model['primordia_centers'][-1]
    organ_center = meristem_model.shape_model['primordia_centers'][0]

    angular_offset = meristem_model.parameters['primordium_1_angle'] - golden_angle

    point_vectors = (positions.values()-dome_apex[np.newaxis,:])
    model_vertical_axis = (dome_apex-dome_center)/np.linalg.norm(dome_apex-dome_center)

    model_radial_axis = (organ_center-dome_apex)
    model_radial_axis -= np.dot(model_radial_axis,model_vertical_axis)*model_vertical_axis
    model_radial_axis = model_radial_axis/np.linalg.norm(model_radial_axis)

    model_coradial_axis = orientation*np.cross(model_vertical_axis,model_radial_axis)

    model_points_vertical = np.dot(point_vectors,model_vertical_axis)

    point_normal_vectors = point_vectors - model_points_vertical[:,np.newaxis]*model_vertical_axis 

    model_points_radial = np.linalg.norm(point_normal_vectors,axis=1)

    model_points_angular = 180.*np.arccos(np.dot(point_normal_vectors,model_radial_axis)/model_points_radial)/np.pi
    model_points_angular *= np.sign(np.dot(point_normal_vectors,model_coradial_axis))
    model_points_angular += angular_offset
    #model_points_angular += (organ_gap+n_organs-1)*golden_angle 
    model_points_angular += golden_angle 
    model_points_angular += organ_gap*golden_angle 
    model_points_angular += (meristem_model.parameters['primordium_offset'])*golden_angle


    dome_sphere = ParametricShapeModel()
    dome_sphere.parameters['radius'] = meristem_model.parameters['dome_radius'] 
    dome_sphere.parameters['center'] = dome_center
    dome_sphere.parameters['scales'] = meristem_model.shape_model['dome_scales']
    dome_sphere.parameters['axes'] = np.array([model_radial_axis,model_coradial_axis,model_vertical_axis])
        
    dome_points_density = sphere_density_function(dome_sphere,k=membership_density_k)(*tuple(positions.values().transpose()))

    organ_points_vertical = {}
    organ_points_radial = {}
    organ_points_angular = {}
    organ_points_density = {}

    organ_points_vertical[0] = model_points_vertical
    organ_points_radial[0] = model_points_radial
    organ_points_angular[0] = model_points_angular
    organ_points_density[0] = dome_points_density

    for p in xrange(n_organs):
        organ_center = meristem_model.shape_model['primordia_centers'][p]
        organ_radius = meristem_model.parameters['primordium_'+str(p+1)+'_radius']
        organ_apex = organ_center + organ_radius*model_vertical_axis

        point_organ_vectors = (positions.values()-organ_apex[np.newaxis,:])

        organ_radial_axis = (organ_apex-dome_apex)
        organ_radial_axis -= np.dot(organ_radial_axis,model_vertical_axis)*model_vertical_axis
        organ_radial_axis = organ_radial_axis/np.linalg.norm(organ_radial_axis)

        organ_coradial_axis = orientation*np.cross(model_vertical_axis,organ_radial_axis)

        organ_points_vertical[p+1] = np.dot(point_organ_vectors,model_vertical_axis)

        point_organ_normal_vectors = point_organ_vectors - organ_points_vertical[p+1][:,np.newaxis]*model_vertical_axis
        
        organ_points_radial[p+1] = np.linalg.norm(point_organ_normal_vectors,axis=1)

        organ_points_angular[p+1] = 180.*np.arccos(np.dot(point_organ_normal_vectors,organ_radial_axis)/organ_points_radial[p+1])/np.pi
        organ_points_angular[p+1] *= np.sign(np.dot(point_organ_normal_vectors,organ_coradial_axis))
        organ_points_angular[p+1] += organ_gap*golden_angle

        organ_points_density[p+1] = nuclei_density_function(dict([(p,organ_center)]),organ_radius,membership_density_k)(*tuple(positions.values().transpose()))

    points_density = np.array(organ_points_density.values()).sum(axis=0)

    model_organ_coordinates = array_dict(np.transpose([[organ_points_angular[p],organ_points_radial[p],organ_points_vertical[p]] for p in xrange(n_organs+1)],(2,0,1)),keys=positions.keys())
    model_organ_memberships = array_dict(np.transpose([organ_points_density[p] for p in xrange(n_organs+1)])/points_density[:,np.newaxis],keys=positions.keys())

    parameters = deepcopy(meristem_model.parameters)
    parameters['orientation'] = orientation
    parameters['dome_apex_x'] = 0
    parameters['dome_apex_y'] = 0
    parameters['dome_apex_z'] = 0
    parameters['dome_phi'] = 0
    parameters['dome_psi'] = 0
    parameters['initial_angle'] = 0 
    parameters['initial_angle'] += meristem_model.parameters['primordium_offset']*golden_angle
    parameters['initial_angle'] += organ_gap*golden_angle
    for p in parameters.keys():
        if ('primordium' in p) and ('angle' in p):
            #parameters[p] += meristem_model.parameters['primordium_offset']*golden_angle
            parameters[p] *= meristem_model.parameters['orientation']

    aligned_meristem_model = create_meristem_model(parameters)

    return model_organ_coordinates, model_organ_memberships, aligned_meristem_model


def compose_meristem_model_cylindrical_coordinates(meristem_model, organ_coordinates, organ_memberships, organ_gap=0, orientation=None):

    if orientation is None:
        orientation = meristem_model.parameters['orientation']
    golden_angle = np.sign(orientation)*(2.*np.pi)/((np.sqrt(5)+1)/2.+1)
    golden_angle = 180.*golden_angle/np.pi

    print orientation

    n_organs = meristem_model.parameters['n_primordia']

    organ_apices = []
    dome_apex = np.array([meristem_model.parameters['dome_apex_'+axis] for axis in ['x','y','z']])
    dome_center = meristem_model.shape_model['dome_center']
    model_vertical_axis = (dome_apex-dome_center)/np.linalg.norm(dome_apex-dome_center)

    organ_apices += [dome_apex]

    organ_angles = []
    organ_angles += [0.]

    point_organ_memberships = organ_memberships.values()

    for p in xrange(n_organs):

        mapped_organ = p-organ_gap

        if mapped_organ in range(n_organs):
            organ_center = meristem_model.shape_model['primordia_centers'][mapped_organ]
            organ_radius = meristem_model.parameters['primordium_'+str(mapped_organ+1)+'_radius']
            organ_apex = organ_center + organ_radius*model_vertical_axis
            organ_apices += [organ_apex]
            organ_angle = meristem_model.parameters['primordium_'+str(mapped_organ+1)+'_angle']
            organ_angles += [organ_angle]
        else:
            point_organ_memberships[:,0] += point_organ_memberships[:,p+1]
            point_organ_memberships[:,p+1] = 0
            organ_apices += [np.zeros(3)]
            organ_angles += [0.]

    organ_apices = np.array(organ_apices)
    organ_angles = np.array(organ_angles)

    print organ_angles

    organ_x = organ_apices[:,0] + organ_coordinates.values()[:,:,1]*np.cos(np.pi*(organ_coordinates.values()[:,:,0] + organ_angles)/180.)
    organ_y = organ_apices[:,1] + organ_coordinates.values()[:,:,1]*np.sin(np.pi*(organ_coordinates.values()[:,:,0] + organ_angles)/180.)
    organ_z = organ_apices[:,2] + organ_coordinates.values()[:,:,2]


    model_x = (organ_x * point_organ_memberships).sum(axis=1)
    model_y = (organ_y * point_organ_memberships).sum(axis=1)
    model_z = (organ_z * point_organ_memberships).sum(axis=1)

    model_points_vertical = model_z
    model_points_radial = np.linalg.norm(np.transpose([model_x,model_y]),axis=1)


    model_points_angular = 180.*np.arccos(model_x/model_points_radial)/np.pi
    model_points_angular *= np.sign(model_y)

    model_coordinates = array_dict(np.transpose([model_points_angular,model_points_radial,model_points_vertical]),keys=organ_coordinates.keys())

    return model_coordinates, meristem_model






def meristem_model_registration(meristem_models, nuclei_positions, reference_dome_apex, nuclei_images=[], signal_images=[], reference_model=None, same_individual=False, model_ids=None):
    if model_ids is None:
        model_ids = np.sort(meristem_models.keys())

    aligned_meristem_models = {}
    aligned_nuclei_positions = {}
    aligned_nuclei_images = {}
    aligned_signal_images = {}

    if reference_model is None:
        if same_individual:
            reference_model = meristem_models[model_ids[0]]
        else:
            reference_model = reference_meristem_model(reference_dome_apex,developmental_time=0)

    if same_individual:
        previous_offset = 0

    for i_model in model_ids:
        model = meristem_models[i_model]
        organ_gap = meristem_model_organ_gap(reference_model,model,same_individual=same_individual)
        if same_individual:
            organ_gap += previous_offset
            # previous_offset = model.parameters['primordium_offset']
            previous_offset = organ_gap
        print "Organ Gap : ",organ_gap
        orientation = reference_model.parameters['orientation']
        positions = nuclei_positions[i_model]

        try:
            nuclei_image = nuclei_images[i_model]
            signal_image = signal_images[i_model]
        except:
            nuclei_image = None
            signal_image = None

        aligned_meristem_model, aligned_positions, aligned_nuclei_image, aligned_signal_image = meristem_model_alignement(model,positions,reference_dome_apex,nuclei_image,signal_image,organ_gap,orientation)
        aligned_meristem_models[i_model] = aligned_meristem_model
        aligned_nuclei_positions[i_model] = aligned_positions
        aligned_nuclei_images[i_model] = aligned_nuclei_image
        aligned_signal_images[i_model] = aligned_signal_image

    return aligned_meristem_models, aligned_nuclei_positions, aligned_nuclei_images, aligned_signal_images

