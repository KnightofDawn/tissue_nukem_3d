# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Nuclei Quantification
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np

from scipy.cluster.vq import vq

from vplants.image.spatial_image import SpatialImage
from vplants.container import array_dict

from vplants.tissue_nukem_3d.nuclei_mesh_tools import nuclei_density_function

def point_position_optimization(points, omega_forces=dict(distance=1,repulsion=1), target_distance=1, sigma_deformation=0.1, n_iterations=100, force_centering=True, center=np.zeros(3)):

    points = array_dict(points)

    for iterations in xrange(100):
        point_vectors = np.array([points[p]- points.values() for p in points.keys()])
        point_distances = np.array([vq(points.values(),np.array([points[p]]))[1] for p in points.keys()])
        point_vectors = point_vectors/(point_distances[...,np.newaxis]+1e-7)

        point_distance_forces = omega_forces['distance']*((target_distance-point_distances)[...,np.newaxis]*point_vectors/target_distance).sum(axis=1)
        point_repulsion_forces = omega_forces['repulsion']*np.power(target_distance,2)*(point_vectors/(np.power(point_distances,2)+1e-7)[...,np.newaxis]).sum(axis=1)
        
        point_forces = np.zeros((len(points),3))
        point_forces += point_distance_forces
        point_forces += point_repulsion_forces
        
        point_forces = np.minimum(1.0,sigma_deformation/np.linalg.norm(point_forces,axis=1))[:,np.newaxis] * point_forces
        
        new_points = points.values() + point_forces
        new_points += center - points.values().mean(axis=0)

        points = array_dict(new_points,points.keys())

    return points


def example_nuclei_image(n_points=100,size=50,voxelsize=(0.25,0.25,0.5),nuclei_radius=1.5,return_points=False):
    size = [size/v for v in voxelsize]

    img = np.zeros(tuple(size))

    center = (np.array(size)*np.array(voxelsize)/2.)

    points = {} 
    for i in xrange(n_points):
        points[i] = center + np.random.rand(3)

    point_target_distance = np.power(n_points,1/3.)*nuclei_radius*1.5
    sigma_deformation = nuclei_radius/5.
    omega_forces = dict(distance=1,repulsion=1)

    points = point_position_optimization(points,omega_forces,point_target_distance,sigma_deformation,force_centering=True,center=center)

    x,y,z = np.ogrid[0:size[0]*voxelsize[0]:voxelsize[0],0:size[1]*voxelsize[1]:voxelsize[1],0:size[2]*voxelsize[2]:voxelsize[2]]

    img = (255.*nuclei_density_function(points,nuclei_radius,2./nuclei_radius)(x,y,z)).astype(np.uint16)

    _return = (SpatialImage(img,voxelsize=voxelsize),)
    if return_points:
        _return += (points,)

    if len(_return)==1:
        return _return[0]
    return _return

def example_nuclei_signal_images(n_points=100,size=50,voxelsize=(0.25,0.25,0.5),nuclei_radius=1.5,signal_type='random',return_points=False,return_signals=False):
    original_size = size
    size = [size/v for v in voxelsize]

    img = np.zeros(tuple(size))

    center = np.array(size)*np.array(voxelsize)/2.

    points = {} 
    for i in xrange(n_points):
        points[i] = center + np.random.rand(3)

    point_target_distance = np.power(n_points,1/3.)*nuclei_radius*1.5
    sigma_deformation = nuclei_radius/5.
    omega_forces = dict(distance=1,repulsion=1)

    points = point_position_optimization(points,omega_forces,point_target_distance,sigma_deformation,force_centering=True,center=center)

    x,y,z = np.ogrid[0:size[0]*voxelsize[0]:voxelsize[0],0:size[1]*voxelsize[1]:voxelsize[1],0:size[2]*voxelsize[2]:voxelsize[2]]

    img = (255.*nuclei_density_function(points,nuclei_radius,2./nuclei_radius)(x,y,z)).astype(np.uint16)

    if signal_type == 'random':
        point_signals = dict([(p,np.random.rand()) for p in points.keys()])
    elif signal_type == 'center':
        point_signals = dict([(p,1.-np.linalg.norm(points[p]-center)/float(original_size)) for p in points.keys()])

    signal_img = np.zeros_like(img).astype(float)
    for p in points.keys():
        point_img = nuclei_density_function(dict([(p,points[p])]),nuclei_radius,2./nuclei_radius)(x,y,z)
        signal_img += 255.*point_signals[p]*point_img
    signal_img = signal_img.astype(np.uint16)

    _return = (SpatialImage(img,voxelsize=voxelsize), SpatialImage(signal_img,voxelsize=voxelsize),)
    if return_points:
        _return += (points,)
    if return_signals:
        _return += (point_signals,)

    if len(_return)==1:
        return _return[0]
    return _return

