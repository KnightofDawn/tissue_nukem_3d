# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Nuclei Quantification
#
#       Copyright 2015 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Sophie Ribes <sophie.ribes@inria.fr>,
#                            Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np

def array_unique(array,return_index=False):
  _,unique_rows = np.unique(np.ascontiguousarray(array).view(np.dtype((np.void,array.dtype.itemsize * array.shape[1]))),return_index=True)
  if return_index:
    return array[unique_rows],unique_rows
  else:
    return array[unique_rows]


def scale_space_transform(image, sigmas):
    from scipy.ndimage.filters import gaussian_filter, gaussian_laplace, laplace

    size = np.array(image.shape)
    spacing = np.array(image.voxelsize)
    scale_space = np.zeros((len(sigmas),size[0],size[1],size[2]),dtype=float)

    for i in xrange(len(sigmas)):
        print "Sigma : ",np.exp(spacing*sigmas[i])
        if i==0:
            previous_gaussian_img = gaussian_filter(image,sigma=np.exp(spacing*sigmas[i]),order=0)
        else:
            gaussian_img = gaussian_filter(np.array(image,np.float64),sigma=np.exp(spacing*sigmas[i]),order=0)
            
            laplace_img = laplace(gaussian_img)
            scale_space[i, : , : , : ] = laplace_img
            # scale_space[i, : , : , : ] = gaussian_img
            previous_gaussian_img = gaussian_img
    
    return scale_space

def detect_peaks_3D_scale_space(scale_space_images,sigmas,threshold=None,voxelsize=[1,1,1]):
    """
    Identify local maxima in a 4D scale-space image
    :scale_space_images: np.ndarray containing scale-space transform of a 3d image
    :sigmas: list of scales used to generate the scale-space
    :threshold: minimal intensity to be reached by a candidate maximum
    :voxelsize: spatial voxelsize of the 3D images
    :returns: np.ndarray containing 4D coordinates of local maxima
    """
    from time import time

    def scale_spatial_local_max(scale_space_images,s,x,y,z,neighborhood=1,voxelsize=(1,1,1)):

        scales = scale_space_images.shape[0] 
        image_neighborhood = np.array(np.ceil(neighborhood/np.array(voxelsize)),int)
        neighborhood_coords = np.mgrid[0:scales+1,-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
        neighborhood_coords = np.concatenate(np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,4,0))))) + np.array([0,x,y,z])
        neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0,0])),np.array(scale_space_images.shape)-1)
        neighborhood_coords = array_unique(neighborhood_coords)
        neighborhood_coords = tuple(np.transpose(neighborhood_coords))

        return scale_space_images[neighborhood_coords].max() == scale_space_images[s,x,y,z]

    start_time = time()
    print "--> Detecting local peaks"
    scales = scale_space_images.shape[0]
    rows = scale_space_images.shape[1]
    cols = scale_space_images.shape[2]
    slices = scale_space_images.shape[3]

    peaks = []
    if threshold is None:
        threshold = np.percentile(scale_space_images,90)
    points = np.array(np.where(scale_space_images>threshold)).transpose()

    print points.shape[0]," possible peaks (thresholding)"

    x_left_max = np.concatenate([scale_space_images[:,:-1,:,:]-scale_space_images[:,1:,:,:],np.zeros_like(scale_space_images)[:,0:1,:,:]],axis=1)>0
    y_left_max = np.concatenate([scale_space_images[:,:,:-1,:]-scale_space_images[:,:,1:,:],np.zeros_like(scale_space_images)[:,:,0:1,:]],axis=2)>0
    z_left_max = np.concatenate([scale_space_images[:,:,:,:-1]-scale_space_images[:,:,:,1:],np.zeros_like(scale_space_images)[:,:,:,0:1]],axis=3)>0
    x_right_max = np.concatenate([np.zeros_like(scale_space_images)[:,0:1,:,:],scale_space_images[:,1:,:,:]-scale_space_images[:,:-1,:,:]],axis=1)>0
    y_right_max = np.concatenate([np.zeros_like(scale_space_images)[:,:,0:1,:],scale_space_images[:,:,1:,:]-scale_space_images[:,:,:-1,:]],axis=2)>0
    z_right_max = np.concatenate([np.zeros_like(scale_space_images)[:,:,:,0:1],scale_space_images[:,:,:,1:]-scale_space_images[:,:,:,:-1]],axis=3)>0
    local_max = x_left_max & x_right_max & y_left_max & y_right_max & z_left_max & z_right_max

    points = np.array(np.where((scale_space_images>threshold) & local_max)).transpose()
    print points.shape[0]," possible peaks (local max)"

    for p,point in enumerate(points):
        if p%10000 == 0:
            print p,"/",points.shape[0]
        if scale_spatial_local_max(scale_space_images,point[0],point[1],point[2],point[3],neighborhood=sigmas[point[0]],voxelsize=voxelsize):
            peaks.append(point)

    end_time = time()
    print "<-- Detecting local peaks      [",end_time-start_time,"s]"

    print np.array(peaks).shape[0]," detected peaks"

    return np.array(peaks)

def detect_nuclei(nuclei_img, threshold = 3000., radius_range=(1.8,2.2)):
    """
    Detect nuclei positions in a (16-bit) nuclei marker SpatialImage
    """

    voxelsize = np.array(nuclei_img.voxelsize)
    size = np.array(nuclei_img.shape)

    step = 0.1
    # sigmas = np.arange(float(size_range_start),float(size_range_end),step)
    # sigmas = np.linspace(size_range_start,size_range_end-step,np.round((size_range_end-size_range_start)/step))
    sigmas = np.linspace(np.log(radius_range[0]),np.log(radius_range[1]),np.ceil(np.log(radius_range[1]/radius_range[0])/step))
    print sigmas
    scale_space = scale_space_transform(nuclei_img,sigmas)

    scale_space_DoG = []
    scale_space_sigmas = []
    for i in xrange(len(sigmas)):
        if i>1:
            print "_______________________"
            print ""
            print "Sigma = ",np.exp(sigmas[i])," - ",np.exp(sigmas[i-1])," -> ",np.power(np.exp(sigmas[i-1]),2)
            print "(k-1)*sigma^2 : ",(np.exp(step)-1)*np.power(np.exp(sigmas[i-1]),2)

            DoG_image = np.power(np.exp(sigmas[i-1]),2)*scale_space[i] - np.power(np.exp(sigmas[i]),2)*scale_space[i-1]
            # DoG_image = scale_space[i] - scale_space[i-1]
            scale_space_DoG.append(DoG_image)
            scale_space_sigmas.append(np.exp(sigmas[i-1]))
    scale_space_DoG = np.array(scale_space_DoG)
    scale_space_sigmas = np.array(scale_space_sigmas)
    print "Scale Space Size : ",scale_space_DoG.shape

    peaks = detect_peaks_3D_scale_space(scale_space_DoG,scale_space_sigmas,threshold=threshold,voxelsize=np.array(nuclei_img.voxelsize))

    # scale_space_LoG = []
    # scale_space_sigmas = []
    # for i in xrange(len(sigmas)):
    #     scale_space_LoG.append(np.power(np.exp(sigmas[i]),2)*scale_space[i])
    #     scale_space_sigmas.append(np.exp(sigmas[i]))
    # scale_space_LoG = np.array(scale_space_LoG)
    # scale_space_sigmas = np.array(scale_space_sigmas)

    # peaks = detect_peaks_3D_scale_space(scale_space_LoG,scale_space_sigmas,threshold=threshold,voxelsize=np.array(nuclei_img.voxelsize))

    peak_scales = dict(zip(np.arange(peaks.shape[0]),scale_space_sigmas[peaks[:,0]]))
    peak_positions = dict(zip(np.arange(peaks.shape[0]),peaks[:,1:]*voxelsize))

    return peak_positions

def compute_fluorescence_ratios(nuclei_img, signal_img, nuclei_points, nuclei_sigma=3.0, negative=False, truncate=False):
    """
    """
    from scipy.ndimage.filters import gaussian_filter

    voxelsize = np.array(nuclei_img.voxelsize)

    filtered_nuclei_img = gaussian_filter(nuclei_img.astype(float),sigma=voxelsize*nuclei_sigma,order=0)
    filtered_signal_img = gaussian_filter(signal_img.astype(float),sigma=voxelsize*nuclei_sigma,order=0)

    coords = np.array(np.array(nuclei_points.values())/voxelsize,int)
    print coords

    points_signal = filtered_signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
    points_nuclei = filtered_nuclei_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]

    # nuclei_ratio = dict(zip(nuclei_points.keys(),list(np.minimum((points_signal+0.001)/(points_nuclei+0.001),1.0))))
    # nuclei_ratio = dict(zip(nuclei_points.keys(),list((points_signal+0.001)/(points_nuclei+0.001))))

    if truncate:
        nuclei_ratio = np.minimum((points_signal+0.001)/(points_nuclei+0.001),1.0)
    else:
        nuclei_ratio = (points_signal+0.001)/(points_nuclei+0.001)
    if negative:
        nuclei_ratio = 1. - nuclei_ratio
    nuclei_ratio = dict(zip(nuclei_points.keys(),list(nuclei_ratio)))

    return nuclei_ratio

def write_nuclei_points(nuclei_positions, nuclei_filename, data_name="data"):
    """
    """

    if "TriangularMesh" in str(nuclei_positions.__class__):
        nuclei_data = nuclei_positions.point_data
        nuclei_positions = nuclei_positions.points
    else:
        nuclei_data = {}

    nuclei_file =  open(nuclei_filename,'w+',1)
    nuclei_file.write("Cell id;x;y;z")
    if len(nuclei_data)>0:
        nuclei_file.write(";"+data_name)
    nuclei_file.write("\n")

    for p in nuclei_positions.keys():
        nuclei_file.write(str(p)+";"+str(nuclei_positions[p][0])+";"+str(nuclei_positions[p][1])+";"+str(nuclei_positions[p][2]))
        if len(nuclei_data)>0:
            nuclei_file.write(";"+str(nuclei_data[p]))
        nuclei_file.write("\n")
    nuclei_file.flush()
    nuclei_file.close()

def read_nuclei_points(nuclei_filename, return_data=False):
    """
    """

    try:
        import csv
        nuclei_data = csv.reader(open(nuclei_filename,"rU"),delimiter=';')
        column_names = np.array(nuclei_data.next())

        nuclei_cells = []
        for data in nuclei_data:
            nuclei_cells.append([float(d) for d in data])
        nuclei_cells = np.array(nuclei_cells)

        points = list(np.array(nuclei_cells[:,0],int)+2)
        points_coordinates = list(nuclei_cells[:,1:4])
        if return_data and nuclei_cells.shape[1]>4:
            points_data = list(nuclei_cells[:,4])
            return dict(zip(points,points_coordinates)), dict(zip(points,points_data))
        else:
            return dict(zip(points,points_coordinates))
    except IOError:
        return None
    


