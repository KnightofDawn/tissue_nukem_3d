from openalea.image.all import SpatialImage
from openalea.container import array_dict

from scipy.cluster.vq import vq 
import numpy as np

def array_unique(array,return_index=False):
  import numpy as np
  _,unique_rows = np.unique(np.ascontiguousarray(array).view(np.dtype((np.void,array.dtype.itemsize * array.shape[1]))),return_index=True)
  if return_index:
    return array[unique_rows],unique_rows
  else:
    return array[unique_rows]

def scale_space_transform(image, sigmas):
    import numpy as np
    from scipy.ndimage.filters import gaussian_filter, gaussian_laplace, laplace

    size = np.array(image.shape)
    spacing = np.array(image.resolution)
    scale_space = np.zeros((len(sigmas),size[0],size[1],size[2]),dtype=float)

    for i in xrange(len(sigmas)):
        if i==0:
            previous_gaussian_img = gaussian_filter(image,sigma=np.exp(spacing*sigmas[i]),order=0)
        else:
            gaussian_img = gaussian_filter(np.array(image,np.float64),sigma=np.exp(spacing*sigmas[i]),order=0)
            
            #laplace_img = gaussian_laplace(gaussian_img,sigma=np.exp(spacing/spacing.min()))
            #laplace_img = gaussian_laplace(gaussian_img,sigma=spacing)
            laplace_img = laplace(gaussian_img)
            #laplace_img = gaussian_laplace(np.array(image,np.float64),sigma=np.exp(spacing*sigmas[i]))
            scale_space[i, : , : , : ] = laplace_img
            previous_gaussian_img = gaussian_img
    
    return scale_space

def detect_peaks_3D_scale_space(scale_space_images,sigmas,threshold=None,resolution=[1,1,1]):
    """
    """
    from time import time
    import numpy as np
    from vplants.meshing.array_tools import array_unique


    def scale_spatial_local_max(scale_space_images,s,x,y,z,neighborhood=1,resolution=(1,1,1)):
        import numpy as np

        scales = scale_space_images.shape[0] 
        image_neighborhood = np.array(np.ceil(neighborhood/np.array(resolution)),int)
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

    print points.shape[0]," possible peaks"
    raw_input()

    for p,point in enumerate(points):
        if p%100000 == 0:
            print p,"/",points.shape[0]
        if scale_spatial_local_max(scale_space_images,point[0],point[1],point[2],point[3],neighborhood=sigmas[point[0]],resolution=resolution):
            peaks.append(point)

    end_time = time()
    print "<-- Detecting local peaks      [",end_time-start_time,"s]"

    print np.array(peaks).shape[0]," detected peaks"

    return np.array(peaks)

world.clear()

def generate_random_nuclei_image(n_points,shape=(512,512,20),resolution=(0.519,0.519,1.064),min_distance=5.0,cell_radius=1.0,density_k=0.75):
    import numpy as np
    from scipy.cluster.vq import vq 
    
    from openalea.image.all import SpatialImage
    from vplants.meshing.array_tools import array_unique
    

    nuclei_positions = {}

    nuclei_img = SpatialImage(np.zeros(shape,np.uint16),resolution=resolution)
    size = np.array(nuclei_img.shape)

    for p in xrange(n_points):
        point_id = p+2

        #point_position = 0.1*np.array([np.random.randint(10.*size[0]),np.random.randint(10.*size[1]),np.random.randint(10.*size[2])])*resolution
        point_position = np.array([np.random.normal(size[0]/2,nuclei_img.resolution[0]*size[0]/2),np.random.normal(size[1]/2,nuclei_img.resolution[1]*size[1]/2),np.random.normal(0,nuclei_img.resolution[2]*size[2]/4)])
        point_validity = ((point_position>0).all())&((point_position<size-1).all())
        point_position = point_position*np.array(resolution)
        if len(nuclei_positions) > 0:
            point_distance = vq(np.array([point_position]),np.array(nuclei_positions.values()))[1][0]
        else:
            point_distance = 2.*min_distance

        while (point_distance < min_distance) or (not point_validity):
            #point_position = 0.1*np.array([np.random.randint(10.*size[0]),np.random.randint(10.*size[1]),np.random.randint(10.*size[2])])*resolution
            point_position = np.array([np.random.normal(size[0]/2,nuclei_img.resolution[0]*size[0]/2),np.random.normal(size[1]/2,nuclei_img.resolution[1]*size[1]/2),np.random.normal(0,nuclei_img.resolution[2]*size[2]/4)])
            point_validity = ((point_position>[0,0,0]).all())&((point_position<size-1).all())
            point_position = point_position*np.array(resolution)
            if len(nuclei_positions) > 0:
                point_distance = vq(np.array([point_position]),np.array(nuclei_positions.values()))[1][0]
            else:
                point_distance = 2.*min_distance

        nuclei_positions[point_id] = point_position


    for p in nuclei_positions.keys():
        point_coords = np.array(nuclei_positions[p]/np.array(resolution),int)
        image_neighborhood = np.array(np.ceil(16.*cell_radius/np.array(nuclei_img.resolution)),int)
        neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
        neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + point_coords
        neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),np.array(nuclei_img.shape)-1)
        neighborhood_coords = array_unique(neighborhood_coords)

        neighborhood_distance = np.linalg.norm(neighborhood_coords*np.array(resolution) - nuclei_positions[p],axis=1)
        neighborhood_intensity = (1. - np.tanh(density_k*(neighborhood_distance - cell_radius)))

        neighborhood_coords = tuple(np.transpose(neighborhood_coords))
        nuclei_img[neighborhood_coords] = np.maximum(np.array(255*255*np.minimum(neighborhood_intensity,1),int),nuclei_img[neighborhood_coords])

    return nuclei_positions, nuclei_img

# real_positions = {}
# n_points = 1000                                        
# min_distance = 5
# cell_radius = 1.0
# k = 0.75

# size = np.array(tag_img.shape)
# resolution = np.array(tag_img.resolution)*np.array([-1,-1,-1])


# for p in xrange(n_points):
#     point_id = p+2

#     #point_position = 0.1*np.array([np.random.randint(10.*size[0]),np.random.randint(10.*size[1]),np.random.randint(10.*size[2])])*resolution
#     point_position = np.array([np.random.normal(size[0]/2,tag_img.resolution[0]*size[0]/2),np.random.normal(size[1]/2,tag_img.resolution[1]*size[1]/2),np.random.normal(0,tag_img.resolution[2]*size[2]/4)])
#     point_validity = ((point_position>0).all())&((point_position<size-1).all())
#     point_position = point_position*resolution
#     if len(real_positions) > 0:
#         point_distance = vq(np.array([point_position]),np.array(real_positions.values()))[1][0]
#     else:
#         point_distance = 2.*min_distance

#     while (point_distance < min_distance) or (not point_validity):
#         #point_position = 0.1*np.array([np.random.randint(10.*size[0]),np.random.randint(10.*size[1]),np.random.randint(10.*size[2])])*resolution
#         point_position = np.array([np.random.normal(size[0]/2,tag_img.resolution[0]*size[0]/2),np.random.normal(size[1]/2,tag_img.resolution[1]*size[1]/2),np.random.normal(0,tag_img.resolution[2]*size[2]/4)])
#         point_validity = ((point_position>[0,0,0]).all())&((point_position<size-1).all())
#         point_position = point_position*resolution
#         if len(real_positions) > 0:
#             point_distance = vq(np.array([point_position]),np.array(real_positions.values()))[1][0]
#         else:
#             point_distance = 2.*min_distance

#     real_positions[point_id] = point_position


# for p in real_positions.keys():
#     point_coords = np.array(real_positions[p]/resolution,int)
#     image_neighborhood = np.array(np.ceil(16.*cell_radius/np.array(tag_img.resolution)),int)
#     neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
#     neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + point_coords
#     neighborhood_coords = np.minimum(np.maximum(neighborhood_coords,np.array([0,0,0])),np.array(tag_img.shape)-1)
#     neighborhood_coords = array_unique(neighborhood_coords)

#     neighborhood_distance = np.linalg.norm(neighborhood_coords*resolution - real_positions[p],axis=1)
#     neighborhood_intensity = (1. - np.tanh(k*(neighborhood_distance - cell_radius)))

#     neighborhood_coords = tuple(np.transpose(neighborhood_coords))
#     tag_img[neighborhood_coords] = np.maximum(np.array(255*255*np.minimum(neighborhood_intensity,1),int),tag_img[neighborhood_coords])

n_points = 1000
real_positions, tag_img = generate_random_nuclei_image(n_points)
resolution = np.array(tag_img.resolution)

world.add(tag_img,'nuclei_image',position=[0,0,0],resolution=resolution,colormap='invert_grey',volume=True,cut_planes=False)
raw_input()


world.clear()

step = 0.1
start = 0.4
end = 0.7
sigmas = np.arange(start,end,step)

from vplants.nuclei_segmentation.scale_space_detection import basic_scale_space

scale_space_function = {}
scale_space_function['sitk_scale_space'] = basic_scale_space
scale_space_function['scipy_scale_space'] = scale_space_transform

scale_spaces = {}

detected_peaks = {}

for s_id,s in enumerate(scale_space_function.keys()):

    world.add(tag_img,s+'_nuclei_image',position=[0,s_id*2.5*tag_img.resolution[1]*tag_img.shape[1],0],resolution = np.array(tag_img.resolution),colormap='invert_grey',volume=True,cut_planes=False)
    #world[s+'_nuclei_image'].set_attribute('volume',False)
    
    if 'sitk' in s:
        import SimpleITK as sitk
        scale_space = scale_space_function[s](sitk.GetImageFromArray(tag_img.transpose((2,0,1))),sigmas)
    else:
        scale_space = scale_space_function[s](tag_img,sigmas)

    scale_space_DoG = []
    scale_space_sigmas = []
    for i in xrange(len(sigmas)):
        if i>1:
            print "_______________________"
            print ""
            print "Sigma = ",np.exp(sigmas[i])," - ",np.exp(sigmas[i-1])," -> ",np.power(np.exp(sigmas[i-1]),2)
            print "(k-1)*sigma^2 : ",(np.exp(step)-1)*np.power(np.exp(sigmas[i-1]),2)

            DoG_image = np.power(np.exp(sigmas[i-1]),2)*scale_space[i] - np.power(np.exp(sigmas[i]),2)*scale_space[i-1]
            scale_space_DoG.append(DoG_image)
            scale_space_sigmas.append(np.exp(sigmas[i-1]))

    scale_space_DoG = np.array(scale_space_DoG)
    scale_space_sigmas = np.array(scale_space_sigmas)

    scale_spaces[s] = scale_space_DoG

    print "Scale Space Size : ",scale_space_DoG.shape

    peaks = detect_peaks_3D_scale_space(scale_space_DoG,scale_space_sigmas,threshold=3000,resolution=np.array(tag_img.resolution))

    resolution = np.array(tag_img.resolution)
    size = np.array(tag_img.shape)

    peak_scales = array_dict(scale_space_sigmas[peaks[:,0]],np.arange(peaks.shape[0]))
    peak_positions = array_dict(peaks[:,1:]*resolution,np.arange(peaks.shape[0]))

    points = peak_positions.keys()
    points_coordinates = peak_positions.values()

    positions = array_dict(points_coordinates,points)
    
    detected_peaks[s] = positions

    from vplants.meshing.triangular_mesh import TriangularMesh

    detected_cells = TriangularMesh()
    detected_cells.points = positions
    detected_cells.point_radius = 2
    world.add(detected_cells,s+'_detected_cells',colormap='glasbey',position=[0,s_id*2.5*tag_img.resolution[1]*tag_img.resolution[1]*tag_img.shape[1],0])


print (2.*np.abs(scale_spaces.values()[1] - scale_spaces.values()[0])/(np.abs(scale_spaces.values()[1]) + np.abs(scale_spaces.values()[0])))[np.abs(scale_spaces.values()[1]) + np.abs(scale_spaces.values()[0])>1].mean()


for s in detected_peaks.keys():
    print "________________________"
    print " "
    print s
    print " "
    print len(detected_peaks[s])," / ",n_points
    cell_real_matching = vq(np.array(detected_peaks[s].values()),np.array(real_positions.values()))
    print "Detected -> Real : ",cell_real_matching[1].mean()," [",cell_real_matching[1].max(),"]"
    
    real_cell_matching = vq(np.array(real_positions.values()),np.array(detected_peaks[s].values()))
    print "Real -> Detected : ",real_cell_matching[1].mean()," [",real_cell_matching[1].max(),"]"
    
    print "Haussdorff       : ",np.max([cell_real_matching[1].max(),real_cell_matching[1].max()])
    print " "
print "________________________"

import vplants.sam4dmaps.nuclei_detection
reload(vplants.sam4dmaps.nuclei_detection)
from vplants.sam4dmaps.nuclei_detection import detect_nuclei

world.add(tag_img,'detect_nuclei_nuclei_image',position=[0,2*2.5*tag_img.resolution[1]*tag_img.shape[1],0],resolution = np.array(tag_img.resolution),colormap='invert_grey',volume=True,cut_planes=False)
positions = detect_nuclei(tag_img,3000,start,end)
detected_cells = TriangularMesh()
detected_cells.points = positions
detected_cells.point_radius = 2
world.add(detected_cells,'detect_nuclei_detected_cells',colormap='glasbey',position=[0,2*2.5*tag_img.resolution[1]*tag_img.resolution[1]*tag_img.shape[1],0])


# nuclei_file =  open(inputfile,'w+',1)
# nuclei_file.write("Cell id;x;y;z\n")
# for p in peak_positions.keys():
#     nuclei_file.write(str(p)+";"+str(peak_positions[p][0])+";"+str(peak_positions[p][1])+";"+str(peak_positions[p][2])+"\n")
# nuclei_file.flush()
# nuclei_file.close()
