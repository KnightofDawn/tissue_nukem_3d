from openalea.tissue_nukem_3d.microscopy_images import imread

import openalea.tissue_nukem_3d.nuclei_image_topomesh
reload(openalea.tissue_nukem_3d.nuclei_image_topomesh)
from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh

channel_names=['DIIV','TagBFP','CLV3']
img_filename = '/Volumes/signal/Carlos/LSM710/20170501 MS-E13 LD qDII CLV3/qDII-11c-CLV-CH-MS-E13-LD-SAM1.czi'

img_dict = imread(img_filename,channel_names=channel_names)
                  
topomesh = nuclei_image_topomesh(img_dict)

world.add(topomesh,'topomesh')

