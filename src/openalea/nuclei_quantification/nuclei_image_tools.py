import numpy as np
import pandas as pd

import scipy.ndimage as nd

def view_image_projection(figure, img, colormap="Greys", intensity_range=None, center=np.array([0,0]), microscope_orientation=-1):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as patch
    from matplotlib import cm
    from matplotlib.colors import Normalize
    
    size = np.array(img.shape)
    resolution = microscope_orientation*np.array(img.resolution)
                
    xx, yy = np.mgrid[0:size[0]*resolution[0]:resolution[0],0:size[1]*resolution[1]:resolution[1]]
    depth = np.power(np.tile(np.tile(np.arange(size[2]),(size[1],1)),(size[0],1,1))/float(size[2]),2)
    #depth = np.zeros_like(img).astype(float)
    extent = yy.min()-center[0],yy.max()-center[0],xx.min()-center[1],xx.max()-center[1]

    if intensity_range is None:
        intensity_range = (np.percentile(img,2),np.percentile(img,98))

    view = cm.ScalarMappable(cmap=colormap,norm=Normalize(vmin=intensity_range[0],vmax=intensity_range[1])).to_rgba(np.transpose((img*(1-depth)).max(axis=2))[:,::-1])

    if figure is not None:
        figure.gca().imshow(view,extent=extent)

    return view,xx,yy
            

