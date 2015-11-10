import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq                       import kmeans, vq

from openalea.image.serial.all              import imread, imsave
from openalea.container                     import array_dict

from openalea.deploy.shared_data import shared_data

import pickle
import csv
from copy import deepcopy

import vtk

world.clear()
#viewer.ren.SetBackground(1,1,1)

import vplants.meshing_data

filename = "r2DII_1.2_141202_sam03_t28"
dirname = shared_data(vplants.meshing_data)


tag_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_tdT.inr.gz"
tag_img = imread(tag_file)

size = np.array(tag_img.shape)
resolution = np.array(tag_img.resolution)*np.array([-1.,-1.,-1.])

position = np.array([0,0,0])
#position = size*resolution/2.

world.add(tag_img,'nuclei_image',position=position/resolution,resolution=resolution,colormap='invert_grey',volume=False,cut_planes=True)

#Load the DII-venus channel
venus_file = dirname+"/nuclei_images/"+filename+"/"+filename+"_DIIV.inr.gz"
venus_img = imread(venus_file)

#Add the signal image to the world
world.add(venus_img,'venus_image',position=[0,0,0],resolution=resolution,colormap='vegetation',volume=False,cut_planes=True)
raw_input()

inputfile = dirname+"/nuclei_images/"+filename+"/cells.csv"

nuclei_data = csv.reader(open(inputfile,"rU"),delimiter=';')
column_names = np.array(nuclei_data.next())

nuclei_cells = []
for data in nuclei_data:
	nuclei_cells.append([float(d) for d in data])
nuclei_cells = np.array(nuclei_cells)

points = np.array(nuclei_cells[:,0],int)
n_points = points.shape[0]  

points_coordinates = nuclei_cells[:,1:4]

positions = array_dict(points_coordinates,points)

from vplants.meshing.triangular_mesh import TriangularMesh

detected_cells = TriangularMesh()
detected_cells.points = positions
world.add(detected_cells,'detected_cells',colormap='grey',position=position)
#raw_input()


# from vplants.mars_alt.mars.segmentation     import filtering
# filtered_signal_img = filtering(venus_img,"gaussian",3.0)
# filtered_tag_img = filtering(tag_img,"gaussian",3.0)

# coords = np.array(points_coordinates/resolution,int)

# points_signal = filtered_signal_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]
# points_tag = filtered_tag_img[tuple([coords[:,0],coords[:,1],coords[:,2]])]

# cell_ratio = array_dict(1.0-np.minimum((points_signal+0.001)/(points_tag+0.001),1.0),points)

# detected_cells.point_data = cell_ratio
# detected_cells.point_radius= 2
# world.add(detected_cells,'fluorescence_ratios',position=[0,0,0],colormap='vegetation',intensity_range=(0.0,1.2))


# import matplotlib.pyplot     as plt
# plt.figure(0)
# plt.clf()
# plt.hist2d(cell_ratio.values(),points_signal,cmap="hot")
# plt.show()


# import tissuelab.gui.vtkviewer.point_editor
# reload(tissuelab.gui.vtkviewer.point_editor)
# from tissuelab.gui.vtkviewer.point_editor import SelectCellPoint

# cell_picker = SelectCellPoint(world_object=world['detected_cells'])
# #cell_picker.SetRenderWindow(viewer.vtkWidget)
# viewer.vtkWidget.SetInteractorStyle(cell_picker)
# raw_input()

# updated_positions = array_dict(world['detected_cells'].data.points)
# viewer.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())




