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

from openalea.tissue_nukem_3d.example_image import example_nuclei_image, example_nuclei_signal_images

def test_detection():
	n_points = 10
	nuclei_radius = 1.5
	img = example_nuclei_image(n_points=n_points,size=15,nuclei_radius=nuclei_radius)

	from openalea.tissue_nukem_3d.nuclei_detection import detect_nuclei

	pos = detect_nuclei(img,threshold=1)

	assert len(pos) == n_points

def test_segmentation():
	n_points = 10
	nuclei_radius = 1.5
	img, img_nuclei = example_nuclei_image(n_points=n_points,size=15,nuclei_radius=nuclei_radius,return_points=True)

	from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_detection
	from openalea.tissue_nukem_3d.nuclei_segmentation import nuclei_positions_from_segmented_image
	from openalea.image.spatial_image import SpatialImage

	pos = nuclei_detection(img,threshold=1,subsampling=1)

	assert len(pos) == n_points

	match = vq(pos.values(),img_nuclei.values())

	assert np.all(match[1]<nuclei_radius/2.)

def test_quantification():
	n_points = 10
	nuclei_radius = 1.5
	img, signal_img, img_nuclei, img_signals = example_nuclei_signal_images(n_points=n_points,size=15,nuclei_radius=nuclei_radius,signal_type='random',return_points=True,return_signals=True)

	from openalea.tissue_nukem_3d.nuclei_detection import compute_fluorescence_ratios
	from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_detection

	pos = nuclei_detection(img,threshold=1,subsampling=1)

	assert len(pos) == n_points

	match = vq(pos.values(),img_nuclei.values())
	point_match = dict(zip(pos.keys(),match[0]))

	assert np.all(match[1]<nuclei_radius/2.)

	sig = compute_fluorescence_ratios(img, signal_img, pos, nuclei_sigma=2.*nuclei_radius)

	print [(sig[p],img_signals[point_match[p]]) for p in pos.keys()]
	assert np.all([np.isclose(sig[p],img_signals[point_match[p]],atol=0.005) for p in pos.keys()])
