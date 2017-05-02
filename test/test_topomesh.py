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

def test_topomesh():
    n_points = 12
    nuclei_radius = 1.5
    img, signal_img, img_nuclei, img_signals = example_nuclei_signal_images(n_points=n_points,size=20,nuclei_radius=nuclei_radius,signal_type='random',return_points=True,return_signals=True)

    from openalea.tissue_nukem_3d.nuclei_image_topomesh import nuclei_image_topomesh

    image_dict = dict(tag=img, sig=signal_img)
    topomesh = nuclei_image_topomesh(image_dict,reference_name='tag',signal_names=['sig'], compute_ratios=[True], microscope_orientation=1, threshold=1, subsampling=2)
    
    assert topomesh.nb_wisps(0) == n_points

    pos = topomesh.wisp_property('barycenter',0)

    match = vq(pos.values(),img_nuclei.values())
    point_match = dict(zip(pos.keys(),match[0]))

    assert np.all(match[1]<nuclei_radius/2.)

    sig = topomesh.wisp_property('sig',0)

    print [(sig[p],img_signals[point_match[p]]) for p in pos.keys()]
    assert np.all([np.isclose(sig[p],img_signals[point_match[p]],atol=0.005) for p in pos.keys()])