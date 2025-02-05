# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
# ==============================================================================
import openalea.phenomenal.data as phm_data
import openalea.phenomenal.segmentation as phm_seg
# ==============================================================================
plant_1_dir = "plant_1"

def test_running():
    bin_images = phm_data.bin_images(plant_1_dir)
    calibrations = phm_data.calibrations(plant_1_dir)
    voxel_grid = phm_data.voxel_grid(plant_1_dir, 32)

    # Load images binarize

    graph = phm_seg.graph_from_voxel_grid(voxel_grid)

    voxel_skeleton = phm_seg.skeletonize(voxel_grid, graph)

    image_projection = list()
    for angle in [0, 120, 270]:
        projection = calibrations["side"].get_projection(angle)
        image_projection.append((bin_images["side"][angle], projection))

    voxel_skeleton = phm_seg.segment_reduction(
        voxel_skeleton, image_projection, required_visible=1, nb_min_pixel=100
    )
