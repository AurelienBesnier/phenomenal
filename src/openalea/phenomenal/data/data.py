# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
# ==============================================================================
from __future__ import division, print_function, absolute_import

import orjson
import os
from collections import defaultdict

import numpy
import cv2
from importlib_resources import files, path

from openalea.phenomenal.mesh import read_ply_to_vertices_faces
from openalea.phenomenal.calibration import (
    Chessboard,
    Chessboards,
    Calibration,
    CalibrationSetup,
    OldCalibrationCamera,
)
from openalea.phenomenal.object import VoxelGrid

# ==============================================================================

# datadir = str(files("openalea.phenomenal_data")._paths[0])
anchor = "openalea.phenomenal_data"


def data_dir(name_dir, dtype="bin"):
    return os.path.join(datadir, name_dir, f"{dtype}/")


def raw_images(name_dir):
    """
    According to the plant number return a dict[id_camera][angle] of
    numpy array of the loader raw image.

    :return: dict[id_camera][angle] of loaded RGB image
    """

    cameras = [f.name for f in files(f"{anchor}.{name_dir}.raw").iterdir()]
    d = defaultdict(dict)
    for id_camera in cameras:
        im_paths = [f for f in files(f"{anchor}.{name_dir}.raw.{id_camera}").iterdir()]
        for p in im_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            d[id_camera][int(p.stem)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return d


def bin_images(name_dir):
    """
    According to the plant number return a dict[id_camera][angle] of
    numpy array of the loader binary image.
    A binary image is a numpy array of uint8 type.

    :return: dict[id_camera][angle] of loaded grayscale image
    """

    cameras = [f.name for f in files(f"{anchor}.{name_dir}.bin").iterdir()]
    d = defaultdict(dict)
    for id_camera in cameras:
        im_paths = [f for f in files(f"{anchor}.{name_dir}.bin.{id_camera}").iterdir()]
        for p in im_paths:
            d[id_camera][int(p.stem)] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return d


def chessboard_images(name_dir):
    """
    According to the plant number return a dict[id_camera][angle] of
    numpy array of the loader binary image.
    A binary image is a numpy array of uint8 type.

    :return: dict[id_camera][angle] of loaded grayscale image
    """

    cameras = [
        f.name
        for f in files(f"{anchor}.{name_dir}.chessboard").iterdir()
        if f.name != "points"
    ]
    d = defaultdict(dict)
    for id_camera in cameras:
        im_paths = [
            f for f in files(f"{anchor}.{name_dir}.chessboard.{id_camera}").iterdir()
        ]
        for p in im_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            d[id_camera][int(p.stem)] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return d


# ==============================================================================


def chessboards(name_dir):
    """
    According to name_dir return a dict[id_camera] of camera
    calibration object

    :return: dict[id_camera] of camera calibration object
    """
    chessboards = []
    for id_chessboard in [1, 2]:
        with path(
            f"{anchor}.{name_dir}", f"chessboard/points/chessboard_{id_chessboard}.json"
        ) as p:
            chessboards.append(Chessboard.load(p))

    return chessboards


def image_points(name_dir):
    """
    According to name_dir return a dict[id_camera] of camera
    calibration object

    :return: dict[id_camera] of camera calibration object
    """
    chessboards = {}
    keep = [42] + list(range(0, 360, 30))
    for id_chessboard in ["target_1", "target_2"]:
        with path(
            f"{anchor}.{name_dir}",
            f"chessboard/points/image_points_{id_chessboard}.json",
        ) as p:
            chessboard = Chessboard.load(p)
            for rotation in list(chessboard.image_points["side"]):
                if rotation not in keep:
                    chessboard.image_points["side"].pop(rotation)
            chessboards[id_chessboard] = chessboard

    return chessboards


def do_calibration(name_dir):
    """Regenerate calibration of cameras"""
    data_directory = os.path.join(name_dir, "calibration")

    cbs = dict(zip(("target_1", "target_2"), chessboards(name_dir)))
    # add missing info
    cb = cbs["target_1"]
    cb.facing_angles = {"side": 48, "top": 48}
    cb.image_sizes = {"side": (2056, 2454), "top": (2454, 2056)}
    cb.check_order()
    #
    cb = cbs["target_2"]
    cb.facing_angles = {"side": 228, "top": 228}
    cb.image_sizes = {"side": (2056, 2454), "top": (2454, 2056)}
    cb.check_order()

    chess_targets = Chessboards(cbs)
    image_sizes = chess_targets.image_sizes()
    image_resolutions = chess_targets.image_resolutions()
    facings = chess_targets.facings()
    target_points = chess_targets.target_points()
    image_points = chess_targets.image_points()
    # distance and inclination of objects
    cameras_guess = {"side": (5500, 90), "top": (2500, 0)}
    targets_guess = {"target_1": (100, 45), "target_2": (100, 45)}
    setup = CalibrationSetup(
        cameras_guess,
        targets_guess,
        image_resolutions,
        image_sizes,
        facings,
        clockwise_rotation=True,
    )
    cameras, targets = setup.setup_calibration(
        reference_camera="side", reference_target="target_1"
    )
    calibration = Calibration(
        targets=targets,
        cameras=cameras,
        target_points=target_points,
        image_points=image_points,
        reference_camera="side",
        clockwise_rotation=True,
    )
    calibration.calibrate()
    calibration.dump(os.path.join(data_directory, "calibration_cameras.json"))


def calibrations(name_dir):
    """
    According to name_dir return a dict[id_camera] of camera
    calibration object

    :return: dict[id_camera] of camera calibration object
    """

    calibration = {}
    for id_camera in ["side", "top"]:
        with path(
            f"{anchor}.{name_dir}", f"calibration/calibration_camera_{id_camera}.json"
        ) as p:
            calibration[id_camera] = OldCalibrationCamera.load(p)

    return calibration


def new_calibrations(name_dir):
    """
    According to name_dir return a camera
    calibration object

    """

    with path(f"{anchor}.{name_dir}", "calibration/calibration_cameras.json") as p:
        return Calibration.load(p)


def voxel_grid(name_dir, voxels_size=4):
    """
    According to the plant number and the voxel size desired return the
    voxel_grid of the plant.

    :param voxels_size: diameter of each voxel in mm (int)
    :return: voxel_grid object
    """
    with path(f"{anchor}", f"{name_dir}/voxels/{voxels_size}.npz") as p:
        vg = VoxelGrid.read(str(p))

        return vg


# ==============================================================================


def tutorial_data_binarization_mask(name_dir):
    """
    Return the list of required images to process the notebook tutorial on
    binarization. The images are already load with opencv in unchanged format.
    images = ["mask_hsv.png", "mask_clean_noise.png", "mask_mean_shift.png"]

    :return: list of image
    """
    masks = []

    with path(f"{anchor}.{name_dir}", "mask") as p:
        for filename in ["mask_hsv.png", "mask_mean_shift.png"]:
            masks.append(cv2.imread(p.joinpath(filename), flags=cv2.IMREAD_GRAYSCALE))

    return masks


# ==============================================================================


def synthetic_plant(name_dir, registration_point=(0, 0, 750)):
    """According to name_dir return the mesh plant and skeleton of the
     synthetic plant.

    Parameters
    ----------
    name_dir : str
        Name of the synthetic plant directory

    registration_point: 3-tuple, optional
        Position of the pot in the scene
    Returns
    -------
        out : vertices, faces, meta_data

    """
    with path(f"{anchor}", f"{name_dir}/synthetic_plant.ply") as filename:
        vertices, faces, _ = read_ply_to_vertices_faces(filename)
        vertices = numpy.array(vertices) * 10 - numpy.array([registration_point])

        with open(str(filename).replace("ply", "json"), "r", encoding="UTF8") as infile:
            meta_data = orjson.loads(infile.read())

        return vertices, faces, meta_data


# ==============================================================================


def mesh_mccormik_plant(name_dir):
    """According to name_dir return the mesh of plant from the McCormik paper"""

    with path(f"{anchor}", f"{name_dir}/segmentedMesh.ply") as filename:
        vertices, faces, colors = read_ply_to_vertices_faces(filename)

        return vertices, faces, colors
