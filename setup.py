# -*- python -*-
#
#       Copyright INRIA - CIRAD - INRA
#
#       File author(s):
#
#       File contributor(s):
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
#
# ==============================================================================
""" """

# ==============================================================================
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension
# ==============================================================================

extensions = [
    Extension(
        "openalea.phenomenal.segmentation._c_skeleton",
        sources=[
            "src/openalea/phenomenal/segmentation/src/skeleton.pyx",
            "src/openalea/phenomenal/segmentation/src/skel.cpp",
        ],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),
    Extension(
        "openalea.phenomenal.multi_view_reconstruction._c_mvr",
        sources=[
            "src/openalea/phenomenal/multi_view_reconstruction/src/c_mvr.pyx",
            "src/openalea/phenomenal/multi_view_reconstruction/src/integral_image.cpp",
        ],
        include_dirs=[numpy.get_include()],
        language="c++",
    ),
]

setup(
    # package installation
    ext_modules=cythonize(extensions, language_level="3"),
)
