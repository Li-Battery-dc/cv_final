"""VGGT + BA + 3D Gaussian Splatting project.

Sets up sys.path to allow importing VGGT modules without modifying upstream code.
"""
import os
import sys

_VGGT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vggt'))
if _VGGT_PATH not in sys.path:
    sys.path.insert(0, _VGGT_PATH)
