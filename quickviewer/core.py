"""Core functions for loading image files."""

import os
import nibabel
import numpy as np
import glob
import fnmatch
import matplotlib as mpl


def load_image(im, affine=None, voxel_sizes=None, origin=None):
    """Load image from either:
        (a) a numpy array;
        (b) an nibabel nifti object;
        (c) a file containing a numpy array;
        (d) a nifti file.

    Returns image data, tuple of voxel sizes, tuple of origin points, 
    and path to image file (None if image was not from a file)."""

    # Try loading from numpy array
    path = None
    if isinstance(im, np.ndarray):
        data = im

    else:

        # Load from file
        if isinstance(im, str):
            path = os.path.expanduser(im)
            try:
                nii = nibabel.load(path)
                data = nii.get_fdata()
                affine = nii.affine

            except FileNotFoundError:
                print(f"Warning: file {path} not found!")
                return None, None, None, None

            except nibabel.filebasedimages.ImageFileError:
                try:
                    data = np.load(path)
                except (IOError, ValueError):
                    raise RuntimeError("Input file <nii> must be a valid "
                                       ".nii or .npy file.")

        # Load nibabel object
        elif isinstance(im, nibabel.nifti1.Nifti1Image):
            data = im.get_fdata()
            affine = im.affine

        else:
            raise TypeError("Image input must be a string, nibabel object, or "
                            "numpy array.")

    # Get voxel sizes and origin from affine
    if affine is not None:
        voxel_sizes = np.diag(affine)[:-1]
        origin = affine[:-1, -1]
    return data, np.array(voxel_sizes), np.array(origin), path


def find_files(paths, ext=""):
    """Find files from a path, list of paths, directory, or list of 
    directories. If <paths> contains directories, files with extension <ext>
    will be searched for inside those directories."""

    paths = [paths] if isinstance(paths, str) else paths
    files = []
    for path in paths:

        # Find files
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            files.extend(glob.glob(f"{path}/*{ext}"))
        else:
            matches = glob.glob(path)
            for m in matches:
                if os.path.isdir(m):
                    files.extend(glob.glob(f"{m}/*{ext}"))
                elif os.path.isfile(m):
                    files.append(m)

    return files


def to_inches(size):
    """Convert a size string to a size in inches. If a float is given, it will 
    be returned. If a string is given, the last two characters will be used to 
    determine the units:
        - "in": inches
        - "cm": cm
        - "mm": mm
        - "px": pixels
    """

    if not isinstance(size, str):
        return size

    val = float(size[:-2])
    units = size[-2:]
    inches_per_cm = 0.394
    if units == "in":
        return val
    elif units == "cm":
        return inches_per_cm * val
    elif units == "mm":
        return inches_per_cm * val / 10
    elif units == "px":
        return val / mpl.rcParams["figure.dpi"]
