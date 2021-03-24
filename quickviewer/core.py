"""Core functions for loading image files."""

import os
import nibabel
import numpy as np
import glob
import fnmatch
import matplotlib as mpl
import shutil
import configparser


user_settings_dir = os.path.expanduser("~/.quickviewer")
user_settings = os.path.join(user_settings_dir, "settings.ini")


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


def get_unique_path(p1, p2):
    """Get the part of path p1 that is unique from path p2."""

    # Get absolute path
    p1 = os.path.abspath(os.path.expanduser(p1))
    p2 = os.path.abspath(os.path.expanduser(p2))

    # Identical paths: can't find unique path
    if p1 == p2:
        return

    # Different basenames
    if os.path.basename(p1) != os.path.basename(p2):
        return os.path.basename(p1)

    # Find unique section
    left, right = os.path.split(p1)
    left2, right2 = os.path.split(p2)
    unique = ""
    while right != "":
        if right != right2:
            if unique == "":
                unique = right
            else:
                unique = os.path.join(right, unique)
        left, right = os.path.split(left)
        left2, right2 = os.path.split(left2)
    return unique


def is_list(var):
    """Check whether a variable is a list/tuple."""

    return isinstance(var, list) or isinstance(var, tuple)


def check_settings_file():

    default_settings_dir = os.path.join(os.path.dirname(__file__), 
                                        "../settings")

    if not os.path.exists(user_settings_dir):
        os.mkdir(user_settings_dir)
        to_copy = [os.path.join(default_settings_dir, f) 
                   for f in os.listdir(default_settings_dir) 
                   if not f.startswith(".")]
        for f in to_copy:
            shutil.copy(f, user_settings_dir)

    elif not os.path.exists(user_settings):
        shutil.copy(os.path.join(default_settings_dir, "settings.ini"),
                    user_settings_dir)


def get_config():
    
    check_settings_file()
    config = configparser.ConfigParser()
    config.read(user_settings)
    return config


def is_nested(d):
    """Check whether a dict <d> has further dicts nested inside."""

    return all([isinstance(val, dict) for val in d.values()])


def make_three(var):
    """Ensure a variable is a tuple with 3 entries."""
    
    if is_list(var):
        return var

    return [var, var, var]
