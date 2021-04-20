"""Helper functions for loading DICOM images."""

import pydicom
import numpy as np
import os

def load_image(path):
    """Load a DICOM image array and affine matrix from a path."""

    # Single file
    if os.path.isfile(path):

        try:
            ds = pydicom.read_file(path)
            if int(ds.ImagesInAcquisition) == 1:
                data, affine = load_image_single_file(ds)
            
            # Look for other files from same image
            else:
                uid = ds.SeriesInstanceUID
                dirname = os.path.dirname(path)
                paths = [os.path.join(dirname, p) for p in os.listdir(dirname) 
                         if not os.path.isdir(os.path.join(dirname, p))]
                data, affine = load_image_multiple_files(paths, uid=uid)

        except pydicom.errors.InvalidDicomError:
            raise TypeError("Not a valid dicom file!")

    # Directory
    elif os.path.isdir(path):
        paths = [os.path.join(path, p) for p in 
                 os.listdir(path) if not os.path.isdir(os.path.join(path, p))]
        data, affine = load_image_multiple_files(paths)

    else:
        raise TypeError("Must provide a valid path to a file or directory!")

    return data, affine


def load_image_single_file(ds):
    """Load DICOM image from a single DICOM object."""

    vx, vy = ds.PixelSpacing
    vz = ds.SliceThickness
    px, py, pz = ds.ImagePositionPatient
    affine = np.array([
        [vx, 0, 0, px],
        [0, vy, 0, py],
        [0, 0, vz, pz],
        [0, 0, 0, 1]
    ])
    data = ds.pixel_array
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)[:, ::-1, ::-1]
    else:
        data = data.transpose(1, 0)[:, ::-1]

    # Rescale data values
    if hasattr(ds, "RescaleSlope"):
        data = data * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # Adjust for consistency with dcm2nii
    affine[0, 0] *= -1
    affine[0, 3] *= -1
    affine[1, 3] = -(affine[1, 3] + affine[1, 1] * float(data.shape[1] - 1))
    if data.ndim == 3:
        affine[2, 3] = affine[2, 3] - affine[2, 2] * float(data.shape[2] - 1)

    return data, affine


def load_image_multiple_files(paths, uid=None):
    """Load a single dicom image from multiple files."""


    data_slices = {}
    for path in paths:
        try:
            ds = pydicom.read_file(path)
            if uid is not None and ds.SeriesInstanceUID != uid:
                continue
            slice_num = ds.SliceLocation
            data, affine = load_image_single_file(ds)
            data_slices[float(slice_num)] = data

        except pydicom.errors.InvalidDicomError:
            continue

    # Sort and stack image slices
    vz = affine[2, 2]
    data_list = [data_slices[sl] for sl in sorted(list(data_slices.keys()),
        reverse=(vz >= 0))]
    data = np.stack(data_list, axis=-1)

    # Get z origin
    func = max if vz >= 0 else min
    affine[2, 3] = - func(list(data_slices.keys()))

    return data, affine


def load_structs(path):
    """Load structures from a DICOM file."""

    return


