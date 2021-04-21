"""Helper functions for loading DICOM images."""

import pydicom
import numpy as np
import os
from shapely import geometry

def load_image(path, rescale=True):
    """Load a DICOM image array and affine matrix from a path."""

    # Single file
    if os.path.isfile(path):

        try:
            ds = pydicom.read_file(path)
            if int(ds.ImagesInAcquisition) == 1:
                data, affine = load_image_single_file(ds, rescale=rescale)
            
            # Look for other files from same image
            else:
                uid = ds.SeriesInstanceUID
                dirname = os.path.dirname(path)
                paths = [os.path.join(dirname, p) for p in os.listdir(dirname) 
                         if not os.path.isdir(os.path.join(dirname, p))]
                data, affine = load_image_multiple_files(paths, uid=uid, 
                                                         rescale=rescale)

        except pydicom.errors.InvalidDicomError:
            raise TypeError("Not a valid dicom file!")

    # Directory
    elif os.path.isdir(path):
        paths = [os.path.join(path, p) for p in 
                 os.listdir(path) if not os.path.isdir(os.path.join(path, p))]
        data, affine = load_image_multiple_files(paths, rescale=rescale)

    else:
        raise TypeError("Must provide a valid path to a file or directory!")

    return data, affine


def load_image_single_file(ds, rescale=True):
    """Load DICOM image from a single DICOM object."""

    data = ds.pixel_array
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)[:, ::-1, ::-1]
    else:
        data = data.transpose(1, 0)[:, ::-1]

    # Rescale data values
    rescale_intercept = float(ds.RescaleIntercept) if \
            hasattr(ds, "RescaleIntercept") else 0
    if rescale == True and hasattr(ds, "RescaleSlope"):
        data = data * float(ds.RescaleSlope) + rescale_intercept
    elif rescale == "dose" and hasattr(ds, "DoseGridScaling"):
        data = data * float(ds.DoseGridScaling) + rescale_intercept

    # Get affine matrix
    vx, vy = ds.PixelSpacing
    vz = ds.SliceThickness
    px, py, pz = ds.ImagePositionPatient
    affine = np.array([
        [vx, 0, 0, px],
        [0, vy, 0, py],
        [0, 0, vz, pz],
        [0, 0, 0, 1]
    ])

    # Adjust for consistency with dcm2nii
    affine[0, 0] *= -1
    affine[0, 3] *= -1
    affine[1, 3] = -(affine[1, 3] + affine[1, 1] * float(data.shape[1] - 1))
    if data.ndim == 3:
        affine[2, 3] = affine[2, 3] - affine[2, 2] * float(data.shape[2] - 1)

    return data, affine


def load_image_multiple_files(paths, uid=None, rescale=True):
    """Load a single dicom image from multiple files."""


    data_slices = {}
    for path in paths:
        try:
            ds = pydicom.read_file(path)
            if uid is not None and ds.SeriesInstanceUID != uid:
                continue
            slice_num = ds.SliceLocation
            data, affine = load_image_single_file(ds, rescale=rescale)
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

    try:
        ds = pydicom.read_file(path)
    except pydicom.errors.InvalidDicomError:
        raise TypeError("Not a valid DICOM file!")

    # Check it's a structure file
    if not (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"):
        print(f"Warning: {path} is not a DICOM structure set file!")
        return

    # Get structure names
    seq = get_dicom_sequence(ds, "StructureSetROI")
    structs = {}
    for struct in seq:
        structs[int(struct.ROINumber)] = {"name": struct.ROIName}

    # Load contour data
    roi_seq = get_dicom_sequence(ds, "ROIContour")
    for roi in roi_seq:

        number = roi.ReferencedROINumber
        data = {"contours": {}}

        # Get colour
        if "ROIDisplayColor" in roi:
            data["color"] = [int(c) / 255 for c in list(roi.ROIDisplayColor)]
        else:
            data["color"] = None

        # Get contours
        contour_seq = get_dicom_sequence(roi, "Contour")
        if contour_seq:
            contour_data = {}
            for c in contour_seq:
                plane_data = [
                    [float(p) for p in c.ContourData[i * 3: i * 3 + 3]]
                    for i in range(c.NumberOfContourPoints)
                ]
                z = float(c.ContourData[2])
                if z not in data["contours"]:
                    data["contours"][z] = []
                data["contours"][z].append(np.array(plane_data))

        structs[number].update(data)

    return structs


def get_dicom_sequence(ds=None, basename=""):

    sequence = []

    for suffix in ["Sequence", "s"]:
        attribute = f"{basename}{suffix}"
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break

    return sequence


def contours_to_indices(contours, origin, voxel_sizes, shape):
    """Convert contours from positions in mm to array indices."""

    converted = {}
    for z, conts in contours.items():

        # Convert z position
        zi = (z - origin[2]) / voxel_sizes[2]
        converted[zi] = []

        # Convert points on each contour
        for points in conts:
            pi = np.zeros(points.shape)
            pi[:, 0] = shape[0] - 1 - (points[:, 0] - origin[0]) / voxel_sizes[0]
            pi[:, 1] = shape[1] - 1 - (points[:, 1] - origin[1]) / voxel_sizes[1]
            pi[:, 2] = zi
            converted[zi].append(pi)

    return converted


def contours_to_mask(contours, shape, level=0.25):
    """Convert contours to mask."""

    mask = np.zeros(shape)

    # Loop over slices
    for iz, conts in contours.items():

        # Loop over contours on each slice
        for c in conts:

            # Make polygon from (x, y) points
            polygon = geometry.Polygon(c[:, 0:2])

            # Get the polygon's bounding box
            ix1, iy1, ix2, iy2 = [int(xy) for xy in polygon.bounds]
            ix1 = max(0, ix1)
            ix2 = min(ix2 + 1, shape[0])
            iy1 = max(0, iy1)
            iy2 = min(iy2 + 1, shape[1])

            # Loop over pixels
            for ix in range(ix1, ix2):
                for iy in range(iy1, iy2):

                    # Make polygon of current pixel
                    pixel = geometry.Polygon([
                        [ix - 0.5, iy - 0.5], [ix - 0.5, iy + 0.5],
                        [ix + 0.5, iy + 0.5], [ix + 0.5, iy - 0.5]
                    ])
                    
                    # Compute overlap
                    overlap = polygon.intersection(pixel).area
                    mask[ix, iy, int(iz)] += overlap

    # Convert mask to boolean
    return mask > level
