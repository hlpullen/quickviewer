# File: quickviewer/structure_comparisons.py
# -*- coding: future_fstrings -*-
"""
Functions for computing comparison metrics for pairs of structures.
"""

import nibabel
import numpy as np
import skimage.measure

from quickviewer.core import *


class StructMask:

    def __init__(self, nii):
        """Take an existing nibabel nifti object, or load from a file."""

        # Load NIfTI object
        if isinstance(nii, str):
            self.nii = nibabel.load(nii)
        elif isinstance(nii, nibabel.nifti1.Nifti1Image):
            self.nii = nii
        else:
            raise TypeError("Input parameter <nii> must be either a NIfTI "
                            "object or a path to a NIfTI file.")

        # Assign properties
        self.affine = self.nii.affine
        self.data = self.nii.get_fdata()
        self.assign_geometry()
        self.make_contours()

    def assign_geometry(self):
        """Compute various geometric properties."""

        self.voxel_sizes = {
            "x": self.affine[1, 1],
            "y": self.affine[0, 0],
            "z": self.affine[2, 2]
        }
        self.origin = {            
            "x": self.affine[1, 3],
            "y": self.affine[0, 3],
            "z": self.affine[2, 3]
        }
        self.n_voxels = {
            "x": self.data.shape[1],
            "y": self.data.shape[0],
            "z": self.data.shape[2]
        }
        self.volume_voxels = self.data.astype(bool).sum()
        self.volume_mm = self.volume_voxels * abs(np.prod(
            list(self.voxel_sizes.values())))
        self.volume_ml = self.volume_mm * (0.1 ** 3)
        self.lims = {
            ax: (self.origin[ax], 
                 self.origin[ax] + self.n_voxels[ax] * self.voxel_sizes[ax])
            for ax in self.voxel_sizes
        }

    def make_contours(self):
        """Make contours in all three orientations."""
     
        self.contours = {}
        self.contours_voxels = {}
        self.n_slices = {}
        for view in _slider_axes:
            self.make_contours_for_view(view)
            self.n_slices[_slider_axes[view]] = len(self.contours[view])

    def make_contours_for_view(self, view):
        """Convert a structure mask to a dictionary of contours for a given
        orientation."""

        # Loop through layers
        points = {}
        points_voxels = {}
        x_ax, y_ax = _plot_axes[view]
        z_ax = _slider_axes[view]
        for i in range(self.n_voxels[z_ax]):

            # Get layer of image
            im_slice = get_image_slice(self.data, view, i)

            # Ignore slices with no structure mask
            if im_slice.max() < 0.5:
                continue

            # Find contours
            contours = skimage.measure.find_contours(im_slice, 0.5, "low", 
                                                     "low")
            if contours:
                points[i] = []
                points_voxels[i] = []
                for contour in contours:
                    contour_points = []
                    contour_points_voxels = []
                    for (y, x) in contour:
                        contour_points_voxels.append((x, y))
                        x_mm = min(self.lims[x_ax]) \
                                + (x + 0.5) * abs(self.voxel_sizes[x_ax])
                        y_mm = min(self.lims[y_ax]) \
                                + (y + 0.5) * abs(self.voxel_sizes[y_ax])
                        contour_points.append((x_mm, y_mm))
                    points[i].append(contour_points)
                    points_voxels[i].append(contour_points_voxels)

        self.contours[view] = points
        self.contours_voxels[view] = points_voxels

    def get_volume(self, units="voxels"):
        """Return volume in given units."""

        if units == "voxels":
            return self.volume_voxels
        elif units == "mm":
            return self.volume_mm
        elif units == "ml":
            return self.volume_ml

    def get_volume_string(self, units="voxels", fmt="{:.1f}"):
        """Return volume as a string. Units can be mm, voxels, or ml."""

        if units == "voxels":
            fmt = "{}"
        return fmt.format(self.get_volume(units))

    def get_contours(self, scale_in_mm):
        """Return contours in mm or voxels."""

        if scale_in_mm:
            return self.contours
        return self.contours_voxels

    def get_extent(self, view, sl, scale_in_mm):
        """Get the extent along the x/y axes in the selected view on a given
        slice."""

        contours = self.get_contours(scale_in_mm)[view]
        if sl in contours:
            all_points = []
            for c in contours[sl]:
                all_points.extend(c)
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            return (min(xs), max(xs)), (min(ys), max(ys))

    def get_extent_string(self, view, sl, scale_in_mm, fmt_x="{:.1f}",
                          fmt_y="{:.1f}"):
        """Return the extent along the x/y axes in the selected view on a 
        given slice as a formatted string."""

        lims = self.get_extent(view, sl, scale_in_mm)
        if lims is not None:
            x_lims = lims[0]
            y_lims = lims[1]
            return [fmt_x.format(x) for x in x_lims], \
                    [fmt_y.format(y) for y in y_lims]


