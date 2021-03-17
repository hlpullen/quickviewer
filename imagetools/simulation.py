# File: voxtox/test/__init__.py
# -*- coding: future_fstrings -*-

"""Classes for creating synthetic images with simple geometries."""

import os
import numpy as np
import nibabel
import logging

class GeometricNifti():
    """Class for creating a NIfTI file containing synthetic image data."""

    def __init__(self, shape, filename=None, origin=(0, 0, 0),
                 voxel_sizes=(1, 1, 1), noise_range=None):
        """Create data to write to a NIfTI file, initially containing a 
        blank image array.

        Parameters
        ----------
        shape : int/tuple
            Dimensions of the image array to create. If an int is given, the 
            image will be created with dimensions (shape, shape, shape).

        filename : str, default=None
            Name of output NIfTI file. If given, the NIfTI file will 
            automatically be written; otherwise, no file will be written until
            the "write" method is called.

        origin : tuple, default=(0, 0, 0)
            Origin in mm for the image.

        voxel_sizes : tuple, default=(1, 1, 1)
            Voxel sizes in mm for the image.
        """

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create image properties
        self.shape = shape if isinstance(shape, tuple) \
            else (shape, shape, shape)
        self.origin = origin
        self.voxel_sizes = voxel_sizes
        self.affine = np.array([
            [self.voxel_sizes[0], 0, 0, self.origin[0]],
            [0, self.voxel_sizes[1], 0, self.origin[1]],
            [0, 0, self.voxel_sizes[2], self.origin[2]],
            [0, 0, 0, 1]
        ])
        self.data = self.make_data(noise_range)

        # Write to file if a filename is given
        if filename is not None:
            self.filename = os.path.expanduser(filename)
            self.write()

    def write(self, filename=None):
        """Write to a NIfTI file.

        Parameters
        ----------
        filename : str, default=None
            Name of file to write to. If None and the "filename" argument was
            used when creating the object, the filename given at creation will
            be used. If no filename was provided, a RuntimeError will be 
            thrown.
        """

        # Check filename
        if filename is None:
            if hasattr(self, 'filename'):
                filename = self.filename
            else:
                raise RuntimeError("Filename must be specified in __init__() "
                                   "or write()!")

        # Write to file
        self.nii = nibabel.Nifti1Image(self.data, self.affine)
        self.nii.to_filename(os.path.expanduser(filename))
        print(f"Wrote NIfTI image to file {filename}")

    def get_image_centre(self):
        """Get coordinates (in voxels) of the centre of the image."""

        return (float(self.shape[0] / 2), 
                float(self.shape[1] / 2), 
                float(self.shape[2] / 2))

    def make_data(self, noise_range=None):
        """Make blank image array or noisy array."""
        
        if noise_range is None:
            return np.zeros(self.shape)
        else:
            noise = np.random.rand(*self.shape)
            noise = noise_range[0] + noise * (noise_range[1] - noise_range[0])
            return noise

    def make_cuboid_data(self, side_length, centre=None):
        """Make array of 1s and 0s, where 1 lies inside a cuboid.

        Parameters
        ----------
        side_length : float or tuple
            Length of the sides of the cuboid. If a float is given, a cube
            will be created.

        centre : tuple, default=None
            Coordinates (in voxels) of the centre of the cuboid. If None,
            the cuboid will be placed at the centre of the image.
        """

        # Set shape parameters
        side_lengths = side_length if isinstance(side_length, tuple) \
                else (side_length, side_length, side_length)
        cuboid_centre = centre if centre is not None \
                else self.get_image_centre()

        # Make array
        data = np.ones(self.shape)
        indices = np.indices(self.shape)
        for i in range(3):
            distance_to_centre = abs(indices[i] - cuboid_centre[i])
            data *= distance_to_centre <= side_lengths[i] / 2.0
        return data

    def make_sphere_data(self, radius, centre=None):
        """Return array of 1s and 0s, where 1 lies inside the sphere.

        Parameters
        ----------
        radius : float 
            Radius of the sphere.

        centre : tuple, default=None
            Coordinates (in voxels) of the centre of the sphere. If None,
            the sphere will be placed at the centre of the image.

        """

        sphere_centre = centre if centre is not None \
                else self.get_image_centre()
        indices = np.indices(self.shape)
        distance_to_centre = np.sqrt(
            (indices[0] - sphere_centre[0]) ** 2
            + (indices[1] - sphere_centre[1]) ** 2
            + (indices[2] - sphere_centre[2]) ** 2
        )
        return (distance_to_centre <= radius).astype(float)

    def make_grid_data(self, spacing, thickness=None, axis=None, 
                       spacing_in_mm=True):
        """Create array with 1s in a grid pattern.

        Parameters
        ----------
        spacing : int/float/tuple
            Grid spacing. If in mm, this will be converted to the nearest 
            distance corresponding to an integer number of voxels.

        thickness : int/float/tuple, default=None
            Gridline thickness in voxes.

        axis : int, default=None
            Axis along which the gridlines should be continuous. If None, 
            gridlines will be produced along all 3 axes.

        spacing_in_mm : bool, default=True
            If True, grid spacing will be in mm. If False, it will be in 
            number of voxels.
        """

        # Convert spacing/thickness to tuples
        if isinstance(spacing, float) or isinstance(spacing, int):
            spacing = [spacing, spacing, spacing]
        if isinstance(thickness, float) or isinstance(thickness, int):
            thickness = [thickness, thickness, thickness]
        elif thickness is None:
            thickness = [1, 1, 1]

        # Convert spacing to number of voxels
        if spacing_in_mm:
            spacing = [int(spacing[i] / abs(self.voxel_sizes[i]))
                       for i in range(3)]

        # Make grid
        coords = np.meshgrid(
            np.arange(0, self.shape[0]),
            np.arange(0, self.shape[1]),
            np.arange(0, self.shape[2])
        )
        if axis is not None:
            ax1, ax2 = [i for i in [0, 1, 2] if i != axis]
            return ((coords[ax1] % spacing[ax1] < thickness[ax1]) | 
                    (coords[ax2] % spacing[ax2] < thickness[ax2])).astype(int)
        else:
            return ((coords[0] % spacing[0] < thickness[0]) | 
                    (coords[1] % spacing[1] < thickness[1]) | 
                    (coords[2] % spacing[2] < thickness[2])).astype(int)

    def add_data(self, data_to_add, intensity=1.0, above=True):
        """Add a numpy array to the current image array.

        Parameters
        ----------
        data_to_add : numpy ndarray
            Array to add to current image data. Must be the same shape as the
            image data.

        intensity : float, default=1.0
            Value by which data_to_add will be multiplied before adding.

        above : bool, default=True
            If True, any nonzero voxels in data_to_add will overwrite voxels
            in the existing image array. If False, nonzero voxels in the 
            existing image array will be unchanged.
        """

        # Check same shape
        assert data_to_add.shape == self.shape

        # Find overlap with existing array
        overlap = self.data.astype(bool) * data_to_add.astype(bool)
        if above:
            self.data *= ~overlap
        else:
            data_to_add *= ~overlap

        # Add the data
        self.data += intensity * data_to_add

    def add_cuboid(self, side_length, centre=None, intensity=1.0, above=True):
        """Add a cuboid to the image array."""

        self.add_data(self.make_cuboid_data(side_length, centre), intensity, 
                      above)

    def add_sphere(self, radius, centre=None, intensity=1.0, above=True):
        """Add a sphere to the image array."""

        self.add_data(self.make_sphere_data(radius, centre), intensity, 
                      above)

    def add_grid(self, spacing, thickness=None, intensity=1.0, above=True, 
                 axis=None, in_mm=True):
        """Add a sphere to the image array."""

        self.add_data(self.make_grid_data(spacing, thickness, axis, in_mm), 
                      intensity, above)


class CuboidNifti(GeometricNifti):
    """Class for creating a NIfTI image file containing a single cuboid."""

    def __init__(self, shape, side_length, centre=None, filename=None):

        self.side_length = side_length
        self.centre = centre
        GeometricNifti.__init__(self, shape, filename)

    def make_data(self):
        return self.make_cuboid_data(self.side_length, self.centre)


class SphereNifti(GeometricNifti):
    """Class for creating a NIfTI image containing a single sphere."""

    def __init__(self, shape, radius, centre=None, filename=None):

        self.radius = radius
        self.centre = centre
        GeometricNifti.__init__(self, shape, filename)

    def make_data(self):
        return self.make_sphere_data(self.radius, self.centre)
