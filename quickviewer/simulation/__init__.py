# File: voxtox/test/__init__.py
# -*- coding: future_fstrings -*-

"""Classes for creating synthetic images with simple geometries."""

import os
import numpy as np
import nibabel
import logging

from quickviewer.viewer.core import make_three, is_list


class GeometricNifti():
    """Class for creating a NIfTI file containing synthetic image data."""

    def __init__(self, shape, filename=None, origin=(0, 0, 0),
                 voxel_sizes=(1, 1, 1), intensity=0, noise_std=None):
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

        intensity : float, default=0
            Intensity in HU for the background of the image.

        noise_std : float, default=None
            Standard deviation of Gaussian noise to apply to the image.
            If None, no noise will be applied.
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
        self.max_hu = 0 if noise_std is None else noise_std * 3
        self.min_hu = -self.max_hu
        self.noise_std = noise_std
        self.background = self.make_background()
        self.shapes = []
        self.structs = []
        self.shape_count = {}

        # Write to file if a filename is given
        if filename is not None:
            self.filename = os.path.expanduser(filename)
            self.write()

    def view(self, **kwargs):
        """View with QuickViewer."""

        from quickviewer import QuickViewer
        qv_kwargs = {
            "hu_limits": [self.min_hu, self.max_hu],
            "title": "",
            "origin": self.origin,
            "voxel_sizes": self.voxel_sizes
        }
        qv_kwargs.update(kwargs)
        QuickViewer(self.get_data(), **qv_kwargs)

    def get_data(self):
        """Get data in orientation consistent with dcm2nii."""

        data = self.background.copy()
        for shape in self.shapes:
            data[shape.get_data()] = shape.intensity

        if self.noise_std is not None:
            data += np.random.normal(0, self.noise_std, self.shape)

        return data[:, ::-1, :]

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

        # Write data to file
        self.nii = nibabel.Nifti1Image(self.get_data(), self.affine)
        self.nii.to_filename(os.path.expanduser(filename))
        print(f"Wrote NIfTI image to file {filename}")

        # Write all structures to files
        dirname = os.path.dirname(filename)
        struct_dir = os.path.join(dirname, "structs")
        if not os.path.exists(struct_dir):
            os.mkdir(struct_dir)
        for struct in self.structs:
            struct.write(self.affine, struct_dir)

    def get_image_centre(self):
        """Get coordinates (in voxels) of the centre of the image."""

        return [float(self.shape[i] - 1) / 2 for i in range(3)]

    def make_background(self, noise_std=None):
        """Make blank image array or noisy array."""
        
        if self.noise_std is None:
            return np.zeros(self.shape)
        else:
            noise = np.random.normal(0, self.noise_std, self.shape)
            return noise

    def add_shape(self, shape, shape_type, is_struct, above):

        if above:
            self.shapes.append(shape)
        else:
            self.shapes.insert(0, shape)

        if is_struct:
            self.structs.append(shape)

        if shape_type not in self.shape_count:
            self.shape_count[shape_type] = 1
        else:
            self.shape_count[shape_type] += 1

        if shape.name is None:
            shape.name = f"{shape_type}{self.shape_count[shape_type]}"

        self.min_hu = min([shape.intensity, self.min_hu])
        self.max_hu = max([shape.intensity, self.max_hu])

    def add_sphere(self, radius, centre=None, intensity=1, is_struct=False,
                   name=None, above=True):

        if centre is None:
            centre = self.get_image_centre()
        sphere = Sphere(self.shape, radius, centre, intensity, name)
        self.add_shape(sphere, "sphere", is_struct, above)

    def add_cuboid(self, side_length, centre=None, intensity=1, 
                   is_struct=False, name=None, above=True):

        if centre is None:
            centre = self.get_image_centre()
        cuboid = Cuboid(self.shape, side_length, centre, intensity, name)
        self.add_shape(cuboid, "cuboid", is_struct, above)

    def add_grid(self, spacing, thickness=1, intensity=1, axis=None,
                 name=None, above=True):
    
        grid = Grid(self.shape, spacing, thickness, intensity, axis, name)
        self.add_shape(grid, "grid", False, above)

        
class Shape:

    def write(self, affine, dirname="."):

        path = os.path.join(os.path.expanduser(dirname), f"{self.name}.nii.gz")
        self.nii = nibabel.Nifti1Image(self.get_data(), self.affine)
        self.nii.to_filename(path)


class Sphere(Shape):
    
    def __init__(self, shape, radius, centre, intensity, name=None):

        self.name = name
        self.radius = radius
        self.centre = centre
        self.intensity = intensity
        self.shape = shape

    def get_data(self):

        indices = np.indices(self.shape)
        distance_to_centre = np.sqrt(
            (indices[0] - self.centre[0]) ** 2
            + (indices[1] - self.centre[1]) ** 2
            + (indices[2] - self.centre[2]) ** 2
        )
        return distance_to_centre <= self.radius


class Cuboid(Shape):

    def __init__(self, shape, side_length, centre, intensity, name=None):

        self.name = name
        self.side_length = side_length
        if not is_list(self.side_length):
            self.side_length = make_three(self.side_length)
        self.centre = centre
        self.intensity = intensity
        self.shape = shape

    def get_data(self):

        coords = np.meshgrid(
            np.arange(0, self.shape[0]),
            np.arange(0, self.shape[1]),
            np.arange(0, self.shape[2])
        )
        data = (
            (np.absolute(coords[0] - self.centre[0]) <= self.side_length[0] / 2) &
            (np.absolute(coords[1] - self.centre[1]) <= self.side_length[1] / 2) &
            (np.absolute(coords[2] - self.centre[2]) <= self.side_length[2] / 2)
        )
        return data


class Cylinder(Shape):

    def __init__(self, shape, radius, length, centre, axis, intensity,
                 name=None):

        self.radius = radius
        self.length = length
        self.centre = centre
        self.axis = axis
        self.intensity = intensity
        self.name = name

    def get_data(self):

        # get cylinder data
        pass


class Grid(Shape):

    def __init__(self, shape, spacing, thickness, intensity, axis=None, 
                 name=None):

        self.name = name
        self.spacing = spacing
        if not is_list(self.spacing):
            self.spacing = make_three(self.spacing)
        self.thickness = thickness
        if not is_list(self.thickness):
            self.thickness = make_three(self.thickness)
        self.intensity = intensity
        self.axis = axis
        self.shape = shape

    def get_data(self):

        coords = np.meshgrid(
            np.arange(0, self.shape[0]),
            np.arange(0, self.shape[1]),
            np.arange(0, self.shape[2])
        )
        if self.axis is not None:
            ax1, ax2 = [i for i in [0, 1, 2] if i != axis]
            return ((coords[ax1] % self.spacing[ax1] < self.thickness[ax1]) | 
                    (coords[ax2] % self.spacing[ax2] < self.thickness[ax2]))
        else:
            return ((coords[0] % self.spacing[0] < self.thickness[0]) | 
                    (coords[1] % self.spacing[1] < self.thickness[1]) | 
                    (coords[2] % self.spacing[2] < self.thickness[2]))

