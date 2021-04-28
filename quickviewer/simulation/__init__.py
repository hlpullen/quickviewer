# File: voxtox/test/__init__.py
# -*- coding: future_fstrings -*-

"""Classes for creating synthetic images with simple geometries."""

import os
import numpy as np
import nibabel
import logging
import shutil

from quickviewer.viewer.image import NiftiImage
from quickviewer.viewer.core import make_three, is_list


class GeometricNifti(NiftiImage):
    """Class for creating a NIfTI file containing synthetic image data."""

    def __init__(self, shape, filename=None, origin=(0, 0, 0),
                 voxel_sizes=(1, 1, 1), intensity=-1000, noise_std=None):
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

        intensity : float, default=-1000
            Intensity in HU for the background of the image.

        noise_std : float, default=None
            Standard deviation of Gaussian noise to apply to the image.
            If None, no noise will be applied.
        """

        # Create image properties
        self.shape = make_three(shape)
        voxel_sizes = [abs(v) for v in make_three(voxel_sizes)]
        self.affine = np.array([
            [-voxel_sizes[0], 0, 0, origin[0] + (self.shape[0] - 1) * 
             voxel_sizes[0]],
            [0, voxel_sizes[1], 0, origin[1]],
            [0, 0, voxel_sizes[2], origin[2]],
            [0, 0, 0, 1]
        ])
        self.max_hu = 0 if noise_std is None else noise_std * 3
        self.min_hu = -self.max_hu if self.max_hu != 0 else -20
        self.noise_std = noise_std
        self.bg_intensity = intensity
        self.background = self.make_background()
        self.shapes = []
        self.structs = []
        self.groups = {}
        self.shape_count = {}

        # Initialise as NiftiImage
        NiftiImage.__init__(self, self.background, affine=self.affine)

        # Write to file if a filename is given
        if filename is not None:
            self.filename = os.path.expanduser(filename)
            self.write()

    def view(self, **kwargs):
        """View with QuickViewer."""

        from quickviewer import QuickViewer
        qv_kwargs = {
            "hu": [self.min_hu, self.max_hu],
            "title": "",
            "origin": self.origin,
            "voxel_sizes": self.voxel_sizes
        }
        qv_kwargs.update(kwargs)
        structs = {shape.name: shape.get_data(self.get_coords())
                   for shape in self.structs}
        QuickViewer(self.get_data(), structs=structs, **qv_kwargs)

    def get_data(self):
        """Get data in orientation consistent with dcm2nii."""

        data = self.background.copy()
        for shape in self.shapes:
            data[shape.get_data(self.get_coords())] = shape.intensity

        if self.noise_std is not None:
            data += np.random.normal(0, self.noise_std, self.shape)

        return data

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
        if len(self.structs):
            dirname = os.path.dirname(filename)
            prefix = os.path.basename(filename).split(".")[0]
            struct_dir = os.path.join(dirname, f"{prefix}_structs")
            shutil.rmtree(struct_dir)
            os.mkdir(struct_dir)
            for struct in self.structs:
                struct.write(self.affine, self.get_coords(), struct_dir)
            print("Wrote structures to", struct_dir)

    def get_image_centre(self, in_mm=True):
        """Get coordinates of the centre of the image."""

        centre = [float(self.shape[i] - 1) / 2 for i in range(3)]
        if in_mm:
            centre = self.idx_to_pos_3d(centre)
        return centre

    def make_background(self):
        """Make blank image array or noisy array."""
        
        return np.ones(self.shape) * self.bg_intensity

    def add_shape(self, shape, shape_type, is_struct, above, group):

        if above:
            self.shapes.append(shape)
        else:
            self.shapes.insert(0, shape)

        if is_struct is None and group is not None:
            is_struct = True

        if is_struct:
            if group is not None:
                if group not in self.groups:
                    self.groups[group] = ShapeGroup([shape], name=group)
                    self.structs.append(self.groups[group])
                else:
                    self.groups[group].add_shape(shape)
            else:
                self.structs.append(shape)

        if shape_type not in self.shape_count:
            self.shape_count[shape_type] = 1
        else:
            self.shape_count[shape_type] += 1

        if shape.name is None:
            shape.name = f"{shape_type}{self.shape_count[shape_type]}"

        self.min_hu = min([shape.intensity, self.min_hu])
        self.max_hu = max([shape.intensity, self.max_hu])

    def add_sphere(self, radius, centre=None, intensity=0, is_struct=None,
                   name=None, above=True, group=None):

        in_mm = True
        if centre is None:
            centre = self.get_image_centre(in_mm)

        # Get radius and centre in mm
        if not in_mm:
            centre = self.idx_to_pos_3d(centre)
            radius = self.length_in_mm(radius, "x")

        sphere = Sphere(self.shape, radius, centre, intensity, name)
        self.add_shape(sphere, "sphere", is_struct, above, group)

    def add_cylinder(self, radius, length, axis="z", centre=None, intensity=0, 
                     is_struct=None, name=None, above=True, 
                     group=None):

        in_mm = True
        if centre is None:
            centre = self.get_image_centre(in_mm)

        # Get radius, centre and length in mm
        if not in_mm:
            centre = self.idx_to_pos_3d(centre)
            rad_axis = "x" if axis != "x" else "y"
            radius = self.length_in_mm(radius, rad_axis)
            length = self.length_in_mm(length, axis)

        cylinder = Cylinder(self.shape, radius, length, axis, centre, intensity,
                            name)
        self.add_shape(cylinder, "cylinder", is_struct, above, group)

    def add_cube(self, side_length, centre=None, intensity=0, is_struct=None,
                 name=None, above=True, group=None):

        # Convert to mm, ensuring side lengths are the same
        in_mm = True
        if not in_mm:
            side_length = self.length_in_mm(side_length, "x")
            if centre is not None:
                centre = self.idx_to_pos_3d(centre)

        self.add_cuboid(side_length, centre, intensity, is_struct, name,
                        above, group=group)

    def add_cuboid(self, side_length, centre=None, intensity=0, 
                   is_struct=None, name=None, above=True, 
                   group=None):

        in_mm = True
        if centre is None:
            centre = self.get_image_centre(in_mm)
        side_length = make_three(side_length)

        # Get side lengths and centre in mm
        if not in_mm:
            centre = self.idx_to_pos_3d(centre)
            side_length = [
                self.length_in_mm(side_length[0], "x"),
                self.length_in_mm(side_length[1], "y"),
                self.length_in_mm(side_length[2], "z")
            ]

        cuboid = Cuboid(self.shape, side_length, centre, intensity, name)
        self.add_shape(cuboid, "cuboid", is_struct, above, group)

    def add_grid(self, spacing, thickness=1, intensity=0, axis=None,
                 name=None, above=True, in_mm=None):
    
        if in_mm is None:
            in_mm = self.in_mm
        grid = Grid(self.shape, spacing, thickness, intensity, axis, name)
        self.add_shape(grid, "grid", False, above, group=None)

    def length_in_voxels(self, length, axis):
        """Convert a length in mm to a length in voxels."""

        return length / abs(self.voxel_sizes[axis])

    def length_in_mm(self, length, axis):
        """Convert a length in voxels to a length in mm."""

        return length * abs(self.voxel_sizes[axis])

    def idx_to_pos_3d(self, idx):
        """Transform a 3D position in array indices to a position in mm."""

        return (
            self.idx_to_pos(idx[0], "x"),
            self.idx_to_pos(idx[1], "y"),
            self.idx_to_pos(idx[2], "z")
        )

    def get_coords(self):
        """Get grids of x, y, and z coordinates for this image."""

        if not hasattr(self, "coords"):
            coords_1d = []
            for ax in ["y", "x", "z"]:
                coords_1d.append(np.arange(
                    self.origin[ax], 
                    self.origin[ax] + self.voxel_sizes[ax] * self.n_voxels[ax],
                    self.voxel_sizes[ax]
                ))
            self.coords = np.meshgrid(*coords_1d)
        return self.coords

        
class Shape:


    def get_data(self, coords):
        data = self.get_shape_data(coords)
        return data[::-1, ::-1, :]

    def write(self, affine, coords, dirname="."):

        path = os.path.join(os.path.expanduser(dirname), f"{self.name}.nii.gz")
        data = self.get_data(coords).astype(int)
        self.nii = nibabel.Nifti1Image(data, affine)
        self.nii.to_filename(path)


class ShapeGroup(Shape):

    def __init__(self, shapes, name):
        
        self.name = name
        self.shapes = shapes

    def add_shape(self, shape):
        self.shapes.append(shape)

    def get_data(self, coords):

        data = self.shapes[0].get_data(coords)
        for shape in self.shapes[1:]:
            data += shape.get_data(coords)
        return data


class Sphere(Shape):
    
    def __init__(self, shape, radius, centre, intensity, name=None):

        self.name = name
        self.radius = radius
        self.centre = centre
        self.intensity = intensity

    def get_shape_data(self, coords):

        distance_to_centre = np.sqrt(
            (coords[0] - self.centre[1]) ** 2
            + (coords[1] - self.centre[0]) ** 2
            + (coords[2] - self.centre[2]) ** 2
        )
        return distance_to_centre <= self.radius


class Cuboid(Shape):

    def __init__(self, shape, side_length, centre, intensity, name=None):

        self.name = name
        self.side_length = make_three(side_length)
        self.centre = centre
        self.intensity = intensity

    def get_shape_data(self, coords):

        try:
            data = (
                (np.absolute(coords[0] - self.centre[1]) <= self.side_length[1] / 2) &
                (np.absolute(coords[1] - self.centre[0]) <= self.side_length[0] / 2) &
                (np.absolute(coords[2] - self.centre[2]) <= self.side_length[2] / 2)
            )
            return data
        except TypeError:
            print("centre:", self.centre)
            print("side length:", self.side_length)


class Cylinder(Shape):

    def __init__(self, shape, radius, length, axis, centre, intensity, 
                 name=None):

        self.radius = radius
        self.length = length
        self.centre = centre
        self.axis = axis
        self.intensity = intensity
        self.name = name

    def get_shape_data(self, coords):

        # Get coordinates in each direction
        axis_idx = {"x": 1, "y": 0, "z": 2}[self.axis]
        circle_idx = [i for i in range(3) if i != axis_idx]
        coords_c1 = coords[circle_idx[0]]
        coords_c2 = coords[circle_idx[1]]
        coords_length = coords[axis_idx]

        # Get centre in each direction
        centre = [self.centre[1], self.centre[0], self.centre[2]]
        centre_c1 = centre[circle_idx[0]]
        centre_c2 = centre[circle_idx[1]]
        centre_length = centre[axis_idx]

        # Make cylinder array
        data = (
            (np.sqrt((coords_c1 - centre_c1) ** 2
                     + (coords_c2 - centre_c2) ** 2) <= self.radius)
            & 
            (np.absolute(coords_length - centre_length) <= self.length / 2)
        )
        return data


class Grid(Shape):

    def __init__(self, shape, spacing, thickness, intensity, axis=None, 
                 name=None):

        self.name = name
        self.spacing = make_three(spacing)
        self.thickness = make_three(thickness)
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

