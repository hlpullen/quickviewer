# Synthetic images

The `SyntheticImage` class enables the creation of images containing simple geometric shapes.

<img src="images/synthetic_example.png" alt="synthetic image with a sphere and a cube" height="500"/>

## Creating a synthetic image

To create an empty image, load the `SyntheticImage` class and specify the desired image shape in order (x, y, z), e.g.

```
from quickviewer.prototype.simulation import SyntheticImage

sim = SyntheticImage((250, 250, 50))
```

The following arguments can be used to adjust the image's properties:
- `voxel_size`: voxel sizes in mm in order (x, y, z); default (1, 1, 1).
- `origin`: position of the top-left voxel in mm; default (0, 0, 0).
- `intensity`: value of the background voxels of the image.
- `noise_std`: standard deviation of Gaussian noise to apply to the image. This noise will also be added on top of any shapes.

### Adding shapes

The `SyntheticImage` object has various methods for adding geometric shapes. Each shape has the following arguments:
- `intensity`: intensity value with which to fill the voxels of this shape.
- `above`: if `True`, this shape will be overlaid on top of all existing shapes; otherwise, it will be added below all other shapes.

The available shapes and their specific arguments are:
- *Sphere*: `sim.add_sphere(radius, centre=None)`
    - `radius`: radius of the sphere in mm.
    - `centre`: position of the centre of the sphere in mm (if `None`, the sphere will be placed in the centre of the image).
- *Cuboid*: `sim.add_cuboid(side_length, centre=None)`
    - `side_length`: side length in mm. Can either be a single value (to create a cube) or a list of the (x, y, z) side lengths.
    - `centre`: position of the centre of the cuboid in mm (if `None`, the cuboid will be placed in the centre of the image).
- *Cube*: `sim.add_cube(side_length, centre=None)`
    - Same as `add_cuboid`.
- *Cylinder*: `sim.add_cylinder(radius, length, axis='z', centre=None)`
    - `radius`: radius of the cylinder in mm.
    - `length`: length of the cylinder in mm.
    - `axis`: either `'x'`, `'y'`, or `'z'`; direction along which the length of the cylinder should lie.
    - `centre`: position of the centre of the cylinder in mm (if `None`, the cylinder will be placed in the centre of the image).
- *Grid*: `sim.add_grid(spacing, thickness=1, axis=None)`
    - `spacing`: grid spacing in mm. Can either be a single value, or list of (x, y, z) spacings.
    - `thickenss`: gridline thickness in mm. Can either be a single value, or list of (x, y, z) thicknesses.
    - `axis`: if None, gridlines will be created in all three directions; if set to `'x'`, `'y'`, or `'z'`, grids will only be created in the plane orthogonal to the chosen axis, such that a solid grid runs through the image along that axis.

### Plotting

### Adding noise

### Rotations and translations

### Getting the image array

### Saving


## Adding structures

### Single structures

### Grouped structures

### Getting structures

### Saving structures
