"""Classes and functions for loading, plotting, and comparing medical images."""

import re
import copy
import dateutil.parser
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm
import nibabel
import numpy as np
import os
import pydicom
import shutil
from scipy import interpolate, ndimage
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FormatStrFormatter


# Shared parameters
_axes = {"x": 0, "y": 1, "z": 2}
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [1, 0, 2]}
_n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
_default_figsize = 6


class Image:
    """Load and plot image arrays from NIfTI files or NumPy objects."""

    def __init__(
        self,
        nii,
        affine=None,
        voxel_sizes=(1, 1, 1),
        origin=(0, 0, 0),
        title=None,
        scale_in_mm=True,
        downsample=None,
        orientation="x-y",
        rescale=True,
    ):
        """Initialise from a NIfTI file, NIfTI object, or numpy array.

        Parameters
        ----------
        nii : str/array/nifti
            Source of the image data to load. This can be either:
                (a) The path to a NIfTI or DICOM file;
                (b) A nibabel.nifti1.Nifti1Image object;
                (c) The path to a file containing a NumPy array;
                (d) A NumPy array.

        affine : 4x4 array, default=None
            Affine matrix to be used if <nii> is a NumPy array. If <nii> is a
            file path or a nibabel object, this parameter is ignored. If None,
            the arguments <voxel_sizes> and <origin> will be used to set the
            affine matrix.

        voxel_sizes : tuple, default=(1, 1, 1)
            Voxel sizes in mm, given in the order (y, x, z). Only used if
            <nii> is a numpy array and <affine> is None.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm, given in the order (y, x, z). Only used if
            <nii> is a numpy array and <affine> is None.

        title : str, default=None
            Title to use when plotting. If None and the image was loaded from
            a file, the filename will be used.

        scale_in_mm : bool, default=True
            If True, plot axes will be in mm; if False, plot axes will be in
            voxels.

        orientation : str, default="x-y"
            String specifying the orientation of the image if a 2D array is
            given for <nii>. Must be "x-y", "y-z", or "x-z".
        """

        # Assign settings
        self.title = title
        self.scale_in_mm = scale_in_mm
        self.data_mask = None
        if nii is None:
            self.valid = False
            return

        # Load image
        self.data, voxel_sizes, origin, self.path = load_image(
            nii, affine, voxel_sizes, origin, rescale
        )
        if self.data is None:
            self.valid = False
            return
        self.valid = True
        self.data = np.nan_to_num(self.data)
        self.shape = self.data.shape
        if self.title is None and self.path is not None:
            self.title = os.path.basename(self.path)

        # Convert 2D image to 3D
        self.dim2 = self.data.ndim == 2
        if self.dim2:
            voxel_sizes, origin = self.convert_to_3d(voxel_sizes, origin, orientation)

        # Assign geometric properties
        self.voxel_sizes = {ax: voxel_sizes[n] for ax, n in _axes.items()}
        self.origin = {ax: origin[n] for ax, n in _axes.items()}
        self.set_geom()
        self.set_shift(0, 0, 0)
        self.set_plotting_defaults()

        # Apply downsampling
        if downsample is not None:
            self.downsample(downsample)

    def convert_to_3d(self, voxel_sizes, origin, orientation):
        """Convert own image array to 3D and fill voxel sizes/origin."""

        if self.data.ndim != 2:
            return

        self.orientation = orientation
        self.data = self.data[..., np.newaxis]
        voxel_sizes = np.array(voxel_sizes)
        origin = np.array(origin)
        np.append(voxel_sizes, 1)
        np.append(origin, 0)

        # Transpose
        transpose = {
            "x-y": [0, 1, 2],
            "y-x": [1, 0, 2],
            "x-z": [0, 2, 1],
            "z-x": [1, 2, 0],
            "y-z": [2, 0, 1],
            "z-y": [2, 1, 0],
        }.get(self.orientation, "x-y")
        self.data = np.transpose(self.data, transpose)
        voxel_sizes = list(voxel_sizes[transpose])
        origin = list(origin[transpose])
        return voxel_sizes, origin

    def set_geom(self):
        """Assign geometric properties based on image data, origin, and
        voxel sizes."""

        # Number of voxels in each direction
        self.n_voxels = {ax: self.data.shape[n] for ax, n in _axes.items()}
        self.centre = [n / 2 for n in self.n_voxels.values()]

        # Min and max voxel position
        self.lims = {
            ax: (
                self.origin[ax],
                self.origin[ax] + (self.n_voxels[ax] - 1) * self.voxel_sizes[ax],
            )
            for ax in _axes
        }

        # Extent and aspect for use in matplotlib.pyplot.imshow
        self.ax_lims = {}
        self.extent = {}
        self.aspect = {}
        for view, (x, y) in _plot_axes.items():

            z = _slider_axes[view]
            if self.scale_in_mm:
                vx = self.voxel_sizes[x]
                vy = self.voxel_sizes[y]
                self.ax_lims[view] = [
                    [min(self.lims[x]) - abs(vx / 2), max(self.lims[x]) + abs(vx / 2)],
                    [max(self.lims[y]) + abs(vy / 2), min(self.lims[y]) - abs(vy / 2)],
                ]
                self.extent[view] = self.ax_lims[view][0] + self.ax_lims[view][1]
                self.aspect[view] = 1
            else:
                x_lim = [
                    self.idx_to_slice(0, x),
                    self.idx_to_slice(self.n_voxels[x] - 1, x),
                ]
                x_lim[x_lim.index(max(x_lim))] += 0.5
                x_lim[x_lim.index(min(x_lim))] -= 0.5
                self.ax_lims[view] = [x_lim, [self.n_voxels[y] + 0.5, 0.5]]
                self.extent[view] = self.ax_lims[view][0] + self.ax_lims[view][1]
                self.aspect[view] = abs(self.voxel_sizes[y] / self.voxel_sizes[x])

    def get_coords(self):
        """Get lists of coordinates in each direction."""

        x = np.linspace(self.lims["x"][0], self.lims["x"][1], self.shape[0])
        y = np.linspace(self.lims["y"][0], self.lims["y"][1], self.shape[1])
        z = np.linspace(self.lims["z"][0], self.lims["z"][1], self.shape[2])
        return x, y, z

    def match_size(self, im, fill_value=-1024):
        """Match shape to that of another Image."""

        if not hasattr(self, "shape") or self.shape == im.shape:
            return

        x, y, z = self.get_coords()
        if x[0] > x[-1]:
            x = x[::-1]
        interpolant = interpolate.RegularGridInterpolator(
            (x, y, z), self.data, method="linear", bounds_error=False,
            fill_value=fill_value)
        stack = np.vstack(np.meshgrid(*im.get_coords(), indexing="ij"))
        points = stack.reshape(3, -1).T.reshape(*im.shape, 3)

        self.data = interpolant(points)[::-1, :, :]
        self.shape = im.shape
        self.voxel_sizes = im.voxel_sizes
        self.origin = im.origin
        self.set_geom()

    def resample(self, data, v, round_up=True):
        """Resample an image to have particular voxel sizes."""


        # Make interpolant
        x, y, z = [
            np.linspace(self.lims["x"][0], self.lims["x"][1], data.shape[0]),
            np.linspace(self.lims["y"][0], self.lims["y"][1], data.shape[1]),
            np.linspace(self.lims["z"][0], self.lims["z"][1], data.shape[2])
        ]
        if x[0] > x[-1]:
            x = x[::-1]
        interpolant = interpolate.RegularGridInterpolator(
            (x, y, z), data, method="linear", bounds_error=False,
            fill_value=self.get_min())

        # Calculate desired limits and numbers of voxels
        lims = []
        shape = []
        for i, ax in enumerate(_axes):
            vx = v[i]
            if vx * self.voxel_sizes[ax] < 0:
                vx = -vx
            lim1 = self.lims[ax][0]
            lim2 = self.lims[ax][1] + self.voxel_sizes[ax]
            length = lim2 - lim1
            n = abs(length / vx)
            if round_up:
                shape.append(np.ceil(n))
            else:
                shape.append(np.floor(n))
            remainder = abs(length) % vx
            if not round_up:
                remainder = vx - remainder
            if length > 0:
                lim2 += remainder
            else:
                lim2 -= remainder
            lim2 -= vx
            lims.append((lim1, lim2))
        shape = [int(s) for s in shape]

        # Interpolate to new set of coordinates
        new_coords = [
            np.linspace(lims[i][0], lims[i][1], shape[i])
            for i in range(3)
        ]
        stack = np.vstack(np.meshgrid(*new_coords, indexing="ij"))
        points = stack.reshape(3, -1).T.reshape(*shape, 3)
        return interpolant(points)[::-1, :, :]

    def get_lengths(self, view):
        """Get the x and y lengths of the image in a given orientation."""

        x_length = abs(self.ax_lims[view][0][1] - self.ax_lims[view][0][0])
        y_length = abs(self.ax_lims[view][1][1] - self.ax_lims[view][1][0])
        if self.scale_in_mm:
            return x_length, y_length
        else:
            x, y = _plot_axes[view]
            return (
                x_length * abs(self.voxel_sizes[x]),
                y_length * abs(self.voxel_sizes[y]),
            )

    def get_image_centre(self, view):
        """Get midpoint of a given orientation."""

        mid_x = np.mean(self.ax_lims[view][0])
        mid_y = np.mean(self.ax_lims[view][1])
        return [mid_x, mid_y]

    def set_shift(self, dx, dy, dz):
        """Set the current translation to apply, where dx/dy/dz are in voxels."""

        self.shift = {"x": dx, "y": dy, "z": dz}
        self.shift_mm = {
            ax: d * abs(self.voxel_sizes[ax]) for ax, d in self.shift.items()
        }

    def same_frame(self, im):
        """Check whether this image is in the same frame of reference as
        another Image <im> (i.e. same origin and shape)."""

        same = self.shape == im.shape
        origin1 = [f"{x:.2f}" for x in self.origin.values()]
        origin2 = [f"{x:.2f}" for x in im.origin.values()]
        same *= origin1 == origin2
        vx1 = [f"{x:.2f}" for x in self.voxel_sizes.values()]
        vx2 = [f"{x:.2f}" for x in im.voxel_sizes.values()]
        same *= vx1 == vx2
        return same

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments."""

        self.mpl_kwargs = {"cmap": "gray", "vmin": -300, "vmax": 200}
        self.mask_color = "black"

    def get_relative_width(self, view, zoom=None, n_colorbars=0, figsize=None):
        """Get width:height ratio for this plot.

        Parameters
        ----------
        view : str
            Orientation ("x-y"/"y-z"/"x-z").

        zoom : float/tuple/dict, default=None
            Zoom factor.

        n_colorbars : int, default=0
            Number of colorbars to account for in computing the plot width.
        """

        if figsize is None:
            figsize = _default_figsize

        # Get x and y lengths
        x_length, y_length = self.get_lengths(view)

        # Account for axis labels and title
        font = mpl.rcParams["font.size"] / 72
        y_pad = 2 * font
        if self.title:
            y_pad += 1.5 * font
        y_ax = _plot_axes[view][1]
        max_y = np.max([abs(lim) for lim in self.lims[y_ax]])
        max_y_digits = np.floor(np.log10(max_y))
        minus = any([lim < 0 for lim in self.lims[y_ax]])
        x_pad = (0.7 * max_y_digits + 1.2 * minus + 1) * font

        # Account for zoom
        zoom = self.get_ax_dict(zoom)
        if zoom is not None:
            x, y = _plot_axes[view]
            y_length /= zoom[y]
            x_length /= zoom[x]

        # Add extra width for colorbars
        colorbar_frac = 0.4 * 5 / figsize
        x_length *= 1 + (n_colorbars * colorbar_frac)
        #  x_pad += 7 * font * n_colorbars

        # Get width ratio
        total_y = figsize + y_pad
        total_x = figsize * x_length / y_length + x_pad
        width = total_x / total_y

        return width

    def set_ax(self, view, ax=None, gs=None, figsize=None, zoom=None, n_colorbars=0):
        """Assign axes to self or create new axes if needed.

        Parameters
        ----------
        view : str
            Orientation ("x-y"/"y-z"/"x-z")

        ax : matplotlib.pyplot.Axes, default=None
            Axes to assign to self for plotting. If None, new axes will be
            created.

        gs : matplotlib.gridspec.GridSpec, default=None
            Gridspec to be used to create axes on an existing figure. Only
            used if <ax> is None.

        figsize : float, default=None
            Size of matplotlib figure in inches. Only used if <ax> and <gs>
            are both None.

        zoom : int/float/tuple, default=None
            Factor by which to zoom in. If a single int or float is given,
            the same zoom factor will be applied in all directions. If a tuple
            of three values is given, these will be used as the zoom factors
            in each direction in the order (x, y, z). If None, the image will
            not be zoomed in.

        n_colorbars : int, default=0
            Number of colorbars that will be plotted on these axes. Used to
            determine the relative width of the axes. Only used if <ax> and
            <gs> are both None.
        """

        # Set axes from gridspec
        if ax is None and gs is not None:
            ax = plt.gcf().add_subplot(gs)

        # Assign existing axes to self
        if ax is not None:
            self.fig = ax.figure
            self.ax = ax
            return

        # Get relative width
        rel_width = self.get_relative_width(view, zoom, n_colorbars, figsize)

        # Create new figure and axes
        figsize = _default_figsize if figsize is None else figsize
        figsize = to_inches(figsize)
        self.fig = plt.figure(figsize=(figsize * rel_width, figsize))
        self.ax = self.fig.add_subplot()

    def get_kwargs(self, mpl_kwargs, default=None):
        """Return a dict of matplotlib keyword arguments, combining default
        values with custom values. If <default> is None, the class
        property self.mpl_kwargs will be used as default."""

        if default is None:
            custom_kwargs = self.mpl_kwargs.copy()
        else:
            custom_kwargs = default.copy()
        if mpl_kwargs is not None:
            custom_kwargs.update(mpl_kwargs)
        return custom_kwargs

    def idx_to_pos(self, idx, ax):
        """Convert an index to a position in mm along a given axis."""

        if ax != "z":
            return (
                self.origin[ax] + (self.n_voxels[ax] - 1 - idx) * self.voxel_sizes[ax]
            )
        else:
            return self.origin[ax] + idx * self.voxel_sizes[ax]

    def pos_to_idx(self, pos, ax, force_int=True):
        """Convert a position in mm to an index along a given axis."""

        if ax != "z":
            idx = self.n_voxels[ax] - 1 + (self.origin[ax] - pos) / self.voxel_sizes[ax]
        else:
            idx = (pos - self.origin[ax]) / self.voxel_sizes[ax]

        if idx < 0 or idx >= self.n_voxels[ax]:
            if idx < 0:
                idx = 0
            if idx >= self.n_voxels[ax]:
                idx = self.n_voxels[ax] - 1

        if force_int:
            idx = round(idx)
        return idx

    def idx_to_slice(self, idx, ax):
        """Convert an index to a slice number along a given axis."""

        if self.voxel_sizes[ax] < 0:
            return idx + 1
        else:
            return self.n_voxels[ax] - idx

    def slice_to_idx(self, sl, ax):
        """Convert a slice number to an index along a given axis."""

        if self.voxel_sizes[ax] < 0:
            idx = sl - 1
        else:
            idx = self.n_voxels[ax] - sl

        if idx < 0 or idx >= self.n_voxels[ax]:
            if idx < 0:
                idx = 0
            if idx >= self.n_voxels[ax]:
                idx = self.n_voxels[ax] - 1

        return idx

    def pos_to_slice(self, pos, ax):
        """Convert a position in mm to a slice number."""

        return self.idx_to_slice(self.pos_to_idx(pos, ax), ax)

    def slice_to_pos(self, sl, ax):
        """Convert a slice number to a position in mm."""

        return self.idx_to_pos(self.slice_to_idx(sl, ax), ax)

    def set_mask(self, mask, threshold=0.5):
        """Set a mask for this image. Can be a single mask array or a
        dictionary of mask arrays. This mask will be used when self.plot()
        is called with masked=True. Note: mask_threshold only used if the
        provided mask is not already a boolean array."""

        if not self.valid:
            return
        if mask is None:
            self.data_mask = None
            return

        # Apply mask from NumPy array
        elif isinstance(mask, np.ndarray):
            self.data_mask = self.process_mask(mask, threshold)

        # Apply mask from Image
        elif isinstance(mask, Image):
            if not mask.valid:
                self.data_mask = None
                return
            self.data_mask = self.process_mask(mask.data, threshold)

        # Dictionary of masks
        elif isinstance(mask, dict):
            self.data_mask = mask
            for view in _orient:
                if view in self.data_mask:
                    self.data_mask[view] = self.process_mask(
                        self.data_mask[view], threshold
                    )
                else:
                    self.data_mask[view] = None
            return
        else:
            raise TypeError("Mask must be a numpy array or an Image.")

    def process_mask(self, mask, threshold=0.5):
        """Convert a mask to boolean and downsample if needed."""

        if mask.dtype != bool:
            mask = mask > threshold
        if mask.shape != self.data.shape and mask.shape == self.shape:
            mask = self.downsample_array(mask)
        return mask

    def get_idx(self, view, sl, pos, default_centre=True):
        """Convert a slice number or position in mm to an array index. If
        <default_centre> is set and <sl> and <pos> are both None, the middle
        slice will be taken; otherwise, an error will be raised."""

        z = _slider_axes[view]
        if sl is None and pos is None:
            if default_centre:
                idx = np.ceil(self.n_voxels[z] / 2)
            else:
                raise TypeError("Either <sl> or <pos> must be provided!")
        elif sl is not None:
            idx = self.slice_to_idx(sl, z)
        else:
            idx = self.pos_to_idx(pos, z)

        return int(idx)

    def get_min_hu(self):
        """Get the minimum HU in the image."""

        if not hasattr(self, "min_hu"):
            self.min_hu = self.data.min()
        return self.min_hu

    def set_slice(self, view, sl=None, pos=None, masked=False, invert_mask=False):
        """Assign a 2D array corresponding to a slice of the image in a given
        orientation to class variable self.current_slice. If the variable
        self.shift contains nonzero elements, the slice will be translated by
        the amounts in self.shift.

        Parameters
        ----------
        view : str
            Orientation ("x-y"/"y-z"/"x-z").

        sl : int, default=None
            Slice number. If <sl> and <pos> are both None, the middle slice
            will be plotted.

        pos : float, default=None
            Position in mm of the slice to plot (will be rounded to the nearest
            slice). If <sl> and <pos> are both None, the middle slice will be
            plotted. If <sl> and <pos> are both given, <sl> supercedes <pos>.

        masked : bool, default=False
            If True and self.data_mask is not None, the mask in data_mask
            will be applied to the image. Voxels above the mask threshold
            in self.data_mask will be masked.

        invert_mask : bool, default=False
            If True, values below the mask threshold will be used to
            mask the image instead of above threshold. Ignored if masked is
            False.
        """

        # Assign current orientation and slice index
        idx = self.get_idx(view, sl, pos)
        self.view = view
        self.idx = idx
        self.sl = self.idx_to_slice(idx, _slider_axes[view])

        # Get array index of the slice to plot
        # Apply mask if needed
        mask = (
            self.data_mask[view] if isinstance(self.data_mask, dict) else self.data_mask
        )
        if masked and mask is not None:
            if invert_mask:
                data = np.ma.masked_where(mask, self.data)
            else:
                data = np.ma.masked_where(~mask, self.data)
        else:
            data = self.data

        # Apply shift to slice index
        z = _slider_axes[view]
        slice_shift = self.shift[z]
        if slice_shift:
            if z == "y":
                idx += slice_shift
            else:
                idx -= slice_shift
            if idx < 0 or idx >= self.n_voxels[_slider_axes[view]]:
                self.current_slice = (
                    np.ones(
                        (
                            self.n_voxels[_plot_axes[view][0]],
                            self.n_voxels[_plot_axes[view][1]],
                        )
                    )
                    * self.get_min_hu()
                )
                return

        # Get 2D slice and adjust orientation
        im_slice = np.transpose(data, _orient[view])[:, :, idx]
        x, y = _plot_axes[view]
        if y != "x":
            im_slice = im_slice[::-1, :]

        # Apply 2D translation
        shift_x = self.shift[x]
        shift_y = self.shift[y]
        if shift_x:
            im_slice = np.roll(im_slice, shift_x, axis=1)
            if shift_x > 0:
                im_slice[:, :shift_x] = self.get_min_hu()
            else:
                im_slice[:, shift_x:] = self.get_min_hu()
        if shift_y:
            im_slice = np.roll(im_slice, shift_y, axis=0)
            if shift_y > 0:
                im_slice[:shift_y, :] = self.get_min_hu()
            else:
                im_slice[shift_y:, :] = self.get_min_hu()

        # Assign 2D array to current slice
        self.current_slice = im_slice

    def get_min(self):
        if not hasattr(self, "min_val"):
            self.min_val = self.data.min()
        return self.min_val

    def length_to_voxels(self, length, ax):
        return length / self.voxel_sizes[ax]

    def translate(self, dx=0, dy=0, dz=0):
        """Apply a translation to the image data."""

        # Convert mm to voxels
        if self.scale_in_mm:
            dx = self.length_to_voxels(dx, "x")
            dy = self.length_to_voxels(dy, "y")
            dz = -self.length_to_voxels(dz, "z")

        transform = get_translation_matrix(dx, dy, dz)
        if not hasattr(self, "original_data"):
            self.original_data = self.data
            self.original_centre = self.centre
        self.data = ndimage.affine_transform(self.data, transform,
                                             cval=self.get_min())
        self.centre = [self.centre[i] + [dx, dy, dz][i] for i in range(3)]

    def rotate(self, yaw=0, pitch=0, roll=0):
        """Rotate image data."""

        # Resample image to have equal voxel sizes
        vox = []
        if yaw or pitch:
            vox.append(abs(self.voxel_sizes["x"]))
        if yaw or roll:
            vox.append(abs(self.voxel_sizes["y"]))
        if pitch or roll:
            vox.append(abs(self.voxel_sizes["z"]))
        to_resample = len(set(vox)) > 1
        if to_resample:
            v = min(vox)
            data = self.resample(self.data, (v, v, v))
        else:
            data = self.data

        # Make 3D rotation matrix
        centre = [n / 2 for n in data.shape]
        transform = get_rotation_matrix(yaw, pitch, roll, centre)

        # Rotate around image centre
        if not hasattr(self, "original_data"):
            self.original_data = self.data
        rotated = ndimage.affine_transform(data, transform,
                                           cval=self.get_min())
        if to_resample:
            self.data = self.resample(rotated, list(self.voxel_sizes.values()), 
                                      round_up=False)
        else:
            self.data = rotated

    def reset(self):
        if hasattr(self, "original_data"):
            self.data = self.original_data
        if hasattr(self, "original_centre"):
            self.centre = self.original_centre

    def plot(
        self,
        view="x-y",
        sl=None,
        pos=None,
        idx=None,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        mpl_kwargs=None,
        show=True,
        colorbar=False,
        colorbar_label="HU",
        masked=False,
        invert_mask=False,
        mask_color="black",
        no_ylabel=False,
        no_title=False,
        annotate_slice=None,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
    ):
        """Plot a 2D slice of the image.

        Parameters
        ----------
        view : str, default="x-y"
            Orientation in which to plot ("x-y"/"y-z"/"x-z").

        sl : int, default=None
            Slice number. If <sl> and <pos> are both None, the middle slice
            will be plotted.

        pos : float, default=None
            Position in mm of the slice to plot (will be rounded to the nearest
            slice). If <sl> and <pos> are both None, the middle slice will be
            plotted. If <sl> and <pos> are both given, <sl> supercedes <pos>.

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot. If None, new axes will be created.

        gs : matplotlib.gridspec.GridSpec, default=None
            If not None and <ax> is None, new axes will be created on the
            current matplotlib figure with this gridspec.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        show : bool, default=True
            If True, the plotted figure will be shown via
            matplotlib.pyplot.show().

        figsize : float, default=None
            Figure height in inches; only used if <ax> and <gs> are None. If
            None, the value in _default_figsize will be used.

        zoom : int/float/tuple, default=None
            Factor by which to zoom in. If a single int or float is given,
            the same zoom factor will be applied in all directions. If a tuple
            of three values is given, these will be used as the zoom factors
            in each direction in the order (x, y, z). If None, the image will
            not be zoomed in.

        zoom_centre : tuple, default=None
            Position around which zooming is applied. If None, the centre of
            the image will be used.

        colorbar : bool, default=True
            If True, a colorbar will be drawn alongside the plot.

        colorbar_label : str, default="HU"
            Label for the colorbar, if drawn.

        masked : bool, default=False
            If True and this object has attribute self.data_mask assigned,
            the image will be masked with the array in self.data_mask.

        invert_mask : bool, default=True
            If True and a mask is applied, the mask will be inverted.

        mask_color : matplotlib color, default="black"
            color in which to plot masked areas.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will
            be added. If True, the default color (white) will be used.
        """

        if not self.valid:
            return

        # Get slice
        zoom = self.get_ax_dict(zoom)
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        self.ax.set_facecolor("black")
        self.set_slice(view, sl, pos, masked, invert_mask)

        # Get colormap
        kwargs = self.get_kwargs(mpl_kwargs)
        if "interpolation" not in kwargs and masked:
            kwargs["interpolation"] = "none"
        cmap = copy.copy(matplotlib.cm.get_cmap(kwargs.pop("cmap")))
        cmap.set_bad(color=mask_color)

        # Plot image
        mesh = self.ax.imshow(
            self.current_slice,
            extent=self.extent[view],
            aspect=self.aspect[view],
            cmap=cmap,
            **kwargs,
        )

        self.label_ax(view, no_ylabel, no_title, annotate_slice)
        self.adjust_ax(view, zoom, zoom_centre)
        if major_ticks:
            self.ax.xaxis.set_major_locator(MultipleLocator(major_ticks))
            self.ax.yaxis.set_major_locator(MultipleLocator(major_ticks))
        if minor_ticks:
            self.ax.xaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
            self.ax.yaxis.set_minor_locator(AutoMinorLocator(minor_ticks))
        if ticks_all_sides:
            self.ax.tick_params(bottom=True, top=True, left=True, right=True)
            if minor_ticks:
                self.ax.tick_params(
                    which="minor", bottom=True, top=True, left=True, right=True
                )

        # Draw colorbar
        if colorbar and kwargs.get("alpha", 1) > 0:
            clb = self.fig.colorbar(mesh, ax=self.ax, label=colorbar_label)
            clb.solids.set_edgecolor("face")

        # Display image
        if show:
            plt.tight_layout()
            plt.show()

    def label_ax(self, view, no_ylabel=False, no_title=False, annotate_slice=None):
        """Assign x/y axis labels and title to the plot."""

        units = " (mm)" if self.scale_in_mm else ""
        self.ax.set_xlabel(_plot_axes[view][0] + units, labelpad=0)
        if not no_ylabel:
            self.ax.set_ylabel(_plot_axes[view][1] + units)
        else:
            self.ax.set_yticks([])

        if self.title and not no_title:
            self.ax.set_title(self.title, pad=8)

        # Slice annotation
        if annotate_slice is not None:

            # Make annotation string
            ax = _slider_axes[view]
            if self.scale_in_mm:
                z_str = "{} = {:.1f} mm".format(ax, self.idx_to_pos(self.idx, ax))
            else:
                z_str = f"{ax} = {self.idx_to_slice(self.idx, ax)}"

            # Add annotation
            if matplotlib.colors.is_color_like(annotate_slice):
                col = annotate_slice
            else:
                col = "white"
            self.ax.annotate(
                z_str,
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                color=col,
                fontsize="large",
            )

    def adjust_ax(self, view, zoom=None, zoom_centre=None):
        """Adjust axis limits."""

        lims = self.get_zoomed_lims(view, zoom, zoom_centre)
        self.ax.set_xlim(lims[0])
        self.ax.set_ylim(lims[1])

    def get_zoomed_lims(self, view, zoom=None, zoom_centre=None):
        """Get axis limits zoomed in."""

        init_lims = self.ax_lims[view]
        if zoom is None:
            return init_lims

        zoom = self.get_ax_dict(zoom)
        zoom_centre = self.get_ax_dict(zoom_centre, default=None)

        # Get mid point
        x, y = _plot_axes[view]
        mid_x, mid_y = self.get_image_centre(view)
        if zoom_centre is not None:
            if zoom_centre[x] is not None:
                mid_x = zoom_centre[x]
            if zoom_centre[y] is not None:
                mid_y = zoom_centre[y]

        # Adjust axis limits
        xlim = [
            mid_x - (init_lims[0][1] - init_lims[0][0]) / (2 * zoom[x]),
            mid_x + (init_lims[0][1] - init_lims[0][0]) / (2 * zoom[x]),
        ]
        ylim = [
            mid_y - (init_lims[1][1] - init_lims[1][0]) / (2 * zoom[y]),
            mid_y + (init_lims[1][1] - init_lims[1][0]) / (2 * zoom[y]),
        ]
        return [xlim, ylim]

    def get_ax_dict(self, val, default=1):
        """Convert a single value or tuple of values in order (x, y, z) to a
        dictionary containing x/y/z as keys."""

        if val is None:
            return None
        if isinstance(val, dict):
            return val
        try:
            val = float(val)
            return {ax: val for ax in _axes}
        except TypeError:
            ax_dict = {ax: val[n] for ax, n in _axes.items()}
            for ax in ax_dict:
                if ax_dict[ax] is None:
                    ax_dict[ax] = default
            return ax_dict

    def downsample(self, d):
        """Downsample image by amount d = (dx, dy, dz) in the (x, y, z)
        directions. If <d> is a single value, the image will be downsampled
        equally in all directions."""

        self.downsample = self.get_ax_dict(d)
        for ax, d_ax in self.downsample.items():
            self.voxel_sizes[ax] *= d_ax
        self.data = self.downsample_array(self.data)
        self.n_voxels = {ax: self.data.shape[n] for ax, n in _axes.items()}
        self.set_geom()

    def downsample_array(self, data_array):
        """Downsample a NumPy array by amount set in self.downsample."""

        return data_array[
            :: round(self.downsample["x"]),
            :: round(self.downsample["y"]),
            :: round(self.downsample["z"]),
        ]


class ImageComparison(Image):
    """Class for loading data from two arrays and plotting comparison images."""

    def __init__(self, nii1, nii2, title=None, plot_type=None, **kwargs):
        """Load data from two arrays. <nii1> and <nii2> can either be existing
        Image objects, or objects from which Images can be created.
        """

        # Load Images
        self.ims = []
        self.standalone = True
        for nii in [nii1, nii2]:

            # Load existing Image
            if issubclass(type(nii), Image):
                self.ims.append(nii)

            # Create new Image
            else:
                self.standalone = False
                self.ims.append(Image(nii, **kwargs))

        self.scale_in_mm = self.ims[0].scale_in_mm
        self.ax_lims = self.ims[0].ax_lims
        self.valid = all([im.valid for im in self.ims])
        self.override_title = title
        self.gs = None
        self.plot_type = plot_type if plot_type else "chequerboard"

    def get_relative_width(self, view, colorbar=False, figsize=None):
        """Get relative width first image."""

        return Image.get_relative_width(
            self.ims[0], view, n_colorbars=colorbar, figsize=figsize
        )

    def plot(
        self,
        view=None,
        sl=None,
        invert=False,
        ax=None,
        mpl_kwargs=None,
        show=True,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        plot_type=None,
        cb_splits=2,
        overlay_opacity=0.5,
        overlay_legend=False,
        overlay_legend_loc=None,
        colorbar=False,
        colorbar_label="HU",
        show_mse=False,
        dta_tolerance=None,
        dta_crit=None,
        diff_crit=None,
    ):

        """Create a comparison plot of the two images.

        Parameters
        ----------
        view : str, default=None
            Orientation to plot ("x-y"/"y-z"/"x-z"). If <view> and <sl> are
            both None, they will be taken from the current orientation and
            slice of the images to be compared.

        sl : int, default=None
            Index of the slice to plot. If <view> and <sl> are both None, they
            will be taken from the current orientation and slice of the images
            to be compared.

        invert : bool, default=False
            If True, the plotting order of the two images will be reversed.

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot the comparison. If None, new axes will be
            created.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib for plotting
            the two images.

        show : bool, default=True
            If True, the figure will be shown via matplotlib.pyplot.show().

        figsize : float, default=None
            Figure height in inches to be used if a new figure is created. If
            None, the value in _default_figsize will be used.
        """

        if not self.valid:
            return

        # Use default plot type if not provided
        if plot_type is None:
            plot_type = self.plot_type

        # By default, use comparison type as title
        if self.override_title is None:
            self.title = plot_type[0].upper() + plot_type[1:]
        else:
            self.title = self.override_title

        # Get image slices
        if view is None and sl is None:
            for im in self.ims:
                if not hasattr(im, "current_slice"):
                    raise RuntimeError(
                        "Must provide a view and slice number "
                        "if input images do not have a current "
                        "slice set!"
                    )
            self.view = self.ims[0].view

        else:
            self.view = view
            for im in self.ims:
                im.set_slice(view, sl)

        self.slices = [im.current_slice for im in self.ims]

        # Plot settings
        self.set_ax(view, ax, self.gs, figsize, zoom)
        self.plot_kwargs = self.ims[0].get_kwargs(mpl_kwargs)
        self.cmap = copy.copy(matplotlib.cm.get_cmap(self.plot_kwargs.pop("cmap")))

        # Produce comparison plot
        if plot_type == "chequerboard":
            mesh = self.plot_chequerboard(invert, cb_splits)
        elif plot_type == "overlay":
            mesh = self.plot_overlay(
                invert, overlay_opacity, overlay_legend, overlay_legend_loc
            )
        elif plot_type == "difference":
            mesh = self.plot_difference(invert)
        elif plot_type == "absolute difference":
            mesh = self.plot_difference(invert, ab=True)
        elif plot_type == "distance to agreement":
            mesh = self.plot_dta(dta_tolerance)
        elif plot_type == "gamma index":
            mesh = self.plot_gamma(invert, dta_crit, diff_crit)
        elif plot_type == "image 1":
            self.title = self.ims[0].title
            mesh = self.ax.imshow(
                self.slices[0],
                extent=self.ims[0].extent[self.view],
                aspect=self.ims[0].aspect[self.view],
                cmap=self.cmap,
                **self.plot_kwargs,
            )
        elif plot_type == "image 2":
            self.title = self.ims[1].title
            mesh = self.ax.imshow(
                self.slices[1],
                extent=self.ims[1].extent[self.view],
                aspect=self.ims[1].aspect[self.view],
                cmap=self.cmap,
                **self.plot_kwargs,
            )

        # Draw colorbar
        if colorbar:
            clb_label = colorbar_label
            if plot_type in ["difference", "absolute difference"]:
                clb_label += " difference"
            elif plot_type == "distance to agreement":
                clb_label = "Distance (mm)"
            elif plot_type == "gamma index":
                clb_label = "Gamma index"

            clb = self.fig.colorbar(mesh, ax=self.ax, label=clb_label)
            clb.solids.set_edgecolor("face")

        # Adjust axes
        self.label_ax(self.view)
        self.adjust_ax(self.view, zoom, zoom_centre)

        # Annotate with mean squared error
        if show_mse:
            mse = np.sqrt(((self.slices[1] - self.slices[0]) ** 2).mean())
            mse_str = f"Mean sq. error = {mse:.2f}"
            if matplotlib.colors.is_color_like(show_mse):
                col = show_mse
            else:
                col = "white"
            self.ax.annotate(
                mse_str,
                xy=(0.05, 0.93),
                xycoords="axes fraction",
                color=col,
                fontsize="large",
            )

    def plot_chequerboard(self, invert=False, n_splits=2):
        """Produce a chequerboard plot with <n_splits> squares in each
        direction."""

        # Get masked image
        i1 = int(invert)
        i2 = 1 - i1
        size_x = int(np.ceil(self.slices[i2].shape[0] / n_splits))
        size_y = int(np.ceil(self.slices[i2].shape[1] / n_splits))
        cb_mask = np.kron(
            [[1, 0] * n_splits, [0, 1] * n_splits] * n_splits, np.ones((size_x, size_y))
        )
        cb_mask = cb_mask[: self.slices[i2].shape[0], : self.slices[i2].shape[1]]
        to_show = {
            i1: self.slices[i1],
            i2: np.ma.masked_where(cb_mask < 0.5, self.slices[i2]),
        }

        # Plot
        for i in [i1, i2]:
            mesh = self.ax.imshow(
                to_show[i],
                extent=self.ims[i].extent[self.view],
                aspect=self.ims[i].aspect[self.view],
                cmap=self.cmap,
                **self.plot_kwargs,
            )
        return mesh

    def plot_overlay(self, invert=False, opacity=0.5, legend=False, legend_loc="auto"):
        """Produce an overlay plot with a given opacity."""

        order = [0, 1] if not invert else [1, 0]
        cmaps = ["Reds", "Blues"]
        alphas = [1, opacity]
        self.ax.set_facecolor("w")
        handles = []
        for n, i in enumerate(order):

            # Show image
            mesh = self.ax.imshow(
                self.slices[i],
                extent=self.ims[i].extent[self.view],
                aspect=self.ims[i].aspect[self.view],
                cmap=cmaps[n],
                alpha=alphas[n],
                **self.plot_kwargs,
            )

            # Make handle for legend
            if legend:
                patch_color = cmaps[n].lower()[:-1]
                alpha = 1 - opacity if alphas[n] == 1 else opacity
                handles.append(
                    mpatches.Patch(
                        color=patch_color, alpha=alpha, label=self.ims[i].title
                    )
                )

        # Draw legend
        if legend:
            self.ax.legend(
                handles=handles, loc=legend_loc, facecolor="white", framealpha=1
            )
        return mesh

    def plot_difference(self, invert=False, ab=False):
        """Produce a difference plot."""

        diff = (
            self.slices[1] - self.slices[0]
            if not invert
            else self.slices[0] - self.slices[1]
        )
        if ab:
            diff = np.absolute(diff)
        return self.ax.imshow(
            diff,
            extent=self.ims[0].extent[self.view],
            aspect=self.ims[0].aspect[self.view],
            cmap=self.cmap,
            **self.plot_kwargs,
        )

    def plot_dta(self, tolerance=5):
        """Produce a distance-to-agreement plot."""

        dta = self.get_dta(tolerance)
        return self.ax.imshow(
            dta,
            extent=self.ims[0].extent[self.view],
            aspect=self.ims[0].aspect[self.view],
            cmap="viridis",
            interpolation=None,
            **self.plot_kwargs,
        )

    def plot_gamma(self, invert=False, dta_crit=None, diff_crit=None):
        """Produce a distance-to-agreement plot."""

        gamma = self.get_gamma(invert, dta_crit, diff_crit)
        return self.ax.imshow(
            gamma,
            extent=self.ims[0].extent[self.view],
            aspect=self.ims[0].aspect[self.view],
            cmap="viridis",
            interpolation=None,
            **self.plot_kwargs,
        )

    def get_dta(self, tolerance=None):
        """Compute distance to agreement array on current slice."""

        sl = self.ims[0].sl
        view = self.ims[0].view
        if not hasattr(self, "dta"):
            self.dta = {}
        if view not in self.dta:
            self.dta[view] = {}

        if sl not in self.dta[view]:

            x_ax, y_ax = _plot_axes[self.ims[0].view]
            vx = abs(self.ims[0].voxel_sizes[x_ax])
            vy = abs(self.ims[0].voxel_sizes[y_ax])

            im1, im2 = self.slices
            if tolerance is None:
                tolerance = 5
            abs_diff = np.absolute(im2 - im1)
            agree = np.transpose(np.where(abs_diff <= tolerance))
            disagree = np.transpose(np.where(abs_diff > tolerance))
            dta = np.zeros(abs_diff.shape)
            for coords in disagree:
                dta_vec = agree - coords
                dta_val = np.sqrt(
                    vy * dta_vec[:, 0] ** 2 + vx * dta_vec[:, 1] ** 2
                ).min()
                dta[coords[0], coords[1]] = dta_val

            self.dta[view][sl] = dta

        return self.dta[view][sl]

    def get_gamma(self, invert=False, dta_crit=None, diff_crit=None):
        """Get gamma index on current slice."""

        im1, im2 = self.slices
        if invert:
            im1, im2 = im2, im1

        if dta_crit is None:
            dta_crit = 1
        if diff_crit is None:
            diff_crit = 15

        diff = im2 - im1
        dta = self.get_dta()
        return np.sqrt((dta / dta_crit) ** 2 + (diff / diff_crit) ** 2)


def load_dicom(path, rescale=True):
    """Load a DICOM image array and affine matrix from a path."""

    # Single file
    if os.path.isfile(path):

        try:
            ds = pydicom.read_file(path)
            if int(ds.ImagesInAcquisition) == 1:
                data, affine = load_image_single_file(ds, rescale=rescale)

            # Look for other files from same image
            else:
                num = ds.SeriesNumber
                dirname = os.path.dirname(path)
                paths = [
                    os.path.join(dirname, p)
                    for p in os.listdir(dirname)
                    if not os.path.isdir(os.path.join(dirname, p))
                ]
                data, affine = load_image_multiple_files(
                    paths, series_num=num, rescale=rescale
                )

        except pydicom.errors.InvalidDicomError:
            raise TypeError("Not a valid dicom file!")

    # Directory
    elif os.path.isdir(path):
        paths = [
            os.path.join(path, p)
            for p in os.listdir(path)
            if not os.path.isdir(os.path.join(path, p))
        ]
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
    rescale_intercept = (
        float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0
    )
    if rescale == True and hasattr(ds, "RescaleSlope"):
        data = data * float(ds.RescaleSlope) + rescale_intercept
    elif rescale == "dose" and hasattr(ds, "DoseGridScaling"):
        data = data * float(ds.DoseGridScaling) + rescale_intercept

    # Get affine matrix
    vx, vy = ds.PixelSpacing
    vz = ds.SliceThickness
    px, py, pz = ds.ImagePositionPatient
    affine = np.array([[vx, 0, 0, px], [0, vy, 0, py], [0, 0, vz, pz], [0, 0, 0, 1]])

    # Adjust for consistency with dcm2nii
    affine[0, 0] *= -1
    affine[0, 3] *= -1
    affine[1, 3] = -(affine[1, 3] + affine[1, 1] * float(data.shape[1] - 1))
    if data.ndim == 3:
        affine[2, 3] = affine[2, 3] - affine[2, 2] * float(data.shape[2] - 1)

    return data, affine


def load_image_multiple_files(paths, series_num=None, rescale=True):
    """Load a single dicom image from multiple files."""

    data_slices = {}
    for path in paths:
        try:
            ds = pydicom.read_file(path)
            if series_num is not None and ds.SeriesNumber != series_num:
                continue
            slice_num = ds.SliceLocation
            data, affine = load_image_single_file(ds, rescale=rescale)
            data_slices[float(slice_num)] = data

        except pydicom.errors.InvalidDicomError:
            continue

    # Sort and stack image slices
    vz = affine[2, 2]
    data_list = [
        data_slices[sl] for sl in sorted(list(data_slices.keys()), reverse=(vz >= 0))
    ]
    data = np.stack(data_list, axis=-1)

    # Get z origin
    func = max if vz >= 0 else min
    affine[2, 3] = -func(list(data_slices.keys()))

    return data, affine


def load_image(im, affine=None, voxel_sizes=None, origin=None, rescale=True):
    """Load image from either:
        (a) a numpy array;
        (b) an nibabel nifti object;
        (c) a file containing a numpy array;
        (d) a nifti or dicom file.

    Returns image data, tuple of voxel sizes, tuple of origin points,
    and path to image file (None if image was not from a file)."""

    # Ensure voxel sizes and origin are lists
    if isinstance(voxel_sizes, dict):
        voxel_sizes = list(voxel_sizes.values())
    if isinstance(origin, dict):
        origin = list(origin.values())

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
                    data, affine = load_dicom(path, rescale)
                except TypeError:
                    try:
                        data = np.load(path)
                    except (IOError, ValueError):
                        raise RuntimeError(
                            "Input file <nii> must be a valid "
                            "NIfTI, DICOM, or NumPy file."
                        )

        # Load nibabel object
        elif isinstance(im, nibabel.nifti1.Nifti1Image):
            data = im.get_fdata()
            affine = im.affine

        else:
            raise TypeError(
                "Image input must be a string, nibabel object, or " "numpy array."
            )

    # Get voxel sizes and origin from affine
    if affine is not None:
        voxel_sizes = np.diag(affine)[:-1]
        origin = affine[:-1, -1]
    return data, np.array(voxel_sizes), np.array(origin), path


def find_files(paths, ext="", allow_dirs=False):
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

    if allow_dirs:
        return files
    return [f for f in files if not os.path.isdir(f)]


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


def is_nested(d):
    """Check whether a dict <d> has further dicts nested inside."""

    return all([isinstance(val, dict) for val in d.values()])


def make_three(var):
    """Ensure a variable is a tuple with 3 entries."""

    if is_list(var):
        return var

    return [var, var, var]


def find_date(s):
    """Find a date-like object in a string."""

    # Split into numeric strings
    nums = re.findall("[0-9]+", s)

    # Look for first date-like object
    for num in nums:
        try:
            return dateutil.parser.parse(num)
        except dateutil.parser.ParserError:
            continue

def get_translation_matrix(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

def get_rotation_matrix(yaw, pitch, roll, centre):

    # Convert angles to radians
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    cx, cy, cz = centre
    r1 = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, cx - cx * np.cos(yaw) + cy * np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw), 0, cy - cx * np.sin(yaw) - cy * np.cos(yaw)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    r2 = np.array([
        [np.cos(pitch), 0, np.sin(pitch), cx - cx * np.cos(pitch) - cz * np.sin(pitch)],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), cz + cx * np.sin(pitch) - cz * np.cos(pitch)],
        [0, 0, 0, 1]
    ])
    r3 = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), cy - cy * np.cos(roll) + cz * np.sin(roll)],
        [0, np.sin(roll), np.cos(roll), cz - cy * np.sin(roll) - cz * np.cos(roll)],
        [0, 0, 0, 1]
    ])
    return r1.dot(r2).dot(r3)
