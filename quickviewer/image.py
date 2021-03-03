"""Classes for plotting images from NIfTI files or NumPy arrays."""

import copy
import fnmatch
import glob
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import nibabel
import numpy as np
import os
import re
import skimage.measure
import matplotlib.patches as mpatches
from timeit import default_timer as timer


# Shared parameters
_axes = {"x": 0, "y": 1, "z": 2}
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [1, 0, 2]}
_n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}
_df_plot_types = ["grid", "quiver", "none"]
_struct_plot_types = ["contour", "mask", "none"]
_default_figsize = 6
_default_spacing = 30


class NiftiImage:
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
        zoom=None,
        orientation="x-y"
    ):
        """Initialise from a NIfTI file, NIfTI object, or numpy array.

        Parameters
        ----------
        nii : str/array/nifti
            Source of the image data to load. This can be either:
                (a) The path to a NIfTI file;
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

        zoom : int/float/tuple, default=None
            Factor by which to zoom in when plotting this image. If a single
            int or float is given, the same zoom factor will be applied in all
            directions. If a tuple of three values is given, these will be used
            as the zoom factors in each direction in the order (x, y, z). If
            None, the image will not be zoomed in.

        orientation : str, default="x-y"
            String specifying the orientation of the image if a 2D array is 
            given for <nii>. Must be "x-y", "y-z", or "x-z".
        """

        # Assign settings
        self.title = title
        self.scale_in_mm = scale_in_mm
        self.data_mask = None
        self.zoom = self.get_ax_dict(zoom)
        if nii is None:
            self.valid = False
            return

        # Load from numpy array
        if isinstance(nii, np.ndarray):
            self.data = nii

        # Load from file or nifti object
        else:
            if isinstance(nii, str):
                self.path = os.path.expanduser(nii)
                try:
                    self.nii = nibabel.load(self.path)
                    self.data = self.nii.get_fdata()
                    affine = self.nii.affine
                except FileNotFoundError:
                    print(f"Warning: file {self.path} not found!")
                    self.valid = False
                    return
                except nibabel.filebasedimages.ImageFileError:
                    try:
                        self.data = np.load(self.path)
                    except (IOError, ValueError):
                        raise RuntimeError("Input file <nii> must be a valid "
                                           ".nii or .npy file.")
                if self.title is None:
                    self.title = os.path.basename(self.path)
            elif isinstance(nii, nibabel.nifti1.Nifti1Image):
                self.nii = nii
                self.data = self.nii.get_fdata()
                affine = self.nii.affine
            else:
                raise TypeError("<nii> must be a string, nibabel object, or "
                                "numpy array.")

        # Assign geometric properties
        self.data = np.nan_to_num(self.data)
        self.shape = self.data.shape
        self.valid = True

        # Check number of dimensions
        self.dim2 = self.data.ndim == 2
        if self.dim2:
            affine, voxel_sizes, origin = self.convert_to_3d(
                affine, voxel_sizes, origin, orientation)

        # Assign geometric properties
        if affine is not None:
            self.voxel_sizes = {ax: affine[n, n] for ax, n in _axes.items()}
            self.origin = {ax: affine[n, 3] for ax, n in _axes.items()}
        else:
            self.voxel_sizes = {ax: voxel_sizes[n] for ax, n in _axes.items()}
            self.origin = {ax: origin[n] for ax, n in _axes.items()}
        self.set_geom()
        self.set_shift(0, 0, 0)
        self.set_plotting_defaults()

        # Apply downsampling
        if downsample is not None:
            self.downsample(downsample)

    def convert_to_3d(self, affine, voxel_sizes, origin, orientation):
        """Convert own image array to 3D and fill voxel sizes/origin."""

        if self.data.ndim != 2:
            return

        self.orientation = orientation
        self.data = self.data[..., np.newaxis]

        if affine is not None:
            voxel_sizes = [affine[0, 0], affine[1, 1]]
            origin = [affine[0, 2], affine[1,2]]
            affine = None
        if orientation == "x-y":
            voxel_sizes = [voxel_sizes[0], voxel_sizes[1], 1]
            origin = [origin[0], origin[2], 0]
        elif orientation == "x-z":
            self.data = np.transpose(self.data, [0, 2, 1])
            voxel_sizes = [voxel_sizes[0], 1, voxel_sizes[1]]
            origin = [0, origin[0], origin[2]]
        elif orientation == "y-z":
            self.data = np.transpose(self.data, [2, 0, 1])
            voxel_sizes = [1, voxel_sizes[0], voxel_sizes[1]]
            origin = [0, origin[0], origin[2]]
        else:
            raise TypeError("Orientation of 2D image must be x-y, y-z, "
                            "or x-z.")

        return affine, voxel_sizes, origin

    def set_geom(self):
        """Assign geometric properties based on image data, origin, and
        voxel sizes."""

        # Number of voxels in each direction
        self.n_voxels = {ax: self.data.shape[n] for ax, n in _axes.items()}

        # Min and max voxel position
        self.lims = {
            ax: (self.origin[ax], self.origin[ax] + (self.n_voxels[ax] - 1) *
                 self.voxel_sizes[ax])
            for ax in _axes
        }

        # Extent and aspect for use in matplotlib.pyplot.imshow
        self.extent = {}
        self.aspect = {}
        for view, (x, y) in _plot_axes.items():
            if self.scale_in_mm:
                vx = self.voxel_sizes[x]
                vy = self.voxel_sizes[y]
                self.extent[view] = (
                    min(self.lims[x]) - abs(vx / 2), 
                    max(self.lims[x]) + abs(vx / 2),
                    max(self.lims[y]) + abs(vy / 2),
                    min(self.lims[y]) - abs(vy / 2)
                )
                self.aspect[view] = 1
            else:
                self.extent[view] = (
                    self.idx_to_slice(0, x) - 0.5, 
                    self.idx_to_slice(self.n_voxels[x], x) + 0.5,
                    self.n_voxels[y] + 0.5, 0.5
                )
                self.aspect[view] = abs(self.voxel_sizes[y] /
                                        self.voxel_sizes[x])

    def get_length(self, ax):
        """Get the length of the image along a given axis. If self.zoom is not
        None, the length will be divided by the factor given in self.zoom
        for the chosen axis."""

        length = abs(self.lims[ax][0] - self.lims[ax][1]) \
                + abs(self.voxel_sizes[ax])
        if self.zoom is not None:
            length /= self.zoom[ax]
        return length

    def set_shift(self, dx, dy, dz):
        """Set the current translation to apply, where dx/dy/dz are in voxels.
        """

        self.shift = {"x": dx, "y": dy, "z": dz}
        self.shift_mm = {ax: d * abs(self.voxel_sizes[ax]) for ax, d in 
                         self.shift.items()}

    def same_frame(self, im):
        """Check whether this image is in the same frame of reference as 
        another NiftiImage <im> (i.e. same origin and shape)."""

        same = self.shape == im.shape
        same *= list(self.origin.values()) == list(im.origin.values())
        same *= list(self.voxel_sizes.values()) \
            == list(im.voxel_sizes.values())
        return same

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments."""

        self.mpl_kwargs = {"cmap": "gray",
                           "vmin": -300,
                           "vmax": 200}
        self.mask_colour = "black"

    def get_relative_width(self, view, n_colorbars=0):
        """Get width:height ratio for this plot.

        Parameters
        ----------
        view : str
            Orientation ("x-y"/"y-z"/"x-z").
    
        n_colorbars : int, default=0
            Number of colorbars to account for in computing the plot width.
        """

        # Get relative x/y length
        x, y = _plot_axes[view]
        x_length = self.get_length(x)
        y_length = self.get_length(y)
        width = x_length / y_length

        # Account for zoom
        if self.zoom is not None:
            width *= self.zoom[x] / self.zoom[y]

        # Add extra width for colorbars
        colorbar_frac = 0.3
        width *= 1 + n_colorbars * colorbar_frac / width

        return width

    def set_ax(self, view, ax=None, gs=None, figsize=None, n_colorbars=0):
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

        # Create new figure and axes
        rel_width = self.get_relative_width(view, n_colorbars)
        figsize = _default_figsize if figsize is None else figsize
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

        if ax == "z":
            return self.origin[ax] + idx * self.voxel_sizes[ax]
        else:
            return self.origin[ax] \
                    + (self.n_voxels[ax] - 1 - idx) * self.voxel_sizes[ax]

    def pos_to_idx(self, pos, ax):
        """Convert a position in mm to an index along a given axis."""

        if ax == "z":
            idx = round((pos - self.origin[ax]) / self.voxel_sizes[ax])
        else:
            idx = round(self.n_voxels[ax] - 1 + (self.origin[ax] - pos) / 
                        self.voxel_sizes[ax])

        if idx < 0 or idx >= self.n_voxels[ax]:
            if idx < 0:
                idx = 0
            if idx >= self.n_voxels[ax]:
                idx = self.n_voxels[ax] -1
            print(f"Warning: position {pos} outside valid range. Will "
                  f"plot slice at {self.idx_to_pos(idx, ax):.1f}")

        return idx

    def idx_to_slice(self, idx, ax):
        """Convert an index to a slice number along a given axis."""

        if ax == "x":
            return idx + 1
        else:
            return self.n_voxels[ax] - idx

    def slice_to_idx(self, sl, ax):
        """Convert a slice number to an index along a given axis."""

        if ax == "x":
            idx = sl - 1
        else:
            idx = self.n_voxels[ax] - sl

        if idx < 0 or idx >= self.n_voxels[ax]:
            if idx < 0:
                idx = 0
            if idx >= self.n_voxels[ax]:
                idx = self.n_voxels[ax] -1
            print(f"Warning: slice {sl} outside valid range. Will "
                  f"plot slice {self.idx_to_slice(idx, ax)}.")

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

        # Apply mask from NiftiImage
        elif isinstance(mask, NiftiImage):
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
                        self.data_mask[view], threshold)
                else:
                    self.data_mask[view] = None
            return
        else:
            raise TypeError("Mask must be a numpy array or a NiftiImage.")

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

        return idx

    def get_min_hu(self):
        """Get the minimum HU in the image."""

        if not hasattr(self, "min_hu"):
            self.min_hu = self.data.min()
        return self.min_hu

    def set_slice(self, view, sl=None, pos=None, masked=False, 
                  invert_mask=False):
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
        self.view = view
        idx = self.get_idx(view, sl, pos)
        self.idx = idx
        self.sl = self.idx_to_slice(idx, _slider_axes[view])

        # Get array index of the slice to plot
        # Apply mask if needed
        mask = self.data_mask[view] if isinstance(self.data_mask, dict) \
                else self.data_mask
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
                self.current_slice = np.ones((
                    self.n_voxels[_plot_axes[view][0]],
                    self.n_voxels[_plot_axes[view][1]])) * self.get_min_hu()
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

    def plot(self, 
             view="x-y",
             sl=None, 
             pos=None,
             idx=None,
             ax=None, 
             gs=None, 
             figsize=None, 
             mpl_kwargs=None, 
             show=True, 
             colorbar=False, 
             colorbar_label="HU", 
             masked=False,
             invert_mask=False, 
             mask_colour="black", 
             no_ylabel=False,
             no_title=False,
             annotate_slice=None
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

        colorbar : bool, default=True
            If True, a colorbar will be drawn alongside the plot.

        colorbar_label : str, default="HU"
            Label for the colorbar, if drawn.

        masked : bool, default=False
            If True and this object has attribute self.data_mask assigned,
            the image will be masked with the array in self.data_mask.

        invert_mask : bool, default=True
            If True and a mask is applied, the mask will be inverted.

        mask_colour : matplotlib colour, default="black"
            Colour in which to plot masked areas.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will 
            be added. If True, the default colour (white) will be used.
        """

        if not self.valid:
            return

        # Get slice
        self.set_ax(view, ax, gs, figsize, colorbar)
        self.ax.set_facecolor("black")
        self.set_slice(view, sl, pos, masked, invert_mask)

        # Get colourmap
        kwargs = self.get_kwargs(mpl_kwargs)
        cmap = copy.copy(matplotlib.cm.get_cmap(kwargs.pop("cmap")))
        cmap.set_bad(color=mask_colour)

        # Plot image
        mesh = self.ax.imshow(self.current_slice,
                              extent=self.extent[view],
                              aspect=self.aspect[view],
                              cmap=cmap, **kwargs)

        self.label_ax(view, no_ylabel, no_title, annotate_slice)
        self.apply_zoom(view)

        # Draw colorbar
        if colorbar and kwargs.get("alpha", 1) > 0:
            clb = self.fig.colorbar(mesh, ax=self.ax, label=colorbar_label)
            clb.solids.set_edgecolor("face")

        # Display image
        if show:
            plt.tight_layout()
            plt.show()

    def label_ax(self, view, no_ylabel=False, no_title=False, 
                 annotate_slice=None):
        """Assign x/y axis labels and title to the plot."""

        units = " (mm)" if self.scale_in_mm else ""
        self.ax.set_xlabel(_plot_axes[view][0] + units)
        if not no_ylabel:
            self.ax.set_ylabel(_plot_axes[view][1] + units)
        else:
            self.ax.set_yticks([])
        if self.title is not None and not no_title:
            self.ax.set_title(self.title)

        # Slice annotation
        if annotate_slice is not None:

            # Make annotation string
            ax = _slider_axes[view]
            if self.scale_in_mm:
                z_str = "{} = {:.1f} mm".format(
                    ax, self.idx_to_pos(self.idx, ax))
            else:
                z_str = f"{ax} = {self.idx_to_slice(self.idx, ax)}"

            # Add annotation
            if matplotlib.colors.is_color_like(annotate_slice):
                col = annotate_slice
            else:
                col = "white"
            self.ax.annotate(z_str, xy=(0.05, 0.93), xycoords='axes fraction',
                             color=col)

    def get_ax_dict(self, val, default=1):
        """Convert a single value or tuple of values in order (x, y, z) to a 
        dictionary containing x/y/z as keys."""

        if val is None:
            return None
        try:
            val = float(val)
            return {ax: val for ax in _axes}
        except TypeError:
            ax_dict = {ax: val[n] for ax, n in _axes.items()}
            for ax in ax_dict:
                if ax_dict[ax] is None:
                    ax_dict[ax] = default
            return ax_dict

    def apply_zoom(self, view, zoom_x=True, zoom_y=True):
        """Zoom in on axes by the factors in self.zoom after axes have 
        been created. If <zoom_x> or <zoom_y> are False, the x or y axes, 
        respectively, will not be zoomed in on."""

        if not hasattr(self, "ax"):
            raise RuntimeError("Trying to zoom before axes have been set!")
        if self.zoom is None:
            return

        # Adjust the axes
        orig = {"x": self.ax.get_xlim(), "y": self.ax.get_ylim()}
        axes = {"x": _plot_axes[view][0], "y": _plot_axes[view][1]}
        new = {}
        for ax in ["x", "y"]:
            mid = np.mean(orig[ax])
            new_min = mid - (mid - orig[ax][0]) / self.zoom[axes[ax]]
            new_max = mid + (orig[ax][1] - mid) / self.zoom[axes[ax]]
            new[ax] = new_min, new_max
        if zoom_x:
            self.ax.set_xlim(new["x"])
        if zoom_y:
            self.ax.set_ylim(new["y"])

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

        return data_array[::round(self.downsample["x"]),
                          ::round(self.downsample["y"]),
                          ::round(self.downsample["z"])]


class DeformationImage(NiftiImage):
    """Class for loading a plotting a deformation field."""

    def __init__(self, nii, spacing=_default_spacing, plot_type="grid", 
                 **kwargs):
        """Load deformation field.

        Parameters
        ----------
        nii : str/array/nifti
            Source of the image data to load. This can be either:
                (a) The path to a NIfTI file;
                (b) A nibabel.nifti1.Nifti1Image object;
                (c) The path to a file containing a NumPy array;
                (d) A NumPy array.
        """

        NiftiImage.__init__(self, nii, **kwargs)
        if not self.valid:
            return
        if self.data.ndim != 5:
            raise RuntimeError(f"Deformation field in {nii} must contain a "
                               "five-dimensional array!")
        self.data = self.data[:, :, :, 0, :]
        self.set_spacing(spacing)

    def set_spacing(self, spacing):
        """Assign grid spacing in each direction. If spacing in given in mm,
        convert it to number of voxels."""

        if spacing is None:
            return

        spacing = self.get_ax_dict(spacing, _default_spacing)
        if self.scale_in_mm:
            self.spacing = {ax: abs(round(sp / self.voxel_sizes[ax]))
                            for ax, sp in spacing.items()}
        else:
            self.spacing = spacing

        # Ensure spacing is at least 2 voxels
        for ax, sp in self.spacing.items():
            if sp < 2:
                self.spacing[ax] = 2

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments for grid 
        plots and quiver plots."""

        self.quiver_kwargs = {"cmap": "jet"}
        self.grid_kwargs = {"color": "green"}

    def set_slice(self, view, sl=None, pos=None):
        """Set 2D array corresponding to a slice of the deformation field in 
        a given orientation."""

        idx = self.get_idx(view, sl, pos, default_centre=False)
        im_slice = np.transpose(self.data, _orient[view] + [3])[:, :, idx, :]
        x, y = _plot_axes[view]
        if y != "x":
            im_slice = im_slice[::-1, :, :]
        self.current_slice = im_slice

    def get_deformation_slice(self, view, sl=None, pos=None):
        """Get voxel positions and displacement vectors on a 2D slice."""

        self.set_slice(view, sl, pos)
        x_ax, y_ax = _plot_axes[view]

        # Get x/y displacement vectors
        df_x = np.squeeze(self.current_slice[:, :, _axes[x_ax]])
        df_y = np.squeeze(self.current_slice[:, :, _axes[y_ax]])
        if not self.scale_in_mm:
            df_x /= self.voxel_sizes[x_ax]
            df_y /= self.voxel_sizes[y_ax]

        # Get x/y coordinates of each point on the slice
        xs = np.arange(0, self.current_slice.shape[1])
        ys = np.arange(0, self.current_slice.shape[0])
        if self.scale_in_mm:
            xs = self.origin[x_ax] + xs * self.voxel_sizes[x_ax]
            ys = self.origin[y_ax] + ys * self.voxel_sizes[y_ax]
        y, x = np.meshgrid(ys, xs)
        x = x.T
        y = y.T
        return x, y, df_x, df_y

    def plot(
        self, 
        view, 
        sl=None, 
        pos=None, 
        ax=None, 
        mpl_kwargs=None, 
        plot_type="grid", 
        spacing=30
    ):
        """Plot deformation field.

        Parameters
        ----------
        view : str
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

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        show : bool, default=True
            If True, the plotted figure will be shown via 
            matplotlib.pyplot.show().

        plot_type : str, default="grid"
            Type of plot to produce. Can either be "grid" to produce a grid 
            plot, "quiver" to produce a quiver (arrow) plot.

        spacing : int/float/tuple, default=30
            Spacing between gridpoints when the deformation field is plotted.
            If scale_in_mm=True, spacing will be in mm; otherwise, it will be
            in number of voxels. If a single value is given, this value will
            be used for the spacing in all directions. A tuple of three
            separate spacing values in order (x, y, z) can also be given.
        """

        if not self.valid:
            return

        self.set_spacing(spacing)
        if plot_type == "grid":
            self.plot_grid(view, sl, pos, ax, mpl_kwargs)
        elif plot_type == "quiver":
            self.plot_quiver(view, sl, pos, ax, mpl_kwargs)

    def plot_quiver(
        self, 
        view, 
        sl, 
        pos, 
        ax, 
        mpl_kwargs=None
    ):
        """Draw a quiver plot on a set of axes."""

        # Get arrow positions and lengths
        self.set_ax(view, ax)
        x_ax, y_ax = _plot_axes[view]
        x, y, df_x, df_y = self.get_deformation_slice(view, sl, pos)
        arrows_x = df_x[::self.spacing[y_ax], ::self.spacing[x_ax]]
        arrows_y = -df_y[::self.spacing[y_ax], ::self.spacing[x_ax]]
        plot_x = x[::self.spacing[y_ax], ::self.spacing[x_ax]]
        plot_y = y[::self.spacing[y_ax], ::self.spacing[x_ax]]

        # Plot arrows
        if arrows_x.any() or arrows_y.any():
            M = np.hypot(arrows_x, arrows_y)
            ax.quiver(plot_x, plot_y, arrows_x, arrows_y, M,
                             **self.get_kwargs(mpl_kwargs, self.quiver_kwargs))
        else:
            # If arrow lengths are zero, plot dots
            ax.scatter(plot_x, plot_y, c="navy", marker=".")
        self.apply_zoom(view)

    def plot_grid(
        self, 
        view, 
        sl, 
        pos, 
        ax, 
        mpl_kwargs=None
    ):
        """Draw a grid plot on a set of axes."""

        # Get gridline positions
        self.set_ax(view, ax)
        self.ax.autoscale(False)
        x_ax, y_ax = _plot_axes[view]
        x, y, df_x, df_y = self.get_deformation_slice(view, sl, pos)
        grid_x = x + df_x
        grid_y = y + df_y

        # Plot gridlines
        kwargs = self.get_kwargs(mpl_kwargs, default=self.grid_kwargs)
        for i in np.arange(0, x.shape[0], self.spacing[y_ax]):
            self.ax.plot(grid_x[i, :], grid_y[i, :], **kwargs)
        for j in np.arange(0, x.shape[1], self.spacing[x_ax]):
            self.ax.plot(grid_x[:, j], grid_y[:, j], **kwargs)
        self.apply_zoom(view)


class StructImage(NiftiImage):
    """Class to load and plot a structure mask."""

    def __init__(self, nii, name=None, color=None, **kwargs):
        """Load structure mask.

        Parameters
        ----------
        nii : str/array/nifti
            Source of the image data to load. This can be either:
                (a) The path to a NIfTI file;
                (b) A nibabel.nifti1.Nifti1Image object;
                (c) The path to a file containing a NumPy array;
                (d) A NumPy array.

        name : str, default=None
            Name to assign to this structure. If the structure is loaded 
            from a file and name is None, the name will be inferred from
            the filename.

        color : matplotlib color, default=None
            Colour in which to plot this structure. If None, a random 
            colour will be assigned. Can also be set later using 
            self.assign_color(color).
        """

        # Load the mask
        NiftiImage.__init__(self, nii)
        if not self.valid:
            return

        # Convert to boolean mask
        self.data = self.data > 0.5

        # Set name
        if name is not None:
            self.name = name
        else:
            basename = os.path.basename(nii).strip(".gz").strip(".nii")
            self.name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "",
                               basename).replace(" ", "_")
        nice = self.name.replace("_", " ")
        self.name_nice = nice[0].upper() + nice[1:]

        # Get contours
        self.visible = True
        self.set_contours()

        # Check whether contours were found
        self.empty = not sum([len(contours) for contours in 
                              self.contours.values()])
        if self.empty:
            self.name_nice += " (empty)"

        # Assign a random colour
        self.assign_color(np.random.rand(3, 1).flatten())

    def set_geom_properties(self, volume=True, length=True):
        """Find structure volume and length in each direction."""

        # Empty structure: return zeros
        if self.empty:
            self.volume = {"voxels": 0, "mm": 0, "ml": 0}
            zeros = {ax: 0 for ax in _axes}
            self.lengths = {"voxels": zeros, "mm": zeros}
            return

        # Volume
        self.volume = {
            "voxels": self.data.astype(bool).sum()
        }
        self.volume["mm"] = self.volume["voxels"] \
            * abs(np.prod(list(self.voxel_sizes.values())))
        self.volume["ml"] = self.volume["mm"] * (0.1 ** 3)

        # Lengths
        self.length = {"voxels": {}, "mm": {}}
        nonzero = np.argwhere(self.data)
        for ax, n in _axes.items():
            vals = nonzero[:, n]
            if len(vals):
                self.length["voxels"][ax] = max(vals) - min(vals)
                self.length["mm"][ax] = self.length["voxels"][ax] \
                    * abs(self.voxel_sizes[ax])
            else:
                self.length["voxels"][ax] = 0
                self.length["mm"][ax] = 0

    def set_plotting_defaults(self):
        """Set default matplotlib plotting keywords for both mask and
        contour images."""

        self.mask_kwargs = {"alpha": 1,
                            "interpolation": "none"}
        self.contour_kwargs = {"linewidth": 2}

    def set_contours(self):
        """Compute positions of contours on each slice in each orientation.
        """

        self.contours = {}
        for view, z in _slider_axes.items():
            self.contours[view] = {}
            for sl in range(1, self.n_voxels[z] + 1):
                contour = self.get_contour_slice(view, sl)
                if contour is not None:
                    self.contours[view][sl] = contour

    def get_contour_slice(self, view, sl):
        """Convert mask to contours on a given slice <sl> in a given
        orientation <view>."""

        # Ignore slices with no structure mask
        self.set_slice(view, sl)
        if self.current_slice.max() < 0.5:
            return

        # Find contours
        x_ax, y_ax = _plot_axes[view]
        contours = skimage.measure.find_contours(self.current_slice, 0.5, 
                                                 "low", "low")
        if contours:
            points = []
            for contour in contours:
                contour_points = []
                for (y, x) in contour:
                    if self.scale_in_mm:
                        x = min(self.lims[x_ax]) + \
                            x * abs(self.voxel_sizes[x_ax])
                        y = min(self.lims[y_ax]) + \
                            y * abs(self.voxel_sizes[y_ax])
                    contour_points.append((x, y))
                points.append(contour_points)
            return points

    def assign_color(self, color):
        """Assign a colour, ensuring that it is compatible with matplotlib."""

        if matplotlib.colors.is_color_like(color):
            self.color = matplotlib.colors.to_rgba(color)
        else:
            print(f"Colour {color} is not a valid colour.")

    def plot(
        self, 
        view, 
        sl=None, 
        pos=None, 
        ax=None, 
        mpl_kwargs=None, 
        plot_type="contour"
    ):
        """Plot structure.

        Parameters
        ----------
        view : str
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

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        plot_type : str, default="contour"
            Type of plot to produce. Can be "contour" for a contour plot,
            "mask" for a mask plot.
        """

        if not self.valid:
            return

        # Make plot
        if plot_type == "contour":
            self.plot_contour(view, sl, pos, ax, mpl_kwargs)
        elif plot_type == "mask":
            self.plot_mask(view, sl, pos, ax, mpl_kwargs)

    def plot_mask(self, view, sl, pos, ax, mpl_kwargs=None):
        """Plot structure as a coloured mask."""

        # Get slice
        self.set_ax(view, ax)
        self.set_slice(view, sl, pos)

        # Make colormap
        norm = matplotlib.colors.Normalize()
        cmap = matplotlib.cm.hsv
        s_colors = cmap(norm(self.current_slice))
        s_colors[self.current_slice > 0, :] = self.color
        s_colors[self.current_slice == 0, :] = (0, 0, 0, 0)

        # Display the mask
        self.ax.imshow(
            s_colors,
            extent=self.extent[view],
            aspect=self.aspect[view],
            **self.get_kwargs(mpl_kwargs, default=self.mask_kwargs)
        )
        self.apply_zoom(view)

    def plot_contour(self, view, sl, pos, ax, mpl_kwargs=None):
        """Plot structure as a contour."""

        if not self.on_slice(view, sl):
            return

        self.set_ax(view, ax)
        kwargs = self.get_kwargs(mpl_kwargs, default=self.contour_kwargs)
        kwargs.setdefault("color", self.color)
        idx = self.get_idx(view, sl, pos, default_centre=False)

        for points in self.contours[view][sl]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            self.ax.plot(points_x, points_y, **kwargs)
        self.apply_zoom(view)

    def on_slice(self, view, sl):
        """Return True if a contour exists for this structure on a given slice.
        """

        return sl in self.contours[view]

    def get_area(self, view, sl, units="voxels"):
        """Get the area on a given slice."""

        if not self.on_slice(view, sl):
            return

        self.set_slice(view, sl)
        non_zero = np.argwhere(self.current_slice)
        area = len(non_zero)
        if units == "mm":
            x, y = _plot_axes[view]
            area *= abs(self.voxel_sizes[x] * self.voxel_sizes[y])
        return area

    def get_extents(self, view, sl, units="voxels"):
        """Get extents along the x/y axes in a given view on a given slice."""

        if not self.on_slice(view, sl):
            return None, None

        self.set_slice(view, sl)
        non_zero = np.argwhere(self.current_slice)
        x, y = _plot_axes[view]
        if len(non_zero):
            mins = non_zero.min(0)
            maxes = non_zero.max(0)
            x_len = abs(maxes[1] - mins[1])
            y_len = abs(maxes[0] - mins[0])
            if units == "mm":
                x_len *= abs(self.voxel_sizes[x])
                y_len *= abs(self.voxel_sizes[x])
            return x_len, y_len
        else:
            return 0, 0


class MultiImage(NiftiImage):
    """Class for loading and plotting an image along with an optional mask,
    dose field, structures, jacobian determinant, and deformation field."""

    def __init__(
        self,
        nii,
        dose=None,
        mask=None,
        jacobian=None,
        df=None,
        structs=None,
        struct_colours=None,
        structs_as_mask=False,
        mask_threshold=0.5,
        **kwargs
    ):
        """Load a MultiImage object.

        Parameters
        ----------
        nii : str/nifti/array
            Path to a .nii/.npy file, or an nibabel nifti object/numpy array.

        title : str, default=None
            Title for this image when plotted. If None and <nii> is loaded from
            a file, the filename will be used.

        dose : str/nifti/array, default=None
            Path or object from which to load dose field.

        mask : str/nifti/array, default=None
            Path or object from which to load mask array.

        jacobian : str/nifti/array, default=None
            Path or object from which to load jacobian determinant field.

        df : str/nifti/array, default=None
            Path or object from which to load deformation field.

        structs : str/list, default=None
            A string containing a path, directory, or wildcard pointing to
            nifti file(s) containing structure(s). Can also be a list of
            paths/directories/wildcards.

        struct_colours : dict, default=None
            Custom colours to use for structures. Dictionary keys can be a
            structure name or a wildcard matching structure name(s). Values
            should be any valid matplotlib colour.

        structs_as_mask : bool, default=False
            If True, structures will be used as masks.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).
        """

        # Load the scan image
        NiftiImage.__init__(self, nii, **kwargs)
        if not self.valid:
            return

        # Load extra overlays
        self.load_to(dose, "dose", kwargs)
        self.load_to(mask, "mask", kwargs)
        self.load_to(jacobian, "jacobian", kwargs)
        self.load_df(df)
        self.load_structs(structs, struct_colours)
        self.structs_as_mask = structs_as_mask
        if self.has_structs and structs_as_mask:
            self.has_mask = True
        self.set_masks(mask_threshold)

    def load_to(self, nii, attr, kwargs):
        """Load image data into a class attribute."""

        # Load single image
        if not isinstance(nii, dict):
            data = NiftiImage(nii, **kwargs)
            setattr(self, attr, data)
            valid = data.valid
        else:
            data = {view: NiftiImage(nii[view], **kwargs) for view in nii}
            for view in _orient:
                if view not in data or not data[view].valid:
                    data[view] = None
            valid = any([d.valid for d in data.values() if d is not None])

        setattr(self, attr, data)
        setattr(self, f"has_{attr}", valid)
        setattr(self, f"{attr}_dict", isinstance(nii, dict))

    def load_df(self, df):
        """Load deformation field data from a path."""

        self.df = DeformationImage(df, scale_in_mm=self.scale_in_mm)
        self.has_df = self.df.valid

    def load_structs(self, structs, struct_colours):
        """Load structures from a path/wildcard or list of paths/wildcards in
        <structs>, and assign the colours in <struct_colours>."""

        self.has_structs = False
        self.structs = []
        if structs is None:
            return

        # Find valid filepaths
        struct_paths = [structs] if isinstance(structs, str) else structs
        files = []
        for path in struct_paths:

            # Find files
            path = os.path.expanduser(path)
            if os.path.isfile(path):
                files.append(path)
            elif os.path.isdir(path):
                files.extend(glob.glob(path + "/*.nii*"))
            else:
                matches = glob.glob(path)
                for m in matches:
                    if os.path.isdir(m):
                        files.extend(glob.glob(m + "/*.nii*"))
                    elif os.path.isfile(m):
                        files.append(m)

        # Load structures from files
        if not len(files):
            print("Warning: no structure files found matching ", structs)
            return
        self.has_structs = True
        files = list(set([os.path.abspath(f) for f in files]))
        self.structs = [StructImage(f) for f in files]

        # Assign colours
        standard_colors = (
            list(matplotlib.cm.Set1.colors)[:-1]
            + list(matplotlib.cm.Set2.colors)[:-1]
            + list(matplotlib.cm.Set3.colors)
            + list(matplotlib.cm.tab20.colors)
        )
        custom = struct_colours if struct_colours is not None else {}
        custom_lower = {standard_str(n): col for n, col in custom.items()}
        for i, struct in enumerate(self.structs):

            # Assign standard colour
            struct.assign_color(standard_colors[i])

            # Check for exact match
            matched = False
            if struct.name.lower() in custom_lower:
                struct.assign_color(custom_lower[struct.name.lower()])
                matched = True

            # If no exact match, check for first matching wildcard
            if not matched:
                for name in custom_lower:
                    if fnmatch.fnmatch(struct.name.lower(), name):
                        struct.assign_color(custom_lower[name])
                        matched = True

            # Otherwise, check for matching filepath
            if not matched:
                for path in custom:
                    if fnmatch.fnmatch(struct.path, os.path.abspath(path)):
                        struct.assign_color(custom[path])
                        break

    def set_plotting_defaults(self):
        """Set default matplotlib plotting options for main image, dose field,
        and jacobian determinant."""

        NiftiImage.set_plotting_defaults(self)
        self.dose_kwargs = {
            "cmap": "jet",
            "alpha": 0.5,
            "vmin": None,
            "vmax": None
        }
        self.jacobian_kwargs = {
            "cmap": "seismic",
            "alpha": 0.5,
            "vmin": 0.8,
            "vmax": 1.2
        }

    def set_masks(self, threshold):
        """Assign mask(s) to self and dose image."""

        if not self.has_mask:
           mask_array = None 

        else:
            # Combine user-input mask with structs
            mask_array = np.zeros(self.shape, dtype=bool)
            if not self.mask_dict and self.mask.valid:
                mask_array += self.mask.data > threshold
            if self.structs_as_mask:
                for struct in self.structs:
                    mask_array += struct.data

            # Get separate masks for each orientation
            if self.mask_dict:
                view_masks = {}
                for view in _orient:
                    mask = self.mask.get(view, None)
                    if mask is not None:
                        if isinstance(mask, NiftiImage):
                            view_masks[view] = mask_array \
                                    + (self.mask[view].data > threshold)
                        else:
                            view_masks[view] = mask_array \
                                    + (self.mask[view] > threshold)
                    else:
                        if self.structs_as_mask:
                            view_masks[view] = self.mask_array
                        else:
                            view_masks[view] = None

                mask_array = view_masks

        # Assign mask to main image and dose field
        self.set_mask(mask_array, threshold)
        self.dose.data_mask = self.data_mask

    def get_n_colorbars(self, colorbar=False):
        """Count the number of colorbars needed for this plot."""

        return colorbar * (1 + self.has_dose + self.has_jacobian)

    def get_relative_width(self, view, colorbar=False):
        """Get the relative width for this plot, including all colorbars."""

        return NiftiImage.get_relative_width(
            self, view, self.get_n_colorbars(colorbar))

    def plot(
        self,
        view="x-y",
        sl=None,
        pos=None,
        ax=None,
        gs=None,
        figsize=None,
        mpl_kwargs=None,
        show=True,
        colorbar=False,
        dose_kwargs=None,
        masked=False,
        invert_mask=False,
        mask_colour="black",
        jacobian_kwargs=None,
        df_kwargs=None,
        df_plot_type="grid",
        df_spacing=30,
        struct_kwargs=None,
        struct_plot_type="contour",
        struct_legend=True,
        legend_loc='lower left',
        annotate_slice=None
    ):
        """Plot a 2D slice of this image and all extra features.

        Parameters
        ----------
        view : str
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

        figsize : float, default=None
            Figure height in inches; only used if <ax> and <gs> are None. If 
            None, the value in _default_figsize will be used.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow() for
            the main image.

        show : bool, default=True
            If True, the plotted figure will be shown via 
            matplotlib.pyplot.show().

        colorbar : bool, default=True
            If True, a colorbar will be drawn alongside the plot.

        dose_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow() for
            the dose field.

        masked : bool, default=False
            If True and this object has attribute self.data_mask assigned,
            the image will be masked with the array in self.data_mask.

        invert_mask : bool, default=True
            If True and a mask is applied, the mask will be inverted.

        mask_colour : matplotlib colour, default="black"
            Colour in which to plot masked areas.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).

        jacobian_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow() for
            the jacobian determinant.

        df_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow() for
            the deformation field.

        df_plot_type : str, default="grid"
            Type of plot ("grid"/"quiver") to produce for the deformation 
            field.

        df_spacing : int/float/tuple, default=30
            Grid spacing for the deformation field plot. If self.scale_in_mm is
            true, the spacing will be in mm; otherwise in voxels. Can be either
            a single value for all directions, or a tuple of values for
            each direction in order (x, y, z).

        struct_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib for structure
            plotting.

        struct_plot_type : str, default="contour"
            Plot type for structures ("contour"/"mask")

        struct_legend : bool, default=True
            If True, a legend will be drawn labelling any structrues visible on
            this slice.

        legend_loc : str, default='lower left'
            Position for the structure legend, if used.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will 
            be added. If True, the default colour (white) will be used.
        """

        # Plot image
        self.set_ax(view, ax, gs, figsize, colorbar)
        NiftiImage.plot(
            self, view, sl, pos, ax=self.ax, mpl_kwargs=mpl_kwargs,
            show=False, colorbar=colorbar, masked=masked,
            invert_mask=invert_mask, mask_colour=mask_colour, figsize=figsize)

        # Plot dose field
        self.dose.plot(
            view, self.sl, ax=self.ax,
            mpl_kwargs=self.get_kwargs(dose_kwargs, default=self.dose_kwargs),
            show=False, masked=masked, invert_mask=invert_mask,
            mask_colour=mask_colour, colorbar=colorbar, 
            colorbar_label="Dose (Gy)")

        # Plot jacobian
        self.jacobian.plot(
            view, self.sl, ax=self.ax, mpl_kwargs=self.get_kwargs(
                jacobian_kwargs, default=self.jacobian_kwargs),
            show=False, colorbar=colorbar,
            colorbar_label="Jacobian determinant")

        # Plot structures
        struct_handles = []
        for struct in self.structs:
            if not struct.visible:
                continue
            struct.plot(view, self.sl, ax=self.ax, mpl_kwargs=struct_kwargs, 
                        plot_type=struct_plot_type)
            if struct.on_slice(view, self.sl) and struct_plot_type != "none":
                struct_handles.append(mpatches.Patch(color=struct.color,
                                                     label=struct.name_nice))

        # Plot deformation field
        self.df.plot(view, self.sl, ax=self.ax,
                     mpl_kwargs=df_kwargs,
                     plot_type=df_plot_type,
                     spacing=df_spacing)

        # Draw legend
        if struct_legend and len(struct_handles):
            self.ax.legend(handles=struct_handles, loc=legend_loc,
                           facecolor="white", framealpha=1)

        self.label_ax(view, annotate_slice=annotate_slice)

        # Display image
        if show:
            plt.tight_layout()
            plt.show()


class OrthogonalImage(MultiImage):
    """MultiImage to be displayed with an orthogonal view of the main image
    next to it."""

    def __init__(self, *args, **kwargs):
        """Initialise a MultiImage and set default orthogonal slice 
        positions."""

        MultiImage.__init__(self, *args, **kwargs)
        self.orthog_slices = {ax: int(self.n_voxels[ax] / 2)
                              for ax in _axes}

    def get_relative_width(self, view, colorbar=False):
        """Get width:height ratio for the full plot (main plot + orthogonal
        view)."""

        width_own = MultiImage.get_relative_width(self, view, colorbar)
        width_orthog = MultiImage.get_relative_width(self, _orthog[view])
        return width_own + width_orthog

    def set_axes(self, view, ax=None, gs=None, figsize=None, 
                 colorbar=False):
        """Set up axes for the plot. If <ax> is not None and <orthog_ax> has
        already been set, these axes will be used. Otherwise if <gs> is not 
        None, the axes will be created within a gridspec on the current 
        matplotlib figure.  Otherwise, a new figure with height <figsize> 
        will be produced."""

        if ax is not None and hasattr(self, "orthog_ax"):
            self.ax = ax

        width_ratios = [
            MultiImage.get_relative_width(self, view, colorbar),
            MultiImage.get_relative_width(self, _orthog[view])
        ]
        if gs is None:
            figsize = _default_figsize if figsize is None else figsize
            fig = plt.figure(figsize=(figsize * sum(width_ratios), figsize))
            self.gs = fig.add_gridspec(1, 2, width_ratios=width_ratios)
        else:
            fig = plt.gcf()
            self.gs = gs.subgridspec(1, 2, width_ratios=width_ratios)

        self.ax = fig.add_subplot(self.gs[0])
        self.orthog_ax = fig.add_subplot(self.gs[1])

    def plot(self,
             view,
             sl=None,
             pos=None,
             ax=None,
             gs=None,
             figsize=None,
             mpl_kwargs=None,
             show=True,
             colorbar=False,
             struct_kwargs=None,
             struct_plot_type=None,
             **kwargs
            ):
        """Plot MultiImage and orthogonal view of main image and structs."""

        self.set_axes(view, ax, gs, figsize, colorbar)

        # Plot the MultiImage
        MultiImage.plot(self, view, sl=sl, pos=pos, ax=self.ax, 
                        colorbar=colorbar, show=False, mpl_kwargs=mpl_kwargs,
                        struct_kwargs=struct_kwargs,
                        struct_plot_type=struct_plot_type,
                        **kwargs)

        # Plot orthogonal view
        orthog_view = _orthog[view]
        orthog_sl = self.orthog_slices[_slider_axes[orthog_view]]
        NiftiImage.plot(self,
                        orthog_view,
                        sl=orthog_sl,
                        ax=self.orthog_ax,
                        mpl_kwargs=mpl_kwargs,
                        show=False,
                        colorbar=False,
                        no_ylabel=True,
                        no_title=True)

        # Plot structures on orthogonal image
        for struct in self.structs:
            if not struct.visible:
                continue
            struct.plot(orthog_view, sl=orthog_sl, ax=self.orthog_ax, 
                        mpl_kwargs=struct_kwargs, plot_type=struct_plot_type)

        # Plot indicator line
        pos = sl if not self.scale_in_mm else self.slice_to_pos(
            sl, _slider_axes[view])
        if view == "x-y":
            full_y = self.extent[orthog_view][2:] if self.scale_in_mm else \
                [0, self.n_voxels[_plot_axes[orthog_view][1]] - 1]
            self.orthog_ax.plot([pos, pos], full_y, 'r')
        else:
            full_x = self.extent[orthog_view][:2] if self.scale_in_mm else \
                [0, self.n_voxels[_plot_axes[orthog_view][0]] - 1]
            self.orthog_ax.plot(full_x, [pos, pos], 'r')

        if show:
            plt.tight_layout()
            plt.show()


class ComparisonImage(NiftiImage):
    """Class for loading data from two arrays and plotting comparison images.
    The implementation of these comparison plots is handled within inherited
    classes."""

    def __init__(self, nii1, nii2, title=None, **kwargs):
        """Load data from two arrays. <nii1> and <nii2> can either be existing
        NiftiImage objects, or objects from which NiftiImages can be created.
        """

        # Load NiftiImages
        self.ims = []
        self.standalone = True
        for nii in [nii1, nii2]:

            # Load existing NiftiImage
            if issubclass(type(nii), NiftiImage):
                self.ims.append(nii)

            # Create new NiftiImage
            else:
                self.standalone = False
                self.ims.append(NiftiImage(nii, **kwargs))

        self.scale_in_mm = self.ims[0].scale_in_mm
        self.zoom = self.ims[0].zoom
        self.valid = all([im.valid for im in self.ims])
        self.title = title
        self.gs = None

    def get_relative_width(self, view, n_colorbars=0):
        """Get relative width of widest of the two images."""
        
        x, y = _plot_axes[view]
        height = max([im.get_length(y) for im in self.ims])
        width = max([im.get_length(x) for im in self.ims])
        return width / height

    def plot(self, view=None, sl=None, invert=False, ax=None,
             mpl_kwargs=None, show=True, figsize=None, **kwargs):
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

        # Get image slices
        if view is None and sl is None:
            for im in self.ims:
                if not hasattr(im, "current_slice"):
                    raise RuntimeError("Must provide a view and slice number "
                                       "if input images do not have a current "
                                       "slice set!")
            self.view = self.ims[0].view

        else:
            self.view = view
            for im in self.ims:
                im.set_slice(view, sl)

        self.slices = [im.current_slice for im in self.ims]

        # Plot settings
        self.set_ax(view, ax, self.gs, figsize)
        self.plot_kwargs = self.ims[0].get_kwargs(mpl_kwargs)
        self.cmap = copy.copy(matplotlib.cm.get_cmap(
            self.plot_kwargs.pop("cmap")))
        
        # Produce the plot
        self.plot_comparison(invert=invert, **kwargs)
        self.label_ax(self.view)
        self.apply_zoom(self.view)


class ChequerboardImage(ComparisonImage):
    """Class for plotting a chequerboard comparison of two NiftiImages."""

    def plot_comparison(self, invert=False, n_splits=2):
        """Produce a chequerboard plot with <n_splits> squares in each
        direction."""

        # Adjust number of splits for zooming
        if self.zoom is not None and n_splits > 1:
            x, y = _plot_axes[self.view]
            n_splits *= min([z for ax, z in self.zoom.items() if ax in [x, y]])
            n_splits = int(round(n_splits))

        # Get masked image
        i1 = int(invert)
        i2 = 1 - i1
        size_x = int(np.ceil(self.slices[i2].shape[0] / n_splits))
        size_y = int(np.ceil(self.slices[i2].shape[1] / n_splits))
        cb_mask = np.kron([[1, 0] * n_splits, [0, 1] * n_splits] * n_splits, 
                          np.ones((size_x, size_y)))
        cb_mask = cb_mask[:self.slices[i2].shape[0], :self.slices[i2].shape[1]]
        to_show = {
            i1: self.slices[i1],
            i2: np.ma.masked_where(cb_mask < 0.5, self.slices[i2])
        }

        # Plot
        for i in [i1, i2]:
            self.ax.imshow(to_show[i],
                           extent=self.ims[i].extent[self.view],
                           aspect=self.ims[i].aspect[self.view],
                           cmap=self.cmap, **self.plot_kwargs)


class OverlayImage(ComparisonImage):
    """Class for plotting two NiftiImages overlaid in red and blue."""

    def plot_comparison(self, invert=False, opacity=0.5, legend=False,
                        legend_loc='auto'):
        """Produce an overlay plot with a given opacity."""

        order = [0, 1] if not invert else [1, 0]
        cmaps = ["Reds", "Blues"]
        alphas = [1, opacity]
        self.ax.set_facecolor("w")
        handles = []
        for n, i in enumerate(order):

            # Show image
            self.ax.imshow(self.slices[i],
                           extent=self.ims[i].extent[self.view],
                           aspect=self.ims[i].aspect[self.view],
                           cmap=cmaps[n],
                           alpha=alphas[n],
                           **self.plot_kwargs)

            # Make handle for legend
            if legend:
                patch_color = cmaps[n].lower()[:-1]
                alpha = 1 - opacity if alphas[n] == 1 else opacity
                handles.append(mpatches.Patch(
                    color=patch_color, alpha=alpha,
                    label=self.ims[i].title))
            
        # Draw legend
        if legend:
            self.ax.legend(handles=handles, loc=legend_loc, facecolor="white", 
                           framealpha=1)


class DiffImage(ComparisonImage):
    """Class for plotting the difference between two NiftiImages."""

    def plot_comparison(self, invert=False):
        """Produce a difference plot."""

        diff = self.slices[1] - self.slices[0] if not invert \
                else self.slices[0] - self.slices[1]
        self.ax.imshow(diff,
                       extent=self.ims[0].extent[self.view],
                       aspect=self.ims[0].aspect[self.view],
                       cmap=self.cmap, 
                       **self.plot_kwargs)


def standard_str(string):
    """Convert a string to lowercase and replace all spaces with 
    underscores."""

    try:
        return string.lower().replace(" ", "_")
    except AttributeError:
        return
