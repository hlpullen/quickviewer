"""Classes for plotting images from NIfTI files or NumPy arrays."""

import dateutil.parser
import copy
import fnmatch
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
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, \
        FormatStrFormatter
from shapely import geometry

from quickviewer.viewer import core, dicom


# Shared parameters
_axes = {"x": 0, "y": 1, "z": 2}
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [1, 0, 2]}
_n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}
_df_plot_types = ["grid", "quiver", "none"]
_struct_plot_types = ["contour", "mask", "filled", "centroid", 
                      "filled centroid", "none"]
_default_figsize = 6
_default_spacing = 30


# Standard list of colours for structures
standard_colors = (
    list(matplotlib.cm.Set1.colors)[:-1]
    + list(matplotlib.cm.Set2.colors)[:-1]
    + list(matplotlib.cm.Set3.colors)
    + list(matplotlib.cm.tab20.colors)
)
to_del = [9, 10]
for d in to_del:
    del standard_colors[d]


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
        orientation="x-y",
        rescale=True
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
        self.data, voxel_sizes, origin, self.path = core.load_image(
            nii, affine, voxel_sizes, origin, rescale)
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
            voxel_sizes, origin = self.convert_to_3d(voxel_sizes, origin, 
                                                     orientation)

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
            "z-y": [2, 1, 0]
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

        # Min and max voxel position
        self.lims = {
            ax: (self.origin[ax], self.origin[ax] + (self.n_voxels[ax] - 1) *
                 self.voxel_sizes[ax])
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
                    [min(self.lims[x]) - abs(vx / 2), 
                     max(self.lims[x]) + abs(vx / 2)],
                    [max(self.lims[y]) + abs(vy / 2),
                     min(self.lims[y]) - abs(vy / 2)]
                ]
                self.extent[view] = self.ax_lims[view][0] \
                        + self.ax_lims[view][1]
                self.aspect[view] = 1
            else:
                x_lim = [self.idx_to_slice(0, x), 
                         self.idx_to_slice(self.n_voxels[x] - 1, x)]
                x_lim[x_lim.index(max(x_lim))] += 0.5
                x_lim[x_lim.index(min(x_lim))] -= 0.5
                self.ax_lims[view] = [
                    x_lim, [self.n_voxels[y] + 0.5, 0.5]
                ]
                self.extent[view] = self.ax_lims[view][0] \
                        + self.ax_lims[view][1]
                self.aspect[view] = abs(self.voxel_sizes[y] /
                                        self.voxel_sizes[x])

    def get_lengths(self, view):
        """Get the x and y lengths of the image in a given orientation."""

        x_length = abs(self.ax_lims[view][0][1] - self.ax_lims[view][0][0])
        y_length = abs(self.ax_lims[view][1][1] - self.ax_lims[view][1][0])
        if self.scale_in_mm:
            return x_length, y_length
        else:
            x, y = _plot_axes[view]
            return (x_length * abs(self.voxel_sizes[x]),
                    y_length * abs(self.voxel_sizes[y]))

    def get_image_centre(self, view):
        """Get midpoint of a given orientation."""

        mid_x = np.mean(self.ax_lims[view][0])
        mid_y = np.mean(self.ax_lims[view][1])
        return [mid_x, mid_y]
            
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
        origin1 = [f"{x:.2f}" for x in self.origin.values()]
        origin2 = [f"{x:.2f}" for x in im.origin.values()]
        same *= origin1 == origin2
        vx1 = [f"{x:.2f}" for x in self.voxel_sizes.values()]
        vx2 = [f"{x:.2f}" for x in im.voxel_sizes.values()]
        same *= vx1 == vx2
        return same

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments."""

        self.mpl_kwargs = {"cmap": "gray",
                           "vmin": -300,
                           "vmax": 200}
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

    def set_ax(self, view, ax=None, gs=None, figsize=None, zoom=None, 
               n_colorbars=0):
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
        figsize = core.to_inches(figsize)
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
            return self.origin[ax] \
                    + (self.n_voxels[ax] - 1 - idx) * self.voxel_sizes[ax]
        else:
            return self.origin[ax] + idx * self.voxel_sizes[ax]

    def pos_to_idx(self, pos, ax):
        """Convert a position in mm to an index along a given axis."""

        if ax != "z":
            idx = round(self.n_voxels[ax] - 1 + (self.origin[ax] - pos) / 
                        self.voxel_sizes[ax])
        else:
            idx = round((pos - self.origin[ax]) / self.voxel_sizes[ax])

        if idx < 0 or idx >= self.n_voxels[ax]:
            if idx < 0:
                idx = 0
            if idx >= self.n_voxels[ax]:
                idx = self.n_voxels[ax] - 1

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

        return int(idx)

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
        idx = self.get_idx(view, sl, pos)
        self.view = view
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
        cmap = copy.copy(matplotlib.cm.get_cmap(kwargs.pop("cmap")))
        cmap.set_bad(color=mask_color)

        # Plot image
        mesh = self.ax.imshow(self.current_slice,
                              extent=self.extent[view],
                              aspect=self.aspect[view],
                              cmap=cmap, **kwargs)

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
                self.ax.tick_params(which="minor", bottom=True, top=True, 
                                    left=True, right=True)

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
                             color=col, fontsize="large")

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
        self.grid_kwargs = {"color": "green",
                            "linewidth": 2}

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
        spacing=30,
        zoom=None,
        zoom_centre=None
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
            self.plot_grid(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre)
        elif plot_type == "quiver":
            self.plot_quiver(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre)

    def plot_quiver(
        self, 
        view, 
        sl, 
        pos, 
        ax, 
        mpl_kwargs=None,
        zoom=None,
        zoom_centre=None
    ):
        """Draw a quiver plot on a set of axes."""

        # Get arrow positions and lengths
        self.set_ax(view, ax, zoom=zoom)
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
        self.adjust_ax(view, zoom, zoom_centre)

    def plot_grid(
        self, 
        view, 
        sl, 
        pos, 
        ax, 
        mpl_kwargs=None,
        zoom=None,
        zoom_centre=None
    ):
        """Draw a grid plot on a set of axes."""

        # Get gridline positions
        self.set_ax(view, ax, zoom=zoom)
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
        self.adjust_ax(view, zoom, zoom_centre)


class StructImage(NiftiImage):
    """Class to load and plot a structure mask."""

    def __init__(self, nii=None, name=None, color=None, label="", load=True, 
                 contours=None, shape=None, origin=None, voxel_sizes=None,
                 **kwargs):
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
            color in which to plot this structure. If None, a random 
            color will be assigned. Can also be set later using 
            self.assign_color(color).
        """

        # Assign variables
        if nii is None and contours is None:
            raise TypeError("Must provide either <nii> or <contours> to "
                            "StructImage!")
        self.nii = nii
        self.nii_kwargs = kwargs if kwargs is not None else {}
        if isinstance(voxel_sizes, dict):
            voxel_sizes = list(voxel_sizes.values())
        if isinstance(origin, dict):
            origin = list(origin.values())
        self.nii_kwargs.update({"voxel_sizes": voxel_sizes, "origin": origin})
        self.visible = True
        self.path = nii if isinstance(nii, str) else None
        self.contours = contours
        self.origin = origin
        self.shape = shape
        self.voxel_sizes = voxel_sizes

        # Set name
        if name is not None:
            self.name = name
        elif isinstance(nii, str):
            basename = os.path.basename(nii).replace(".gz", "").replace(".nii", "")
            self.name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "",
                               basename).replace(" ", "_")
        else:
            self.name = "Structure"
        self.set_label(label)

        # Assign a random color
        self.custom_color_set = False
        if color is None:
            self.assign_color(np.random.rand(3, 1).flatten(), custom=False)
        else:
            self.assign_color(color)

        # Load data
        self.loaded = False
        if load:
            self.load()

    def set_label(self, label):
        """Set the label for this structure and use to generate nice name."""

        self.label = label
        nice = self.name.replace("_", " ")
        self.name_nice = nice[0].upper() + nice[1:]
        self.name_nice_nolabel = self.name_nice
        if self.label:
            self.name_nice += f" ({self.label})"

    def load(self):
        """Load struct data and create contours in all orientations."""

        if self.loaded:
            return

        # Create mask from initial set of contours if needed
        if self.nii is None:
            contours_idx = dicom.contours_to_indices(
                self.contours, self.origin, self.voxel_sizes, self.shape)
            self.nii = dicom.contours_to_mask(contours_idx, self.shape)

        NiftiImage.__init__(self, self.nii, **self.nii_kwargs)
        if not self.valid:
            return

        # Load contours
        self.set_contours()

        # Convert to boolean mask
        if not self.data.dtype == "bool":
            self.data = self.data > 0.5

        # Check whether structure is empty
        self.empty = not sum([len(contours) for contours in 
                              self.contours.values()])
        if self.empty:
            self.name_nice += " (empty)"

        self.loaded = True

    def __lt__(self, other):
        """Compare structures by name."""

        n1 = self.name_nice
        n2 = other.name_nice
        if n1.split()[:-1] == n2.split()[:-1]:
            try:
                num1 = int(n1.split()[-1])
                num2 = int(n2.split()[-1])
                return num1 < num2
            except ValueError:
                return n1 < n2
        else:
            if n1 == n2:
                if self.label and other.label:
                    return self.label < other.label
                if self.path is not None and other.path is not None:
                    return self.path < other.path

            return n1 < n2

    def get_centroid(self, units="voxels"):
        """Get the centroid position in 3D."""

        if not hasattr(self, "centroid"):
            self.centroid = {}
            non_zero = np.argwhere(self.data)
            cx, cy, cz = non_zero.mean(0)
            centroid = [cx, cy, cz]
            axes = ["x", "y", "z"]
            self.centroid["voxels"] = [self.idx_to_slice(c, axes[i]) 
                                       for i, c in enumerate(centroid)]
            self.centroid["mm"] = [self.idx_to_pos(c, axes[i]) 
                                   for i, c in enumerate(centroid)]

        return np.array(self.centroid[units])

    def get_centroid_2d(self, view, sl, units="voxels"):
        """Get the centroid position on a 2D slice."""

        if not self.on_slice(view, sl):
            return None, None
        self.set_slice(view, sl)
        non_zero = np.argwhere(self.current_slice)
        cy, cx = non_zero.mean(0)
        x_ax, y_ax = _plot_axes[view]
        conversion = self.idx_to_slice if units == "voxels" else \
                self.idx_to_pos
        if y_ax == "y":
            cy = self.n_voxels[y_ax] - 1 - cy
        return (conversion(cx, x_ax), 
                conversion(cy, y_ax))

    def get_volume(self, units):
        """Get total structure volume in voxels, mm, or ml."""

        self.load()
        if self.empty:
            return 0

        if not hasattr(self, "volume"):
            self.volume = {
                "voxels": self.data.astype(bool).sum()
            }
            self.volume["mm"] = self.volume["voxels"] \
                * abs(np.prod(list(self.voxel_sizes.values())))
            self.volume["ml"] = self.volume["mm"] * (0.1 ** 3)

        return self.volume[units]

    def get_struct_length(self, units):
        """Get the total x, y, z length in voxels or mm."""

        self.load()
        if self.empty:
            return (0, 0, 0)

        if not hasattr(self, "length"):
            self.length = {"voxels": [], "mm": []}
            nonzero = np.argwhere(self.data)
            for ax, n in _axes.items():
                vals = nonzero[:, n]
                if len(vals):
                    self.length["voxels"].append(max(vals) - min(vals))
                    self.length["mm"].append(self.length["voxels"][n]
                                             * abs(self.voxel_sizes[ax]))
                else:
                    self.length["voxels"].append(0)
                    self.length["mm"].append(0)

        return self.length[units]

    def get_struct_centre(self, units=None):
        """Get the centre of this structure in voxels or mm. If no
        units are given, units will be mm if <self_in_mm> is True."""

        self.load()
        if self.empty:
            return None, None, None

        if not hasattr(self, "centre"):
            self.centre = {"voxels": [], "mm": []}
            nonzero = np.argwhere(self.data)
            for ax, n in _axes.items():
                vals = nonzero[:, n]
                if len(vals):
                    mid_idx = np.mean(vals)
                    if ax == "y":
                        mid_idx = self.n_voxels[ax] - 1 - mid_idx
                    self.centre["voxels"].append(
                        self.idx_to_slice(mid_idx, ax))
                    self.centre["mm"].append(
                        self.idx_to_pos(mid_idx, ax))
                else:
                    self.centre["voxels"].append(None)
                    self.centre["mm"].append(None)

        if units is None:
            units = "mm" if self.scale_in_mm else "voxels"
        return self.centre[units]

    def set_plotting_defaults(self):
        """Set default matplotlib plotting keywords for both mask and
        contour images."""

        self.mask_kwargs = {"alpha": 1,
                            "interpolation": "none"}
        self.contour_kwargs = {"linewidth": 2}

    def convert_xy_contours(self, contours):
        """Convert index number to position or slice number for a set of 
        contours in the x-y plane."""

        contours_converted = {}

        for z, conts in contours.items():

            # Convert z key to slice number
            z_sl = self.pos_to_slice(z, "z")
            contours_converted[z_sl] = []

            # Convert x/y to either position or slice number
            for c in conts:
                points = []
                for p in c:
                    x, y = p[0], p[1]
                    if not self.scale_in_mm:
                        x = self.pos_to_slice(x, "x")
                        y = self.pos_to_slice(y, "y")
                    points.append((x, y))
                contours_converted[z_sl].append(points)

        # Set as x-y contours
        self.contours = {"x-y": contours_converted}

    def set_contours(self):
        """Compute positions of contours on each slice in each orientation.
        """

        if self.contours is None:
            self.contours = {}
        else:
            self.convert_xy_contours(self.contours)

        for view, z in _slider_axes.items():
            if view in self.contours:
                continue
            self.contours[view] = {}
            for sl in range(1, self.n_voxels[z] + 1):
                contour = self.get_contour_slice(view, sl)
                if contour is not None:
                    self.contours[view][sl] = contour

    def set_mask(self):
        """Compute structure mask using contours."""

        pass

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
                    else:
                        x = self.idx_to_slice(x, x_ax)
                        if self.voxel_sizes[y_ax] < 0:
                            y = self.idx_to_slice(y, y_ax)
                        else:
                            y = self.idx_to_slice(self.n_voxels[y_ax] - y, 
                                                  y_ax) + 1
                    contour_points.append((x, y))
                points.append(contour_points)
            return points

    def assign_color(self, color, custom=True):
        """Assign a color, ensuring that it is compatible with matplotlib."""

        if matplotlib.colors.is_color_like(color):
            self.color = matplotlib.colors.to_rgba(color)
            self.custom_color_set = custom
        else:
            print(f"color {color} is not a valid color.")

    def plot(
        self, 
        view, 
        sl=None, 
        pos=None, 
        ax=None, 
        mpl_kwargs=None, 
        plot_type="contour",
        zoom=None,
        zoom_centre=None,
        show=False,
        no_title=False
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

        zoom : float/tuple, default=None
            Factor by which to zoom in.
        """

        self.load()
        if not self.valid or not self.visible:
            return

        mpl_kwargs = {} if mpl_kwargs is None else mpl_kwargs
        linewidth = mpl_kwargs.get("linewidth", 2)
        contour_kwargs = {"linewidth": linewidth}
        centroid = "centroid" in plot_type
        if centroid:
            contour_kwargs["markersize"] = mpl_kwargs.get(
                "markersize", 7 * np.sqrt(linewidth))
            contour_kwargs["markeredgewidth"] = mpl_kwargs.get(
                "markeredgewidth", np.sqrt(linewidth))

        # Make plot
        if plot_type in ["contour", "centroid"]:
            self.plot_contour(view, sl, pos, ax, contour_kwargs, zoom, 
                              zoom_centre, no_title=no_title, 
                              centroid=centroid)
        elif plot_type == "mask":
            self.plot_mask(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre,
                           no_title=no_title)
        elif plot_type in ["filled", "filled centroid"]:
            mask_kwargs = {"alpha": mpl_kwargs.get("alpha", 0.3)}
            self.plot_mask(view, sl, pos, ax, mask_kwargs, zoom, zoom_centre,
                           no_title=no_title)
            self.plot_contour(view, sl, pos, self.ax, contour_kwargs,
                              zoom, zoom_centre, no_title=no_title,
                              centroid=centroid)

        if show:
            plt.show()

    def plot_mask(self, view, sl, pos, ax, mpl_kwargs=None, zoom=None,
                  zoom_centre=None, no_title=False):
        """Plot structure as a colored mask."""

        # Get slice
        self.load()
        self.set_ax(view, ax, zoom=zoom)
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
        self.adjust_ax(view, zoom, zoom_centre)
        self.label_ax(view, no_title=no_title)

    def plot_contour(self, view, sl, pos, ax, mpl_kwargs=None, zoom=None,
                     zoom_centre=None, centroid=False, no_title=False):
        """Plot structure as a contour."""

        self.load()
        self.set_ax(view, ax, zoom=zoom)
        if not self.on_slice(view, sl):
            return

        kwargs = self.get_kwargs(mpl_kwargs, default=self.contour_kwargs)
        kwargs.setdefault("color", self.color)
        idx = self.get_idx(view, sl, pos, default_centre=False)

        for points in self.contours[view][sl]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            self.ax.plot(points_x, points_y, **kwargs)

        if centroid:
            units = "voxels" if not self.scale_in_mm else "mm"
            x, y = self.get_centroid_2d(view, sl, units)
            self.ax.plot(x, y, '+', **kwargs)

        self.adjust_ax(view, zoom, zoom_centre)
        self.label_ax(view, no_title=no_title)

    def on_slice(self, view, sl):
        """Return True if a contour exists for this structure on a given slice.
        """

        self.load()
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

    def get_full_extent(self, units="voxels"):
        """Get the full extent along x, y, z."""

        self.load()
        if self.empty:
            return 0

        if not hasattr(self, "full_extent"):
            non_zero = np.argwhere(self.data)
            mins = non_zero.min(0)
            maxes = non_zero.max(0)
            x_len = abs(maxes[1] - mins[1]) + 1
            y_len = abs(maxes[0] - mins[0]) + 1
            z_len = abs(maxes[2] - mins[2]) + 1
            self.full_extent = {
                "voxels": [x_len, y_len, z_len]
            }
            self.full_extent["mm"] = [
                x_len * abs(self.voxel_sizes["x"]),
                y_len * abs(self.voxel_sizes["y"]),
                z_len * abs(self.voxel_sizes["z"]),
            ]

        return self.full_extent[units]

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
            x_len = abs(maxes[1] - mins[1]) + 1
            y_len = abs(maxes[0] - mins[0]) + 1
            if units == "mm":
                x_len *= abs(self.voxel_sizes[x])
                y_len *= abs(self.voxel_sizes[y])
            return x_len, y_len
        else:
            return 0, 0

    def get_centre(self, view, sl):
        """Get the coordinates of the centre of this structure in a given view
        on a given slice."""

        if not self.on_slice(view, sl):
            return [None, None]

        self.set_slice(view, sl)
        non_zero = np.argwhere(self.current_slice)
        x_ax, y_ax = _plot_axes[view]
        if len(non_zero):
            y, x = (non_zero.max(0) + non_zero.min(0)) / 2
            convert = self.idx_to_pos if self.scale_in_mm \
                else self.idx_to_slice
            if y_ax == "y":
                y = self.n_voxels[y_ax] - 1 - y
            return [convert(x, x_ax), convert(y, y_ax)]
        else:
            return [0, 0]


class MultiImage(NiftiImage):
    """Class for loading and plotting an image along with an optional mask,
    dose field, structures, jacobian determinant, and deformation field."""

    def __init__(
        self,
        nii=None,
        dose=None,
        mask=None,
        jacobian=None,
        df=None,
        structs=None,
        multi_structs=None,
        timeseries=None,
        struct_colors=None,
        structs_as_mask=False,
        struct_names=None,
        compare_structs=False,
        comp_type="auto",
        ignore_empty_structs=False,
        ignore_unpaired_structs=False,
        structs_to_keep=None,
        structs_to_ignore=None,
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

        struct_colors : dict, default=None
            Custom colors to use for structures. Dictionary keys can be a
            structure name or a wildcard matching structure name(s). Values
            should be any valid matplotlib color.

        structs_as_mask : bool, default=False
            If True, structures will be used as masks.

        struct_names : list/dict, default=None
            For multi_structs, this parameter will be used to name
            the structures. Can either be a list (i.e. the first structure in 
            the file will be given the first name in the list and so on), or a 
            dict of numbers and names (e.g. {1: "first structure"} etc).

        compare_structs : bool, default=False
            If True, structures will be paired together into comparisons.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).
        """

        # Flags for image type
        self.dose_as_im = False
        self.dose_comp = False
        self.timeseries = False

        # Load the scan image
        if nii is not None:
            NiftiImage.__init__(self, nii, **kwargs)
            self.timeseries = False

        # Load a dose field only
        elif dose is not None and timeseries is None:
            self.dose_as_im = True
            NiftiImage.__init__(self, dose, **kwargs)
            dose = None

        # Load a timeseries of images
        elif timeseries is not None:
            self.timeseries = True
            dates = self.get_date_dict(timeseries)
            self.dates = list(dates.keys())
            if "title" in kwargs:
                kwargs.pop("title")
            self.ims = {
                date: NiftiImage(file, title=date, **kwargs) for date, file in 
                dates.items()
            }
            NiftiImage.__init__(self, dates[self.dates[0]], 
                                title=self.dates[0], **kwargs)
            self.date = self.dates[0]
        else:
            raise TypeError("Must provide either <nii>, <dose>, or "
                            "<timeseries!>")
        if not self.valid:
            return

        # Load extra overlays
        self.load_to(dose, "dose", kwargs)
        self.load_to(mask, "mask", kwargs)
        self.load_to(jacobian, "jacobian", kwargs)
        self.load_df(df)

        # Load structs
        self.comp_type = comp_type
        self.load_structs(structs, 
                          multi_structs,
                          names=struct_names, 
                          colors=struct_colors, 
                          compare_structs=compare_structs, 
                          ignore_empty=ignore_empty_structs,
                          ignore_unpaired=ignore_unpaired_structs,
                          comp_type=comp_type,
                          to_keep=structs_to_keep,
                          to_ignore=structs_to_ignore
                         )

        # Mask settings
        self.structs_as_mask = structs_as_mask
        if self.has_structs and structs_as_mask:
            self.has_mask = True
        self.mask_threshold = mask_threshold
        self.set_masks()

    def load_to(self, nii, attr, kwargs):
        """Load image data into a class attribute."""

        # Load single image
        rescale = "dose" if attr == "dose" else True
        if not isinstance(nii, dict):
            data = NiftiImage(nii, rescale=rescale, **kwargs)
            setattr(self, attr, data)
            valid = data.valid
        else:
            data = {view: NiftiImage(nii[view], rescale=rescale, **kwargs) 
                    for view in nii}
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

    def load_structs(self, 
                     structs=None,
                     multi_structs=None,
                     names=None, 
                     colors=None, 
                     compare_structs=False, 
                     ignore_empty=False,
                     ignore_unpaired=False,
                     comp_type="auto",
                     to_keep=None,
                     to_ignore=None
                    ):
        """Load structures from a path/wildcard or list of paths/wildcards in
        <structs>, and assign the colors in <colors>."""

        self.has_structs = False
        self.struct_timeseries = False
        if not (structs or multi_structs):
            self.structs = []
            self.struct_comparisons = []
            self.standalone_structs = []
            return

        # Check whether a timeseries of structs is being used
        if self.timeseries:
            try:
                struct_dates = self.get_date_dict(structs, True)
                self.struct_timeseries = len(struct_dates) > 1
            except TypeError:
                pass

        # No timeseries: load single set of structs
        if not self.struct_timeseries:
            loader = StructLoader(structs, multi_structs, names, colors,
                                  comp_type=comp_type,
                                  struct_kwargs={"scale_in_mm": 
                                                 self.scale_in_mm},
                                  image=self,
                                  to_keep=to_keep,
                                  to_ignore=to_ignore
                                 )
            self.structs = loader.get_structs(ignore_unpaired, ignore_empty)

            if compare_structs:
                self.struct_comparisons = loader.get_comparisons(ignore_empty)
                self.standalone_structs = loader.get_standalone_structs(
                    ignore_unpaired, ignore_empty)
            else:
                self.standalone_structs = self.structs
                self.struct_comparisons = []

            self.has_structs = bool(len(self.structs))

        # Load timeseries of structs
        else:
            self.dated_structs = {}
            self.dated_comparisons = {}
            self.dated_standalone_structs = {}
            struct_colors = {}
            for date, structs in struct_dates.items():

                if date not in self.dates:
                    continue

                loader = StructLoader(structs, names=names, colors=colors,
                                      comp_type=comp_type,
                                      struct_kwargs={"scale_in_mm": 
                                                     self.scale_in_mm},
                                      image=self, to_keep=to_keep,
                                      to_ignore=to_ignore)
                struct_colors = loader.reassign_colors(struct_colors)
                self.dated_structs[date] = loader.get_structs(
                    ignore_unpaired, ignore_empty, sort=True)

                if compare_structs:
                    self.dated_comparisons[date] = \
                            loader.get_comparisons(ignore_empty)
                    self.dated_standalone_structs = \
                            loader.get_standalone_structs(ignore_unpaired,
                                                          ignore_empty)
                else:
                    self.dated_comparisons[date] = []
                    self.dated_standalone_structs[date] = \
                            self.dated_structs[date]

            self.has_structs = any([len(s) for s in 
                                    self.dated_structs.values()])

            # Set to current date
            if self.date in self.dated_structs:
                self.structs = self.dated_structs[date]
                self.struct_comparisons = self.dated_comparisons[date]
                self.standalone_structs = self.dated_standalone_structs[date]

    def get_date_dict(self, timeseries, single_layer=False):
        """Convert list/dict/directory to sorted dict of dates and files."""

        if isinstance(timeseries, dict):
            dates = {dateutil.parser.parse(key): val for key, val in 
                     timeseries.items()}

        else:
            if isinstance(timeseries, str):
                files = core.find_files(timeseries)
            elif core.is_list(timeseries):
                files = timeseries
            else:
                raise TypeError("Timeseries must be a list, dict, or str.")

            # Find date-like string in filenames
            dates = {}
            for file in files:
                dirname = os.path.basename(os.path.dirname(file))
                date = core.find_date(dirname)
                if not date:
                    base = os.path.basename(file)
                    date = core.find_date(base)
                if not date:
                    raise TypeError("Date-like string could not be found in "
                                    f"filename of dirname of {file}!")
                dates[date] = file

        # Sort by date
        dates_sorted = sorted(list(dates.keys()))
        date_strs = {date: f"{date.day}/{date.month}/{date.year}"
                     for date in dates_sorted}
        return {date_strs[date]: dates[date] for date in dates_sorted}

    def set_date(self, n):
        """Go to the nth image in series."""

        if n < 1:
            n = 1
        if n > len(self.dates) or n == -1:
            n = len(self.dates)
        self.date = self.dates[n - 1]
        self.data = self.ims[self.date].data
        self.title = self.date

        # Set structs
        if self.struct_timeseries:
            self.structs = self.dated_structs.get(self.date, [])
            self.struct_comparisosn = self.dated_comparisons.get(self.date, [])
            self.standalone_structs = \
                    self.dated_standalone_structs.get(self.date, [])

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

    def set_masks(self):
        """Assign mask(s) to self and dose image."""

        if not self.has_mask:
           mask_array = None 

        else:
            # Combine user-input mask with structs
            mask_array = np.zeros(self.shape, dtype=bool)
            if not self.mask_dict and self.mask.valid:
                mask_array += self.mask.data > self.mask_threshold
            if self.structs_as_mask:
                for struct in self.structs:
                    if struct.visible:
                        mask_array += struct.data

            # Get separate masks for each orientation
            if self.mask_dict:
                view_masks = {}
                for view in _orient:
                    mask = self.mask.get(view, None)
                    if mask is not None:
                        if isinstance(mask, NiftiImage):
                            view_masks[view] = mask_array \
                                    + (self.mask[view].data > 
                                       self.mask_threshold)
                        else:
                            view_masks[view] = mask_array \
                                    + (self.mask[view] > 
                                       self.mask_threshold)
                    else:
                        if self.structs_as_mask:
                            view_masks[view] = self.mask_array
                        else:
                            view_masks[view] = None

                mask_array = view_masks

        # Assign mask to main image and dose field
        self.set_mask(mask_array, self.mask_threshold)
        self.dose.data_mask = self.data_mask

    def get_n_colorbars(self, colorbar=False):
        """Count the number of colorbars needed for this plot."""

        return colorbar * (1 + self.has_dose + self.has_jacobian)

    def get_relative_width(self, view, zoom=None, colorbar=False, 
                           figsize=None):
        """Get the relative width for this plot, including all colorbars."""

        return NiftiImage.get_relative_width(
            self, view, zoom, self.get_n_colorbars(colorbar), figsize)

    def plot(
        self,
        view="x-y",
        sl=None,
        pos=None,
        ax=None,
        gs=None,
        figsize=None,
        zoom=None,
        zoom_centre=None,
        mpl_kwargs=None,
        n_date=1,
        show=True,
        colorbar=False,
        colorbar_label="HU",
        dose_kwargs=None,
        masked=False,
        invert_mask=False,
        mask_color="black",
        jacobian_kwargs=None,
        df_kwargs=None,
        df_plot_type="grid",
        df_spacing=30,
        struct_kwargs=None,
        struct_plot_type="contour",
        struct_legend=True,
        legend_loc='lower left',
        struct_plot_grouping=None,
        struct_to_plot=None,
        annotate_slice=None,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False
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

        zoom : int/float/tuple, default=None
            Factor by which to zoom in. If a single int or float is given, 
            the same zoom factor will be applied in all directions. If a tuple 
            of three values is given, these will be used as the zoom factors 
            in each direction in the order (x, y, z). If None, the image will 
            not be zoomed in.

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

        mask_color : matplotlib color, default="black"
            color in which to plot masked areas.

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
            Plot type for structures ("contour"/"mask"/"filled")

        struct_legend : bool, default=True
            If True, a legend will be drawn labelling any structrues visible on
            this slice.

        legend_loc : str, default='lower left'
            Position for the structure legend, if used.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will 
            be added. If True, the default color (white) will be used.
        """

        # Set date
        if self.timeseries:
            self.set_date(n_date)

        # Plot image
        self.set_ax(view, ax, gs, figsize, zoom, colorbar)
        NiftiImage.plot(
            self, view, sl, pos, ax=self.ax, mpl_kwargs=mpl_kwargs,
            show=False, colorbar=colorbar, colorbar_label=colorbar_label,
            masked=masked, invert_mask=invert_mask, mask_color=mask_color, 
            figsize=figsize, major_ticks=major_ticks, minor_ticks=minor_ticks,
            ticks_all_sides=ticks_all_sides)

        # Plot dose field
        self.dose.plot(
            view, self.sl, ax=self.ax,
            mpl_kwargs=self.get_kwargs(dose_kwargs, default=self.dose_kwargs),
            show=False, masked=masked, invert_mask=invert_mask,
            mask_color=mask_color, colorbar=colorbar, 
            colorbar_label="Dose (Gy)")

        # Plot jacobian
        self.jacobian.plot(
            view, self.sl, ax=self.ax, mpl_kwargs=self.get_kwargs(
                jacobian_kwargs, default=self.jacobian_kwargs),
            show=False, colorbar=colorbar,
            colorbar_label="Jacobian determinant")

        # Plot standalone structures and comparisons
        for s in self.standalone_structs:
            s.plot(view, self.sl, ax=self.ax, mpl_kwargs=struct_kwargs, 
                   plot_type=struct_plot_type)
        for s in self.struct_comparisons:
            if struct_plot_grouping == "group others":
                if s.s1.name_unique != struct_to_plot:
                    continue
            s.plot(view, self.sl, ax=self.ax, mpl_kwargs=struct_kwargs, 
                   plot_type=struct_plot_type, 
                   plot_grouping=struct_plot_grouping)

        # Plot deformation field
        self.df.plot(view, self.sl, ax=self.ax,
                     mpl_kwargs=df_kwargs,
                     plot_type=df_plot_type,
                     spacing=df_spacing)

        # Draw structure legend
        if struct_legend and struct_plot_type != "none":
            handles = []
            for s in self.structs:
                if struct_plot_grouping == "group others":
                    if s.name_unique == struct_to_plot:
                        handles.append(mpatches.Patch(color=s.color, 
                                                      label=s.name_nice))
                        handles.append(mpatches.Patch(color="white",
                                                      label="Others"))
                elif s.visible and s.on_slice(view, self.sl):
                    handles.append(mpatches.Patch(color=s.color, 
                                                  label=s.name_nice))
            if len(handles):
                self.ax.legend(handles=handles, loc=legend_loc,
                               facecolor="white", framealpha=1)

        self.adjust_ax(view, zoom, zoom_centre)
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

    def get_relative_width(self, view, zoom=None, colorbar=False, 
                           figsize=None):
        """Get width:height ratio for the full plot (main plot + orthogonal
        view)."""

        width_own = MultiImage.get_relative_width(self, view, zoom, colorbar,
                                                  figsize)
        width_orthog = MultiImage.get_relative_width(self, _orthog[view],
                                                     figsize)
        return width_own + width_orthog

    def set_axes(self, view, ax=None, gs=None, figsize=None, zoom=None,
                 colorbar=False):
        """Set up axes for the plot. If <ax> is not None and <orthog_ax> has
        already been set, these axes will be used. Otherwise if <gs> is not 
        None, the axes will be created within a gridspec on the current 
        matplotlib figure.  Otherwise, a new figure with height <figsize> 
        will be produced."""

        if ax is not None and hasattr(self, "orthog_ax"):
            self.ax = ax

        width_ratios = [
            MultiImage.get_relative_width(self, view, zoom, colorbar, figsize),
            MultiImage.get_relative_width(self, _orthog[view], figsize)
        ]
        if gs is None:
            figsize = _default_figsize if figsize is None else figsize
            figsize = core.to_inches(figsize)
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
             zoom=None,
             zoom_centre=None,
             mpl_kwargs=None,
             show=True,
             colorbar=False,
             colorbar_label="HU",
             struct_kwargs=None,
             struct_plot_type=None,
             major_ticks=None,
             minor_ticks=None,
             ticks_all_sides=False,
             **kwargs
            ):
        """Plot MultiImage and orthogonal view of main image and structs."""

        self.set_axes(view, ax, gs, figsize, zoom, colorbar)

        # Plot the MultiImage
        MultiImage.plot(self, view, sl=sl, pos=pos, ax=self.ax, zoom=zoom,
                        zoom_centre=zoom_centre, colorbar=colorbar, 
                        show=False, colorbar_label=colorbar_label,
                        mpl_kwargs=mpl_kwargs, struct_kwargs=struct_kwargs,
                        struct_plot_type=struct_plot_type, 
                        major_ticks=major_ticks, minor_ticks=minor_ticks, 
                        ticks_all_sides=ticks_all_sides, 
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
                        no_ylabel=False,
                        no_title=True,
                        major_ticks=major_ticks,
                        minor_ticks=minor_ticks,
                        ticks_all_sides=ticks_all_sides,
                       )

        # Plot structures on orthogonal image
        for struct in self.structs:
            if not struct.visible:
                continue
            plot_type = struct_plot_type
            if plot_type == "centroid":
                plot_type = "contour"
            elif plot_type == "filled centroid":
                plot_type = "filled"
            struct.plot(orthog_view, sl=orthog_sl, ax=self.orthog_ax, 
                        mpl_kwargs=struct_kwargs, plot_type=plot_type,
                        no_title=True)

        # Plot indicator line
        pos = sl if not self.scale_in_mm else self.slice_to_pos(
            sl, _slider_axes[view])
        if view == "x-y":
            full_y = self.extent[orthog_view][2:] if self.scale_in_mm else \
                [0.5, self.n_voxels[_plot_axes[orthog_view][1]] + 0.5]
            self.orthog_ax.plot([pos, pos], full_y, 'r')
        else:
            full_x = self.extent[orthog_view][:2] if self.scale_in_mm else \
                [0.5, self.n_voxels[_plot_axes[orthog_view][0]] + 0.5]
            self.orthog_ax.plot(full_x, [pos, pos], 'r')

        if show:
            plt.tight_layout()
            plt.show()


class ComparisonImage(NiftiImage):
    """Class for loading data from two arrays and plotting comparison images.
    """

    def __init__(self, nii1, nii2, title=None, plot_type=None, **kwargs):
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
        self.ax_lims = self.ims[0].ax_lims
        self.valid = all([im.valid for im in self.ims])
        self.override_title = title
        self.gs = None
        self.plot_type = plot_type if plot_type else "chequerboard"

    def get_relative_width(self, view, colorbar=False, figsize=None):
        """Get relative width first image."""

        return NiftiImage.get_relative_width(self.ims[0], view, 
                                             n_colorbars=colorbar,
                                             figsize=figsize)

    def plot(self, view=None, sl=None, invert=False, ax=None,
             mpl_kwargs=None, show=True, figsize=None, zoom=None, 
             zoom_centre=None, plot_type=None, cb_splits=2,
             overlay_opacity=0.5, overlay_legend=False, 
             overlay_legend_loc=None, colorbar=False, colorbar_label="HU",
             show_mse=False, dta_tolerance=None, dta_crit=None, diff_crit=None
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
        self.set_ax(view, ax, self.gs, figsize, zoom)
        self.plot_kwargs = self.ims[0].get_kwargs(mpl_kwargs)
        self.cmap = copy.copy(matplotlib.cm.get_cmap(
            self.plot_kwargs.pop("cmap")))
        
        # Produce comparison plot
        if plot_type == "chequerboard":
            mesh = self.plot_chequerboard(invert, cb_splits)
        elif plot_type == "overlay":
            mesh = self.plot_overlay(invert, overlay_opacity, overlay_legend,
                                     overlay_legend_loc)
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
            mesh = self.ax.imshow(self.slices[0], 
                                  extent=self.ims[0].extent[self.view],
                                  aspect=self.ims[0].aspect[self.view],
                                  cmap=self.cmap, **self.plot_kwargs)
        elif plot_type == "image 2":
            self.title = self.ims[1].title
            mesh = self.ax.imshow(self.slices[1], 
                                  extent=self.ims[1].extent[self.view],
                                  aspect=self.ims[1].aspect[self.view],
                                  cmap=self.cmap, **self.plot_kwargs)

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
            self.ax.annotate(mse_str, xy=(0.05, 0.93), 
                             xycoords='axes fraction', 
                             color=col, fontsize="large")

    def plot_chequerboard(self, invert=False, n_splits=2):
        """Produce a chequerboard plot with <n_splits> squares in each
        direction."""

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
            mesh = self.ax.imshow(to_show[i],
                                  extent=self.ims[i].extent[self.view],
                                  aspect=self.ims[i].aspect[self.view],
                                  cmap=self.cmap, **self.plot_kwargs)
        return mesh

    def plot_overlay(self, invert=False, opacity=0.5, legend=False,
                        legend_loc='auto'):
        """Produce an overlay plot with a given opacity."""

        order = [0, 1] if not invert else [1, 0]
        cmaps = ["Reds", "Blues"]
        alphas = [1, opacity]
        self.ax.set_facecolor("w")
        handles = []
        for n, i in enumerate(order):

            # Show image
            mesh = self.ax.imshow(self.slices[i],
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
        return mesh

    def plot_difference(self, invert=False, ab=False):
        """Produce a difference plot."""

        diff = self.slices[1] - self.slices[0] if not invert \
                else self.slices[0] - self.slices[1]
        if ab:
            diff = np.absolute(diff)
        return self.ax.imshow(diff,
                              extent=self.ims[0].extent[self.view],
                              aspect=self.ims[0].aspect[self.view],
                              cmap=self.cmap, 
                              **self.plot_kwargs)

    def plot_dta(self, tolerance=5):
        """Produce a distance-to-agreement plot."""

        dta = self.get_dta(tolerance)
        return self.ax.imshow(dta,
                              extent=self.ims[0].extent[self.view],
                              aspect=self.ims[0].aspect[self.view],
                              cmap="viridis", interpolation=None,
                              **self.plot_kwargs)

    def plot_gamma(self, invert=False, dta_crit=None, diff_crit=None):
        """Produce a distance-to-agreement plot."""

        gamma = self.get_gamma(invert, dta_crit, diff_crit)
        return self.ax.imshow(gamma,
                              extent=self.ims[0].extent[self.view],
                              aspect=self.ims[0].aspect[self.view],
                              cmap="viridis", interpolation=None,
                              **self.plot_kwargs)

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
                dta_val = np.sqrt(vy * dta_vec[:, 0] ** 2 
                                  + vx * dta_vec[:, 1] ** 2).min()
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


def standard_str(string):
    """Convert a string to lowercase and replace all spaces with 
    underscores."""

    try:
        return str(string).lower().replace(" ", "_")
    except AttributeError:
        return


class StructComparison:
    """Class for computing comparison metrics for two structures and plotting
    the structures together."""

    def __init__(self, struct1, struct2, name="", comp_type=None, **kwargs):
        """Initialise from a pair of StructImages, or load new StructImages.
        """

        self.name = name
        self.comp_type = comp_type
        if self.comp_type == "others":
            self.comp_type = "majority vote"

        # Two structures
        self.s2_is_list = core.is_list(struct2)
        if not self.s2_is_list:
            for i, s in enumerate([struct1, struct2]):
                struct = s if isinstance(s, StructImage) \
                    else StructImage(s, **kwargs)
                setattr(self, f"s{i + 1}", s)

        # List of structures
        else:
            self.s1 = struct1
            self.s2_list = struct2
            self.s2_voxel_sizes = list(self.s1.voxel_sizes.values())
            self.s2_origin = list(self.s1.origin.values())
            self.s2_name = f"{self.comp_type} of others"
            self.update_s2_data()

        mean_sq_col = (
            np.array(self.s1.color) ** 2 
            + np.array(self.s2.color) ** 2) / 2
        self.color = np.sqrt(mean_sq_col)

    def update_s2_data(self, comp_type=None):
        """Update the data in struct2 using struct visibility and potential
        new comp type."""

        if not self.s2_is_list:
            return

        if comp_type:
            self.comp_type = comp_type

        structs_to_use = [s for s in self.s2_list if s.visible]
        data = structs_to_use[0].data.copy()
        if self.comp_type == "majority vote":
            data = data.astype(int)
        for s in structs_to_use[1:]:
            if self.comp_type == "sum":
                data += s.data
            elif self.comp_type == "overlap":
                data *= s.data
            elif self.comp_type == "majority vote":
                data += s.data.astype(int)
        if self.comp_type == "majority vote":
            data = data >= len(structs_to_use) / 2

        self.s2 = StructImage(data, name=self.s2_name, 
                              load=True, voxel_sizes=self.s2_voxel_sizes, 
                              origin=self.s2_origin)
        self.s2.color = matplotlib.colors.to_rgba("white")
        self.s2.name_unique = f"vs. {self.comp_type} of others"

        self.centroid_distance(force=True)
        self.global_dice_score(force=True)

    def is_valid(self):
        """Check both structures are valid and in same reference frame."""

        self.s1.load()
        self.s2.load()
        if not self.s1.same_frame(self.s2):
            raise TypeError(f"Comparison structures {self.s1.name} and "
                            f"{self.s2.name} are not in the same reference "
                            "frame!")
        self.valid = self.s1.valid and self.s2.valid
        return self.valid

    def plot(
        self, 
        view, 
        sl=None, 
        pos=None, 
        ax=None, 
        mpl_kwargs=None, 
        plot_type="contour",
        zoom=None,
        zoom_centre=None,
        show=False,
        plot_grouping=None
    ):
        """Plot comparison structures."""

        if not self.is_valid():
            return
        if mpl_kwargs is None:
            mpl_kwargs = {}

        if self.s2_is_list and plot_grouping != "group others":
            self.s1.plot(view=view, sl=sl, pos=pos, ax=ax, 
                         mpl_kwargs=mpl_kwargs, plot_type=plot_type, zoom=zoom, 
                         zoom_centre=zoom_centre, show=show)
            return

        # If one structure isn't currently visible, only plot the other
        if not self.s1.visible or not self.s2.visible:
            s_vis = [s for s in [self.s1, self.s2] if s.visible]
            if len(s_vis):
                s_vis[0].plot(view, sl, pos, ax, mpl_kwargs, plot_type, zoom,
                              zoom_centre, show)
            return

        # Make plot
        linewidth = mpl_kwargs.get("linewidth", 2)
        contour_kwargs = {"linewidth": linewidth}
        centroid = "centroid" in plot_type
        if centroid:
            contour_kwargs["markersize"] = mpl_kwargs.get(
                "markersize", 7 * np.sqrt(linewidth))
            contour_kwargs["markeredgewidth"] = mpl_kwargs.get(
                "markeredgewidth", np.sqrt(linewidth))

        if plot_type in ["contour", "centroid"]:
            centroid = plot_type != "contour"
            self.s2.plot_contour(view, sl, pos, ax, contour_kwargs, 
                                 zoom, zoom_centre, centroid=centroid)
            self.s1.plot_contour(view, sl, pos, self.s2.ax, contour_kwargs, zoom, 
                                 zoom_centre, centroid=centroid)
        elif plot_type == "mask":
            self.plot_mask(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre)
        elif plot_type in ["filled", "filled centroid"]:
            mask_kwargs = {"alpha": mpl_kwargs.get("alpha", 0.3)}
            self.plot_mask(view, sl, pos, ax, mask_kwargs, zoom, zoom_centre)
            self.s2.plot_contour(view, sl, pos, self.s2.ax, contour_kwargs, 
                                 zoom, zoom_centre, centroid=centroid)
            self.s1.plot_contour(view, sl, pos, self.s2.ax, contour_kwargs, 
                                 zoom, zoom_centre, centroid=centroid)

        if show:
            plt.show()

    def plot_mask(self, view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre):
        """Plot two masks, with intersection in different colour."""

        # Set slice for both images
        self.s2.set_ax(view, ax, zoom=zoom)
        self.s2.set_slice(view, sl, pos)
        self.s1.set_slice(view, sl, pos)

        # Get differences and overlap
        diff1 = self.s1.current_slice & ~self.s2.current_slice
        diff2 = self.s2.current_slice & ~self.s1.current_slice
        overlap = self.s1.current_slice & self.s2.current_slice
        to_plot = [
            (diff1, self.s1.color),
            (diff2, self.s2.color),
            (overlap, self.color)
        ]

        for im, color in to_plot:

            # Make colormap
            norm = matplotlib.colors.Normalize()
            cmap = matplotlib.cm.hsv
            s_colors = cmap(norm(im))
            s_colors[im > 0, :] = color
            s_colors[im == 0, :] = (0, 0, 0, 0)

            # Display mask
            self.s2.ax.imshow(
                s_colors,
                extent=self.s1.extent[view],
                aspect=self.s1.aspect[view],
                **self.s1.get_kwargs(mpl_kwargs, default=self.s1.mask_kwargs)
            )

        self.s1.adjust_ax(view, zoom, zoom_centre)

    def on_slice(self, view, sl):
        """Check whether both structures are on a given slice."""
        
        if not self.is_valid():
            return False
        return self.s1.on_slice(view, sl) and self.s2.on_slice(view, sl)

    def centroid_distance(self, units="voxels", force=False):
        """Get total centroid distance."""

        if not hasattr(self, "centroid_dist") or force:
            self.centroid_dist = {
                units: np.linalg.norm(self.s1.get_centroid(units) 
                                      - self.s2.get_centroid(units)) 
                for units in ["mm", "voxels"]}

        return self.centroid_dist[units]

    def centroid_distance_2d(self, view, sl, units="voxels"):
        """Get distances between centroid in x, y directions for current "
        slice."""

        cx1, cy1 = self.s1.get_centroid_2d(view, sl, units)
        cx2, cy2 = self.s2.get_centroid_2d(view, sl, units)
        if cx1 is None or cx2 is None:
            return None, None
        return cx1 - cx2, cy1 - cy2

    def dice_score(self, view, sl):
        """Get dice score on a given slice."""

        if not self.on_slice(view, sl):
            return
        self.s1.set_slice(view, sl)
        self.s2.set_slice(view, sl)
        slice1 = self.s1.current_slice
        slice2 = self.s2.current_slice
        return (slice1 & slice2).sum() / np.mean([slice1.sum(), slice2.sum()])

    def global_dice_score(self, force=False):
        """Global dice score for entire structures."""

        if not hasattr(self, "global_dice") or force:
            self.global_dice = (self.s1.data & self.s2.data).sum() / \
                    np.mean([self.s1.data.sum(), self.s2.data.sum()])

        return self.global_dice

    def vol_ratio(self):
        """Get relative volume of the two structures."""

        v1 = self.s1.get_volume("voxels")
        v2 = self.s2.get_volume("voxels")
        return v1 / v2

    def relative_vol(self):
        """Get relative structure volume difference."""

        v1 = self.s1.get_volume("voxels")
        v2 = self.s2.get_volume("voxels")
        return (v1 - v2) / v1

    def relative_area(self, view, sl):
        """Get relative structure area difference on a slice."""

        a1 = self.s1.get_area(view, sl)
        a2 = self.s2.get_area(view, sl)
        if a1 is None or a2 is None:
            return None
        return (a1 - a2) / a1

    def area_ratio(self, view, sl):

        if not self.on_slice(view, sl):
            return
        a1 = self.s1.get_area(view, sl)
        a2 = self.s2.get_area(view, sl)
        return a1 / a2

    def extent_ratio(self, view, sl):

        if not self.on_slice(view, sl):
            return
        x1, y1 = self.s1.get_extents(view, sl)
        x2, y2 = self.s2.get_extents(view, sl)
        return x1 / x2, y1 / y2


class StructLoader:
    """Class for loading and storing multiple StructImages."""

    def __init__(self, structs=None, multi_structs=None, names=None, 
                 colors=None, comp_type="auto", struct_kwargs=None,
                 image=None, to_keep=None, to_ignore=None):
        """Load structures. 

        Parameters
        ----------
        
        structs : str/list/dict
            Sources of structures files to load structures from. Can be:
                (a) A string containing a filepath, directory path, or wildcard
                to a file or directory path. If a directory is given, all 
                .nii and .nii.gz files within that directory will be loaded.
                (b) A list of strings as described in (a). All 
                files/directories in the list will be loaded.
                (c) A dictionary, where keys are labels and values are strings
                or lists as described in (a) and (b). The files loaded for 
                each entry will be given the label in the key.
                (d) A list of pairs filepaths or wildcard filepaths, which
                should point to one file only. These pairs of files will then
                be used for comparisons.

        multi_structs : str/list/dict

        names : list/dict, default=None
            A dictionary where keys are filenames or wildcards matching 
            filenames, and values are names to give the structure(s) in those
            files. Keys can also be lists of several potential filenames. If 
            None, defaults will be taken from the file given in the 
            default_struct_names parameter ~/.quickviewer/settings.ini, if it 
            exists.

            If using multiple structures per file, this can also be either:
                (a) A list of names, where the order reflects the order of 
                structure masks in the file (i.e. the nth item in the list
                will refer to the structure with label mask n + 1).
                (b) A dictionary where the keys are integers referring to the
                label masks and values are structure names.

            This can also be nested in a dictionary to give multiple naming
            options for different labels.

        colors : dict, default=None
            A dictionary of colours to assign to structures with a given name.
            Can also be a nested dictionary inside a dictionary where keys
            are labels, so that different sets of structure colours can be used
            for different labels. If None, defaults will be taken from the file
            given in ~/.quickviewer/settings.ini, if it exists.

        struct_kwargs : dict, default=None
            Keyword arguments to pass to any created StructImage objects.

        comp_type : str, default="auto"
            Option for method of comparing any loaded structures. Can be:
            - "auto": Structures will be matched based on name if many are
              loaded, pairs if a list of pairs is given, or simply matched
              if only two structs are loaded.
            - "pairs": Every possible pair of loaded structs will be compared.
            - "other": Each structure will be compared to the consenues of all 
            of the others.
            - "overlap": Each structure will be comapred to the overlapping
            region of all of the others.

        to_keep : list, default=None
            List of structure names/wildcards to keep. If this argument is set,
            all otehr structures will be ignored.

        to_ignore : list, default=None
            List of structure names to ignore.
        """

        # Lists for storing structures
        self.loaded = False
        self.structs = []
        self.comparisons = []
        self.comparison_structs = []
        self.comp_type = comp_type
        self.struct_kwargs = struct_kwargs if struct_kwargs is not None \
            else {}
        if not (structs or multi_structs):
            return
        self.image = image
        self.to_keep = to_keep
        self.to_ignore = to_ignore

        # Format colors and names
        names = self.load_settings(names)
        colors = self.load_settings(colors)

        # Load all structs and multi structs
        self.load_structs(structs, names, colors, False)
        self.load_structs(multi_structs, names, colors, True)

    def load_settings(self, settings):
        """Process a settings dict into a standard format."""

        if settings is None:
            return {}
        
        # Convert single list to enumerated dict
        elif core.is_list(settings):
            settings = {value: i + 1 for i, value in enumerate(settings)}

        # Convert label dict of lists into enumerated dicts
        elif isinstance(settings, dict):
            for label, s in settings.items():
                if core.is_list(s):
                    settings[label] = {value: i + 1 for i, value in 
                                       enumerate(s)}

        return settings

    def load_structs(self, structs, names, colors, multi=False):
        """Load a list/dict of structres."""

        if structs is None:
            return

        struct_dict = {}

        # Put into standard format
        # Case where structs are already in a dict of labels and sources
        if isinstance(structs, dict):
            struct_dict = structs
            for label, path in struct_dict.items():
                if not core.is_list(path):
                    struct_dict[label] = [path]

        # Case where structs are in a list
        elif isinstance(structs, list):

            # Special case: pairs of structure sources for comparison
            input_is_pair = [core.is_list(s) and len(s) == 2 for s in structs]
            if all(input_is_pair):
                self.load_struct_pairs(structs, names, colors)
                return
            elif any(input_is_pair):
                raise TypeError

            # Put list of sources into a dictionary
            struct_dict[""] = structs

        # Single structure source
        else:
            struct_dict[""] = [structs]

        # Load all structs in the final dict
        for label, paths in struct_dict.items():
            for p in paths:
                if isinstance(p, str) and p.startswith("multi:"):
                    self.load_structs_from_file(p[6:], label, names, colors,
                                                True)
                else:
                    self.load_structs_from_file(p, label, names, colors, multi)

    def load_structs_from_file(self, paths, label, names, colors, multi=False):
        """Search for filenames matching <paths> and load structs from all
        files."""

        # Get files
        if isinstance(paths, str):
            files = core.find_files(paths)
        else:
            files = paths

        # Get colors and names dicts
        if core.is_nested(colors):
            colors = colors.get(label, {})
        if core.is_nested(names):
            names = names.get(label, {})

        # Load each file
        for f in files:
            self.add_struct_file(f, label, names, colors, multi)

    def find_name_match(self, names, path):
        """Assign a name to a structure based on its path."""

        # Infer name from filepath
        name = None
        if isinstance(path, str):
            basename = os.path.basename(path).replace(".gz", "").replace(".nii", "")
            name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "",
                               basename).replace(" ", "_")

            # See if we can convert this name based on names list
            for name2, matches in names.items():
                if not core.is_list(matches):
                    matches = [matches]
                for match in matches:
                    if fnmatch.fnmatch(standard_str(name), standard_str(match)):
                        return name2

        # See if we can get name from filepath
        for name2, matches in names.items():
            if not core.is_list(matches):
                matches = [matches]
            for match in matches:
                if fnmatch.fnmatch(standard_str(path), standard_str(match)):
                    return name2

        return name

    def find_color_match(self, colors, name):
        """Find the first color in a color dictionary that matches a given
        structure name."""

        for comp_name, color in colors.items():
            if fnmatch.fnmatch(standard_str(name), standard_str(comp_name)):
                return color

    def keep_struct(self, name):
        """Check whether a structure with a given name should be kept or 
        ignored."""

        keep = True
        if self.to_keep is not None:
            if not any([fnmatch.fnmatch(standard_str(name), standard_str(k)) 
                        for k in self.to_keep]):
                keep = False
        if self.to_ignore is not None:
            if any([fnmatch.fnmatch(standard_str(name), standard_str(i)) 
                    for i in self.to_ignore]):
                keep = False
        return keep

    def add_struct_file(self, path, label, names, colors, multi=False):
        """Create StructImage object(s) from file and add to list."""

        self.loaded = False

        # Try loading from a DICOM file
        if self.load_structs_from_dicom(path, label, names, colors):
            return

        # Get custom name
        if isinstance(path, str):
            name = self.find_name_match(names, path)
        else:
            name = f"Structure {len(self.structs) + 1}"

        # Keep or ignore

        # Only one structure per file
        if not multi:
            if self.keep_struct(name):
                struct = StructImage(path, label=label, name=name, load=False, 
                                     **self.struct_kwargs)
                color = self.find_color_match(colors, struct.name)
                if color is not None:
                    struct.assign_color(color)
                self.structs.append(struct)
            return

        # Search for many label masks in one file
        # Load the data
        data, voxel_sizes, origin, path = core.load_image(path)
        kwargs = self.struct_kwargs.copy()
        kwargs.update({"voxel_sizes": voxel_sizes, "origin": origin})
        mask_labels = np.unique(data).astype(int)
        mask_labels = mask_labels[mask_labels != 0]

        # Case with only one structure in that file
        if len(mask_labels) < 2:
            if self.keep_struct(name):
                struct = StructImage(path, label=label, name=name, load=False, 
                                     **self.struct_kwargs)
                color = self.find_color_match(colors, struct.name)
                if color is not None:
                    struct.assign_color(color)
                self.structs.append(struct)
            return

        # Load multiple massk
        for ml in mask_labels:

            name = self.find_name_match(names, ml)
            if name is None:
                name = f"Structure {ml}"
            if self.keep_struct(name):
                color = self.find_color_match(colors, name)

                struct = StructImage(data == ml, name=name, label=label, 
                                     color=color, **kwargs)
                struct.path = path
                self.structs.append(struct)

    def load_struct_pairs(self, structs, names, colors):
        """Load structs from pairs and create a StructComparison for each."""

        self.loaded = False
        for pair in structs:
            s_pair = []
            for path in pair:
                name = self.find_name_match(names, path)
                if not self.keep_struct(name):
                    return
                color = self.find_color_match(colors, name)
                s_pair.append(StructImage(path, name=name, color=color, 
                                          load=False, **self.struct_kwargs))

            self.structs.extend(s_pair)
            self.comparison_structs.extend(s_pair)
            self.comparisons.append(StructComparison(*s_pair))

    def load_structs_from_dicom(self, path, label, names, colors):
        """Attempt to load structures from a DICOM file."""

        try:
            structs = dicom.load_structs(path)
        except TypeError:
            return

        if not self.image:
            raise RuntimeError("Must provide the <image> argument to "
                               "StructLoader in order to load from DICOM!")

        # Load each structure
        for struct in structs.values():

            # Get settings for this structure
            name = self.find_name_match(names, struct["name"])
            if not self.keep_struct(name):
                continue
            color = self.find_color_match(colors, name)
            if color is None:
                color = struct["color"]
            contours = struct["contours"]

            # Adjust contours
            for z, conts in contours.items():
                for c in conts:
                    c[:, 0] -= self.image.voxel_sizes["x"] / 2
                    c[:, 1] += self.image.voxel_sizes["y"] / 2

            # Create structure
            struct = StructImage(contours=contours, label=label, name=name, 
                                 load=False, color=color, 
                                 shape=self.image.data.shape,
                                 origin=self.image.origin,
                                 voxel_sizes=self.image.voxel_sizes,
                                 **self.struct_kwargs)
            self.structs.append(struct)

        return True

    def find_comparisons(self):
        """Find structures suitable for comparison and make a list of 
        StructComparison objects."""
        
        if len(self.comparisons) and self.loaded:
            return

        # Match each to all others
        if self.comp_type == "others":
            for i, s in enumerate(self.structs):
                others = [self.structs[j] for j in range(len(self.structs))
                          if j != i]
                self.comparisons.append(StructComparison(
                    s, others, comp_type=self.comp_type))
            self.comparison_structs = self.structs
            return

        # Case with only two structures
        if len(self.structs) == 2:
            self.comparisons.append(StructComparison(*self.structs))
            self.comparison_structs = self.structs
            return

        # Look for structures with matching names
        use_pairs = False
        n_per_name = {}
        if self.comp_type == "auto":
            unique_names = set([s.name for s in self.structs])
            n_per_name = {n: len([s for s in self.structs if s.name == n])
                          for n in unique_names}
            if max(n_per_name.values()) != 2:
                use_pairs = True

        # Match all pairs
        if self.comp_type == "pairs" or use_pairs:
            for i, s1 in enumerate(self.structs):
                for s2 in self.structs[i + 1:]:
                    self.comparisons.append(StructComparison(s1, s2))
            self.comparison_structs = self.structs
            return

        # Make structure comparisons
        names_to_compare = [name for name in n_per_name 
                            if n_per_name[name] == 2]
        for name in names_to_compare:
            structs = [s for s in self.structs if s.name == name]
            self.comparisons.append(
                StructComparison(*structs, name=structs[0].name_nice_nolabel))
            self.comparison_structs.extend(structs)

    def set_unique_name(self, struct):
        """Create a unique name for a structure with respect to all other
        loaded structures."""

        if struct.path is None or struct.label:
            struct.name_unique = struct.name_nice
            return

        # Find structures with the same name
        same_name = [s for s in self.structs if standard_str(s.name) ==
                     standard_str(struct.name) and s != struct]
        if not len(same_name):
            struct.name_unique = struct.name_nice
            return

        # Get unique part of path wrt those structures
        unique_paths = list(set([core.get_unique_path(struct.path, s.path) 
                                 for s in same_name]))

        # If path isn't unique, just use own name
        if None in unique_paths:
            struct.name_unique = struct.name_nice

        elif len(unique_paths) == 1:
            struct.name_unique = f"{struct.name_nice} ({unique_paths[0]})"

        else:

            # Find unique path wrt all paths
            remaining = unique_paths[1:]
            current = core.get_unique_path(unique_paths[0], remaining)
            while len(remaining) > 1:
                remaining = remaining[1:]
                current = core.get_unique_path(current, remaining[0])
            struct.name_unique = f"{struct.name_nice} ({current})"

    def load_all(self):
        """Load all structures and assign custom colours and unique names."""

        if self.loaded:
            return

        # Assign colors
        for i, s in enumerate(self.structs):
            if not s.custom_color_set:
                s.assign_color(standard_colors[i])

        for s in self.structs:
            s.load()
            self.set_unique_name(s)

        self.loaded = True

    def reassign_colors(self, colors):
        """Reassign colors such that any structures in the <colors> dict are
        given that color, and any not in the dict are given unique colors and 
        added to it."""

        self.load_all()

        for s in self.structs:
            if s.name_unique in colors:
                s.assign_color(colors[s.name_unique])
            else:
                color = standard_colors[len(colors)]
                s.assign_color(color)
                colors[s.name_unique] = color
        return colors

    def get_structs(self, ignore_unpaired=False, ignore_empty=False, 
                    sort=False):
        """Get list of all structures. If <ignore_unpaired> is True, only 
        structures that are part of a comparison pair will be returned."""

        self.load_all()
        s_list = self.structs
        if sort:
            s_list = sorted(s_list)
        if ignore_unpaired:
            self.find_comparisons()
            s_list = self.comparison_structs
        if ignore_empty:
            return [s for s in s_list if not s.empty]
        else:
            return s_list

    def get_comparisons(self, ignore_empty=False):
        """Get list of StructComparison objects."""

        self.load_all()
        self.find_comparisons()
        if ignore_empty:
            return [c for c in self.comparisons if not c.s1.empty 
                    or c.s2.empty]
        else:
            return self.comparisons

    def get_standalone_structs(self, ignore_unpaired=False, ignore_empty=False):
        """Get list of the structures that are not part of a comparison 
        pair."""

        if ignore_unpaired:
            return []

        self.load_all()
        self.find_comparisons()
        standalones = [s for s in self.structs if s not in 
                       self.comparison_structs]
        if ignore_empty:
            return [s for s in standalones if not s.empty]
        else:
            return standalones
