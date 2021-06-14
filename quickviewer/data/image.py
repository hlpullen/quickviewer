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


_axes = ['x', 'y', 'z']
_slice_axes = {
    'x-y': 2,
    'y-z': 0,
    'x-z': 1
}
_plot_axes = {
    'x-y': [0, 1],
    'y-z': [2, 1],
    'x-z': [2, 0]
}
_default_figsize = 6
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [1, 0, 2]}
_n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}


class Image:
    '''Loads and stores a medical image and its geometrical properties, either 
    from a dicom/nifti file or a numpy array.'''

    def __init__(
        self,
        source,
        load=True,
        title=None,
        affine=None,
        voxel_size=(1, 1, 1),
        origin=(0, 0, 0),
        downsample=None
    ):
        '''
        Initialise from a medical image source.

        Parameters
        ----------
        source : str/array/Nifti1Image
            Source of image data. Can be either:
                (a) A string containing the path to a dicom or nifti file;
                (b) A string containing the path to a numpy file containing a
                    2D or 3D array;
                (c) A 2D or 3D numpy array;
                (d) A nibabel.nifti1.Nifti1Image object.

        load : bool, default=True
            If True, the image data will be immediately loaded. Otherwise, it
            can be loaded later with the load_data() method.

        title : str, default=None
            Title to use when plotting the image. If None and <source> is a 
            path, a title will be automatically generated from the filename.

        affine : 4x4 array, default=None
            Array containing the affine matrix to use if <source> is a numpy
            array or path to a numpy file. If not None, this takes precendence 
            over <voxel_size> and <origin>.

        voxel_size : tuple, default=(1, 1, 1)
            Voxel sizes in mm in order (x, y, z) to use if <source> is a numpy
            array or path to a numpy file and <affine> is not provided.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm in order (x, y, z) to use if <source> is a 
            numpy array or path to a numpy file and <affine> is not provided.
        '''

        self.data = None
        self.title = title
        self.source = source
        self.affine = affine
        self.voxel_size = voxel_size
        self.origin = origin
        self.downsampling = downsample
        self.valid = False
        if load:
            self.load_data()

        self.mask = None
        self.mask_per_view = None
        self.shift = None

    def get_data(self):
        '''Return image array.'''

        if self.data is None:
            self.load_data()
        return self.data

    def get_masked_data(self, view=None):
        '''Return image array with mask applied, if self.mask is not None.'''

        self.load_data()
        if self.mask:
            return self.data * self.mask
        elif self.mask_per_view and view in self.mask_per_view:
            return self.data * self.mask_per_view[view]
        return self.data

    def load_data(self, force=False):
        '''Load pixel array from image source.'''

        if self.data is not None and not force:
            return

        window_width = None
        window_centre = None

        # Load image array from source
        # Numpy array
        if isinstance(self.source, np.ndarray):
            self.data = self.source

        # Try loading from nifti file
        else:
            if not os.path.exists(self.source):
                raise RuntimeError(f'Image input {self.source} does not exist!')
            if os.path.isfile(self.source):
                self.data, self.affine = load_nifti(self.source)

        # Try loading from dicom file
        if self.data is None:
            self.data, self.affine, window_centre, window_width \
                    = load_dicom(self.source)

        # Try loading from numpy file
        if self.data is None:
            self.data = load_npy(self.source)

        # Ensure array is 3D
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]

        # Apply downsampling
        if self.downsampling:
            self.downsample(self.downsampling)
        else:
            self.set_geometry()

        # Set default grayscale range
        if window_width and window_centre:
            self.default_window = [
                window_centre - window_width / 2,
                window_centre + window_width / 2
            ]
        else:
            self.default_window = [-300, 200] 

        # Set title from filename
        if self.title is None:
            if os.path.exists(self.source):
                self.title = os.path.basename(self.source)

        if self.data is not None:
            self.valid = True

    def set_geometry(self):
        '''Set geometric properties.'''
    
        # Set affine matrix, voxel sizes, and origin
        if self.affine is None:
            self.affine = np.array([
                [self.voxel_size[0], 0, 0, self.origin[0]],
                [0, self.voxel_size[1], 0, self.origin[1]],
                [0, 0, self.voxel_size[2], self.origin[2]],
                [0, 0, 0, 1]
            ])
        else:
            self.voxel_size = list(np.diag(self.affine)[:-1])
            self.origin = list(self.affine[:-1, -1])

        # Set other geometric properties
        self.n_voxels = self.data.shape
        self.lims = [
            (self.origin[i], 
             self.origin[i] + (self.n_voxels[i] - 1) * self.voxel_size[i])
            for i in range(3)
        ]
        self.image_extent = [
            (self.lims[i][0] - self.voxel_size[i] / 2,
             self.lims[i][1] + self.voxel_size[i] / 2)
            for i in range(3)
        ]

    def plot(
        self, 
        view='x-y', 
        sl=None, 
        idx=None, 
        pos=None,
        scale_in_mm=True,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        zoom=None,
        zoom_centre=None,
        mpl_kwargs=None,
        show=True,
        colorbar=False,
        colorbar_label='HU',
        masked=False,
        invert_mask=False,
        mask_color='black',
        no_title=False,
        no_ylabel=False,
        annotate_slice=False,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False
    ):
        '''Plot a 2D slice of the image.

        Parameters
        ----------
        view : str, default='x-y'
            Orientation in which to plot the image. Can be any of 'x-y', 
            'y-z', and 'x-z'.

        sl : int, default=None
            Slice number to plot. Takes precedence over <idx> and <pos> if not
            None. If all of <sl>, <idx>, and <pos> are None, the central 
            slice will be plotted.

        idx : int, default=None
            Index of the slice in the array to plot. Takes precendence over 
            <pos>.

        pos : float, default=None
            Position in mm of the slice to plot. Will be rounded to the nearest
            slice. Only used if <sl> and <idx> are both None.

        scale_in_mm : bool, default=True
            If True, axis labels will be in mm; otherwise, they will be slice 
            numbers.

        ax : matplotlib.pyplot.Axes, default=None
            Axes on which to plot. If None, new axes will be created.

        gs : matplotlib.gridspec.GridSpec, default=None
            If not None and <ax> is None, new axes will be created on the
            current matplotlib figure with this gridspec.

        figsize : float, default=None
            Figure height in inches; only used if <ax> and <gs> are None.

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

        colorbar_label : str, default='HU'
            Label for the colorbar, if drawn.

        masked : bool, default=False
            If True and this object has attribute self.data_mask assigned,
            the image will be masked with the array in self.data_mask.

        invert_mask : bool, default=True
            If True and a mask is applied, the mask will be inverted.

        mask_color : matplotlib color, default='black'
            color in which to plot masked areas.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        show : bool, default=True
            If True, the plotted figure will be shown via
            matplotlib.pyplot.show().

        no_title : bool, default=False
            If True, the plot will not be given a title.

        no_ylabel : bool, default=False
            If True, the y axis will not be labelled.

        annotate_slice : bool/str, default=False
            Color for annotation of slice number. If False, no annotation will
            be added. If True, the default color (white) will be used.

        major_ticks : float, default=None
            If not None, this value will be used as the interval between major
            tick marks. Otherwise, automatic matplotlib axis tick spacing will
            be used.

        minor_ticks : int, default=None
            If None, no minor ticks will be plotted. Otherwise, this value will
            be the number of minor tick divisions per major tick interval.

        ticks_all_sides : bool, default=False
            If True, major (and minor if using) tick marks will be shown above
            and to the right hand side of the plot as well as below and to the
            left. The top/right ticks will not be labelled.
        '''

        # Set up figure/axes
        if ax is None and  gs is not None:
            ax = plt.gcf().add_subplot(gs)
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            figsize = to_inches(figsize)
            aspect = self.get_plot_aspect_ratio(
                view, zoom, colorbar, figsize
            )
            self.fig = plt.figure(figsize=(figsize * aspect, figsize))
            self.ax = self.fig.add_subplot()

        # Get index of the slice to plot
        self.load_data()
        if sl is not None:
            idx = self.slice_to_idx(sl, _slice_axes[view])
        elif idx is None:
            if pos is not None:
                idx = self.pos_to_idx(pos, _slice_axes[view])
            else:
                centre_pos = self.get_image_centre()[_slice_axes[view]]
                idx = self.pos_to_idx(centre_pos, _slice_axes[view])

        # Get image slice
        transpose = {
            'x-y': (0, 1, 2),
            'y-z': (0, 2, 1),
            'x-z': (1, 2, 0)
        }[view]
        list(_plot_axes[view]) + [_slice_axes[view]]
        image_slice = np.transpose(self.data, transpose)[:, :, idx]

        # Apply masking if needed
        if masked and self.mask:
            mask = self.mask if isinstance(self.mask, np.ndarray) \
                    else self.mask[view]
            mask_slice = np.transpose(mask, transpose)[:, :, idx]
            if invert_mask:
                mask_slice = ~mask_slice
            image_slice *= mask_slice

        # Get colormap
        mpl_kwargs = {} if mpl_kwargs is None else mpl_kwargs
        if 'cmap' in mpl_kwargs:
            cmap = mpl_kwargs.pop('cmap')
        else:
            cmap = 'gray'
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap))
        if masked:
            cmap.set_bad(color=mask_color)

        # Get image extent and aspect ratio
        x_ax, y_ax = _plot_axes[view]
        extent = self.image_extent[x_ax] + self.image_extent[y_ax][::-1]
        if scale_in_mm:
            aspect = 1
        else:
            extent = [
                self.pos_to_slice(extent[0], x_ax, False),
                self.pos_to_slice(extent[1], x_ax, False),
                self.pos_to_slice(extent[2], y_ax, False),
                self.pos_to_slice(extent[3], y_ax, False)
            ]
            aspect = abs(self.voxel_size[y_ax] / self.voxel_size[x_ax])

        # Plot the slice
        mesh = self.ax.imshow(
            image_slice, 
            cmap=cmap,
            extent=extent,
            aspect=aspect,
            vmin=self.default_window[0],
            vmax=self.default_window[1],
            **mpl_kwargs
        )

        # Set title 
        if self.title and not no_title:
            self.ax.set_title(self.title, pad=8)

        # Set axis labels
        units = ' (mm)' if scale_in_mm else ''
        self.ax.set_xlabel(_axes[x_ax] + units, labelpad=0)
        if not no_ylabel:
            self.ax.set_xlabel(_axes[x_ax] + units, labelpad=0)
        else:
            self.ax.set_yticks([])

        # Annotate with slice position
        if annotate_slice:
            z_ax = _axes[_slice_axes[view]]
            if scale_in_mm:
                z_str = '{} = {:.1f} mm'.format(z_ax, self.idx_to_pos(idx, z_ax))
            else:
                z_str = '{} = {}'.format(z_ax, self.idx_to_slice(idx, z_ax))
            if matplotlib.colors.is_color_like(annotate_slice):
                color = annotate_slice
            else:
                color = 'white'
            self.ax.annotate(z_str, xy=(0.05, 0.93), xycoords='axes fraction',
                             color=color, fontsize='large')

        # Adjust tick marks
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
                    which='minor', bottom=True, top=True, left=True, right=True
                )

        # Add colorbar
        if colorbar and mpl_kwargs.get('alpha', 1) > 0:
            clb = self.fig.colorbar(mesh, ax=self.ax, label=colorbar_label)
            clb.solids.set_edgecolor('face')

        # Display image
        if show:
            plt.tight_layout()
            plt.show()

    def idx_to_pos(self, idx, ax):
        '''Convert an array index to a position in mm along a given axis.'''

        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        return self.origin[i_ax] + idx * self.voxel_size[i_ax]
        #  return self.origin[i_ax] + (self.n_voxels[i_ax] - 1 - idx) \
                #  * self.voxel_size[i_ax]

    def pos_to_idx(self, pos, ax, return_int=True):
        '''Convert a position in mm to an array index along a given axis.'''

        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        #  idx = self.n_voxels[i_ax] - 1 + (self.origin[i_ax] - pos) \
                #  / self.voxel_size[i_ax]
        idx = (pos - self.origin[i_ax]) / self.voxel_size[i_ax]
        if return_int:
            return round(idx)
        else:
            return idx

    def idx_to_slice(self, idx, ax):
        '''Convert an array index to a slice number along a given axis.'''
        
        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        if i_ax == 2:
            return self.n_voxels[i_ax] - idx
        else:
            return idx + 1

    def slice_to_idx(self, sl, ax):
        '''Convert a slice number to an array index along a given axis.'''

        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        if i_ax == 2:
            return self.n_voxels[i_ax] - sl
        else:
            return sl - 1

    def pos_to_slice(self, pos, ax, return_int=True):
        '''Convert a position in mm to a slice number along a given axis.'''

        sl = self.idx_to_slice(self.pos_to_idx(pos, ax, return_int), ax)
        if return_int:
            return round(sl)
        else:
            return sl

    def slice_to_pos(self, sl, ax):
        '''Convert a slice number to a position in mm along a given axis.'''

        return self.idx_to_pos(self.slice_to_idx(sl, ax), ax)

    def get_image_centre(self):
        '''Get position in mm of the centre of the image.'''

        return [np.mean(self.lims[i]) for i in range(3)]

    def get_voxel_coords(self):
        '''Get arrays of voxel coordinates in each direction.'''

        return

    def get_plot_aspect_ratio(self, view, zoom=None, n_colorbars=0,
                              figsize=_default_figsize):
        '''Estimate the ideal width/height ratio for a plot of this image 
        in a given orientation.'''

        # Get length of the image in the plot axis directions
        x_ax, y_ax = _plot_axes[view]
        x_len = abs(self.lims[x_ax][1] - self.lims[x_ax][0])
        y_len = abs(self.lims[y_ax][1] - self.lims[y_ax][0])

        # Add padding for axis labels and title
        font = mpl.rcParams['font.size'] / 72
        y_pad = 2 * font
        if self.title:
            y_pad += 1.5 * font
        max_y_digits = np.floor(np.log10(
            max([abs(lim) for lim in self.lims[y_ax]])
        ))
        minus_sign = any([lim < 0 for lim in self.lims[y_ax]])
        x_pad = (0.7 * max_y_digits + 1.2 * minus_sign + 1) * font

        # Account for zoom

        # Add padding for colorbar(s)
        colorbar_frac = 0.4 * 5 / figsize
        x_len *= 1 + (n_colorbars * colorbar_frac)

        # Return estimated width ratio
        total_y = figsize + y_pad
        total_x = figsize * x_len / y_len + x_pad
        return total_x / total_y

    def downsample(self, downsampling):
        '''Apply downsampling to the image array. Can be either a single
        value (to downsampling equally in all directions) or a list of 3 
        values.'''

        # Get downsampling in each direction
        if is_list(downsampling):
            if len(downsampling) != 3:
                raise TypeError('<downsample> must contain 3 elements!')
            dx, dy, dz = downsampling
        else:
            dx = dy = dz = downsampling

        # Apply to image array
        self.data = downsample(self.data, dx, dy, dz)

        # Adjust voxel sizes
        self.voxel_size = [
            v * d for v, d in zip(self.voxel_size, [dx, dy, dz])
        ]
        self.affine = None

        # Reset geometric properties of this image
        self.set_geometry()

    def translate(self, dx=0, dy=0, dz=0):
        '''Apply a translation to the image data.'''

        return

    def rotate(self, yaw=0, pitch=0, roll=0):
        '''Apply a rotation to the image data.'''

        return

    def reset(self):
        '''Return image data to its original state before any translations or
        rotations.'''

        return

    def set_shift(self, nx, ny, nz):
        '''Set a shift amount in voxels to apply to the image before plotting.
        (Faster than translating the entire 3D image).
        '''

        return

    def set_mask(self, mask, threshold=0.5):
        '''Set a mask that can be applied to the image at plotting time.
        Can be either a single array, list of arrays to stack, or dict of 
        arrays for different orientations.'''
 
        # Get single mask or list of masks
        single_mask = None
        mask_per_view = None
        if isinstance(mask, np.ndarray):
            single_mask = mask
        elif is_list(mask):
            single_mask = mask[0]
            for m in mask[1:]:
                single_mask += m
        elif isinstance(mask, dict):
            mask_per_view = mask

        # Convert mask(s) to boolean
        if single_mask:
            single_mask = single_mask > threshold
        if mask_per_view:
            for view, m in mask_per_view.items():
                mask_per_view[view] = m > threshold

        # Assign to class properties
        self.mask = single_mask
        self.mask_per_view = mask_per_view

    def write(self, outname):
        '''Write image data to a file.'''

        # Write to nifti file
        if outname.endswith('.nii') or outname.endswith('.nii.gz'):
            pass

        # Write to numpy file
        elif outname.endswith('.npy'):
            pass

    def get_coords(self):
        '''Get grids of x, y, and z coordinates for each voxel in the image.'''

        if not hasattr(self, 'coords'):

            # Make coordinates
            coords_1d = {}
            for i, ax in enumerate(_axes):
                coords_1d[ax] = np.arange(*self.lims[i], self.voxel_size[i])
            X, Y, Z = np.meshgrid(coords_1['x'], coords_1d['y'], 
                                  coords_1d['z'])

            # Set coords
            self.coords = (X, Y, Z)

        # Apply transformations
        return self.coords


class ImageComparison(Image):
    """Class for loading data from two arrays and plotting comparison images."""

    def __init__(self, im1, im2, title=None, plot_type=None, **kwargs):
        """Load data from two arrays. <im1> and <im2> can either be existing
        Image objects, or objects from which Images can be created.
        """

        # Load Images
        self.ims = []
        self.standalone = True
        for im in [im1, im2]:

            # Load existing Image
            if issubclass(type(im), Image):
                self.ims.append(im)

            # Create new Image
            else:
                self.standalone = False
                self.ims.append(Image(im, **kwargs))

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


def load_nifti(path):
    '''Load an image from a nifti file.'''

    try:
        nii = nibabel.load(path)
        data = nii.get_fdata().transpose(1, 0, 2)[::-1, :, :]
        affine = nii.affine
        return data, affine

    except FileNotFoundError:
        print(f'Warning: file {path} not found! Could not load nifti.')
        return None, None

    except nibabel.filebasedimages.ImageFileError:
        return None, None


def load_dicom(path):
    '''Load a dicom image from one or more dicom files.'''

    # Try loading single dicom file
    paths = []
    if os.path.isfile(path):
        try:
            ds = pydicom.read_file(path)
            if ds.get('ImagesInAcquisition', None) == 1:
                paths = [path]
        except pydicom.errors.InvalidDicomError:
            return None, None

    # Case where there are multiple dicom files for this image
    if not paths:
        if os.path.isdir(path):
            dirname = path
        else:
            dirname = os.path.dirname(path)
        paths = sorted([os.path.join(dirname, p) for p in os.listdir(dirname)])

        # Ensure user-specified file is loaded first
        if path in paths:
            paths.insert(0, paths.pop(paths.index(path)))

    # Load image arrays from all files
    study_uid = None
    series_num = None
    modality = None
    slice_thickness = None
    pixel_size = None
    origin = None
    rescale_slope = None
    rescale_intercept = None
    window_centre = None
    window_width = None
    data_slices = {}
    for dcm in paths:
        try:
            
            # Load file and check it matches the others
            ds = pydicom.read_file(dcm)
            if study_uid is None:
                study_uid = ds.StudyInstanceUID
            if series_num is None:
                series_num = ds.SeriesNumber
            if modality is None:
                modality = ds.Modality
            if (ds.StudyInstanceUID != study_uid 
                or ds.SeriesNumber != series_num
                or ds.Modality != modality):
                continue

            # Get data 
            z = getattr(ds, 'ImagePositionPatient', [0, 0, 0])[2]
            data_slices[z] = ds.pixel_array

            # Get voxel spacings
            if pixel_size is None:
                for attr in ['PixelSpacing', 'ImagerPixelSpacing']:
                    pixel_size = getattr(ds, attr, None)
                    if pixel_size:
                        break
            if slice_thickness is None:
                slice_thickness = getattr(ds, 'SliceThickness', None)

            # Get origin
            if origin is None:
                origin = ds.ImagePositionPatient

            # Get rescale settings
            if rescale_slope is None:
                rescale_slope = getattr(ds, 'RescaleSlope', None)
            if rescale_intercept is None:
                rescale_intercept = getattr(ds, 'RescaleIntercept', None)

            # Get HU window defaults
            if window_centre is None:
                window_centre = getattr(ds, 'WindowCenter', None)
            if window_width is None:
                window_width = getattr(ds, 'WindowWidth', None)

        # Skip any invalid dicom files
        except pydicom.errors.InvalidDicomError:
            continue

    # Case where no data was found
    if not data_slices:
        print(f'Warning: no valid dicom files found in {path}')
        return None, None

    # Case with single image array
    if len(data_slices) == 1:
        data = list(data_slices.values())[0]

    # Combine arrays
    else:

        # Sort by slice position
        sorted_slices = sorted(list(data_slices.keys()))
        sorted_data = [data_slices[z] for z in sorted_slices]
        data = np.stack(sorted_data, axis=-1)

        # Recalculate slice thickness from spacing
        slice_thickness = sorted_slices[1] - sorted_slices[0]

        # z origin = lowest z (idx=0)
        origin[2] = sorted_slices[0]

    # Make affine matrix
    affine =  np.array([
        [pixel_size[0], 0, 0, origin[0]],
        [0, pixel_size[1], 0, origin[1]],
        [0, 0, slice_thickness, origin[2]],
        [0, 0, 0, 1]
    ])

    return data, affine, window_centre, window_width


def load_npy(path):
    '''Load a numpy array from a .npy file.'''

    try:
        data = np.load(path)
        return data

    except (IOError, ValueError):
        return


def is_list(var):
    '''Check whether a variable is a list, tuple, or array.'''

    is_a_list = False
    for t in [list, tuple, np.ndarray]:
        if isinstance(var, t):
            is_a_list = True
    return is_a_list


def downsample(data, dx=None, dy=None, dz=None):
    '''Downsample an array by the factors specified in <dx>, <dy>, and <dz>.
    '''

    if dx is None:
        dx = 1
    if dy is None:
        dy = 1
    if dx is None:
        dz = 1

    return data[::round(dy), ::round(dx), ::round(dz)]


def to_inches(size):
    '''Convert a size string to a size in inches. If a float is given, it will
    be returned. If a string is given, the last two characters will be used to
    determine the units:
        - 'in': inches
        - 'cm': cm
        - 'mm': mm
        - 'px': pixels
    '''

    if not isinstance(size, str):
        return size

    val = float(size[:-2])
    units = size[-2:]
    inches_per_cm = 0.394
    if units == 'in':
        return val
    elif units == 'cm':
        return inches_per_cm * val
    elif units == 'mm':
        return inches_per_cm * val / 10
    elif units == 'px':
        return val / mpl.rcParams['figure.dpi']


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
                            "Input file <im> must be a valid "
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

