"""Classes for plotting images from NIfTI files or arrays."""

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


# Global properties
_axes = {"x": 1, "y": 0, "z": 2}
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [0, 1, 2]}
_n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}

# Plotting properties
_df_plot_types = ["grid", "quiver", "none"]
_struct_plot_types = ["contour", "mask", "none"]
_default_figsize = 6


class NiftiImage:
    """Class to hold properties of an image array with an affine matrix."""

    def __init__(
        self,
        nii,
        affine=None,
        voxel_sizes=(1, 1, 1),
        origin=(0, 0, 0),
        title=None,
        scale_in_mm=True
    ):
        """Initialise from a nifti file, nifti object, or numpy array.

        Parameters
        ----------
        nii : str/array/nifti
            Source of the nifti to load. This can be either:
                (a) A string containing the path to a nifti file;
                (b) A nibabel.nifti1.Nifti1Image object;
                (c) A numpy array;
                (d) A numpy file containing an array.

        affine : 4x4 array, default=None
            Affine matrix to be used if <nii> is a numpy array. If <nii> is a
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
            Title to use when plotting. If None, the filename will be used.

        scale_in_mm : bool, default=True
            If True, image will be plotted in mm rather than voxels.
        """

        # Assign settings
        self.title = title
        self.scale_in_mm = scale_in_mm
        self.data_mask = None
        if nii is None:
            self.valid = False
            return

        # Load from numpy array
        if isinstance(nii, np.ndarray):
            self.data = nii

        # Load from file or nifti object
        else:
            if isinstance(nii, str):
                path = os.path.expanduser(nii)
                try:
                    self.nii = nibabel.load(path)
                    self.data = self.nii.get_fdata()
                    affine = self.nii.affine
                    if self.title == "":
                        self.title = os.path.basename(path)
                except FileNotFoundError:
                    self.valid = False
                    return
                except nibabel.filebasedimages.ImageFileError:
                    try:
                        self.data = np.load(path)
                    except (IOError, ValueError):
                        raise RuntimeError("Input file <nii> must be a valid "
                                           ".nii or .npy file.")
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
        if affine is not None:
            self.voxel_sizes = {ax: affine[n, n] for ax, n in _axes.items()}
            self.origin = {ax: affine[n, 3] for ax, n in _axes.items()}
        else:
            self.voxel_sizes = {ax: voxel_sizes[n] for ax, n in _axes.items()}
            self.origin = {ax: origin[n] for ax, n in _axes.items()}
        self.set_geom()
        self.set_plotting_defaults()

    def set_geom(self):
        """Assign geometric properties based on image data, origin, and
        voxel sizes."""

        self.n_voxels = {ax: self.data.shape[n] for ax, n in _axes.items()}
        self.lims = {
            ax: (self.origin[ax], self.idx_to_pos(self.n_voxels[ax], ax))
            for ax in _axes
        }
        self.extent = {}
        self.aspect = {}
        for view, (x, y) in _plot_axes.items():
            if self.scale_in_mm:
                self.extent[view] = (
                    min(self.lims[x]), max(self.lims[x]),
                    max(self.lims[y]), min(self.lims[y])
                )
                self.aspect[view] = 1
            else:
                self.extent[view] = None
                self.aspect[view] = abs(self.voxel_sizes[y] /
                                        self.voxel_sizes[x])

    def same_frame(self, im):
        """Compare own image to another NiftiImage; check whether the frame of 
        reference is the same."""

        same = self.shape == im.shape
        same *= list(self.origin.values()) == list(im.origin.values())
        same *= list(self.voxel_sizes.values()) == list(im.voxel_sizes.values())
        return same

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments."""

        self.mpl_kwargs = {"cmap": "gray",
                           "vmin": -300,
                           "vmax": 200}
        self.mask_colour = "black"

    def get_relative_width(self, view, n_colorbars=0):
        """Get width:height ratio for this plot in a given orientation with a
        given number of colorbars."""

        x, y = _plot_axes[view]
        x_length = abs(self.lims[x][0] - self.lims[x][1])
        y_length = abs(self.lims[y][0] - self.lims[y][1])
        width = x_length / y_length
        colorbar_frac = 0.3
        width *= 1 + n_colorbars * colorbar_frac / width
        return width

    def set_ax(self, view, ax=None, figsize=None, n_colorbars=0):
        """Assign an axis to self, or create new axis if needed."""

        # Assign external axes
        if ax is not None:
            self.fig = ax.figure
            self.ax = ax
            return

        # Create figure and axes
        rel_width = self.get_relative_width(view, n_colorbars)
        figsize = _default_figsize if figsize is None else figsize
        self.fig = plt.figure(figsize=(figsize * rel_width, figsize))
        self.ax = self.fig.add_subplot()

    def get_kwargs(self, mpl_kwargs, default=None):
        """Return a dict of matplotlib keyword arguments, combining default
        values with custom values."""

        if default is None:
            custom_kwargs = self.mpl_kwargs.copy()
        else:
            custom_kwargs = default.copy()
        if mpl_kwargs is not None:
            custom_kwargs.update(mpl_kwargs)
        return custom_kwargs

    def idx_to_pos(self, idx, ax):
        """Convert an index to a position along a given axis."""

        return self.origin[ax] + idx * self.voxel_sizes[ax]

    def pos_to_idx(self, pos, ax):
        """Convert a position to an index along a given axis."""

        return round((pos - self.origin[ax]) / self.voxel_sizes[ax])

    def set_mask(self, mask):
        """Set the mask for this image. The mask will be used when self.plot()
        is called with masked=True."""

        if not self.valid:
            return
        if mask is None:
            self.data_mask = None
            return
        elif isinstance(mask, np.ndarray):
            data_mask = mask
        elif isinstance(mask, NiftiImage):
            if not mask.valid:
                self.data_mask = None
                return
            data_mask = mask.data
        else:
            raise TypeError("Mask must be a numpy array or a NiftiImage.")

        # Check mask shape is consistent
        if data_mask.shape != self.data.shape:

            # Downsample mask if it matches original image shape
            if data_mask.shape == self.shape:
                self.data_mask = self.downsample_array(data_mask)
            else:
                print("Warning: mask does not match shape of image!")

        else:
            self.data_mask = data_mask

    def get_slice(self, view, sl, masked=False, invert_mask=False):
        """Get 2D array corresponding to a slice of the image in a given
        orientation.

        Parameters
        ----------
        view : str
            Orientation to use. Must be one of "x-y", "y-z", and "x-z".

        sl : int
            Index of the slice to use.

        masked : bool, default=False
            If True and the "data_mask" attribute is not None, a mask will be
            applied.
        """

        # Apply mask if needed
        if masked and self.data_mask is not None:
            if invert_mask:
                data = np.ma.masked_where(self.data_mask > 0.5, self.data)
            else:
                data = np.ma.masked_where(self.data_mask < 0.5, self.data)
        else:
            data = self.data

        # Get 2D slice and adjust orientation
        im_slice = np.transpose(data, _orient[view])[:, :, sl]
        if view == "y-z":
            im_slice = im_slice[:, ::-1]
        elif view == "x-z":
            im_slice = im_slice[::-1, ::-1]
        return np.rot90(im_slice, _n_rot[view])

    def plot(self, view, sl, ax=None, mpl_kwargs=None, show=True, figsize=None,
             colorbar=False, colorbar_label="HU", masked=False,
             invert_mask=False, mask_colour="black", no_ylabel=False):
        """Plot a given slice on a set of axes.

        Parameters
        ----------
        view : str
            Orientation to use. Must be one of "x-y", "y-z", and "x-z".

        sl : int
            Index of the slice to show.

        ax : matplotlib.axes, default=None
            Axes on which to plot. If None, new axes will be created.

        show : bool, default=True
            If True, the plotted figure will be shown immediately.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.imshow().

        masked : bool, default=False
            If True and the "mask" attribute is set, the image will be plotted
            with a mask.

        invert_mask : bool, default=True
            If True, any masks will be inverted.

        mask_colour : matplotlib colour, default="black"
            Colour in which to plot any masked areas
        """

        if not self.valid:
            return

        # Get slice
        self.set_ax(view, ax, figsize, colorbar)
        im_slice = self.get_slice(view, sl, masked, invert_mask)

        # Get colourmap
        kwargs = self.get_kwargs(mpl_kwargs)
        cmap = copy.copy(matplotlib.cm.get_cmap(kwargs.pop("cmap")))
        cmap.set_bad(color=mask_colour)

        # Plot image
        mesh = self.ax.imshow(im_slice,
                              extent=self.extent[view],
                              aspect=self.aspect[view],
                              cmap=cmap, **kwargs)

        # Set labels
        units = " (mm)" if self.scale_in_mm else ""
        self.ax.set_xlabel(_plot_axes[view][0] + units)
        if not no_ylabel:
            self.ax.set_ylabel(_plot_axes[view][1] + units)
        else:
            self.ax.set_yticks([])
        if self.title is not None:
            self.ax.set_title(self.title)

        # Draw colorbar
        if colorbar and kwargs.get("alpha", 1) > 0:
            clb = self.fig.colorbar(mesh, ax=self.ax, label=colorbar_label)
            clb.solids.set_edgecolor("face")

        # Display image
        if show:
            plt.tight_layout()
            plt.show()

    def downsample(self, d):
        """Downsample own image by amount d = (dx, dy, dz) in the (x, y, z)
        directions. If <d> is a single value, the image will be downsampled
        equally in all directions."""

        self.downsample = get_arg_tuple(d)
        for ax, d_ax in zip(_axes, self.downsample):
            self.voxel_sizes[ax] *= d_ax
        self.data = self.downsample_array(self.data)
        self.set_geom()

    def downsample_array(self, data_array):
        """Downsample a numpy array by amount set in self.downsample."""

        return data_array[::self.downsample[1],
                          ::self.downsample[0],
                          ::self.downsample[2]]


class DeformationImage(NiftiImage):
    """NiftiImage containing a deformation field."""

    def __init__(self, path, scale_in_mm=True, spacing=30, plot_type="grid"):
        """Load deformation field from a file."""

        NiftiImage.__init__(self, path, scale_in_mm=scale_in_mm)
        if not self.valid:
            return
        if self.data.ndim != 5:
            raise RuntimeError(f"Deformation field in {path} must contain a "
                               "five-dimensional array!")
        self.data = self.data[:, :, :, 0, :]
        self.plot_type = plot_type if plot_type in _df_plot_types else \
            _df_plot_types[0]
        self.set_spacing(spacing)

    def set_spacing(self, spacing):
        """Assign grid spacing in each direction."""

        if spacing is None:
            return
        spacing = get_arg_tuple(spacing)
        if self.scale_in_mm:
            self.spacing = {ax: abs(round(spacing[i] / self.voxel_sizes[ax]))
                            for i, ax in enumerate(_axes)}
        else:
            self.spacing = {ax: spacing[i] for i, ax in enumerate(_axes)}
        for ax, sp in self.spacing.items():
            if sp < 2:
                self.spacing[ax] = 2  # Ensure spacing is at least 2 voxels

    def set_plotting_defaults(self):
        """Create dict of default matplotlib plotting arguments."""

        self.quiver_kwargs = {"cmap": "jet"}
        self.grid_kwargs = {"color": "green"}

    def get_slice(self, view, sl):
        """Get 2D array corresponding to a slice of the image in a given
        orientation."""

        im_slice = np.transpose(self.data, _orient[view] + [3])[:, :, sl, :]
        if view == "y-z":
            im_slice = im_slice[:, ::-1, :]
        elif view == "x-z":
            im_slice = im_slice[::-1, ::-1, :]
        return np.rot90(im_slice, _n_rot[view])

    def get_deformation_slice(self, view, sl):
        """Get indices and displacements on a 2D slice."""

        im_slice = self.get_slice(view, sl)
        x_ax, y_ax = _plot_axes[view]

        # Get x/y displacement vectors
        df_x = np.squeeze(im_slice[:, :, _axes[x_ax]])
        df_y = np.squeeze(im_slice[:, :, _axes[y_ax]])
        if view == "x-y":
            df_x = -df_x
        elif view == "x-z":
            df_y = -df_y
        if not self.scale_in_mm:
            df_x /= self.voxel_sizes[x_ax]
            df_y /= self.voxel_sizes[y_ax]

        # Get x/y coordinates of each point on the slice
        xs = np.arange(0, im_slice.shape[1])
        ys = np.arange(0, im_slice.shape[0])
        if self.scale_in_mm:
            xs = self.origin[x_ax] + xs * self.voxel_sizes[x_ax]
            ys = self.origin[y_ax] + ys * self.voxel_sizes[y_ax]
        y, x = np.meshgrid(ys, xs)
        x = x.T
        y = y.T
        return x, y, df_x, df_y

    def plot(self, view, sl, ax=None, mpl_kwargs=None, plot_type=None,
             spacing=None):
        """Plot deformation field."""

        if not self.valid:
            return

        # Apply plot settings
        if plot_type in _df_plot_types:
            self.plot_type = plot_type
        self.set_spacing(spacing)

        # Plot grid/quiver
        if self.plot_type == "grid":
            self.plot_grid(view, sl, ax, mpl_kwargs)
        elif self.plot_type == "quiver":
            return self.plot_quiver(view, sl, ax, mpl_kwargs)

    def plot_quiver(self, view, sl, ax=None, mpl_kwargs=None):
        """Draw a quiver plot on a set of axes."""

        # Get arrow positions and lengths
        self.set_ax(view, ax)
        x_ax, y_ax = _plot_axes[view]
        x, y, df_x, df_y = self.get_deformation_slice(view, sl)
        arrows_x = df_x[::self.spacing[y_ax], ::self.spacing[x_ax]]
        arrows_y = -df_y[::self.spacing[y_ax], ::self.spacing[x_ax]]
        plot_x = x[::self.spacing[y_ax], ::self.spacing[x_ax]]
        plot_y = y[::self.spacing[y_ax], ::self.spacing[x_ax]]

        # Plot arrows
        if arrows_x.any() or arrows_y.any():
            M = np.hypot(arrows_x, arrows_y)
            return ax.quiver(plot_x, plot_y, arrows_x, arrows_y, M,
                             **self.get_kwargs(mpl_kwargs, self.quiver_kwargs))
        else:
            # If arrow lengths are zero, plot dots
            return ax.scatter(plot_x, plot_y, c="navy", marker=".")

    def plot_grid(self, view, sl, ax=None, mpl_kwargs=None):
        """Draw a grid plot on a set of axes."""

        # Get gridline positions
        self.set_ax(view, ax)
        self.ax.autoscale(False)
        x_ax, y_ax = _plot_axes[view]
        x, y, df_x, df_y = self.get_deformation_slice(view, sl)
        grid_x = x + df_x
        grid_y = y + df_y

        # Plot gridlines
        kwargs = self.get_kwargs(mpl_kwargs, default=self.grid_kwargs)
        for i in np.arange(0, x.shape[0], self.spacing[y_ax]):
            self.ax.plot(grid_x[i, :], grid_y[i, :], **kwargs)
        for j in np.arange(0, x.shape[1], self.spacing[x_ax]):
            self.ax.plot(grid_x[:, j], grid_y[:, j], **kwargs)


class StructImage(NiftiImage):
    """NiftiImage containing a structure mask."""

    def __init__(self, path, name=None, plot_type="contour"):
        """Load structure mask from a file, optionally setting the structure's
        name. If <name> is None, the name will be extracted from the filename.
        """

        # Load the mask
        self.path = path
        NiftiImage.__init__(self, path)
        if not self.valid:
            return

        # Set name
        if name is not None:
            self.name = name
        else:
            basename = os.path.basename(path).strip(".gz").strip(".nii")
            self.name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "",
                               basename).replace(" ", "_")
        nice = self.name.replace("_", " ")
        self.name_nice = nice[0].upper() + nice[1:]

        # Set plot type
        self.plot_type = plot_type if plot_type in _struct_plot_types else \
            _struct_plot_types[0]
        self.visible = True

        # Assign geometric properties and contours
        self.set_geom_properties()
        self.set_contours()

        # Assign a random colour
        self.assign_color(np.random.rand(3, 1).flatten())

    def set_geom_properties(self):
        """Set volume and length in each direction."""

        # Volume
        self.volume = {
            "voxels": self.data.astype(bool).sum()
        }
        self.volume["mm"] = self.volume["voxels"] \
            * abs(np.prod(list(self.voxel_sizes.values())))
        self.volume["ml"] = self.volume["mm"] * (0.1 ** 3)

        # Lengths
        self.length = {"voxels": {}, "mm": {}}
        nonzero = np.argwhere(self.data > 0.5)
        for ax, n in _axes.items():
            self.length["voxels"][ax] = max(nonzero[:, n]) - min(nonzero[:, n])
            self.length["mm"][ax] = self.length["voxels"][ax] \
                * abs(self.voxel_sizes[ax])

    def set_plotting_defaults(self):
        """Set default matplotlib plotting keywords."""

        self.mask_kwargs = {"alpha": 1,
                            "interpolation": "none"}
        self.contour_kwargs = {"linewidth": 2}

    def set_contours(self):
        """Compute positions of contours on each slice in each orientation.
        """

        self.contours = {}
        for view, z in _slider_axes.items():
            self.contours[view] = {}
            for i in range(self.n_voxels[z]):
                contour = self.get_contour_slice(view, i)
                if contour is not None:
                    self.contours[view][i] = contour

    def get_contour_slice(self, view, sl):
        """Convert mask to contours on a given slice <sl> in a given
        orientation <view>."""

        # Ignore slices with no structure mask
        im_slice = self.get_slice(view, sl)
        if im_slice.max() < 0.5:
            return

        # Find contours
        x_ax, y_ax = _plot_axes[view]
        contours = skimage.measure.find_contours(im_slice, 0.5, "low", "low")
        if contours:
            points = []
            for contour in contours:
                contour_points = []
                for (y, x) in contour:
                    if self.scale_in_mm:
                        x = min(self.lims[x_ax]) + \
                            (x + 0.5) * abs(self.voxel_sizes[x_ax])
                        y = min(self.lims[y_ax]) + \
                            (y + 0.5) * abs(self.voxel_sizes[y_ax])
                    contour_points.append((x, y))
                points.append(contour_points)
            return points

    def assign_color(self, color):
        """Assign a colour, ensuring that it is compatible with matplotlib."""

        if matplotlib.colors.is_color_like(color):
            self.color = matplotlib.colors.to_rgba(color)
        else:
            print(f"Colour {color} is not a valid colour.")

    def plot(self, view, sl, ax, mpl_kwargs=None, plot_type=None):
        """Plot structure."""

        if not self.valid:
            return
        if plot_type in _struct_plot_types:
            self.plot_type = plot_type
        if self.plot_type == "contour":
            self.plot_contour(view, sl, ax, mpl_kwargs)
        elif self.plot_type == "mask":
            return self.plot_mask(view, sl, ax, mpl_kwargs)

    def plot_mask(self, view, sl, ax, mpl_kwargs=None):
        """Plot structure as a coloured mask."""

        # Get slice
        self.set_ax(view, ax)
        im_slice = self.get_slice(view, sl)

        # Make colormap
        norm = matplotlib.colors.Normalize()
        cmap = matplotlib.cm.hsv
        s_colors = cmap(norm(im_slice))
        s_colors[im_slice > 0, :] = self.color
        s_colors[im_slice == 0, :] = (0, 0, 0, 0)

        # Display the mask
        return self.ax.imshow(
            s_colors,
            extent=self.extent[view],
            aspect=self.aspect[view],
            **self.get_kwargs(mpl_kwargs, default=self.mask_kwargs)
        )

    def plot_contour(self, view, sl, ax, mpl_kwargs=None):
        """Plot a contour for a given orientation and slice number on an
        existing set of axes."""

        if not self.on_slice(sl, view):
            return
        self.set_ax(view, ax)
        kwargs = self.get_kwargs(mpl_kwargs, default=self.contour_kwargs)
        kwargs.setdefault("color", self.color)
        for points in self.contours[view][sl]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            self.ax.plot(points_x, points_y, **kwargs)

    def on_slice(self, sl, view):
        return sl in self.contours[view]


class MultiImage(NiftiImage):
    """Class for containing information for an image which can have a dose map,
    masks, structures, jacobian determinant, and deformation field."""

    def __init__(
        self,
        nii,
        title=None,
        scale_in_mm=True,
        downsample=None,
        dose=None,
        mask=None,
        jacobian=None,
        df=None,
        structs=None,
        struct_colours=None,
        structs_as_mask=False
    ):
        """Load a QuickViewerImage.

        Parameters
        ----------
        nii : str/nifti/array
            Path to a .nii/.npy file, or an nibabel nifti object/numpy array.

        title : str, default=None
            Title for this image when plotted. If None, the filename will be
            used.

        scale_in_mm : bool, default=True
            If True, image will be plotted in mm rather than voxels.

        downsample : int/tuple, default=None
            Amount by which to downsample in the (x, y, z) directions. If a
            single value is given, the image will be downsampled equally in
            all directions.

        dose : str, default=None
            Path to a dose file.

        mask : str, default=None
            Path to a mask file.

        jacobian : str, default=None
            Path to a jacobian determinant file.

        df : str, default=None
            Path to a deformation field file.

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

        """

        # Load the scan image
        NiftiImage.__init__(self, nii, title=title, scale_in_mm=scale_in_mm)
        if not self.valid:
            return
        if downsample is not None:
            self.downsample(downsample)

        # Load extra overlays
        self.load_to(dose, "dose")
        self.load_to(mask, "mask")
        self.load_to(jacobian, "jacobian")
        self.load_df(df)
        self.load_structs(structs, struct_colours)
        self.structs_as_mask = structs_as_mask
        if self.has_structs and structs_as_mask:
            self.has_mask = True

    def load_to(self, path, attr):
        """Load image data from a path into a class attribute."""

        data = NiftiImage(path, scale_in_mm=self.scale_in_mm)
        setattr(self, attr, data)
        valid = data.valid and data.data.shape == self.shape
        setattr(self, f"has_{attr}", valid)

    def load_df(self, df):
        """Load deformation field data from a path."""

        self.df = DeformationImage(df, scale_in_mm=self.scale_in_mm)
        self.has_df = self.df.valid and self.df.shape[:3] == self.shape

    def load_structs(self, structs, struct_colours):
        """Load structures from a path/wildcard or list of paths/wildcards."""

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
        custom = {n.lower().replace(" ", "_"): col for n, col in
                  custom.items()}
        for i, struct in enumerate(self.structs):

            # Assign standard colour
            struct.assign_color(standard_colors[i])

            # Check for exact match
            matched = False
            for name in custom:
                if name == struct.name.lower():
                    struct.assign_color(custom[name])
                    matched = True

            # If no exact match, check for first matching wildcard
            if not matched:
                for name in custom:
                    if fnmatch.fnmatch(struct.name.lower(), name):
                        struct.assign_color(custom[name])
                        break

    def set_plotting_defaults(self):
        """Set default matplotlib plotting options."""

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

        if self.has_mask:
            mask_array = self.mask.data
            if self.structs_as_mask:
                for struct in self.structs:
                    mask_array += struct.data
        elif self.structs_as_mask and self.has_structs:
            mask_array = self.structs[0].data
            for struct in self.structs[1:]:
                mask_array += struct.data
        else:
            mask_array = None
        self.set_mask(mask_array)
        self.dose.set_mask(mask_array)

    def get_n_colorbars(self, colorbar=False):
        return colorbar * (1 + self.has_dose + self.has_jacobian)

    def set_ax(self, view, ax=None, gs=None, figsize=None, colorbar=False):

        if ax is None and gs is not None:
            ax = plt.gcf().add_subplot(gs)

        NiftiImage.set_ax(self, view, ax, figsize,
                          self.get_n_colorbars(colorbar))

    def get_relative_width(self, view, colorbar=False):
        return NiftiImage.get_relative_width(
            self, view, self.get_n_colorbars(colorbar))

    def plot(
        self,
        view,
        sl,
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
        df_plot_type=None,
        df_spacing=None,
        struct_kwargs=None,
        struct_plot_type=None,
        struct_legend=True,
        legend_loc='lower left'
    ):
        """Plot image slice and any extra overlays."""

        # Plot image
        self.set_ax(view, ax, gs, figsize, colorbar)
        self.set_masks()
        NiftiImage.plot(self, view, sl, self.ax, mpl_kwargs, show=False,
                        colorbar=colorbar, masked=masked,
                        invert_mask=invert_mask,
                        mask_colour=mask_colour, figsize=figsize)

        # Plot dose field
        self.dose.plot(view, sl, self.ax,
                       self.get_kwargs(dose_kwargs, default=self.dose_kwargs),
                       show=False, masked=masked, invert_mask=invert_mask,
                       mask_colour=mask_colour, colorbar=colorbar,
                       colorbar_label="Dose (Gy)")

        # Plot jacobian
        self.jacobian.plot(view, sl, self.ax,
                           self.get_kwargs(jacobian_kwargs,
                                           default=self.jacobian_kwargs),
                           show=False, colorbar=colorbar,
                           colorbar_label="Jacobian determinant")

        # Plot structures
        struct_handles = []
        for struct in self.structs:
            if not struct.visible:
                continue
            struct.plot(view, sl, self.ax, struct_kwargs, struct_plot_type)
            if struct.on_slice(sl, view) and struct_plot_type != "none":
                struct_handles.append(mpatches.Patch(color=struct.color,
                                                     label=struct.name_nice))

        # Plot deformation field
        self.df.plot(view, sl, self.ax,
                     mpl_kwargs=df_kwargs,
                     plot_type=df_plot_type,
                     spacing=df_spacing)

        # Draw legend
        if struct_legend:
            self.ax.legend(handles=struct_handles, loc=legend_loc,
                           facecolor="white", framealpha=1)

        # Display image
        if show:
            plt.tight_layout()
            plt.show()


class OrthogonalImage(MultiImage):
    """MultiImage with an orthogonal view next to it."""

    def __init__(self, *args, **kwargs):
        MultiImage.__init__(self, *args, **kwargs)
        self.orthog_slices = {ax: int(self.n_voxels[ax] / 2)
                              for ax in _axes}

    def get_relative_width(self, view, colorbar=False):
        """Get width:height ratio for this plot in a given orientation with a
        given number of colorbars."""

        width_own = MultiImage.get_relative_width(self, view, colorbar)
        width_orthog = MultiImage.get_relative_width(self, _orthog[view])
        return width_own + width_orthog

    def set_axes(self, view, gs=None, figsize=None, colorbar=False):

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

        self.own_ax = fig.add_subplot(self.gs[0])
        self.orthog_ax = fig.add_subplot(self.gs[1])

    def plot(self,
             view,
             sl,
             gs=None,
             figsize=None,
             mpl_kwargs=None,
             show=True,
             colorbar=False,
             struct_kwargs=None,
             struct_plot_type=None,
             **kwargs
            ):
        """Plot MultiImage next to orthogonal image."""

        self.set_axes(view, gs, figsize, colorbar)

        # Plot the MultiImage
        MultiImage.plot(self, view, sl, ax=self.own_ax, colorbar=colorbar,
                        show=False, mpl_kwargs=mpl_kwargs,
                        struct_kwargs=struct_kwargs,
                        struct_plot_type=struct_plot_type,
                        **kwargs)

        # Plot orthogonal image
        orthog_view = _orthog[view]
        orthog_sl = self.orthog_slices[_slider_axes[orthog_view]]
        NiftiImage.plot(self,
                        orthog_view,
                        orthog_sl,
                        ax=self.orthog_ax,
                        mpl_kwargs=mpl_kwargs,
                        show=False,
                        colorbar=False,
                        no_ylabel=True)

        # Plot structures on orthogonal image
        for struct in self.structs:
            if not struct.visible:
                continue
            struct.plot(orthog_view, orthog_sl, self.orthog_ax, struct_kwargs,
                        struct_plot_type)

        # Plot indicator line
        pos = sl if not self.scale_in_mm else self.idx_to_pos(
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


def in_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except NameError:
        return False
    return True


def same_shape(imgs):
    """Check whether images in a list all have the same shape (in the
    first 3 dimensions)."""

    for i in range(len(imgs) - 1):
        if imgs[i].shape[:3] != imgs[i + 1].shape[:3]:
            return False
    return True


def get_arg_tuple(arg):
    """Convert an argument to a tuple if it isn't alread."""

    if isinstance(arg, tuple) or isinstance(arg, list):
        return arg
    else:
        return (arg, arg, arg)


def get_image_slice(image, view, sl):
    """Get 2D slice of an image in a given orienation."""

    orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [0, 1, 2]}
    n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
    if image.ndim == 3:
        im_to_show = np.transpose(image, orient[view])[:, :, sl]
        if view == "y-z":
            im_to_show = im_to_show[:, ::-1]
        elif view == "x-z":
            im_to_show = im_to_show[::-1, ::-1]
        return np.rot90(im_to_show, n_rot[view])
    else:
        transpose = orient[view] + [3]
        im_to_show = np.transpose(image, transpose)[:, :, sl, :]
        if view == "y-z":
            im_to_show = im_to_show[:, ::-1, :]
        elif view == "x-z":
            im_to_show = im_to_show[::-1, ::-1, :]
        return np.rot90(im_to_show, n_rot[view])


__all__ = ("_axes", "_plot_axes", "_slider_axes", "same_shape",
           "get_image_slice", "in_notebook")
