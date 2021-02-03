"""Shared variables and functions."""

import fnmatch
import glob
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import nibabel
import numpy as np
import os
import re
import skimage.measure


# Global properties
_style = {"description_width": "initial"}
_axes = {"x": 1, "y": 0, "z": 2}
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}


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
        if nii is None:
            self.valid = False
            return

        # Load from numpy array
        if isinstance(nii, np.ndarray):
            self.data = nii

        # Load from file or nifti object
        else:
            if isinstance(nii, str):
                try:
                    self.nii = nibabel.load(os.path.expanduser(nii))
                    self.data = self.nii.get_fdata()
                    affine = self.nii.affine
                    if self.title == "":
                        self.title = os.path.basename(nii)
                except FileNotFoundError:
                    self.valid = False
                    return
                except nibabel.filebasedimages.ImageFileError:
                    try:
                        self.data = np.load(nii)
                    except (IOError, ValueError):
                        raise RuntimeError(f"Input file <nii> must be a valid "
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

    def idx_to_pos(self, idx, ax):
        """Convert an index to a position along a given axis."""

        return self.origin[ax] + idx * self.voxel_sizes[ax]

    def pos_to_idx(self, pos, idx):
        """Convert a position to an index along a given axis."""

        return int((pos - self.origin[ax]) / self.voxel_sizes[ax])

    def get_slice(self, view, sl):
        """Get 2D array corresponding to a slice of the image in a given 
        orientation.

        Parameters
        ----------
        view : str
            Orientation to use. Must be one of "x-y", "y-z", and "x-z".

        sl : int
            Index of the slice to use.
        """

        orient = {"y-z": [1, 2, 0], "x-z": [0, 2, 1], "x-y": [0, 1, 2]}
        n_rot = {"y-z": 2, "x-z": 2, "x-y": 1}
        im_slice = np.transpose(self.data, orient[view])[:, :, sl]
        if view == "y-z":
            im_slice = im_slice[:, ::-1]
        elif view == "x-z":
            im_slice = im_slice[::-1, ::-1]
        return np.rot90(im_slice, n_rot[view])
  
    def plot_slice(self, view, sl, ax=None, show=True, mpl_kwargs=None):
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

        """
        
        # Create axes if needed
        if ax is None:
            fig, ax = plt.subplots()

        # Plot image
        if mpl_kwargs is None:
            mpl_kwargs = {}
        ax.imshow(self.get_slice(view, sl), 
                  extent=self.extent[view],
                  aspect=self.aspect[view],
                  **mpl_kwargs)

        # Set labels
        units = " (mm)" if self.scale_in_mm else ""
        ax.set_xlabel(_plot_axes[view][0] + units)
        ax.set_ylabel(_plot_axes[view][1] + units)
        if self.title is not None:
            ax.set_title(self.title)

        # Display image
        if show:
            plt.show()

    def downsample(self, d):
        """Downsample image by amount d = (dx, dy, dz) in the (x, y, z) 
        directions. If <d> is a single value, the image will be downsampled
        equally in all directions."""

        ds = (d, d, d) if isinstance(d, int) else d
        for ax, d_ax in zip(_axes, ds):
            self.voxel_sizes[ax] *= d_ax
        self.data = self.data[::d[1], ::d[0], ::d[2]]
        self.set_geom()


class DeformationImage(NiftiImage):
    """NiftiImage containing a deformation field."""

    def __init__(self, path, scale_in_mm=True):
        """Load deformation field from a file."""

        NiftiImage.__init__(self, path, scale_in_mm=scale_in_mm)
        if not self.valid:
            return
        if self.data.ndim != 5:
            raise RuntimeError(f"Deformation field in {path} must contain a "
                               "five-dimensional array!")
        self.data = self.data[:, :, :, 0, :]


class StructImage(NiftiImage):
    """NiftiImage containing a structure mask."""

    def __init__(self, path, name=None):
        """Load structure mask from a file, optionally setting the structure's
        name. If <name> is None, the name will be extracted from the filename.
        """

        # Load the mask
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

        # Assign geometric properties and contours
        self.set_geom_properties()
        self.set_contours()

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

    def assign_colour(self, color):
        """Assign a colour, ensuring that it is compatible with matplotlib."""

        if matplotlib.colors.is_color_like(colour):
            self.color = color
        else:
            print(f"Colour {colour} is not a valid colour.")

    def plot_contour(self, ax, view, sl, mpl_kwargs=None):
        """Plot a contour for a given orientation and slice number on an 
        existing set of axes."""

        if sl not in self.contours[view]:
            return
        kwargs = mpl_kwargs if mpl_kwargs is not None else {}
        if hasattr(self, "color"):
            kwargs["color"] = self.color
        for points in self.contours[view][sl]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            ax.plot(points_x, points_y, **kwargs)


class MultiImage(NiftiImage):
    """Class for containing information for an image which can have a dose map,
    masks, structures, jacobian determinant, and deformation field."""

    def __init__(
        self, 
        path, 
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
        path : str
            Path to a nifti file.

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
        self.path = os.path.expanduser(path)
        NiftiImage.__init__(self, path, title=title, scale_in_mm=scale_in_mm)
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
        if structs is None:
            return

        # Find valid filepaths
        struct_paths = [structs] if isinstance(structs, str) else structs
        files = []
        for path in structs_paths:

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
        custom = {n.lower().replace(" ", "_"): col for n, col in custom}
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
                    if fnmatch.fnmatch(name, struct.name.lower()):
                        struct.assign_color(custom[name])
                        break


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

__all__ = ("_style", "_axes", "_plot_axes", "_slider_axes", "_view_map", 
           "_orthog", "in_notebook", "same_shape", "get_image_slice")
