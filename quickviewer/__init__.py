# File: quickviewer/__init__.py
# -*- coding: future_fstrings -*-

import ipywidgets as ipyw
import glob
import numpy as np
import pydicom
import nibabel
import matplotlib
import matplotlib.pyplot as plt
import os
import re
from matplotlib import colors, cm, gridspec
import matplotlib.ticker as mticker
import fnmatch
import logging
import copy
import skimage.measure
import matplotlib.patches as mpatches


# Global properties
_style = {"description_width": "initial"}
_axes = ["x", "y", "z"]
_slider_axes = {"x-y": "z", "x-z": "y", "y-z": "x"}
_plot_axes = {"x-y": ("x", "y"), "x-z": ("z", "x"), "y-z": ("z", "y")}
_view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}
plt.rcParams['axes.facecolor'] = 'black'


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


class QuickViewerImage():
    """Class for storing data for a single plot for a QuickViewer output."""

    def __init__(self, path, title, scale_in_mm, downsample=None,
                 ch=None, fh=None, count=0):
        """Load an image from a file path."""

        # Set up logging
        self.logger = logging.getLogger(f"QuickViewerImage_{count}")
        if ch is not None:
            self.logger.addHandler(ch)
        if fh is not None:
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)

        # Set path and check it exists
        self.count = count
        self.path = path
        self.valid = os.path.isfile(self.path)
        if not self.valid:
            return
        self.title = title
        self.logger.debug(f"Initialised QuickViewerImage #{self.count}")
        self.logger.debug(f"    Path = {self.path}")
        self.logger.debug(f"    Title = {self.title}")

        # Downsampling settings
        if downsample is None:
            self.downsample = {ax: 1 for ax in _axes}
        else:
            self.downsample = {ax: downsample[i] for i, ax in enumerate(_axes)}

        # Load image data
        self.nii = nibabel.load(self.path)
        self.image = self.process_image(self.nii.get_fdata())
        self.scale_in_mm = scale_in_mm
        self.logger.debug("    Loaded image with dimensions "
                          f"{self.image.shape}")

        # Initial properties
        self.has_mask = False
        self.has_dose = False
        self.has_structs = False
        self.has_jacobian = False
        self.has_df = False
        self.colorbar = False
        self.dose_colorbar = False
        self.jac_colorbar = False
        self.masked_image = {}
        self.masked_dose = {}

        # Get voxel sizes and axis limits
        self.n_voxels = {ax: self.image.shape[i] for i, ax in enumerate(_axes)}
        self.voxel_sizes = {}
        self.axis_min = {}
        self.axis_max = {}
        self.length = {}
        self.origin = {}
        for i, ax in enumerate(["y", "x", "z"]):

            # Set number of voxels and voxel size
            self.nii.affine[i, i] *= self.downsample[ax]
            self.voxel_sizes[ax] = self.nii.affine[i, i] 

            # Get min and max of axis
            self.origin[ax] = self.nii.affine[i, 3]
            lims = [self.origin[ax], self.origin[ax] + self.n_voxels[ax] 
                    * self.voxel_sizes[ax]]
            self.length[ax] = abs(lims[1] - lims[0])
            self.axis_min[ax] = min(lims)
            self.axis_max[ax] = max(lims)
            self.logger.debug(f"Image settings for axis {ax}:")
            self.logger.debug(f"    Number of voxels = {self.n_voxels[ax]}")
            self.logger.debug(f"    Voxel size = {self.voxel_sizes[ax]}")
            self.logger.debug(f"    Origin = {self.origin[ax]}")
            self.logger.debug(f"    Axis range = {self.axis_min[ax]} -- "
                              f"{self.axis_max[ax]}")
            self.logger.debug(f"    Length = {self.length[ax]}")

        # Set aspect ratios and axis extents
        self.aspect = {}
        self.extent = {}
        for view, (x, y) in _plot_axes.items():
            if not scale_in_mm:
                self.extent[view] = None
                self.aspect[view] = abs(self.voxel_sizes[y] / 
                                        self.voxel_sizes[x])
            else:
                self.extent[view] = (self.axis_min[x], self.axis_max[x],
                                     self.axis_max[y], self.axis_min[y])
                self.aspect[view] = 1
            self.logger.debug(f"Plot settings for view {view}:")
            self.logger.debug(f"    Aspect ratio = {self.aspect[view]}")
            self.logger.debug(f"    Axis extent = {self.extent[view]}")

        # Set initial positions for orthogonal views
        self.orthog_slice = {
            view: int(self.n_voxels[_slider_axes[_orthog[view]]] / 2) 
            for view in _slider_axes
        }

        # Create mappings between voxels and positions
        self.make_slice_voxel_maps()

        # Get HU range
        self.hu_min = int(np.min([im.min() for im in self.image]))
        self.hu_max = int(np.max([im.max() for im in self.image]))

    def load_image(self, path, return_nii=False, force_same_shape=True,
                   allow_glob=False):
        """Load an image and check it has the same shape as the scan image."""

        # Deal with None paths
        if path is None:
            self.logger.debug("    Path is None, no image loaded")
            return None if not return_nii else (None, None)

        # Convert to a list if needed
        path_list = path if isinstance(path, list) else [path]
        images = []
        niis = []

        # Load each image in list
        for p in path_list:

            # Check if file exists
            paths_to_load = [p]
            if not os.path.isfile(p):

                # Try globbing for files
                if allow_glob:
                    self.logger.debug(f"Globbing for images in {p}")
                    paths_to_load = glob.glob(p)
                    if not len(paths_to_load):
                        self.logger.warning(f"No files matching {p} were "
                                            "found!")
                        return None if not return_nii else (None, None)
                else: 
                    self.logger.warning(f'File {p} not found!')
                    return None if not return_nii else (None, None)

            # Load all images
            for p_load in set(paths_to_load):
                nii = nibabel.load(p_load)
                image = nii.get_fdata()
                self.logger.debug(f"Loaded NIfTI image from {p_load}")
                if not same_shape([image, self.image]) and force_same_shape:
                    self.logger.warning(f'Image in {p_load} does not have the '
                                        f'same shape as image in {self.path}!')
                    return None if not return_nii else (None, None)
                images.append(image)
                niis.append(nii)
        
        # Return either a list or a single image
        if len(images) > 1:
            return images if not return_nii else (images, niis)
        else:
            return images[0] if not return_nii else (images[0], niis[0])

    def load_mask(self, mask_path, invert=False):
        """Create masked version of the image by applying mask from a given
        file path. If path is None, assign the original image to the masked
        image variable."""

        self.logger.debug("Attempting to load mask(s)...")
        self.mask_per_view = isinstance(mask_path, dict)

        # Assign mask for each view
        use_previous = False
        for view in _slider_axes:
            if not use_previous:
                path = mask_path.get(view) if self.mask_per_view else mask_path
                mask = self.load_image(path)
                if invert and mask is not None:
                    mask = ~(mask.astype(bool))
                use_previous = not self.mask_per_view
            self.apply_mask(mask, view)

    def apply_mask(self, mask, view):
        """Apply mask to image from a mask array or list of mask arrays."""

        # Case where mask provided is empty
        if mask is None and not hasattr(self, "mask_to_apply"):
            self.has_mask = False
            self.masked_image[view] = self.image
            if self.has_dose:
                self.masked_dose[view] = self.dose
            return

        # Case where mask is provided or was provided in the past
        self.logger.debug("Applying mask(s) to scan image")
        self.mask_to_apply = mask
        self.has_mask = True
        total_mask = sum(mask) if isinstance(mask, list) else mask
        self.masked_image[view] = np.ma.masked_where(total_mask < 0.5, 
                                                     self.image)
        if self.has_dose:
            self.logger.debug("Applying mask(s) to dose image")
            self.masked_dose[view] = np.ma.masked_where(total_mask < 0.5, 
                                                        self.dose)

    def load_dose(self, dose_path):
        """Load an associated dose image."""

        self.logger.debug("Attempting to load dose...")
        self.dose = self.load_image(dose_path)
        self.has_dose = self.dose is not None
        if self.has_mask:
            for view in _plot_axes:
                self.apply_mask(None, view)

    def load_jacobian(self, jacobian_path):
        """Load an associated jacobian image."""

        self.logger.debug("Attempting to load jacobian...")
        self.jacobian = self.load_image(jacobian_path)
        self.has_jacobian = self.jacobian is not None

    def load_df(self, df_path):
        """Load an associated deformation field image."""
        
        self.extrapolate_df = False
        if df_path == "regular":
            self.regular_grid = True
            self.logger.debug("Will plot regular grid")
            self.has_df = True
        else:
            self.regular_grid = False
            self.logger.debug("Attempting to load deformation field...")
            self.df, df_nii = self.load_image(df_path, return_nii=True,
                                     force_same_shape=False)
            self.has_df = self.df is not None
            if not self.has_df:
                return
            self.df = self.df[:, :, :, 0, :]
            self.df_affine = df_nii.affine
            if (self.nii.affine - self.df_affine).any():
                if self.scale_in_mm:
                    self.logger.debug("    Deformation field has different "
                                      "shape to image; will be extrapolated.")
                    self.extrapolate_df = True
                else:
                    self.logger.warning("Deformation field has a different "
                                        "shape to image, and extrapolation "
                                        "is not supported when scale_in_mm="
                                        "False. Deformation field will not be "
                                        "plotted.")
                    self.has_df = False
                    self.df = None

    def process_image(self, image):
        """Remove NaNs and downsample image if required."""

        # Convert any NaN to zero
        image = np.nan_to_num(image)
        if True in [ds != 1 for ds in self.downsample.values()]:
            image = self.downsample_image(image)
        return image

    def downsample_image(self, image):
        """Downsample an image."""

        dsx = self.downsample["x"]
        dsy = self.downsample["y"]
        dsz = self.downsample["z"]
        self.logger.debug(f"    Downsample = {self.downsample}")
        return image[::dsx, ::dsy, ::dsz]

    def get_nearest_vox(self, ax, vox):
        """Find the nearest slice number to a given number."""
        
        if vox in self.voxel_to_pos_map[ax]:
            return vox
        distance = [abs(v - vox) for v in self.voxel_to_pos_map[ax]]
        nearest_idx = distance.index(min(distance))
        return list(self.voxel_to_pos_map[ax].keys())[nearest_idx]

    def voxel_to_pos(self, ax, vox, take_nearest=False):
        """Convert a voxel number to a position in mm along a given axis."""

        if vox not in self.voxel_to_pos_map[ax] and take_nearest:
            nearest_vox = self.get_nearest_vox(ax, vox)
            self.voxel_to_pos_map[ax][nearest_vox]
        else:
            return float(self.voxel_to_pos_map[ax][vox])

    def get_nearest_pos(self, ax, pos):
        """Find the key of the nearest slice position to a given position."""

        distance = [abs(float(p) - pos) for p in self.pos_to_voxel_map[ax]]
        nearest_idx = distance.index(min(distance))
        return list(self.pos_to_voxel_map[ax].keys())[nearest_idx]

    def pos_to_voxel(self, ax, pos, take_nearest=False):
        """Convert a position in mm to a voxel number along a given axis.
        If take_nearest is True, the closest slice to the position will be
        given."""
        
        pos_str = self.position_fmt[ax].format(float(pos))
        if pos_str not in self.pos_to_voxel_map[ax]:
            if not take_nearest:
                print(f"Error: {pos_str} not in pos_to_voxel map!")
                print("Positions in the map:")
                print(list(self.pos_to_voxel_map[ax].keys()))
                return 0
            else:
                return self.pos_to_voxel_map[ax][self.get_nearest_pos(ax, pos)]
        voxel = self.pos_to_voxel_map[ax][pos_str]
        return voxel

    def make_slice_voxel_maps(self):
        """Make maps between slice number and position in mm for each axis."""

        self.voxel_to_pos_map = {}
        self.pos_to_voxel_map = {}
        self.position_fmt = {}
        for ax in ['x', 'y', 'z']:

            # Work out precision to which to record positions
            log_size = int(np.floor(np.log10(abs(self.voxel_sizes[ax]))))
            if log_size >= 0:
                prec = 0
            else:
                prec = abs(log_size)
            self.position_fmt[ax] = "{:." + str(prec) + "f}"

            # Create map from voxel number to position
            self.voxel_to_pos_map[ax] = {
                j: self.position_fmt[ax].format(
                    self.origin[ax] + j * self.voxel_sizes[ax]
                ) for j in range(self.n_voxels[ax])
            }

            # Create map in from position to voxel number
            self.pos_to_voxel_map[ax] = {
                pos: vox for vox, pos in self.voxel_to_pos_map[ax].items()
            }

    def make_slice_slider(self, ax, init_idx=None, init_pos=None,
                          continuous_update=False):
        """Create a slice slider for a given initial axis."""

        # Check whether to use non-default initial value
        self.current_pos = {}
        use_init_idx = (init_idx is not None) and (init_pos is None or 
                                                  not self.scale_in_mm)
        use_init_pos = (init_pos is not None) and (init_idx is None or 
                                                   self.scale_in_mm)

        # User-input index as starting position
        if use_init_idx:
            if self.scale_in_mm and ax == "z":
                init_idx = self.n_voxels[ax] - init_idx
            init_idx = self.get_nearest_vox(ax, init_idx)
            self.current_pos[ax] = init_idx if not self.scale_in_mm \
                    else self.voxel_to_pos(ax, init_idx)

        # User-input position in mm as starting position
        elif use_init_pos:
            nearest_pos = self.get_nearest_pos(ax, init_pos)
            self.current_pos[ax] = nearest_pos if self.scale_in_mm else \
                    self.pos_to_voxel(ax, float(nearest_pos))

        # Set initial postions for all axes
        self.logger.debug("Setting initial slice positions:")
        for axis in _axes:
            if axis not in self.current_pos:
                mid_slice = int(self.n_voxels[axis] / 2)
                if self.scale_in_mm:
                    self.current_pos[axis] = self.voxel_to_pos(axis, mid_slice)
                else:
                    self.current_pos[axis] = mid_slice
            self.logger.debug(f"    {axis} = {self.current_pos[axis]}")

        # Make mappings between slider values and slices
        if not self.scale_in_mm:
            self.slider_to_idx = {}
            self.slider_to_idx["z"] = {
                self.n_voxels["z"] - i: i for i in 
                range(self.n_voxels["z"])}
            for a in ["x", "y"]:
                self.slider_to_idx[a] = {
                    i + 1: i for i in range(self.n_voxels[a])}

            # Make reverse map
            self.idx_to_slider = {}
            for a in _axes:
                self.idx_to_slider[a] = {
                    sl: idx for idx, sl in self.slider_to_idx[a].items()}

        # Make slice slider
        self.slice_slider = ipyw.FloatSlider(
            continuous_update=continuous_update,
            style=_style,
            readout_format=".1f"
        )
        self.update_slice_slider(ax)

    def share_slice_slider(self, im):
        """Setup sharing of a slice slider with another QuickViewerImage."""

        self.slice_slider = im.slice_slider
        self.current_pos = im.current_pos
        if not self.scale_in_mm:
            self.slider_to_idx = im.slider_to_idx

    def make_hu_slider(self, hu_min, hu_max, vmin, vmax):
        """Make HU slider."""

        self.hu_slider = ipyw.IntRangeSlider(
            min=hu_min,
            max=hu_max,
            value=[vmin, vmax],
            description="HU range",
            continuous_update=False,
            style=_style
        )

    def update_slice_slider(self, ax):
        """Update the slice slider to match a given axis."""

        # Set max, min, and step
        if self.scale_in_mm:
            self.slice_slider.min = self.axis_min[ax]
            self.slice_slider.max = self.axis_max[ax]
            self.slice_slider.step = abs(self.voxel_sizes[ax])
            self.slice_slider.description = f"{ax} (mm)"
        else:
            self.slice_slider.min = 1
            self.slice_slider.max = self.n_voxels[ax]
            self.slice_slider.step = 1
            pos = self.voxel_to_pos(
                ax, self.slider_to_idx[ax][self.current_pos[ax]])
            self.slice_slider.description = f"{ax} ({pos} mm)"

        # Set current value
        self.slice_slider.value = self.current_pos[ax]

    def load_structs(self, structs, struct_colours={}):
        """Load a dictionary of structures from files matching a pattern.
        Any structures within the keys of <struct_colours> will be assigned
        custom colours; all others will be assigned a standardised set of
        colours."""

        self.has_structs = False
        self.structs = {}
        self.struct_names = {}
        self.struct_names_nice = {}
        self.contours = {'x-y': {},
                         'y-z': {},
                         'x-z': {}}
        self.struct_checkboxes = {}
        if structs is None:
            return

        # Find files
        files = []
        if not isinstance(structs, list):
            structs = [structs]
        for s in structs:

            s = os.path.expanduser(s)
            struct_files = []

            # Take all files from directory
            if os.path.isdir(s):
                struct_files.extend(glob.glob(s + "/*.nii*"))

            # Take a single file
            elif os.path.isfile(s):
                struct_files.append(s)

            # Search for wildcard
            else:
                matches = glob.glob(s)
                for m in matches:
                    if os.path.isdir(m):
                        struct_files.extend(glob.glob(m + "/*.nii*"))
                    elif os.path.isfile(m):
                        struct_files.append(m)

            if not len(struct_files):
                self.logger.warning("No structure files found corresponding to"
                                    f" {s}.")
            else:
                files.extend(struct_files)

        # Remove duplicates
        files = set(files)

        # Load structures from files
        for f in files:

            # Load NIfTI file and image
            struct = self.load_image(f)
            if struct is None:
                continue

            # Add to dictionary
            f_abs = os.path.abspath(f)
            self.structs[f_abs] = struct
            basename = os.path.basename(f).strip(".gz").strip(".nii")
            struct_name = re.sub(
                r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "", basename).replace(
                    " ", "_")
            self.struct_names[f_abs] = struct_name
            nice_name = struct_name.replace("_", " ")
            nice_name = nice_name[0].upper() + nice_name[1:]
            self.struct_names_nice[f_abs] = nice_name

            # Get contours in each plane
            for view in self.contours:
                self.contours[view][f_abs] = self.get_contours(struct, view)

        self.has_structs = bool(len(self.structs))
        self.show_struct = {struct: True for struct in self.structs}

        # Assign standard colours
        standard_colors = (
            list(cm.Set1.colors)[:-1]
            + list(cm.Set2.colors)[:-1]
            + list(cm.Set3.colors)
            + list(cm.tab20.colors)
        )
        self.struct_colours = {f: standard_colors[i] for i, f in 
                               enumerate(self.structs)}

        # Apply custom colours
        for struct, colour in struct_colours.items():

            struct = os.path.expanduser(struct)
    
            # Check for matching filenames
            if os.path.abspath(struct) in self.structs:
                structs_to_colour = [os.path.abspath(struct)]

            # Check for matching structure names
            elif struct.replace(" ", "_") in self.struct_names.values():
                structs_to_colour = [f for f, name in self.struct_names.items() 
                                     if name.lower() == 
                                     struct.lower().replace(" ", "_")]

            # Use wildcard
            else:

                # Search for matches with structure names
                structs_to_colour = [
                    f for f in self.structs if fnmatch.fnmatch(
                        self.struct_names[f].lower(), 
                        struct.lower().replace(" ", "_"))]

                # Search for matches with structure files
                if not len(structs_to_colour):
                    structs_to_colour = [
                        f for f in self.structs if 
                        fnmatch.fnmatch(f, os.path.abspath(struct))]

            # Check colour is valid
            if colors.is_color_like(colour):
                for f in structs_to_colour:
                    self.struct_colours[f] = colour
            else:
                self.logger.warning(f"Colour {colour} provided in "
                                    "struct_colours is not a valid colour.")

    def get_contours(self, struct, view):
        """Convert a structure mask to a dictionary of contours in a given 
        orientation."""

        # Loop through layers
        points = {}
        x_ax, y_ax = _plot_axes[view]
        z_ax = _slider_axes[view]
        for i in range(self.n_voxels[z_ax]):

            # Get layer of image
            im_slice = get_image_slice(struct, view, i)

            # Ignore slices with no structure mask
            if im_slice.max() < 0.5:
                continue

            # Find contours
            contours = skimage.measure.find_contours(
                im_slice, 0.5, "low", "low")
            if contours:
                points[i] = []
                for contour in contours:
                    contour_points = []
                    for (y, x) in contour:
                        if self.scale_in_mm:
                            x = self.axis_min[x_ax] \
                                    + (x + 0.5) * abs(self.voxel_sizes[x_ax])
                            y = self.axis_min[y_ax] \
                                    + (y + 0.5) * abs(self.voxel_sizes[y_ax])
                        contour_points.append((x, y))
                    points[i].append(contour_points)

        return points

    def set_structs_as_mask(self):
        """Create masked image using loaded structures."""

        # Check we can set structs as mask
        if not self.has_structs:
            self.logger.warning("No structures have been loaded! Cannot set "
                                "structures as mask.")
            return

        # Set structs as mask
        self.logger.debug("Setting structures as masks")
        for view in _plot_axes:
            self.apply_mask([struct for key, struct in self.structs.items() 
                             if self.show_struct[key]], view)

    def count_colorbars(self):
        return int(self.colorbar) + int(self.dose_colorbar) \
                + int(self.jac_colorbar)

    def increase_slice(self, n=1):
        """Increase slice slider value by n slices."""

        new_val = self.slice_slider.value + n * self.slice_slider.step
        if new_val <= self.slice_slider.max:
            self.slice_slider.value = new_val
        else:
            self.slice_slider.value = self.slice_slider.max

    def decrease_slice(self, n):
        """Decrease slice slider value by n slices."""

        new_val = self.slice_slider.value - n * self.slice_slider.step
        if new_val >= self.slice_slider.min:
            self.slice_slider.value = new_val
        else:
            self.slice_slider.value = self.slice_slider.min

    def jump_to_struct(self, include_orthog=False):
        """Jump to the midpoint of a structure for a given image."""

        # Get full key of structure to jump
        s_name = self.struct_jump_menu.value
        self.logger.debug(f"Jumping to middle of structure {s_name}")
        self.current_struct = s_name
        if s_name == "":
            return
        idx = self.structs_for_jump.index(s_name) - 1
        s_key = list(self.structs.keys())[idx]

        # Get midpoint in current orientation
        slices = list(self.contours[self.view][s_key].keys())
        mid_slice = int(np.mean(slices))

        # Jump to the slice
        if self.scale_in_mm:
            new_slice = self.voxel_to_pos(_slider_axes[self.view], mid_slice)
        else:
            new_slice = self.idx_to_slider(_slider_axes[self.view], mid_slice)
        self.slice_slider.value = new_slice

        # Set slice number of orthogonal view
        if include_orthog:
            orthog = _orthog[self.view]
            slices_orthog = list(self.contours[orthog][s_key].keys())
            self.orthog_slice[self.view] = int(np.mean(slices_orthog))


class QuickViewer:
    def __init__(
        self,
        file_path,
        init_idx=None,
        v=(-300, 200),
        figsize=5,
        continuous_update=False,
        translation=False,
        translation_file_to_overwrite=None,
        show_cb=False,
        overlay=False,
        show_diff=False,
        title=None,
        mask=None,
        invert_mask=False,
        mask_colour="black",
        dose=None,
        dose_opacity=0.5,
        dose_cmap="jet",
        cmap="gray",
        overlay_opacity=0.5,
        overlay_legend=False,
        share_slider=True,
        structs=None,
        struct_opacity=1,
        struct_linewidth=2,
        struct_plot_type="contour",
        struct_colours=None,
        struct_legend=True,
        legend_loc='lower left',
        zoom=None,
        downsample=None,
        jacobian=None,
        jacobian_cmap="seismic",
        jacobian_opacity=0.5,
        df=None,
        df_spacing=30,
        df_plot_type="quiver",
        df_linespec="g",
        init_view="x-y",
        comparison_only=False,
        cb_splits=2,
        suptitle=None,
        save_as=None,
        scale_in_mm=True,
        match_axes=None,
        init_pos=None,
        structs_as_mask=False,
        colorbar=False,
        interpolation=None,
        orthog_view=False,
        annotate_slice=None,
        plots_per_row=None,
        no_show=False,
        debug=False
    ):

        """ Display an interactive plot of one or more scans in a jupyter 
        notebook.

        Parameters
        ----------
        file_path : string or list of strings 
            Path(s) to either a NIfTI file or a directory containing DICOM 
            files.

        init_idx : integer, default=None
            Slice number in the initial orientation direction at which to 
            display the first image (can be changed interactively later). If
            None, the central slice will be displayed.

        v : tuple, default=(-300, 200)
            HU thresholds at which to display the first image (can be changed 
            interactively later).

        figsize : float, default=5
            Height of the displayed figure. The width is calculated 
            automatically from this. If
            None, the central slice will be displayed.

        continuous_update : bool, default=False
            If True, the x/y/z slice sliders will continuously update the 
            figure as they are adjusted.

        translation : bool, default=False
            If True, an extra set of widgets will be displayed, allowing the 
            user to apply a translation to the image and write this to an 
            elastix transformation file.

        translation_file_to_overwrite : str, default=None
            If not None and the "translation" option is used, this parameter
            will be used to population the "Original file" and "New file" 
            fields in the translation user interface.

        show_cb : bool, default=False
            If True, a chequerboard image will be displayed. This option will 
            only be applied if the number of images in <file_path> is 2.

        overlay : bool, default=False
            If True, a blue/red transparent overlaid image will be displayed.
            This option will only be applied if the number of images in 
            <file_path> is 2.

        overlay_opacity : float, default=0.5
            Initial opacity of overlay.

        overlay_legend : bool, default=False
            If True, a legend will be drawn on the overlay.

        show_diff : bool, default=False
            If True, a the difference between two images will be shown. This 
            option will only be applied if the number of images in <file_path> 
            is 2.

        title : string or list of strings, default=None
            Custom title(s) to use for the image(s) to be displayed. If the 
            number of titles given, n, is less than the number of images, only
            the first n figures will be given custom titles. If any titles are
            None, the name of the image file will be used as the title.

        mask : string/list of strings/dict/list of dicts, default=None
            Path(s) to NIfTI file(s) containing masks to be applied to the 
            image(s) to be displayed. For a single scan image, this can be:
                (a) a path to a single mask file;
                (b) a dictionary containing the three orientations (x-y, x-z,
                    and y-z) as keys, and paths to seperate files for each as
                    values.
            For more than one scan image, a list of paths or dictionaries as 
            described above can be provided. If the input is not a list but
            more than one scan image is loaded, the mask will be overlaid on 
            the first image.

        invert_mask : bool, default=False
            If True, any loaded masks will be inverted (i.e. areas in which 
            mask is nonzero will be masked)

        mask_colour : matplotlib colour, default="black"
            Colour of masked areas.

        dose : string or list of strings, default=None
            Path(s) to NIfTI file(s) containing dose maps to be applied to the
            image(s) to be displayed. If the number of dose maps given, n, is 
            less than the number of images, only the first n figures will have
            a dose map overlaid. If any dose paths are None, no dose mape will 
            be shown for the corresponding image.

        dose_opacity : float, default=0.5
            Initial opacity of overlaid dose field.
    
        cmap : string, default='cmap'
            Colormap used when displaying images.

        dose_cmap : string, default='jet'
            Colormap used when overlaying dose maps.

        share_slider : bool, default=True
            If True and all images have the same number of voxels, a single set 
            of x/y/z slice sliders will be produced for all images. If images
            have different sizes, this option will be ignored.

        structs : string/list of strings/list of list of strings, default=None
            A string or list of strings corresponding to each image to be 
            shown. These strings can be:
                (a) A path to a NIfTI structure file;
                (b) A path to a directory containing NIfTI structure files;
                (c) A wildcard matching path(s) to NIfTI structure file(s);
                (d) A wildcard matching path(s) to directories containing 
                    NIfTI structure file(s).
            For each image, the structures in all NIfTI files matching the 
            contents of the corresponding string or list of strings will be
            overlaid on the image. If the item corresponding to the image is 
            None, no structures will be overlaid on that image.

        struct_opacity : float, default=1
            Initial opacity of overlaid structures if plotting as masks.

        struct_linewidth : float, default=2
            Initial width of overlaid structurs if plotting as contours.
    
        struct_plot_type : string, default='contour'
            Option to initially plot structures. Can be 'contour', 'mask', or
            None.

        struct_colours : dict, default=None
            Map between structures and colours to use for that structure. The
            key can either be: 
                (a) The path corresponding to the structure NIfTI file;
                (b) The name of the structure (i.e. the name of the NIfTI file
                    without the extension, optionally with underscores replaced
                    by spaces);
                (c) A wildcard matching the name(s) of structure(s) to be 
                    coloured, optionally with underscores replace by spaces;
                (d) A wildcard string matching the path(s) of the structure 
                    file(s) to be given the chosen colour.
            Matching structures will be searched for in that order. If more 
            than one structure matches, all matching structures will have that
            colour. (Note: structure names are case insensitive).

        struct_legend : bool, default=True
            If true, a legend will be displayed for any plot with structures.

        legend_loc : str, default='lower left'
            Location for any legends (structure/overlay), if used.

        zoom : double, default=None
            Amount between by which to zoom in (e.g. zoom=2 would give a 
            2x zoom)
            
        downsample : int or tuple of ints, default=None
            Amount by which to downsample an image. If an int is given, the 
            same degree of downsampling will be applied in all 3 dimensions.
            A tuple containing downsampling values for (x, y, z) can also
            be given. If None, no downsampling is applied.

        jacobian : string or list of strings, default=None
            Path(s) to NIfTI file(s) containing Jacobian determinants to be 
            applied to the image(s) to be displayed. If the number of files 
            given, n, is less than the number of images, only the first n 
            figures will have a Jacobian determinant overlaid. If any Jacobian 
            paths are None, no Jacobian determinant will be shown for the 
            corresponding image.

        jacobian_cmap : string, default='seismic'
            Colormap used when overlaying Jacobian determinants.

        jacobian_opacity : float, default=0.5
            Initial opacity of overlaid Jacobian determinant.
    
        df : string or list of strings, default=None
            Path(s) to NIfTI file(s) containing deformation fields to 
            be applied to the image(s) to be displayed. If the number of files 
            given, n, is less than the number of images, only the first n 
            figures will have a deformation field overlaid. If any deformation
            field paths are None, no deformation field will be shown for the
            corresponding image.

        df_spacing : int/tuple, default=30
            Spacing between the arrows on the deformation field quiver plot. 
            Can be a single value for spacing in all directions, or a tuple
            with a value for each direction. Units are the same as the units
            of the plot (default = mm).

        df_plot_type : string, default='quiver'
            Option to initially plot deformation field. Can be 'grid', 
            'quiver', or None.

        df_linespec : string, default='g'
            Linespec for deformation field plot if the "grid" option is used.

        init_view : string, default='x-y'
            Orientation at which to initially display the scan(s).

        comparison_only : bool, False
            If True, only comparison images (overlay/chequerboard/difference)
            will be shown. If no comparison options are selected, this 
            parameter will automatically be set to False.

        cb_splits : int, default=2
            Number of sections to show for chequerboard image. Minimum = 2;
            maximum = 100. The number of splits can also be changed later via
            a slider, where the maximum slider value is set to the larger of
            8 and cb_splits.

        suptitle : string, default=None
            Global title for all subplots. If None, no suptitle will be added.

        save_as : string, default=None
            File to which to save the matplotlib figure upon its creation. If 
            this option is used, a button will be added to the UI allowing the 
            user to overwrite this file at a later point.

        scale_in_mm : bool, default=True
            If True, the axis scales will be shown in mm instead of array
            indices.

        match_axes : str, default=None
            If not None, this must be a string giving an axis-matching option,
            which can be "largest" or "smallest". The axes of all plots will
            have their limits adjusted to be the same as either the largest
            or smallest plot. Only works when scale_in_mm is True.

        init_pos : float, default=None
            Position in mm of the first slice to display. This will be rounded
            to the nearest slice. If init_pos and init_idx are both given,
            init_pos will override init_idx if scale_in_mm is True.

        structs_as_mask : bool, default=False
            If True, any loaded structures will be used as masks as well 
            as being displayed.

        colorbar : bool, default=False
            If True, colorbars will be displayed for HU, dose and Jacobian 
            determinant.

        interpolation : str, default=None
            Interpolation parameter to be passed to matplotlib.pyplot.imshow.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will 
            be added unless viewing outside jupyter, in which case the 
            annotation will be white by default.

        orthog_view : bool, default=False
            If True, a sagittal view with a line indicating the current slice
            will be shown alongside the axial view.

        plots_per_row : int, default=None
            Number of plots to display before starting a new row. If None,
            all plots will be shown on a single row.

        no_show : bool, default=False
            If True, no image will be displayed when a QuickViewer object
            is created. Requested output files will still be created.

        debug : bool, default=False
            If True, debug information will be written to a file called
            quick_viewer.log.
        """

        # Set up logging
        self.logger = logging.getLogger('QuickViewer')
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.ch.setFormatter(formatter)
        self.logger.addHandler(self.ch)
        if debug:
            self.logger.setLevel(logging.DEBUG)
            logfile_name = "quick_viewer.log"
            if os.path.exists(logfile_name):
                os.remove(logfile_name)
            self.fh = logging.FileHandler('quick_viewer.log')
            formatter = logging.Formatter('%(name)s - %(levelname)s - '
                                          '%(message)s')
            self.fh.setLevel(logging.DEBUG)
            self.fh.setFormatter(formatter)
            self.logger.addHandler(self.fh)
        else:
            self.fh = None

        # Parse input filename(s)
        if isinstance(file_path, str):
            self.paths = [file_path]
        elif isinstance(file_path, list):
            self.paths = file_path
        else:
            raise TypeError("<file_path> must be a string or a list")
        self.n = len(self.paths)

        # Assign settings
        self.in_notebook = in_notebook()
        self.init_idx = init_idx
        self.init_pos = init_pos
        self.init_view = init_view
        try: 
            self.vmin, self.vmax = v
        except (TypeError, ValueError):
            self.logger.warning(f"Could not unpack vmin, vmax from v = {v}."
                                " Using default values.")
            self.vmin, self.vmax = -300, 200
        self.figsize = figsize
        self.continuous_update = continuous_update
        self.show_t = translation
        self.tfile = translation_file_to_overwrite
        self.overlay = overlay
        self.overlay_opacity = overlay_opacity
        self.overlay_legend = overlay_legend
        self.show_cb = show_cb
        self.cb_splits = cb_splits
        self.show_diff = show_diff
        self.titles = self.get_arg_list(title)
        self.suptitle = suptitle
        self.save_as = save_as
        self.mask_paths = self.get_arg_list(mask)
        self.invert_mask = invert_mask
        self.mask_colour = mask_colour
        self.dose_paths = self.get_arg_list(dose)
        self.cmap = cmap
        self.dose_opacity = dose_opacity
        self.dose_cmap = dose_cmap
        self.share_slider = share_slider
        self.structs = self.get_arg_list(structs)
        self.struct_legend = struct_legend
        self.legend_loc = legend_loc
        self.valid_struct_plot_types = ["Mask", "Contour", "None"]
        struct_plot_type = str(struct_plot_type).capitalize()
        self.struct_plot_type = (
            struct_plot_type if struct_plot_type in 
            self.valid_struct_plot_types else self.valid_df_plot_types[0]
        )
        self.struct_colours = {} if struct_colours is None else struct_colours
        self.struct_opacity = struct_opacity
        self.struct_linewidth = struct_linewidth
        self.zoom = zoom
        self.downsample = self.get_downsample_settings(downsample)
        self.scale_in_mm = scale_in_mm
        self.match_axes = match_axes if match_axes in ["largest", "smallest"] \
                else None
        if not self.scale_in_mm and self.match_axes is not None:
            self.logger.warning("match_axes option not available when "
                                "scale_in_mm = False. Axes will not be "
                                "matched.")
            self.match_axes = None
        self.comparison_only = comparison_only
        self.jacobian_paths = self.get_arg_list(jacobian)
        self.jacobian_cmap = jacobian_cmap
        self.jacobian_opacity = jacobian_opacity
        self.df_paths = self.get_arg_list(df)
        self.df_spacing = df_spacing if isinstance(df_spacing, tuple) \
                else (df_spacing, df_spacing, df_spacing)
        self.df_linespec = df_linespec
        self.valid_df_plot_types = ["Grid", "Quiver", "None"]
        df_plot_type = str(df_plot_type).capitalize()
        self.df_plot_type = (
            df_plot_type if df_plot_type in 
            self.valid_df_plot_types else self.valid_df_plot_types[0]
        )
        self.colorbar = colorbar
        self.structs_as_mask = structs_as_mask
        self.interp = interpolation
        self.orthog_view = orthog_view
        self.annotate_slice = annotate_slice
        self.plots_per_row = plots_per_row
        self.no_show = no_show

        # Set unspecified figure titles to be filenames
        for i in range(self.n):
            if self.titles[i] is None:
                self.titles[i] = os.path.basename(self.paths[i])

        # General settings
        self.saved = False
        self.no_update = False
        self.plotted_colorbars = []
        if init_view in _view_map:
            self.init_view = _view_map[init_view]
        if self.init_view not in _plot_axes:
            self.logger.warning(
                f"Unrecognised init_view option {init_view}."
                f" Orientation must be one of {list(_plot_axes.keys())}."
                " Default view (x-y) will be used."
            )
            self.init_view = "x-y"

        # Load images
        self.load_scans()
        if not len(self.images):
            print('[ERROR]: No valid image files found!')
            return None

        # Make UI
        self.plot_kw = {}
        self.horizontal_ui = []
        self.check_ui_settings()
        self.make_main_ui()
        self.make_structure_ui()
        self.make_translation_ui()
        self.make_comparison_ui()

        # Calculate figure sizes/width ratios to use for each orientation
        self.extent_to_match = {}
        self.fig_settings = {
            view: self.get_fig_settings(view) for view in _plot_axes.keys()
        }

        # Make single figure if outside notebook
        if not self.in_notebook:
            self.make_figure(self.init_view)
            self.set_callbacks()
            self.plot_current_slice()

        # Display UI and image
        else:
            self.ui = ipyw.HBox([ipyw.VBox(h) for h in self.horizontal_ui])
            self.out = ipyw.interactive_output(self.plot_slice, self.plot_kw)
        if not self.no_show:
            self.display()

    def display(self):
        """Display interactive image in notebook, or in a matplotlib 
        window outside a notebook."""

        if self.in_notebook:
            from IPython.display import display
            display(self.ui, self.out)
        else:
            plt.show()

    def get_downsample_settings(self, downsample):
        """Convert downsampling input argument to a tuple."""

        if isinstance(downsample, int):
            ds = downsample if downsample != 0 else 1
            return (ds, ds, ds)
        elif isinstance(downsample, tuple) or isinstance(downsample, list):
            dsx = downsample[0] if downsample[0] != 0 else 1
            dsy = downsample[1] if downsample[1] != 0 else 1
            dsz = downsample[2] if downsample[2] != 0 else 1
            return (dsx, dsy, dsz)

    def load_scans(self):
        """Load scan images from self.paths."""

        # Load all images
        self.images = []
        for i, p in enumerate(self.paths):

            # Load image
            im = QuickViewerImage(p, self.titles[i], self.scale_in_mm, 
                                  downsample=self.downsample, 
                                  ch=self.ch, fh=self.fh,
                                  count=len(self.images))
            if im.valid:

                # Load overlays
                im.view = self.init_view
                im.load_dose(self.dose_paths[i])
                im.load_mask(self.mask_paths[i], invert=self.invert_mask)
                im.load_jacobian(self.jacobian_paths[i])
                im.load_df(self.df_paths[i])

                # Load structures
                im.load_structs(self.structs[i], self.struct_colours)
                if self.structs_as_mask:
                    im.set_structs_as_mask()

                # Add to list of images
                self.images.append(im)

            else:
                self.logger.warning(f"File {im.path} not found! Image will "
                                    "not be plotted.")

        # Return if no valid images found
        if not len(self.images):
            return

        # Check which extras were loaded
        self.n_masks = sum([int(im.has_mask) for im in self.images])
        self.use_dose = True in [im.has_dose for im in self.images]
        self.use_jacobian = True in [im.has_jacobian for im in self.images]
        self.use_df = True in [im.has_df for im in self.images]
        self.use_structs = True in [im.has_structs for im in self.images]

        # Set global HU range 
        v_lower = min([im.hu_min for im in self.images])
        v_upper = max([im.hu_max for im in self.images])
        self.hu_min = v_lower if v_lower > -2000 else -2000
        self.hu_max = v_upper if v_upper < 2000 else 2000

        # Check whether images have same shape
        self.all_same_shape = same_shape([im.image for im in self.images])
        self.share_slider *= self.all_same_shape

        # Colorbar settings
        if self.colorbar:
            if self.share_slider and self.colorbar:
                self.images[-1].colorbar = True
            else:
                for im in self.images:
                    im.colorbar = True
            for im in self.images:
                if im.has_dose:
                    im.dose_colorbar = True
                if im.has_jacobian:
                    im.jac_colorbar = True

    def make_main_ui(self):
        """Make slice and HU sliders, mask toggle, dose opacity slider,
        and orientation radio buttons."""

        for i, im in enumerate(self.images):

            # Use shared slider if requested
            if self.share_slider and i > 0:

                im.share_slice_slider(self.images[0])
                im.hu_slider = self.images[0].hu_slider

            else:

                # Make new sliders
                im.make_slice_slider(_slider_axes[self.init_view], 
                                     self.init_idx, self.init_pos,
                                     self.continuous_update)
                im.make_hu_slider(self.hu_min, self.hu_max, 
                                  self.vmin, self.vmax)

                # Add to horizontal UI list
                vertical_ui = [im.slice_slider, im.hu_slider]
                if not self.share_slider:
                    vertical_ui.insert(0, ipyw.Label(value=im.title + ":"))
                self.horizontal_ui.append(vertical_ui)

            # Add to keyword arguments for plotting
            self.plot_kw[f"z{i}"] = im.slice_slider
            self.plot_kw[f"v{i}"] = im.hu_slider

        # Add checkbox for turning off masks
        if self.n_masks:
            mask_desc = f'Apply mask{(self.n_masks > 1) * "s"}'
            self.mask_checkbox = ipyw.Checkbox(value=True, 
                                               description=mask_desc)
            self.horizontal_ui[0].append(self.mask_checkbox)
            self.plot_kw["mask"] = self.mask_checkbox

        # Add dose opacity slider
        if self.use_dose:
            self.dose_slider = self.make_opacity_slider(
                "dose", self.dose_opacity)
            self.horizontal_ui[0].append(self.dose_slider)
            self.plot_kw["dose_opacity"] = self.dose_slider

        # Add Jacobian sliders
        if self.use_jacobian:

            # Opacity slider
            self.jacobian_slider = self.make_opacity_slider(
                "jacobian", self.jacobian_opacity)
            self.horizontal_ui[0].append(self.jacobian_slider)
            self.plot_kw["jacobian_opacity"] = self.jacobian_slider

            # Colour range slider
            self.jacobian_range_slider = ipyw.FloatRangeSlider(
                min=-0.5,
                max=2.5,
                step=0.1,
                value=[0.8, 1.2],
                description="Jacobian range",
                continuous_update=False,
                style=_style,
                readout_format=".1f",
            )
            self.horizontal_ui[0].append(self.jacobian_range_slider)
            self.plot_kw["jacobian_range"] = self.jacobian_range_slider

        # Add dropdown menu for changing deformation field display
        if self.use_df:
            self.df_menu = ipyw.Dropdown(
                options=self.valid_df_plot_types,
                value=self.df_plot_type,
                description="Deformation field",
                style=_style,
            )
            self.horizontal_ui[0].append(self.df_menu)
            self.plot_kw["df_plot_type"] = self.df_menu

        # Make radio buttons for selecting orientation
        self.view_radio = ipyw.RadioButtons(
            options=["x-y", "y-z", "x-z"],
            value=self.init_view,
            description="Slice plane selection:",
            disabled=False,
            style=_style,
        )
        self.current_view = self.view_radio.value
        self.plot_kw["view"] = self.view_radio

        if self.show_t or self.use_structs:
            self.horizontal_ui[0].append(self.view_radio)
        else:
            self.horizontal_ui.append([self.view_radio])

        # Make save button
        if self.save_as != None:
            self.save_name = ipyw.Text(description="Output file:", 
                                       value=self.save_as)
            self.save_button = ipyw.Button(description="Save plot")
            self.save_button.on_click(self.savefig)
            self.horizontal_ui[0].append(self.save_name)
            self.horizontal_ui[0].append(self.save_button)

    def make_structure_ui(self):
        """Make structure opacity slider and list of checkboxes to turn each
        structure on/off."""

        # Check we need this
        if not self.use_structs:
            return

        # Make structure UI
        self.struct_menu = ipyw.Dropdown(
            options=self.valid_struct_plot_types,
            value=self.struct_plot_type,
            description="Structure plotting",
            style=_style,
        )
        self.struct_slider = self.make_opacity_slider("Structure", 
                                                      self.struct_opacity)
        self.current_struct_plot_type = self.struct_plot_type
        self.update_struct_slider()
        self.plot_kw["struct_plot_type"] = self.struct_menu
        self.plot_kw["struct_property"] = self.struct_slider
        self.horizontal_ui[0].append(self.struct_menu)
        self.horizontal_ui[0].append(self.struct_slider)

        # Make checkboxes for each scan image
        self.struct_checkboxes = []
        for i, im in enumerate(self.images):

            if not im.has_structs:
                continue

            # Make checkbox for each structure
            im.struct_checkboxes = {}
            for struct in im.structs:

                # Add checkbox for each structure
                im.struct_checkboxes[struct] = ipyw.Checkbox(
                    value=True, description=im.struct_names_nice[struct])
                self.plot_kw[f"show_struct{i}_{struct}"] \
                    = im.struct_checkboxes[struct]
                self.struct_checkboxes.append(im.struct_checkboxes[struct])

            # Add checkboxes to UI
            n_cols = np.ceil(len(im.structs) / 10)
            current_ui = []
            if len(self.images) > 1 or n_cols == 1:
                self.horizontal_ui.append(list(im.struct_checkboxes.values()))
            else:
                max_per_col = np.ceil(len(im.structs) / n_cols)
                for box in im.struct_checkboxes.values():
                    current_ui.append(box)
                    if len(current_ui) == max_per_col:
                        self.horizontal_ui.append(current_ui)
                        current_ui = []
                if len(current_ui):
                    self.horizontal_ui.append(current_ui)

            # Add dropdown for jumping to a structure
            im.structs_for_jump = [""] + list(im.struct_names_nice.values())
            im.struct_jump_menu = ipyw.Dropdown(
                options=im.structs_for_jump,
                value="",
                description="Jump to:",
                style=_style,
            )
            self.horizontal_ui[-1].append(im.struct_jump_menu)
            self.plot_kw[f"jump{i}"] = im.struct_jump_menu
            im.current_struct = ""

    def update_struct_slider(self):
        """Update struct slider to show opacity for mask plotting, or line
        thickness for contour plotting."""

        self.no_update = True
        if self.current_struct_plot_type == "Mask":
            self.struct_slider.min = 0
            self.struct_slider.max = 1
            self.struct_slider.step = 0.1
            self.struct_slider.value = self.struct_opacity
            self.struct_slider.description = 'Structure opacity'
            self.struct_slider.readout_format = '.1f'
        else:
            self.struct_slider.min = 1
            self.struct_slider.max = 8
            self.struct_slider.value = self.struct_linewidth
            self.struct_slider.step = 1
            self.struct_slider.description = 'Structure linewidth'
            self.struct_slider.readout_format = '.0f'
        self.no_update = False

    def make_translation_ui(self):
        """Make sliders for applying voxel translations to images and button
        for writing the translation to a file."""

        # Check we need this
        if not self.show_t:
            return

        # Make label and sliders
        self.tlabel = ipyw.Label(value="Translation:")
        self.tsliders = {}
        for ax in _axes:
            n = self.images[1].n_voxels[ax]
            self.tsliders[ax] = ipyw.IntSlider(
                min=-n,
                max=n,
                value=0,
                description=f"{ax} (0 mm)",
                continuous_update=False,
            )
            if self.show_t:
                self.plot_kw[f"d{ax}"] = self.tsliders[ax]

        # UI for writing translation to a file
        if self.tfile is None:
            infile, outfile = self.find_translation_files()
        else:
            infile, outfile = self.tfile, self.tfile
        self.tfilein = ipyw.Text(description="Original file:", value=infile)
        self.tfileout = ipyw.Text(description="New file:", value=outfile)
        self.tbutton = ipyw.Button(description="Write to file")
        self.tbutton.on_click(self.write_translation_to_file)
        tui = (
            [self.tlabel]
            + list(self.tsliders.values())
            + [self.tfilein, self.tfileout, self.tbutton]
        )
        self.horizontal_ui.append(tui)

    def make_opacity_slider(self, name, init_val):
        """ Make an opacity slider."""
        return ipyw.FloatSlider(
            value=init_val,
            min=0,
            max=1,
            step=0.1,
            description=f"{name.capitalize()} opacity",
            continuous_update=self.continuous_update,
            readout_format=".1f",
            style=_style,
        )

    def check_ui_settings(self):
        """Check for any incompatible UI options and adjust."""

        # Don't share allow difference image if scans have different shapes
        self.show_diff *= self.all_same_shape

        # Only allow comparison images if number of scans images is 2
        if self.n != 2:
            if self.show_cb or self.overlay or self.show_diff:
                self.logger.warning(
                    "Chosen viewing options can only be shown "
                    "when number of files = 2. Some options will not be "
                    "shown."
                )
                self.show_cb = False
                self.overlay = False
                self.show_diff = False

            if self.show_t:
                self.logger.warning(
                    "Custom translation option can only be used "
                    "when number of files = 2. Translation will not be "
                    "enabled."
                )
                self.show_t = False

        # Don't allow comparison_only if no comparisons are being used
        if self.comparison_only and not (
            self.show_cb or self.show_diff or self.overlay
        ):
            self.logger.warning(
                'Option "comparison_only" is not available, as no'
                " comparison images are being displayed."
            )
            self.comparison_only = False

        # Check for valid number of chequerboard splits
        self.cb_split_min = 2
        self.cb_split_max = 8
        if self.show_cb:
            if self.cb_splits < 2:
                self.logger.warning("Minimum number of chequerboard splits is "
                                    "2.")
                self.cb_splits = 2
            elif self.cb_splits > self.cb_split_max:
                if self.cb_splits > 100:
                    self.logger.warning("Maximum number of chequerboard splits"
                                        "is 100.")
                    self.cb_split_max = 100
                else:
                    self.cb_split_max = self.cb_splits + 2

        # Adjust colorbar settings
        self.diff_colorbar = self.show_diff and self.colorbar
        self.cb_colorbar = self.show_cb and not self.show_diff \
                and self.colorbar
        if (self.cb_colorbar or self.diff_colorbar) and self.share_slider:
            for im in self.images:
                im.colorbar = False

    def make_comparison_ui(self):
        """Make opacity/inversion sliders for comparison images."""

        comparison_ui = []

        # Chequerboard UI
        if self.show_cb:

            # Inversion option
            self.invert_cb = ipyw.Checkbox(
                value=False, description="Invert chequerboard"
            )
            self.plot_kw["invert_cb"] = self.invert_cb
            comparison_ui.append(self.invert_cb)

            # Adjust number of splits
            self.cb_split_slider = ipyw.IntSlider(
                min=self.cb_split_min,
                max=self.cb_split_max,
                value=self.cb_splits,
                step=1,
                continuous_update=self.continuous_update,
                description="Chequerboard splits",
                style=_style,
            )
            self.plot_kw["cb_splits"] = self.cb_split_slider
            comparison_ui.append(self.cb_split_slider)

        # Overlay inversion and opacity
        if self.overlay:
            self.invert_overlay = ipyw.Checkbox(
                value=False, description="Invert overlay"
            )
            self.overlay_slider = self.make_opacity_slider(
                "Overlay", self.overlay_opacity)
            self.plot_kw["invert_overlay"] = self.invert_overlay
            self.plot_kw["overlay_opacity"] = self.overlay_slider
            comparison_ui.append(self.invert_overlay)
            comparison_ui.append(self.overlay_slider)

        # Difference image inversion
        if self.show_diff:
            self.invert_diff = ipyw.Checkbox(
                value=False, description="Invert difference"
            )
            self.plot_kw["invert_diff"] = self.invert_diff
            comparison_ui.append(self.invert_diff)

        # Add any extra elements to horizontal UI
        if len(comparison_ui):
            self.horizontal_ui.append(comparison_ui)

    def find_translation_files(self):
        """Try to find input translation file based on image filename and
        create a corresponding output filename."""

        if not self.show_t:
            return "", ""
        im2_dir = os.path.dirname(self.paths[1])
        tfile = os.path.join(im2_dir, "TransformParameters.0.txt")
        if os.path.isfile(tfile):
            infile = tfile
            outfile = os.path.join(im2_dir, "TransformParameters_custom.txt")
            return infile, outfile
        else:
            return "", ""

    def plot_current_slice(self):
        """Plot a slice using the current UI values."""

        plot_kwargs = {kw: w.value for kw, w in self.plot_kw.items()}
        self.plot_slice(**plot_kwargs)

    def plot_slice(self, **kwargs):
        """Produce a matplotlib figure showing scans with additional user-input
        options.

        Parameters
        ----------
        z<img_idx> : int
            Index of the slice to display for a given image index. This can be 
            in the x/y/z direction depending on the current orientation.

        v<img_idx> : tuple
            HU thresholds for the colour map for a given image index.

        view : string (either 'x-y', 'y-x', or 'x-z')
            Orientation in whcih to display the image.

        invert_cb : bool, default=False
            If False, the standard version of the chequerboard image will be 
            shown (first image in top-left corner). If True, the inverted
            chequerboard will be displayed. (If self.show_cb is False, this 
            option is ignored.)

        cb_splits : int, default=2
            Number of splits for chequerboard image.

        invert_overlay : bool, default=False
            If False, the standard version of the overlay image will be 
            shown (first image in blue, second image in red). If True, the 
            inverted overlay will be displayed. (If self.overlay is False, this 
            option is ignored.)

        overlay_opacity : float, default=0.5
            Alpha for the second image overlaid on the first image. (If 
            self.show_cb is False, this option is ignored.)

        mask : bool, default=False
            If True, the masked version of the image(s) will be displayed. (If
            self.apply_masks is False, this option is ignored.)

        dose_opacity : float, default=0.5
            Alpha for the dose fields overlaid on images.

        jacobian_opacity : float, default=0.5
            Alpha for the Jacobian determinants overlaid on images.

        jacobian_range : tuple of floats
            Range of the Jacobian colour map.

        dx : int, default=0
            Translation (number of voxels) to apply in the x direction.

        dy : int, default=0
            Translation (number of voxels) to apply in the y direction.
    
        dx : int, default=0
            Translation (number of voxels) to apply in the z direction.

        show_struct<img_idx>_<ROI> : bool
            Option to display an ROI for a given image index.

        df : bool, default=True
            Whether or not to overlay deformation field(s). 
        """

        # Check whether the plot should be updated
        if self.no_update:
            return

        # Get input args
        z = [kwargs.get(f"z{i}") for i in range(self.n)]
        v = [kwargs.get(f"v{i}") for i in range(self.n)]
        v_default = v[0]
        view = kwargs.get("view")
        invert_cb = kwargs.get("invert_cb", False)
        cb_splits = kwargs.get("cb_splits", 2)
        invert_overlay = kwargs.get("invert_overlay", False)
        invert_diff = kwargs.get("invert_diff", False)
        overlay_opacity = kwargs.get("overlay_opacity", 0.5)
        mask = kwargs.get("mask", False)
        dose_opacity = kwargs.get("dose_opacity", 0.5)
        jacobian_opacity = kwargs.get("jacobian_opacity", 0.5)
        jacobian_range = kwargs.get("jacobian_range")
        dx = kwargs.get("dx", 0)
        dy = kwargs.get("dy", 0)
        dz = kwargs.get("dz", 0)
        save = kwargs.get("save", False)
        struct_plot_type = kwargs.get("struct_plot_type", "None")
        struct_property = kwargs.get("struct_property", 1)
        df_plot_type = kwargs.get("df_plot_type", "None")
        show_df = df_plot_type != "None"
        jump_to = [kwargs.get(f"jump{i}") for i in range(self.n)]

        # Jump to selected structures
        for i, im in enumerate(self.images):
            if im.has_structs:
                if jump_to[i] != im.current_struct:
                    self.no_update = True
                    im.jump_to_struct(self.orthog_view)
                    z[i] = im.slice_slider.value
                    im.struct_jump_menu.value = ""
                    self.no_update = False

        # Check which structures should be shown
        for i, im in enumerate(self.images):
            if not im.has_structs:
                continue
            prev_show_struct = im.show_struct
            im.show_struct = {struct: kwargs.get(f'show_struct{i}_{struct}')
                              for struct in im.structs}

            # Reset structure masks if something has changed
            if self.structs_as_mask:
                for struct in im.show_struct:
                    if im.show_struct[struct] != prev_show_struct[struct]:
                        im.set_structs_as_mask()

        # Check whether view was changed and update sliders if so
        view_changed = view != self.current_view
        if view_changed:
            self.current_view = view
            self.no_update = True
            for im in self.images:
                im.update_slice_slider(_slider_axes[view])
                im.view = view
            self.no_update = False
            z = [im.slice_slider.value for im in self.images]
        else:
            for i, im in enumerate(self.images):
                im.current_pos[_slider_axes[view]] = z[i]

        # Adjust structure UI
        if self.use_structs:

            # Disable parts of UI if plot style is "None"
            disabled = struct_plot_type == "None"
            self.struct_slider.disabled = disabled
            if not self.structs_as_mask:
                for cb in self.struct_checkboxes:
                    cb.disabled = disabled

            # Update slider if needed
            if struct_plot_type != "None" \
               and struct_plot_type != self.current_struct_plot_type:
                self.current_struct_plot_type = struct_plot_type
                self.update_struct_slider()
                struct_property = self.struct_slider.value

            # Store the structure property
            if struct_plot_type == "Contour":
                self.struct_linewidth = struct_property
            else:
                self.struct_opacity = struct_property

        # Update translation slider descriptions
        delta = {"x": dx, "y": dy, "z": dz}
        if self.show_t:
            for ax, d in delta.items():
                length = abs(self.images[1].voxel_sizes[ax]) * d
                self.tsliders[ax].description = f"{ax} ({length:.0f} mm)"

        # Remake figure and subplots
        if self.in_notebook:
            self.make_figure(view)
        else:
            # Keep same axes but clear them
            for ax in self.axlist:
                ax.clear()
            if view_changed:
                fs = self.fig_settings[view]
                self.fig.set_size_inches(fs["width"], fs["height"])

        # Set suptitle
        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)

        # Get translated image
        translated_image = None
        if self.show_t and (dx or dy or dz):
            if mask:
                im_to_translate = self.images[1].masked_image[view]
            else:
                im_to_translate = self.images[1].image
            translated_image = self.apply_translation(im_to_translate,
                                                      dx, dy, dz)

        # Clear any existing colorbars
        for clb in self.plotted_colorbars:
            clb.remove()
        self.plotted_colorbars = []

        # Plot images
        ims_to_show = []
        for i, im in enumerate(self.images):

            # Convert slice position to voxel if needed
            ax = _slider_axes[view]
            sl = (
                im.slider_to_idx[ax][z[i]] if not self.scale_in_mm else 
                im.pos_to_voxel(ax, z[i], take_nearest=True)
            )

            # Update slice slider description
            if not self.scale_in_mm:
                pos = im.voxel_to_pos(ax, sl)
                im.slice_slider.description = f"{ax} ({pos} mm)"

            # Get image to show
            image = im.masked_image[view] if mask else im.image
            if i == 1 and translated_image is not None:
                image = translated_image
            ims_to_show.append(get_image_slice(image, view, sl))

            # Don't plot images if only showing comparisons
            if self.comparison_only:
                continue

            # Plot the image
            if self.interp is not None:
                interp = self.interp
            else:
                interp = 'nearest' if mask else self.interp
            cmap = copy.copy(cm.get_cmap(self.cmap))
            cmap.set_bad(color=self.mask_colour)
            im_mesh = im.ax.imshow(
                ims_to_show[i], cmap=cmap, vmin=v[i][0], vmax=v[i][1],
                aspect=im.aspect[view], interpolation=interp,
                extent=im.extent[view]
            )
            title = self.titles[i]
            im.ax.set_title(title)

            # Plot orthogonal view
            if self.orthog_view:
                orthog = _orthog[view]
                orthog_im = get_image_slice(im.image, orthog, 
                                            im.orthog_slice[view])
                im.orthog_ax.set_visible(True)
                im.orthog_ax.imshow(
                    orthog_im, cmap=cmap, vmin=v[i][0], vmax=v[i][1],
                    aspect=im.aspect[orthog], interpolation=interp,
                    extent=im.extent[orthog]
                )
                if self.scale_in_mm:
                    if view == "x-y":
                        x_pos = [z[i], z[i]]
                        y_pos = im.extent[orthog][2:]
                    else:
                        x_pos = im.extent[orthog][:2]
                        y_pos = [z[i], z[i]]
                else:
                    if view == "x-y":
                        x_pos = [sl, sl]
                        y_pos = [0, im.n_voxels[_plot_axes[orthog][1]] - 1]
                    else:
                        x_pos = [0, im.n_voxels[_plot_axes[orthog][0]] - 1]
                        y_pos = [sl, sl]

            # Turn off orthogonal axes if outside notebook
            else:
                if self.orthog_view and not self.in_notebook:
                    im.orthog_ax.set_visible(False)

            # Add colorbar
            ax_for_clb = im.ax if not self.orthog_view else im.orthog_ax
            if im.colorbar:
                clb = self.fig.colorbar(im_mesh, ax=ax_for_clb)
                clb.solids.set_edgecolor("face")
                clb.ax.set_title("HU")
                self.plotted_colorbars.append(clb)

            # Plot dose map
            if im.has_dose:
                dose = im.masked_dose[view] if mask else im.dose
                dose_cmap = copy.copy(cm.get_cmap(self.dose_cmap))
                dose_cmap.set_bad(color='black')
                dmap_to_show = get_image_slice(dose, view, sl)
                dose_mesh = im.ax.imshow(
                    dmap_to_show, alpha=dose_opacity, cmap=dose_cmap,
                    aspect=im.aspect[view], interpolation=interp,
                    extent=im.extent[view]
                )
                if im.dose_colorbar and dose_opacity:
                    clb = self.fig.colorbar(dose_mesh, ax=ax_for_clb)
                    clb.solids.set_edgecolor("face")
                    clb.ax.set_title("Dose (Gy)")
                    self.plotted_colorbars.append(clb)

            # Plot Jacobian determinant
            if im.has_jacobian:
                jmap_to_show = get_image_slice(im.jacobian, view, sl)
                jac_mesh = im.ax.imshow(
                    jmap_to_show,
                    alpha=jacobian_opacity,
                    cmap=self.jacobian_cmap,
                    vmin=jacobian_range[0],
                    vmax=jacobian_range[1],
                    aspect=im.aspect[view], 
                    extent=im.extent[view]
                )
                if im.jac_colorbar and jacobian_opacity:
                    clb = self.fig.colorbar(jac_mesh, ax=ax_for_clb)
                    clb.solids.set_edgecolor("face")
                    #  clb.ax.set_title("Jacobian determinant")
                    self.plotted_colorbars.append(clb)

            # Plot df
            if im.has_df:
                self.plot_df(im.ax, im, view, sl, df_plot_type)

            # Plot structures
            struct_handles = []
            for struct, struct_img in im.structs.items():
                if im.show_struct[struct]:

                    # Get colour to plot
                    color = colors.to_rgba(im.struct_colours[struct])
                    if sl in im.contours[view][struct] \
                       and struct_plot_type != "None":
                        struct_handles.append(mpatches.Patch(
                            color=color, label=im.struct_names_nice[struct]))

                    # Plot as mask
                    if struct_plot_type == "Mask":
                        self.plot_struct_mask(im.ax, struct_img, im, view,
                                              sl, color)
                        if self.orthog_view:
                            self.plot_struct_mask(im.orthog_ax, struct_img,
                                                  im, _orthog[view], 
                                                  im.orthog_slice[view],
                                                  color)

                    # Plot as contour
                    elif struct_plot_type == "Contour":
                        self.plot_contours(im.ax, im.contours[view][struct], 
                                           sl, color)
                        if self.orthog_view:
                            self.plot_contours(im.orthog_ax, 
                                               im.contours[orthog][struct],
                                               im.orthog_slice[view],
                                               color)

            # Plot indicator line on orthogonal view
            if self.orthog_view:
                im.orthog_ax.plot(x_pos, y_pos, 'r')

            # Add structure legend
            if self.struct_legend and len(struct_handles):
                im.ax.legend(
                    handles=struct_handles, loc=self.legend_loc,
                    facecolor="white", framealpha=1)

            # Annotate with slice number
            if not self.in_notebook or self.annotate_slice is not None:

                # Get string
                ax = _slider_axes[view]
                if self.scale_in_mm:
                    pos_str = im.position_fmt[ax].format(z[i])
                    z_str = f"{ax} = {pos_str} mm"
                else:
                    z_str = f"{ax} = {z[i]:.0f}"

                # Annotate
                col = self.annotate_slice if self.annotate_slice is not None \
                        else "white"
                im.ax.annotate(z_str, xy=(0.05, 0.93),
                                        xycoords='axes fraction',
                                        color=col)

        # Plot chequerboard image
        if self.show_cb:
            i1, i2 = self.get_comparison_idx(invert_cb)
            cb_image = self.get_chequerboard_image(
                ims_to_show[i1], ims_to_show[i2], cb_splits
            )
            ax_idx = -1 - self.overlay - self.show_diff
            cb_mesh = self.axlist[ax_idx].imshow(
                cb_image,
                cmap=self.cmap,
                vmin=v_default[0],
                vmax=v_default[1],
                aspect=self.images[i1].aspect[view],
                extent=self.images[i1].extent[view]
            )
            self.axlist[ax_idx].set_title("Chequerboard")
            if self.cb_colorbar:
                clb = self.fig.colorbar(cb_mesh, ax=self.axlist[ax_idx])
                clb.solids.set_edgecolor("face")
                clb.ax.set_title("HU")
                self.plotted_colorbars.append(clb)

        # Plot difference between the images
        if self.show_diff:
            i1, i2 = self.get_comparison_idx(invert_diff)
            im1 = ims_to_show[i1]
            im2 = ims_to_show[i2]
            diff = im1 - im2
            ax_idx = -1 - self.overlay
            diff_mesh = self.axlist[ax_idx].imshow(
                diff,
                cmap=self.cmap,
                vmin=v_default[0],
                vmax=v_default[1],
                aspect=self.images[i1].aspect[view],
                extent=self.images[i1].extent[view]
            )
            self.axlist[ax_idx].set_title("Difference")
            if self.diff_colorbar:
                clb = self.fig.colorbar(diff_mesh, ax=self.axlist[ax_idx])
                clb.solids.set_edgecolor("face")
                clb.ax.set_title("HU")
                self.plotted_colorbars.append(clb)

        # Plot overlay image
        if self.overlay:
            i1, i2 = self.get_comparison_idx(invert_overlay)
            im1 = ims_to_show[i1]
            im2 = ims_to_show[i2]
            ax_idx = -1
            mesh1 = self.axlist[ax_idx].imshow(
                im1, cmap="Blues", vmin=v_default[0], vmax=v_default[1], 
                aspect=self.images[i1].aspect[view],
                extent=self.images[i1].extent[view]
            )
            mesh2 = self.axlist[ax_idx].imshow(
                im2,
                cmap="Reds",
                vmin=v_default[0],
                vmax=v_default[1],
                aspect=self.images[i2].aspect[view],
                alpha=overlay_opacity,
                extent=self.images[i2].extent[view]
            )
            self.axlist[ax_idx].set_title("Overlay")

            # Add colorbar
            if self.colorbar:
                for m in [mesh1, mesh2]:
                    clb = self.fig.colorbar(m, ax=self.axlist[ax_idx])
                    clb.solids.set_edgecolor("face")
                    clb.ax.set_title("HU")
                    self.plotted_colorbars.append(clb)

            # Add overlay legend
            if self.overlay_legend:
                overlay_handles = [
                    mpatches.Patch(color="blue", alpha=(1 - overlay_opacity),
                                   label=self.images[i1].title),
                    mpatches.Patch(color="red", alpha=overlay_opacity,
                                   label=self.images[i2].title)
                ]
                self.axlist[ax_idx].legend(
                    handles=overlay_handles, loc=self.legend_loc,
                    facecolor="white", framealpha=1)

        # Label all axes
        orthog_axes = [] if not self.orthog_view else [im.orthog_ax for im in
                                                       self.images]
        for ax in self.axlist:
            if ax in orthog_axes:
                self.label_axes(ax, _orthog[view], no_y=False)
            else:
                self.label_axes(ax, view)

        # Adjust axes to have same extent
        if self.match_axes is not None:
            self.adjust_axes(self.axlist, view)

        # Zoom in on axes
        if self.zoom is not None:
            for ax in self.axlist:
                self.zoom_axes(ax, self.zoom)

        # Set unused slots to be blank
        for ax in self.empty_axes:
            ax.set_visible(False)

        # Apply tight layout
        if self.in_notebook:
            plt.tight_layout()
            self.fig.set_tight_layout(True)
        if self.orthog_view:
            plt.subplots_adjust(wspace=0.1)

        # Redraw
        if not self.in_notebook:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        # Save the figure
        if self.save_as != None and not self.saved:
            self.savefig(self.save_button)
            self.saved = True

    def make_figure(self, view):
        """Create figure and list of axes."""

        fs = self.fig_settings[view]
        self.fig, ax = plt.subplots(
            fs["n_rows"],
            fs["n_cols"],
            figsize=(fs["width"], fs["height"]),
            gridspec_kw={"width_ratios": fs["width_ratios"]}
        )
        if fs["n_images"] == 1:
            ax = np.array([ax])
        axlist = ax.flatten()

        # Assign axes to images
        count = 0
        for im in self.images:
            im.ax = axlist[count]
            count += 1
            if self.orthog_view:
                im.orthog_ax = axlist[count]
                count += 1

        # Seperate empty plots
        self.axlist = axlist[:fs["n_images"]]
        self.empty_axes = axlist[fs["n_images"]:]

    def scroll_images(self, up=True, n=1):
        """Loop through all images and scroll through their slices."""

        for im in self.images:
            if up:
                im.increase_slice(n)
            else:
                im.decrease_slice(n)

    def on_key(self, event):
        """Adjust plot settings when certain keys are pressed."""

        # Settings
        n_small = 1
        n_big = 5

        # Press v to change view
        if event.key == "v":
            next_view = {"x-y": "y-z", "y-z": "x-z", "x-z": "x-y"}
            self.view_radio.value = next_view[self.view_radio.value]

        # Press d to change dose opacity
        elif event.key == "d":
            if self.use_dose:
                doses = [0, 0.15, 0.35, 0.5, 1]
                next_dose = {doses[i]: doses[i + 1] if i + 1 < len(doses) 
                             else doses[0] for i in range(len(doses))}
                diffs = [abs(d - self.dose_slider.value) for d in doses]
                current = doses[diffs.index(min(diffs))]
                self.dose_slider.value = next_dose[current]

        # Press m to switch mask on and off
        elif event.key == "m":
            if self.n_masks:
                self.mask_checkbox.value = not self.mask_checkbox.value

        # Press c to change structure plot type
        elif event.key == "c":
            if self.use_structs:
                next_type = {"Mask": "Contour", "Contour": "None", 
                             "None": "Mask"}
                self.struct_menu.value = next_type[self.struct_menu.value]

        # Press i to invert comparisons
        elif event.key == "i":
            if self.show_cb:
                self.invert_cb.value = not self.invert_cb.value
            if self.overlay:
                self.invert_overlay.value = not self.invert_overlay.value
            if self.show_diff:
                self.invert_diff.value = not self.invert_diff.value

        # Press o to change overlay opacity
        elif event.key == "o":
            if self.overlay:
                ops = [0.2, 0.35, 0.5, 0.65, 0.8]
                next_op = {ops[i]: ops[i + 1] if i + 1 < len(ops) 
                             else ops[0] for i in range(len(ops))}
                diffs = [abs(op - self.overlay_slider.value) for op in ops]
                current = ops[diffs.index(min(diffs))]
                self.overlay_slider.value = next_op[current]

        # Press arrow keys to scroll through many slices
        elif event.key == "left":
            self.scroll_images(False, n_small) 
        elif event.key == "right":
            self.scroll_images(True, n_small) 
        elif event.key == "down":
            self.scroll_images(False, n_big) 
        elif event.key == "up":
            self.scroll_images(True, n_big) 

        # Don't redraw for any other key
        else:
            return

        # Redraw plot
        self.plot_current_slice()

    def on_scroll(self, event):
        """Adjust slice when scrolling."""
        
        if event.button == "up":
            self.scroll_images()
        elif event.button == "down":
            self.scroll_images(False)
        self.plot_current_slice()

    def set_callbacks(self):
        """Connect matplotlib callbacks to self.fig."""
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def zoom_axes(self, ax, zoom):
        """Adjust axis limits in order to zoom in on an image by a given 
        amount."""

        if zoom is None:
            return
        if not isinstance(zoom, tuple):
            zoom = (zoom, zoom)
        xlim = ax.get_xlim()
        mid_x = sum(xlim) / 2
        ax.set_xlim(mid_x - (mid_x - xlim[0]) / zoom[0],
                    mid_x + (xlim[1] - mid_x) / zoom[0])
        ylim = ax.get_ylim()
        mid_y = sum(ylim) / 2
        ax.set_ylim(mid_y - (mid_y - ylim[0]) / zoom[1],
                    mid_y + (ylim[1] - mid_y) / zoom[1])

    def savefig(self, button):
        """Save figure to a file."""
        self.fig.savefig(self.save_name.value)

    def adjust_axes(self, axlist, view):
        """Adjust all axes in a list to have the same limits."""

        for ax in axlist:
            zoom_x = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) \
                    / self.extent_to_match[view][0]
            zoom_y = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) \
                    / self.extent_to_match[view][1]
            self.zoom_axes(ax, (zoom_x, zoom_y))

    def label_axes(self, ax, view, no_y=False):
        """Give appropriate labels to a given axis."""

        units = self.scale_in_mm * " (mm)"
        axis_labels = {"x-y": ("x", "y"), "y-z": ("z", "y"), "x-z": ("z", "x")}
        ax.set_xlabel(axis_labels[view][0] + units)
        if not no_y:
            ax.set_ylabel(axis_labels[view][1] + units)
        else:
            ax.set_yticks([])

    def get_arg_list(self, arg):
        """Convert an argument to a list of values for each path."""

        # Return None for all paths if input arg is None or []
        if arg is None or len(arg) == 0:
            return [None for i in range(self.n)]

        # Convert arg to a list
        input_list = []
        if isinstance(arg, list):
            if self.n == 1:
                input_list = [arg]
            else:
                input_list = arg
        else:
            input_list = [arg]

        # Loop through paths
        arg_list = []
        for i in range(self.n):
            if len(input_list) > i:
                arg_list.append(input_list[i])
            else:
                arg_list.append(None)

        return arg_list

    def get_fig_settings(self, view):
        """Calculate horizontal and vertical figure sizes for a given 
        orientation."""

        # Calculate figure sizes
        width_ratios = []
        fig_height = self.figsize
        x, y = _plot_axes[view]
        extra = 1 + 0.25 * (not self.in_notebook)
        for im in self.images:
            width_ratios.append(fig_height * im.length[x] / im.length[y] 
                                * extra)
            if self.orthog_view:
                orthog = _orthog[view]
                width_ratios.append(fig_height 
                                    * im.length[_plot_axes[orthog][0]]
                                    / im.length[_plot_axes[orthog][1]]
                                    * extra)

        # Adjust width ratios if axes are going to be rescaled
        if self.match_axes is not None:
            x_extents = [im.length[x] for im in self.images]
            y_extents = [im.length[y] for im in self.images]
            func = max if self.match_axes == "largest" else min
            self.extent_to_match[view] = (func(x_extents), func(y_extents))
            width_to_use = fig_height * func(x_extents) / func(y_extents) \
                    * extra
            width_ratios = [width_to_use for w in width_ratios]

        # Add extra padding for colorbars
        cb_extra_hu = 0.2 * extra
        cb_extra = 0.25 * extra
        for i, im in enumerate(self.images):
            ax_to_pad = i if not self.orthog_view else 2 * i
            width_ratios[ax_to_pad] *= 1 + im.colorbar * cb_extra_hu \
                    + (im.jac_colorbar + im.dose_colorbar) * cb_extra

        # Set figure width to zero if only comparison images are being shown
        first_im_width = width_ratios[0]
        if self.comparison_only:
            fig_width = 0
            width_ratios = []

        # Add extra figure size for comparison images
        n_images = (
            (not self.comparison_only) * len(self.images) 
            * (1 + self.orthog_view)
            + self.show_cb
            + self.overlay
            + self.show_diff
        )
        if self.show_cb:
            width_ratios.append(
                first_im_width * (1 + cb_extra * self.cb_colorbar))
        if self.show_diff:
            width_ratios.append(
                first_im_width * (1 + cb_extra * self.diff_colorbar))
        if self.overlay:
            width_ratios.append(
                first_im_width * (1 + 2 * cb_extra * self.colorbar))

        # Get rows/columns
        if self.plots_per_row is not None:
            n_cols = int(self.plots_per_row)
            n_rows = int(np.ceil(n_images / n_cols))
            for i in range(len(width_ratios), n_rows * n_cols):
                width_ratios.append(0)
            ratios_per_row = np.array(width_ratios).reshape(n_rows, n_cols)
            width_ratios = np.amax(ratios_per_row, axis=0)
            fig_height *= n_rows
        else:
            n_rows = 1
            n_cols = n_images

        # Return dictionary of settings
        fig_width = sum(width_ratios)
        settings = {
            "width": fig_width,
            "height": fig_height,
            "width_ratios": width_ratios,
            "n_images": n_images,
            "n_cols": n_cols,
            "n_rows": n_rows,
        }
        self.logger.debug(f"Figure settings {view}:")
        self.logger.debug(f"    Height: {fig_height}")
        self.logger.debug(f"    Width: {fig_width}")
        self.logger.debug(f"    Width ratios: {width_ratios}")
        self.logger.debug(f"    Rows: {n_rows}")
        self.logger.debug(f"    Columns: {n_cols}")
        return settings

    def get_comparison_idx(self, invert=False):
        """Get the indices to two images to compare (optionally inverted)."""

        i1, i2 = 0, 1
        if invert:
            i2, i1 = i1, i2
        return i1, i2

    def get_chequerboard_image(self, im1, im2, n):
        """Create a chequerboard of two images for a given number of 
        sections, n."""

        ni = int(im1.shape[0] / n)
        nj = int(im1.shape[1] / n)
        rows = []
        for i in range(n):
            cols = []
            for j in range(n):
                imin = i * ni
                imax = im1.shape[0] if i == n - 1 else (i + 1) * ni
                jmin = j * nj
                jmax = im1.shape[1] if j == n - 1 else (j + 1) * nj
                if (j + i) % 2 == 0:
                    cols.append(im1[imin:imax, jmin:jmax])
                else:
                    cols.append(im2[imin:imax, jmin:jmax])
            rows.append(np.concatenate(cols, axis=1))
        return np.concatenate(rows, axis=0)

    def apply_translation(self, image, dx, dy, dz):
        """Apply a translation (dx, dy, dx) to an image array."""

        # Create image copy
        image_t = image.copy()

        # x translation
        if dx > 0:
            image_t = np.concatenate(
                [np.zeros((dx, image.shape[1], image.shape[2])), image_t[:-dx, :, :]],
                axis=0,
            )
        elif dx < 0:
            image_t = np.concatenate(
                [image_t[-dx:, :, :], np.zeros((-dx, image.shape[1], image.shape[2]))],
                axis=0,
            )

        # y translation
        if dy > 0:
            image_t = np.concatenate(
                [np.zeros((image.shape[0], dy, image.shape[2])), image_t[:, :-dy, :]],
                axis=1,
            )
        elif dy < 0:
            image_t = np.concatenate(
                [image_t[:, -dy:, :], np.zeros((image.shape[0], -dy, image.shape[2]))],
                axis=1,
            )

        # z translation
        if dz > 0:
            image_t = np.concatenate(
                [np.zeros((image.shape[0], image.shape[1], dz)), image_t[:, :, :-dz]],
                axis=2,
            )
        elif dz < 0:
            image_t = np.concatenate(
                [image_t[:, :, -dz:], np.zeros((image.shape[0], image.shape[1], -dz))],
                axis=2,
            )

        # Return the translated image
        return image_t

    def write_translation_to_file(self, button):

        # Get translation
        delta = {ax: slider.value for ax, slider in self.tsliders.items()}

        # Make signs consistent
        for ax in delta.keys():
            if ax != "y":
                delta[ax] = -delta[ax]

        # Get translations in mm
        delta_mm = {}
        for ax, d in delta.items():
            delta_mm[ax] = d * abs(self.images[1].voxel_sizes[ax])

        # Write to file
        write_translation_to_file(
            self.tfilein.value,
            self.tfileout.value,
            delta_mm["x"],
            delta_mm["y"],
            delta_mm["z"],
        )

    def plot_struct_mask(self, ax, struct_img, im, view, sl, color):
        """Plot a structure mask on some axes."""

        # Get the structure mask
        struct_to_show = get_image_slice(struct_img, view, sl)

        # Make colormap
        norm = colors.Normalize()
        cmap = cm.hsv
        s_colors = cmap(norm(struct_to_show))
        s_colors[struct_to_show > 0, :] = color
        s_colors[struct_to_show == 0, :] = (0, 0, 0, 0)

        # Display the mask
        ax.imshow(
            s_colors, aspect=im.aspect[view], 
            alpha=self.struct_opacity, interpolation='nearest',
            extent=im.extent[view]
        )

    def plot_contours(self, ax, contours, sl, color):
        """Plot a structure's contours for a given slice on some axes."""

        if sl not in contours:
            return
        for points in contours[sl]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            ax.plot(points_x, points_y, color=color,
                    linewidth=self.struct_linewidth)

    def plot_df(self, ax, im, view, sl, plot_type):
        """Plot a deformation field on a given axis."""

        # Check whether we need to extrapolate z position
        if im.extrapolate_df:
            self.extrapolate_df(ax, im, view, sl, plot_type)
            return

        # Get spacing in mm
        if not hasattr(self, "spacing_vox"):
            if self.scale_in_mm:
                self.spacing_vox = {
                    x: abs(int(self.df_spacing[i] / im.voxel_sizes[x]))
                    for i, x in enumerate(_axes)}
            else:
                self.spacing_vox = {
                    x: self.df_spacing[i] for i, x in enumerate(_axes)}

            # Ensure spacing is at least 2 voxels
            for x, sp in self.spacing_vox.items():
                if sp < 2:
                    spacing[x] = 2

        # Get spacing in x and y directions
        x_ax, y_ax = _plot_axes[view]
        spacing_x = self.spacing_vox[x_ax]
        spacing_y = self.spacing_vox[y_ax]

        # Check whether to plot a regular grid
        if im.regular_grid:
            if plot_type == "Grid":
                self.plot_regular_grid(ax, im, view, spacing_x, spacing_y)
            return

        # Get vectors for the x and y axes of the plot
        df = get_image_slice(im.df, view, sl)
        vector_axes = {"x-y": (1, 0), "y-z": (2, 0), "x-z": (2, 1)}
        df_x = np.squeeze(df[:, :, vector_axes[view][0]])
        df_y = np.squeeze(df[:, :, vector_axes[view][1]])
        if view == "x-y":
            df_x = -df_x
        elif view == "x-z":
            df_y = -df_y
        if not self.scale_in_mm:
            df_x = df_x / im.voxel_sizes[x_ax]
            df_y = df_y / im.voxel_sizes[y_ax]

        # Get x and y coordinates
        xs = np.arange(0, df.shape[1])
        ys = np.arange(0, df.shape[0])
        if self.scale_in_mm:
            xs = im.origin[x_ax] + xs * im.voxel_sizes[x_ax]
            ys = im.origin[y_ax] + ys * im.voxel_sizes[y_ax]
        y, x = np.meshgrid(ys, xs)
        x = x.T
        y = y.T

        # Plot a grid
        ax.autoscale(False)
        if plot_type == "Grid":
            df_x = df_x + x
            df_y = df_y + y
            for i in np.arange(0, df.shape[0], spacing_y):
                ax.plot(df_x[i, :], df_y[i, :], self.df_linespec)
            for j in np.arange(0, df.shape[1], spacing_x):
                ax.plot(df_x[:, j], df_y[:, j], self.df_linespec)

        # Make a quiver plot
        elif plot_type == "Quiver":

            # Get arrows
            arrows_x = df_x[::spacing_y, ::spacing_x]
            arrows_y = -df_y[::spacing_y, ::spacing_x]

            # Get plotting positions
            plot_x = x[::spacing_y, ::spacing_x]
            plot_y = y[::spacing_y, ::spacing_x]

            # Plot the arrows
            if arrows_x.any() or arrows_y.any():
                M = np.hypot(arrows_x, arrows_y)
                ax.quiver(plot_x, plot_y, arrows_x, arrows_y, M, cmap="jet")
            else:
                # If arrow lengths are zero, plot dots
                ax.scatter(plot_x, plot_y, c="navy", marker=".")

    def plot_regular_grid(self, ax, im, view, spacing_x, spacing_y):
        """Plot a regularly spaced grid on an image."""

        x_ax, y_ax = _plot_axes[view]
        nx = im.n_voxels[x_ax]
        ny = im.n_voxels[y_ax]
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        for i in np.arange(0, nx, spacing_x):
            x = i if not self.scale_in_mm else im.voxel_to_pos(x_ax, i)
            ax.plot([x, x], ylim, self.df_linespec)
        for j in np.arange(0, ny, spacing_y):
            y = j if not self.scale_in_mm else im.voxel_to_pos(y_ax, j)
            ax.plot(xlim, [y, y], self.df_linespec)

    def extrapolate_df(self, ax, im, view, sl, plot_type):
        """Extrapolate a slice from deformed coordinates and plot the 
        deformation field on an axis."""

        # Get spacing (in voxels) in deformation field's frame of reference
        assert self.scale_in_mm
        axes = {"x-y": (1, 0, 2), "y-z": (2, 0, 1), "x-z": (2, 1, 0)}
        xi, yi, zi = axes[view]
        spacing_x = abs(int(self.df_spacing[0] / im.df_affine[xi, xi]))
        spacing_y = abs(int(self.df_spacing[1] / im.df_affine[yi, yi]))
        if spacing_x < 2:
            spacing_x = 2
        if spacing_y < 2:
            spacing_y = 2

        # Get deformation vectors on x-y grid at each z slice in deformation
        # field's frame of reference
        df = np.array([
            get_image_slice(im.df, view, z)[::spacing_x, ::spacing_y]
            for z in range(0, im.df.shape[zi])
        ])

        # Get coordinates on x-y grid on each slice in deformation field's
        # frame of reference
        if self.scale_in_mm:
            spacing = [spacing_x, spacing_y, im.df_affine[zi, zi]]
        else:
            spacing = [spacing_x, spacing_y, im.df_affine[zi, zi]]
        coords = [np.arange(
            im.df_affine[i, 3],
            im.df_affine[i, 3] + im.df.shape[i] * im.df_affine[i, i],
            im.df_affine[i, i]) for i in (xi, zi, yi)
        ]
        coords[0] = coords[0][::spacing_x]
        coords[2] = coords[2][::spacing_y]
        x, z, y = np.meshgrid(*coords)

        # Get transformed coordinates
        vx = df[:, :, :, xi]
        vy = df[:, :, :, yi]
        vz = df[:, :, :, zi]
        xt = x + vx
        yt = y + vy
        zt = z + vz

        # Get the xy points closest to the current z slice
        dz = abs(zt - im.voxel_to_pos(_slider_axes[view], sl))
        min_idx = np.argmin(dz, axis=0)
        shape_xy = (xt.shape[1], xt.shape[2])
        xt_slice = np.empty(shape_xy)
        yt_slice = np.empty(shape_xy)
        vx_slice = np.empty(shape_xy)
        vy_slice = np.empty(shape_xy)
        for i in range(xt.shape[1]):
            for j in range(xt.shape[2]):
                k = min_idx[i, j]
                xt_slice[i, j] = xt[k, i, j]
                yt_slice[i, j] = yt[k, i, j]
                vx_slice[i, j] = vx[k, i, j]
                vy_slice[i, j] = vy[k, i, j]

        # Quiver: plot arrows at locations x', y', with arrows pointing in 
        # direction of -1 * vector
        if plot_type == "Quiver":
            M = np.hypot(vx_slice, vy_slice)
            ax.quiver(xt_slice, yt_slice, -vx_slice, -vx_slice, 
                      M, cmap="jet")

        # Grid: plot x', y', joined up along the rows and columns of df


def write_translation_to_file(
    input_file, output_file, dx=None, dy=None, dz=None, overwrite=False
):

    """Open an existing elastix transformation file and create a new 
    version with the translation parameters either replaced or added to the 
    current user-created translation in the displayed figure.

    Parameters
    ----------
    input_file : string
        Path to an Elastix translation file to use as an input.

    output_file : string
        Name of the output file to produce.

    dx, dy, dz : float, default=None
        Translations (in mm) to add to the initial translations in the 
        input_file.

    overwrite : bool, default=False
        If True, the shifts will be overwritten. If False, they will be added.
    """

    # Get files
    infile = open(input_file, "r")

    # Make dictionary of shifts
    delta = {"x": dx, "y": dy, "z": dz}

    # Create output text
    out_text = ""
    for line in infile:
        if len(line) == 1:
            out_text += "\n"
            continue
        words = line.split()
        if words[0] == "(TransformParameters":
            old_vals = {
                "x": float(words[1]),
                "y": float(words[2]),
                "z": float(words[3][:-1]),
            }
            new_vals = {}
            for ax, old_val in old_vals.items():
                if delta[ax] is None:
                    new_vals[ax] = old_vals[ax]
                else:
                    if overwrite:
                        new_vals[ax] = delta[ax]
                    else:
                        new_vals[ax] = old_vals[ax] + delta[ax]
            new_line = words[0]
            for val in new_vals.values():
                new_line += " " + str(val)
            new_line += ")\n"
            out_text += new_line
        else:
            out_text += line

    # Write to output
    outfile = open(output_file, "w")
    outfile.write(out_text)
    outfile.close()
    if os.path.normpath(input_file) == os.path.normpath(output_file):
        print("Overwrote translation file:", output_file)
    else:
        print("Wrote new translation file:", output_file)


def new_shrunk_to_aspect(self, box_aspect, container=None, fig_aspect=1.0):
    """
    Return a copy of the :class:`Bbox`, shrunk so that it is as
    large as it can be while having the desired aspect ratio,
    *box_aspect*.  If the box coordinates are relative---that
    is, fractions of a larger box such as a figure---then the
    physical aspect ratio of that figure is specified with
    *fig_aspect*, so that *box_aspect* can also be given as a
    ratio of the absolute dimensions, not the relative dimensions.
    """
    if box_aspect <= 0 or fig_aspect <= 0:
        raise ValueError("'box_aspect' and 'fig_aspect' must be positive")
    if container is None:
        container = self
    w, h = container.size
    H = w * box_aspect / fig_aspect
    W = h * fig_aspect / box_aspect
    H = h
    return matplotlib.transforms.Bbox([self._points[0],
                                       self._points[0] + (W, H)])

# Matplotlib settings
matplotlib.rcParams["figure.figsize"] = (7.4, 4.8)
matplotlib.rcParams["font.serif"] = "Times New Roman"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 14.0
#  matplotlib.transforms.Bbox.shrunk_to_aspect = new_shrunk_to_aspect
