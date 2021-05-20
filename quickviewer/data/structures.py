"""Classes and functions for loading, plotting, and comparing structures."""

import fnmatch
import matplotlib.cm
import numpy as np
import os
import pydicom
import re
import skimage.measure
from shapely import geometry

from quickviewer.data.image import (
    find_files,
    get_unique_path,
    is_list,
    is_nested,
    load_image,
    _axes,
    _slider_axes,
    _plot_axes
)
from quickviewer.data.image import Image


# Standard list of colours for structures
_standard_colors = (
    list(matplotlib.cm.Set1.colors)[:-1]
    + list(matplotlib.cm.Set2.colors)[:-1]
    + list(matplotlib.cm.Set3.colors)
    + list(matplotlib.cm.tab20.colors)
)
for i in [9, 10]:  # Remove greys
    del _standard_colors[i]


class Struct(Image):
    """Class to load and plot a structure as a contour or mask."""

    def __init__(
        self,
        nii=None,
        name=None,
        color=None,
        label="",
        load=True,
        contours=None,
        shape=None,
        origin=None,
        voxel_sizes=None,
        **kwargs,
    ):
        """Load structure mask or contour.

        Parameters
        ----------
        nii : str/array/nifti, default=None
            Source of the image data to load. This can be either:
                (a) The path to a NIfTI file;
                (b) A nibabel.nifti1.Nifti1Image object;
                (c) The path to a file containing a NumPy array;
                (d) A NumPy array.
            This mask will be used to generate contours. If None, the
            <contours> argument must be provided instead.

        name : str, default=None
            Name to assign to this structure. If the structure is loaded
            from a file and name is None, the name will be inferred from
            the filename.

        color : matplotlib color, default=None
            color in which to plot this structure. If None, a random
            color will be assigned. Can also be set later using
            self.assign_color(color).

        label : str, default=""
            User-defined category to which this structure belongs.

        load : bool, default=True
            If True, the structure's data will be loaded and its mask/contours
            will be created during initialise. Otherwise, this information can
            be loaded later with the load() function.

        contours : dict, default=None
            Dictionary of contour points in the x-y orientation, where the keys
            are the z positions and values are the 3D contour point
            coordinates. Only used if the <nii> argument is None. These
            contours are used to generate a mask.

        shape : list, default=None
            Number of voxels in the image in the (x, y, z) directions. Used to
            specify the image shape for the structure mask if <contours> is
            used.

        origin : list, default=None
            Origin position in (x, y, z) coordinates. Used if the structure is
            defined through a NumPy array or a coordinate dictionary.

        voxel_sizes : list, default=None
            Voxel sizes in (x, y, z) coordinates. Used if the structure is
            defined through a NumPy array or a coordinate dictionary.

        kwargs : dict
            Keyword arguments passed to initialisation of the parent
            quickviewer.image.Image object.
        """

        # Assign variables
        if nii is None and contours is None:
            raise TypeError("Must provide either <nii> or <contours> to " "Struct!")
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
            self.name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "", basename).replace(
                " ", "_"
            )
            self.name = make_name_nice(self.name)
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
        #  nice = self.name.replace("_", " ")
        #  self.name_nice = nice[0].upper() + nice[1:]
        self.name_nice = self.name
        self.name_nice_nolabel = self.name_nice
        if self.label:
            self.name_nice += f" ({self.label})"

    def load(self):
        """Load struct data and create contours in all orientations."""
        
        if self.loaded:
            return

        # Create mask from initial set of contours if needed
        if self.nii is None:
            contours_idx = contours_to_indices(
                self.contours, self.origin, self.voxel_sizes, self.shape
            )
            affine = np.array(
                [
                    [self.voxel_sizes[0], 0, 0, self.origin[0]],
                    [0, self.voxel_sizes[1], 0, self.origin[1]],
                    [0, 0, self.voxel_sizes[2], self.origin[2]],
                    [0, 0, 0, 1],
                ]
            )
            self.nii = contours_to_mask(contours_idx, self.shape, affine=affine)

        Image.__init__(self, self.nii, **self.nii_kwargs)
        if not self.valid:
            return

        # Load contours
        self.set_contours()

        # Convert to boolean mask
        if not self.data.dtype == "bool":
            self.data = self.data > 0.5

        # Check whether structure is empty
        self.empty = not sum([len(contours) for contours in self.contours.values()])
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
            self.centroid["voxels"] = [
                self.idx_to_slice(c, axes[i]) for i, c in enumerate(centroid)
            ]
            self.centroid["mm"] = [
                self.idx_to_pos(c, axes[i]) for i, c in enumerate(centroid)
            ]

        return np.array(self.centroid[units])

    def get_centroid_2d(self, view, sl, units="voxels"):
        """Get the centroid position on a 2D slice."""

        if not self.on_slice(view, sl):
            return None, None
        self.set_slice(view, sl)
        non_zero = np.argwhere(self.current_slice)
        cy, cx = non_zero.mean(0)
        x_ax, y_ax = _plot_axes[view]
        conversion = self.idx_to_slice if units == "voxels" else self.idx_to_pos
        if y_ax == "y":
            cy = self.n_voxels[y_ax] - 1 - cy
        return (conversion(cx, x_ax), conversion(cy, y_ax))

    def get_volume(self, units):
        """Get total structure volume in voxels, mm, or ml."""

        if not self.loaded or self.empty:
            return 0

        if not hasattr(self, "volume"):
            self.volume = {"voxels": self.data.astype(bool).sum()}
            self.volume["mm"] = self.volume["voxels"] * abs(
                np.prod(list(self.voxel_sizes.values()))
            )
            self.volume["ml"] = self.volume["mm"] * (0.1 ** 3)

        return self.volume[units]

    def get_struct_length(self, units):
        """Get the total x, y, z length in voxels or mm."""

        if not self.loaded or self.empty:
            return (0, 0, 0)

        if not hasattr(self, "length"):
            self.length = {"voxels": [], "mm": []}
            nonzero = np.argwhere(self.data)
            for ax, n in _axes.items():
                vals = nonzero[:, n]
                if len(vals):
                    self.length["voxels"].append(max(vals) - min(vals))
                    self.length["mm"].append(
                        self.length["voxels"][n] * abs(self.voxel_sizes[ax])
                    )
                else:
                    self.length["voxels"].append(0)
                    self.length["mm"].append(0)

        return self.length[units]

    def get_struct_centre(self, units=None):
        """Get the centre of this structure in voxels or mm. If no
        units are given, units will be mm if <self_in_mm> is True."""

        if not self.loaded or self.empty:
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
                    self.centre["voxels"].append(self.idx_to_slice(mid_idx, ax))
                    self.centre["mm"].append(self.idx_to_pos(mid_idx, ax))
                else:
                    self.centre["voxels"].append(None)
                    self.centre["mm"].append(None)

        if units is None:
            units = "mm" if self.scale_in_mm else "voxels"
        return self.centre[units]

    def set_plotting_defaults(self):
        """Set default matplotlib plotting keywords for both mask and
        contour images."""

        self.mask_kwargs = {"alpha": 1, "interpolation": "none"}
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
        """Compute positions of contours on each slice in each orientation."""

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
        contours = skimage.measure.find_contours(self.current_slice, 0.5, "low", "low")
        if contours:
            points = []
            for contour in contours:
                contour_points = []
                for (y, x) in contour:
                    if self.scale_in_mm:
                        x = min(self.lims[x_ax]) + x * abs(self.voxel_sizes[x_ax])
                        y = min(self.lims[y_ax]) + y * abs(self.voxel_sizes[y_ax])
                    else:
                        x = self.idx_to_slice(x, x_ax)
                        if self.voxel_sizes[y_ax] < 0:
                            y = self.idx_to_slice(y, y_ax)
                        else:
                            y = self.idx_to_slice(self.n_voxels[y_ax] - y, y_ax) + 1
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
        no_title=False,
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

        if not self.visible:
            return
        self.load()
        if not self.valid:
            return

        mpl_kwargs = {} if mpl_kwargs is None else mpl_kwargs
        linewidth = mpl_kwargs.get("linewidth", 2)
        contour_kwargs = {"linewidth": linewidth}
        centroid = "centroid" in plot_type
        if centroid:
            contour_kwargs["markersize"] = mpl_kwargs.get(
                "markersize", 7 * np.sqrt(linewidth)
            )
            contour_kwargs["markeredgewidth"] = mpl_kwargs.get(
                "markeredgewidth", np.sqrt(linewidth)
            )

        # Make plot
        if plot_type in ["contour", "centroid"]:
            self.plot_contour(
                view,
                sl,
                pos,
                ax,
                contour_kwargs,
                zoom,
                zoom_centre,
                no_title=no_title,
                centroid=centroid,
            )
        elif plot_type == "mask":
            self.plot_mask(
                view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre, no_title=no_title
            )
        elif plot_type in ["filled", "filled centroid"]:
            mask_kwargs = {"alpha": mpl_kwargs.get("alpha", 0.3)}
            self.plot_mask(
                view, sl, pos, ax, mask_kwargs, zoom, zoom_centre, no_title=no_title
            )
            self.plot_contour(
                view,
                sl,
                pos,
                self.ax,
                contour_kwargs,
                zoom,
                zoom_centre,
                no_title=no_title,
                centroid=centroid,
            )

        if show:
            plt.show()

    def plot_mask(
        self,
        view,
        sl,
        pos,
        ax,
        mpl_kwargs=None,
        zoom=None,
        zoom_centre=None,
        no_title=False,
    ):
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
            **self.get_kwargs(mpl_kwargs, default=self.mask_kwargs),
        )
        self.adjust_ax(view, zoom, zoom_centre)
        self.label_ax(view, no_title=no_title)

    def plot_contour(
        self,
        view,
        sl,
        pos,
        ax,
        mpl_kwargs=None,
        zoom=None,
        zoom_centre=None,
        centroid=False,
        no_title=False,
    ):
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
            self.ax.plot(x, y, "+", **kwargs)

        self.adjust_ax(view, zoom, zoom_centre)
        self.label_ax(view, no_title=no_title)

    def on_slice(self, view, sl):
        """Return True if a contour exists for this structure on a given slice."""

        if not self.loaded:
            return False
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

        if not self.loaded or self.empty:
            return [0, 0, 0]

        if not hasattr(self, "full_extent"):
            non_zero = np.argwhere(self.data)
            mins = non_zero.min(0)
            maxes = non_zero.max(0)
            x_len = abs(maxes[1] - mins[1]) + 1
            y_len = abs(maxes[0] - mins[0]) + 1
            z_len = abs(maxes[2] - mins[2]) + 1
            self.full_extent = {"voxels": [x_len, y_len, z_len]}
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
            convert = self.idx_to_pos if self.scale_in_mm else self.idx_to_slice
            if y_ax == "y":
                y = self.n_voxels[y_ax] - 1 - y
            return [convert(x, x_ax), convert(y, y_ax)]
        else:
            return [0, 0]


class StructComparison:
    """Class for computing comparison metrics for two structures and plotting
    the structures together."""

    def __init__(self, struct1, struct2, name="", comp_type=None, **kwargs):
        """Initialise from a pair of Structs, or load new Structs."""

        self.name = name
        self.comp_type = comp_type
        if self.comp_type == "others":
            self.comp_type = "majority vote"

        # Two structures
        self.s2_is_list = is_list(struct2)
        if not self.s2_is_list:
            for i, s in enumerate([struct1, struct2]):
                struct = s if isinstance(s, Struct) else Struct(s, **kwargs)
                setattr(self, f"s{i + 1}", s)

        # List of structures
        else:
            self.s1 = struct1
            self.s2_list = struct2
            self.s2_voxel_sizes = list(self.s1.voxel_sizes.values())
            self.s2_origin = list(self.s1.origin.values())
            self.s2_name = f"{self.comp_type} of others"
            self.update_s2_data()

        mean_sq_col = (np.array(self.s1.color) ** 2 + np.array(self.s2.color) ** 2) / 2
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

        self.s2 = Struct(
            data,
            name=self.s2_name,
            load=True,
            voxel_sizes=self.s2_voxel_sizes,
            origin=self.s2_origin,
        )
        self.s2.color = matplotlib.colors.to_rgba("white")
        self.s2.name_unique = f"vs. {self.comp_type} of others"

        self.centroid_distance(force=True)
        self.global_dice_score(force=True)

    def is_valid(self):
        """Check both structures are valid and in same reference frame."""

        self.s1.load()
        self.s2.load()
        if not self.s1.same_frame(self.s2):
            raise TypeError(
                f"Comparison structures {self.s1.name} and "
                f"{self.s2.name} are not in the same reference "
                "frame!"
            )
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
        plot_grouping=None,
    ):
        """Plot comparison structures."""

        if not self.is_valid():
            return
        if mpl_kwargs is None:
            mpl_kwargs = {}

        if self.s2_is_list and plot_grouping != "group others":
            self.s1.plot(
                view=view,
                sl=sl,
                pos=pos,
                ax=ax,
                mpl_kwargs=mpl_kwargs,
                plot_type=plot_type,
                zoom=zoom,
                zoom_centre=zoom_centre,
                show=show,
            )
            return

        # If one structure isn't currently visible, only plot the other
        if not self.s1.visible or not self.s2.visible:
            s_vis = [s for s in [self.s1, self.s2] if s.visible]
            if len(s_vis):
                s_vis[0].plot(
                    view, sl, pos, ax, mpl_kwargs, plot_type, zoom, zoom_centre, show
                )
            return

        # Make plot
        linewidth = mpl_kwargs.get("linewidth", 2)
        contour_kwargs = {"linewidth": linewidth}
        centroid = "centroid" in plot_type
        if centroid:
            contour_kwargs["markersize"] = mpl_kwargs.get(
                "markersize", 7 * np.sqrt(linewidth)
            )
            contour_kwargs["markeredgewidth"] = mpl_kwargs.get(
                "markeredgewidth", np.sqrt(linewidth)
            )

        if plot_type in ["contour", "centroid"]:
            centroid = plot_type != "contour"
            self.s2.plot_contour(
                view, sl, pos, ax, contour_kwargs, zoom, zoom_centre, centroid=centroid
            )
            self.s1.plot_contour(
                view,
                sl,
                pos,
                self.s2.ax,
                contour_kwargs,
                zoom,
                zoom_centre,
                centroid=centroid,
            )

        elif plot_type == "mask":
            self.plot_mask(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre)

        elif plot_type in ["filled", "filled centroid"]:
            mask_kwargs = {"alpha": mpl_kwargs.get("alpha", 0.3)}
            self.plot_mask(view, sl, pos, ax, mask_kwargs, zoom, zoom_centre)
            self.s2.plot_contour(
                view,
                sl,
                pos,
                self.s2.ax,
                contour_kwargs,
                zoom,
                zoom_centre,
                centroid=centroid,
            )
            self.s1.plot_contour(
                view,
                sl,
                pos,
                self.s2.ax,
                contour_kwargs,
                zoom,
                zoom_centre,
                centroid=centroid,
            )

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
            (overlap, self.color),
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
                **self.s1.get_kwargs(mpl_kwargs, default=self.s1.mask_kwargs),
            )

        self.s2.adjust_ax(view, zoom, zoom_centre)

    def on_slice(self, view, sl):
        """Check whether both structures are on a given slice."""

        if not self.is_valid():
            return False
        return self.s1.on_slice(view, sl) and self.s2.on_slice(view, sl)

    def centroid_distance(self, units="voxels", force=False):
        """Get total centroid distance."""

        if not hasattr(self, "centroid_dist") or force:
            self.centroid_dist = {
                units: np.linalg.norm(
                    self.s1.get_centroid(units) - self.s2.get_centroid(units)
                )
                for units in ["mm", "voxels"]
            }

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
            self.global_dice = (self.s1.data & self.s2.data).sum() / np.mean(
                [self.s1.data.sum(), self.s2.data.sum()]
            )

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
    """Class for loading and storing multiple Structs."""

    def __init__(
        self,
        structs=None,
        multi_structs=None,
        names=None,
        colors=None,
        comp_type="auto",
        struct_kwargs=None,
        image=None,
        to_keep=None,
        to_ignore=None,
        autoload=True
    ):
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
            Keyword arguments to pass to any created Struct objects.

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

        autoload : bool, default=True
            If True, all structures will be loaded before being returned.
        """

        # Lists for storing structures
        self.loaded = False
        self.autoload = autoload
        self.structs = []
        self.comparisons = []
        self.comparison_structs = []
        self.comp_type = comp_type
        self.struct_kwargs = struct_kwargs if struct_kwargs is not None else {}
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

        parsed_settings = {}

        # Convert single list to enumerated dict
        if is_list(settings):
            parsed_settings = {value: i + 1 for i, value in 
                               enumerate(settings)}

        # Convert label dict of lists into enumerated dicts
        elif isinstance(settings, dict):
            for label, s in settings.items():

                if isinstance(label, int):
                    parsed_settings[s] = label

                elif isinstance(s, dict):
                    parsed_settings[label] = {}
                    for label2, s2 in s.items():
                        if isinstance(label2, int):
                            parsed_settings[label][s2] = label2
                        else:
                            parsed_settings[label][label2] = s2

                elif is_list(s):
                    parsed_settings[label] = {value: i + 1 for i, value in 
                                              enumerate(s)}

                else:
                    parsed_settings[label] = s

        return parsed_settings

    def load_structs(self, structs, names, colors, multi=False):
        """Load a list/dict of structres."""

        if structs is None:
            return

        struct_dict = {}

        # Put into standard format
        # Case where structs are already in a dict of labels and sources
        if isinstance(structs, dict):

            # Load numpy arrays
            array_structs = [
                name for name in structs if isinstance(structs[name], np.ndarray)
            ]
            for name in array_structs:
                self.add_struct_array(structs[name], name, colors)
                del structs[name]

            # Load path/label combos
            struct_dict = structs
            for label, path in struct_dict.items():
                if not is_list(path):
                    struct_dict[label] = [path]

        # Case where structs are in a list
        elif isinstance(structs, list):

            # Special case: pairs of structure sources for comparison
            input_is_pair = [is_list(s) and len(s) == 2 for s in structs]
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
                    self.load_structs_from_file(p[6:], label, names, colors, True)
                else:
                    self.load_structs_from_file(p, label, names, colors, multi)

    def load_structs_from_file(self, paths, label, names, colors, multi=False):
        """Search for filenames matching <paths> and load structs from all
        files."""

        # Get files
        if isinstance(paths, str):
            files = find_files(paths)
        else:
            files = paths

        # Get colors and names dicts
        if is_nested(colors):
            colors = colors.get(label, {})
        if is_nested(names):
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
            name = re.sub(r"RTSTRUCT_[MVCT]+_\d+_\d+_\d+_", "", basename).replace(
                " ", "_"
            )
            name = make_name_nice(name)

            # See if we can convert this name based on names list
            for name2, matches in names.items():
                if not is_list(matches):
                    matches = [matches]
                for match in matches:
                    if fnmatch.fnmatch(standard_str(name), standard_str(match)):
                        return name2

        # See if we can get name from filepath
        for name2, matches in names.items():
            if not is_list(matches):
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
            if not any(
                [
                    fnmatch.fnmatch(standard_str(name), standard_str(k))
                    for k in self.to_keep
                ]
            ):
                keep = False
        if self.to_ignore is not None:
            if any(
                [
                    fnmatch.fnmatch(standard_str(name), standard_str(i))
                    for i in self.to_ignore
                ]
            ):
                keep = False
        return keep

    def add_struct_array(self, array, name, colors):
        """Create Struct object from a NumPy array and add to list."""

        self.loaded = False
        if self.image is None:
            raise RuntimeError(
                "In order to load structs from NumPy array,"
                " StructLoader class must be created with the "
                "<image> argument!"
            )

        color = self.find_color_match(colors, name)
        struct = Struct(
            array,
            name=name,
            color=color,
            origin=self.image.origin,
            voxel_sizes=self.image.voxel_sizes,
        )
        self.structs.append(struct)

    def add_struct_file(self, path, label, names, colors, multi=False):
        """Create Struct object(s) from file and add to list."""

        self.loaded = False

        # Try loading from a DICOM file
        if self.load_structs_from_dicom(path, label, names, colors):
            return

        # Get custom name
        if isinstance(path, str):
            name = self.find_name_match(names, path)
        else:
            name = f"Structure {len(self.structs) + 1}"

        # Only one structure per file
        if not multi:
            if self.keep_struct(name):
                struct = Struct(
                    path, label=label, name=name, load=False, **self.struct_kwargs
                )
                color = self.find_color_match(colors, struct.name)
                if color is not None:
                    struct.assign_color(color)
                self.structs.append(struct)
            return

        # Search for many label masks in one file
        # Load the data
        data, voxel_sizes, origin, path = load_image(path)
        kwargs = self.struct_kwargs.copy()
        kwargs.update({"voxel_sizes": voxel_sizes, "origin": origin})
        mask_labels = np.unique(data).astype(int)
        mask_labels = mask_labels[mask_labels != 0]

        # Load multiple massk
        for ml in mask_labels:

            name = self.find_name_match(names, ml)
            if name is None:
                name = f"Structure {ml}"
            if self.keep_struct(name):
                color = self.find_color_match(colors, name)

                struct = Struct(
                    data == ml, name=name, label=label, color=color, **kwargs
                )
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
                s_pair.append(
                    Struct(
                        path, name=name, color=color, load=False, **self.struct_kwargs
                    )
                )

            self.structs.extend(s_pair)
            self.comparison_structs.extend(s_pair)
            self.comparisons.append(StructComparison(*s_pair))

    def load_structs_from_dicom(self, path, label, names, colors):
        """Attempt to load structures from a DICOM file."""

        try:
            structs = load_structs(path)
        except TypeError:
            return

        if not self.image:
            raise RuntimeError(
                "Must provide the <image> argument to "
                "StructLoader in order to load from DICOM!"
            )

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
            struct = Struct(
                contours=contours,
                label=label,
                name=name,
                load=False,
                color=color,
                shape=self.image.data.shape,
                origin=self.image.origin,
                voxel_sizes=self.image.voxel_sizes,
                **self.struct_kwargs,
            )
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
                others = [self.structs[j] for j in range(len(self.structs)) if j != i]
                self.comparisons.append(
                    StructComparison(s, others, comp_type=self.comp_type)
                )
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
            n_per_name = {
                n: len([s for s in self.structs if s.name == n]) for n in unique_names
            }
            if max(n_per_name.values()) != 2:
                use_pairs = True

        # Match all pairs
        if self.comp_type == "pairs" or use_pairs:
            for i, s1 in enumerate(self.structs):
                for s2 in self.structs[i + 1 :]:
                    self.comparisons.append(StructComparison(s1, s2))
            self.comparison_structs = self.structs
            return

        # Make structure comparisons
        names_to_compare = [name for name in n_per_name if n_per_name[name] == 2]
        for name in names_to_compare:
            structs = [s for s in self.structs if s.name == name]
            self.comparisons.append(
                StructComparison(*structs, name=structs[0].name_nice_nolabel)
            )
            self.comparison_structs.extend(structs)

    def set_unique_name(self, struct):
        """Create a unique name for a structure with respect to all other
        loaded structures."""

        if struct.path is None or struct.label:
            struct.name_unique = struct.name_nice
            return

        # Find structures with the same name
        same_name = [
            s
            for s in self.structs
            if standard_str(s.name) == standard_str(struct.name) and s != struct
        ]
        if not len(same_name):
            struct.name_unique = struct.name_nice
            return

        # Get unique part of path wrt those structures
        unique_paths = list(
            set([get_unique_path(struct.path, s.path) for s in same_name])
        )

        # If path isn't unique, just use own name
        if None in unique_paths:
            struct.name_unique = struct.name_nice

        elif len(unique_paths) == 1:
            struct.name_unique = f"{struct.name_nice} ({unique_paths[0]})"

        else:

            # Find unique path wrt all paths
            remaining = unique_paths[1:]
            current = get_unique_path(unique_paths[0], remaining)
            while len(remaining) > 1:
                remaining = remaining[1:]
                current = get_unique_path(current, remaining[0])
            struct.name_unique = f"{struct.name_nice} ({current})"

    def load_all(self):
        """Load all structures and assign custom colours and unique names."""

        if self.loaded:
            return

        # Assign colors
        for i, s in enumerate(self.structs):
            if not s.custom_color_set:
                s.assign_color(_standard_colors[i])

        for s in self.structs:
            if self.autoload:
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
                color = _standard_colors[len(colors)]
                s.assign_color(color)
                colors[s.name_unique] = color
        return colors

    def get_structs(self, ignore_unpaired=False, ignore_empty=False, sort=False):
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
            return [c for c in self.comparisons if not c.s1.empty or c.s2.empty]
        else:
            return self.comparisons

    def get_standalone_structs(self, ignore_unpaired=False, ignore_empty=False):
        """Get list of the structures that are not part of a comparison
        pair."""

        if ignore_unpaired:
            return []

        self.load_all()
        self.find_comparisons()
        standalones = [s for s in self.structs if s not in self.comparison_structs]
        if ignore_empty:
            return [s for s in standalones if not s.empty]
        else:
            return standalones


def standard_str(string):
    """Convert a string to lowercase and replace all spaces with
    underscores."""

    try:
        return str(string).lower().replace(" ", "_")
    except AttributeError:
        return


def load_structs(path):
    """Load structures from a DICOM file."""

    try:
        ds = pydicom.read_file(path)
    except pydicom.errors.InvalidDicomError:
        raise TypeError("Not a valid DICOM file!")

    # Check it's a structure file
    if not (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"):
        print(f"Warning: {path} is not a DICOM structure set file!")
        return

    # Get structure names
    seq = get_dicom_sequence(ds, "StructureSetROI")
    structs = {}
    for struct in seq:
        structs[int(struct.ROINumber)] = {"name": struct.ROIName}

    # Load contour data
    roi_seq = get_dicom_sequence(ds, "ROIContour")
    for roi in roi_seq:

        number = roi.ReferencedROINumber
        data = {"contours": {}}

        # Get colour
        if "ROIDisplayColor" in roi:
            data["color"] = [int(c) / 255 for c in list(roi.ROIDisplayColor)]
        else:
            data["color"] = None

        # Get contours
        contour_seq = get_dicom_sequence(roi, "Contour")
        if contour_seq:
            contour_data = {}
            for c in contour_seq:
                plane_data = [
                    [float(p) for p in c.ContourData[i * 3 : i * 3 + 3]]
                    for i in range(c.NumberOfContourPoints)
                ]
                z = float(c.ContourData[2])
                if z not in data["contours"]:
                    data["contours"][z] = []
                data["contours"][z].append(np.array(plane_data))

        structs[number].update(data)

    return structs


def get_dicom_sequence(ds=None, basename=""):

    sequence = []

    for suffix in ["Sequence", "s"]:
        attribute = f"{basename}{suffix}"
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break

    return sequence


def contours_to_indices(contours, origin, voxel_sizes, shape):
    """Convert contours from positions in mm to array indices."""

    converted = {}
    for z, conts in contours.items():

        # Convert z position
        zi = (z - origin[2]) / voxel_sizes[2]
        converted[zi] = []

        # Convert points on each contour
        for points in conts:
            pi = np.zeros(points.shape)
            pi[:, 0] = shape[0] - 1 - (points[:, 0] - origin[0]) / voxel_sizes[0]
            pi[:, 1] = shape[1] - 1 - (points[:, 1] - origin[1]) / voxel_sizes[1]
            pi[:, 2] = zi
            converted[zi].append(pi)

    return converted


def contours_to_mask(contours, shape, level=0.25, save_name=None, affine=None):
    """Convert contours to mask."""

    mask = np.zeros(shape)

    # Loop over slices
    for iz, conts in contours.items():

        # Loop over contours on each slice
        for c in conts:

            # Make polygon from (x, y) points
            polygon = geometry.Polygon(c[:, 0:2])

            # Get the polygon's bounding box
            ix1, iy1, ix2, iy2 = [int(xy) for xy in polygon.bounds]
            ix1 = max(0, ix1)
            ix2 = min(ix2 + 1, shape[0])
            iy1 = max(0, iy1)
            iy2 = min(iy2 + 1, shape[1])

            # Loop over pixels
            for ix in range(ix1, ix2):
                for iy in range(iy1, iy2):

                    # Make polygon of current pixel
                    pixel = geometry.Polygon(
                        [
                            [ix - 0.5, iy - 0.5],
                            [ix - 0.5, iy + 0.5],
                            [ix + 0.5, iy + 0.5],
                            [ix + 0.5, iy - 0.5],
                        ]
                    )

                    # Compute overlap
                    overlap = polygon.intersection(pixel).area
                    mask[ix, iy, int(iz)] += overlap

    # Convert mask to boolean
    mask = mask > level

    # Save if needed
    if save_name is not None and affine is not None:
        nii = nibabel.Nifti1Image(mask.astype(int), affine)
        nii.to_filename(save_name + ".nii.gz")

    return mask


def make_name_nice(name):
    """Replace underscores with spaces and make uppercase."""

    name_nice = name.replace("_", " ")
    name_nice = name_nice[0].upper() + name_nice[1:]
    return name_nice
