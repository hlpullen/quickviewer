"""Classes for combining image data for plotting in QuickViewer."""

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches

from quickviewer.data.image import Image, ImageComparison
from quickviewer.data.image import (
    is_list, 
    find_files, 
    find_date, 
    to_inches,
    _axes,
    _plot_axes,
    _slider_axes,
    _orient
)
from quickviewer.data.structures import Struct, StructComparison, StructureSet


# Shared parameters
_orthog = {"x-y": "y-z", "y-z": "x-z", "x-z": "y-z"}
_default_spacing = 30


class MultiImage(Image):
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
        autoload_structs=True,
        mask_threshold=0.5,
        **kwargs,
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
            Image.__init__(self, nii, **kwargs)
            self.timeseries = False

        # Load a dose field only
        elif dose is not None and timeseries is None:
            self.dose_as_im = True
            Image.__init__(self, dose, **kwargs)
            dose = None

        # Load a timeseries of images
        elif timeseries is not None:
            self.timeseries = True
            dates = self.get_date_dict(timeseries)
            self.dates = list(dates.keys())
            if "title" in kwargs:
                kwargs.pop("title")
            self.ims = {
                date: Image(file, title=date, **kwargs) for date, file in dates.items()
            }
            Image.__init__(self, dates[self.dates[0]], title=self.dates[0], **kwargs)
            self.date = self.dates[0]
        else:
            raise TypeError("Must provide either <nii>, <dose>, or " "<timeseries!>")
        if not self.valid:
            return

        # Load extra overlays
        self.load_to(dose, "dose", kwargs)
        self.load_to(mask, "mask", kwargs)
        self.load_to(jacobian, "jacobian", kwargs)
        self.load_df(df)

        # Load structs
        self.comp_type = comp_type
        self.load_structs(
            structs,
            multi_structs,
            names=struct_names,
            colors=struct_colors,
            compare_structs=compare_structs,
            ignore_empty=ignore_empty_structs,
            ignore_unpaired=ignore_unpaired_structs,
            comp_type=comp_type,
            to_keep=structs_to_keep,
            to_ignore=structs_to_ignore,
            autoload=autoload_structs
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
            data = Image(nii, rescale=rescale, **kwargs)
            data.match_size(self, 0)
            valid = data.valid
        else:
            data = {view: Image(nii[view], rescale=rescale, **kwargs) for view in nii}
            for view in _orient:
                if view not in data or not data[view].valid:
                    data[view] = None
                else:
                    data[view].match_size(self, 0)
            valid = any([d.valid for d in data.values() if d is not None])

        setattr(self, attr, data)
        setattr(self, f"has_{attr}", valid)
        setattr(self, f"{attr}_dict", isinstance(nii, dict))

    def load_df(self, df):
        """Load deformation field data from a path."""

        self.df = DeformationImage(df, scale_in_mm=self.scale_in_mm)
        self.has_df = self.df.valid

    def load_structs(
        self,
        structs=None,
        multi_structs=None,
        names=None,
        colors=None,
        compare_structs=False,
        ignore_empty=False,
        ignore_unpaired=False,
        comp_type="auto",
        to_keep=None,
        to_ignore=None,
        autoload=True
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
                struct_dates = self.get_date_dict(structs, True, True)
                self.struct_timeseries = len(struct_dates) > 1
            except TypeError:
                pass

        # No timeseries: load single set of structs
        if not self.struct_timeseries:
            loader = StructureSet(
                structs,
                multi_structs,
                names,
                colors,
                comp_type=comp_type,
                struct_kwargs={"scale_in_mm": self.scale_in_mm},
                image=self,
                to_keep=to_keep,
                to_ignore=to_ignore,
                autoload=autoload
            )
            self.structs = loader.get_structs(ignore_unpaired, ignore_empty)

            if compare_structs:
                self.struct_comparisons = loader.get_comparisons(ignore_empty)
                self.standalone_structs = loader.get_standalone_structs(
                    ignore_unpaired, ignore_empty
                )
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

                loader = StructureSet(
                    structs,
                    names=names,
                    colors=colors,
                    comp_type=comp_type,
                    struct_kwargs={"scale_in_mm": self.scale_in_mm},
                    image=self,
                    to_keep=to_keep,
                    to_ignore=to_ignore,
                )
                struct_colors = loader.reassign_colors(struct_colors)
                self.dated_structs[date] = loader.get_structs(
                    ignore_unpaired, ignore_empty, sort=True
                )

                if compare_structs:
                    self.dated_comparisons[date] = loader.get_comparisons(ignore_empty)
                    self.dated_standalone_structs = loader.get_standalone_structs(
                        ignore_unpaired, ignore_empty
                    )
                else:
                    self.dated_comparisons[date] = []
                    self.dated_standalone_structs[date] = self.dated_structs[date]

            self.has_structs = any([len(s) for s in self.dated_structs.values()])

            # Set to current date
            if self.date in self.dated_structs:
                self.structs = self.dated_structs[date]
                self.struct_comparisons = self.dated_comparisons[date]
                self.standalone_structs = self.dated_standalone_structs[date]

    def get_date_dict(self, timeseries, single_layer=False, allow_dirs=False):
        """Convert list/dict/directory to sorted dict of dates and files."""

        if isinstance(timeseries, dict):
            dates = {dateutil.parser.parse(key): val for key, val in timeseries.items()}

        else:
            if isinstance(timeseries, str):
                files = find_files(timeseries, allow_dirs=allow_dirs)
            elif is_list(timeseries):
                files = timeseries
            else:
                raise TypeError("Timeseries must be a list, dict, or str.")

            # Find date-like string in filenames
            dates = {}
            for file in files:
                dirname = os.path.basename(os.path.dirname(file))
                date = find_date(dirname)
                if not date:
                    base = os.path.basename(file)
                    date = find_date(base)
                if not date:
                    raise TypeError(
                        "Date-like string could not be found in "
                        f"filename of dirname of {file}!"
                    )
                dates[date] = file

        # Sort by date
        dates_sorted = sorted(list(dates.keys()))
        date_strs = {
            date: f"{date.day}/{date.month}/{date.year}" for date in dates_sorted
        }
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
            self.standalone_structs = self.dated_standalone_structs.get(self.date, [])

    def set_plotting_defaults(self):
        """Set default matplotlib plotting options for main image, dose field,
        and jacobian determinant."""

        Image.set_plotting_defaults(self)
        self.dose_kwargs = {"cmap": "jet", "alpha": 0.5, "vmin": None, "vmax": None}
        self.jacobian_kwargs = {
            "cmap": "seismic",
            "alpha": 0.5,
            "vmin": 0.8,
            "vmax": 1.2,
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
                        if isinstance(mask, Image):
                            view_masks[view] = mask_array + (
                                self.mask[view].data > self.mask_threshold
                            )
                        else:
                            view_masks[view] = mask_array + (
                                self.mask[view] > self.mask_threshold
                            )
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

    def get_relative_width(self, view, zoom=None, colorbar=False, figsize=None):
        """Get the relative width for this plot, including all colorbars."""

        return Image.get_relative_width(
            self, view, zoom, self.get_n_colorbars(colorbar), figsize
        )

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
        legend_loc="lower left",
        struct_plot_grouping=None,
        struct_to_plot=None,
        annotate_slice=None,
        major_ticks=None,
        minor_ticks=None,
        ticks_all_sides=False,
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
        Image.plot(
            self,
            view,
            sl,
            pos,
            ax=self.ax,
            mpl_kwargs=mpl_kwargs,
            show=False,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
            masked=masked,
            invert_mask=invert_mask,
            mask_color=mask_color,
            figsize=figsize,
            major_ticks=major_ticks,
            minor_ticks=minor_ticks,
            ticks_all_sides=ticks_all_sides,
        )

        # Plot dose field
        self.dose.plot(
            view,
            self.sl,
            ax=self.ax,
            mpl_kwargs=self.get_kwargs(dose_kwargs, default=self.dose_kwargs),
            show=False,
            masked=masked,
            invert_mask=invert_mask,
            mask_color=mask_color,
            colorbar=colorbar,
            colorbar_label="Dose (Gy)",
        )

        # Plot jacobian
        self.jacobian.plot(
            view,
            self.sl,
            ax=self.ax,
            mpl_kwargs=self.get_kwargs(jacobian_kwargs, default=self.jacobian_kwargs),
            show=False,
            colorbar=colorbar,
            colorbar_label="Jacobian determinant",
        )

        # Plot standalone structures and comparisons
        for s in self.standalone_structs:
            s.plot(
                view,
                self.sl,
                ax=self.ax,
                mpl_kwargs=struct_kwargs,
                plot_type=struct_plot_type,
            )
        for s in self.struct_comparisons:
            if struct_plot_grouping == "group others":
                if s.s1.name_unique != struct_to_plot:
                    continue
            s.plot(
                view,
                self.sl,
                ax=self.ax,
                mpl_kwargs=struct_kwargs,
                plot_type=struct_plot_type,
                plot_grouping=struct_plot_grouping,
            )

        # Plot deformation field
        self.df.plot(
            view,
            self.sl,
            ax=self.ax,
            mpl_kwargs=df_kwargs,
            plot_type=df_plot_type,
            spacing=df_spacing,
        )

        # Draw structure legend
        if struct_legend and struct_plot_type != "none":
            handles = []
            for s in self.structs:
                if struct_plot_grouping == "group others":
                    if s.name_unique == struct_to_plot:
                        handles.append(mpatches.Patch(color=s.color, label=s.name_nice))
                        handles.append(mpatches.Patch(color="white", label="Others"))
                elif s.visible and s.on_slice(view, self.sl):
                    handles.append(mpatches.Patch(color=s.color, label=s.name_nice))
            if len(handles):
                self.ax.legend(
                    handles=handles, loc=legend_loc, facecolor="white", framealpha=1
                )

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
        self.orthog_slices = {ax: int(self.n_voxels[ax] / 2) for ax in _axes}

    def get_relative_width(self, view, zoom=None, colorbar=False, figsize=None):
        """Get width:height ratio for the full plot (main plot + orthogonal
        view)."""

        width_own = MultiImage.get_relative_width(self, view, zoom, colorbar, figsize)
        width_orthog = MultiImage.get_relative_width(self, _orthog[view], figsize)
        return width_own + width_orthog

    def set_axes(self, view, ax=None, gs=None, figsize=None, zoom=None, colorbar=False):
        """Set up axes for the plot. If <ax> is not None and <orthog_ax> has
        already been set, these axes will be used. Otherwise if <gs> is not
        None, the axes will be created within a gridspec on the current
        matplotlib figure.  Otherwise, a new figure with height <figsize>
        will be produced."""

        if ax is not None and hasattr(self, "orthog_ax"):
            self.ax = ax

        width_ratios = [
            MultiImage.get_relative_width(self, view, zoom, colorbar, figsize),
            MultiImage.get_relative_width(self, _orthog[view], figsize),
        ]
        if gs is None:
            figsize = _default_figsize if figsize is None else figsize
            figsize = to_inches(figsize)
            fig = plt.figure(figsize=(figsize * sum(width_ratios), figsize))
            self.gs = fig.add_gridspec(1, 2, width_ratios=width_ratios)
        else:
            fig = plt.gcf()
            self.gs = gs.subgridspec(1, 2, width_ratios=width_ratios)

        self.ax = fig.add_subplot(self.gs[0])
        self.orthog_ax = fig.add_subplot(self.gs[1])

    def plot(
        self,
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
        **kwargs,
    ):
        """Plot MultiImage and orthogonal view of main image and structs."""

        self.set_axes(view, ax, gs, figsize, zoom, colorbar)

        # Plot the MultiImage
        MultiImage.plot(
            self,
            view,
            sl=sl,
            pos=pos,
            ax=self.ax,
            zoom=zoom,
            zoom_centre=zoom_centre,
            colorbar=colorbar,
            show=False,
            colorbar_label=colorbar_label,
            mpl_kwargs=mpl_kwargs,
            struct_kwargs=struct_kwargs,
            struct_plot_type=struct_plot_type,
            major_ticks=major_ticks,
            minor_ticks=minor_ticks,
            ticks_all_sides=ticks_all_sides,
            **kwargs,
        )

        # Plot orthogonal view
        orthog_view = _orthog[view]
        orthog_sl = self.orthog_slices[_slider_axes[orthog_view]]
        Image.plot(
            self,
            orthog_view,
            sl=orthog_sl,
            ax=self.orthog_ax,
            mpl_kwargs=mpl_kwargs,
            show=False,
            colorbar=False,
            no_ylabel=True,
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
            struct.plot(
                orthog_view,
                sl=orthog_sl,
                ax=self.orthog_ax,
                mpl_kwargs=struct_kwargs,
                plot_type=plot_type,
                no_title=True,
            )

        # Plot indicator line
        pos = sl if not self.scale_in_mm else self.slice_to_pos(sl, _slider_axes[view])
        if view == "x-y":
            full_y = (
                self.extent[orthog_view][2:]
                if self.scale_in_mm
                else [0.5, self.n_voxels[_plot_axes[orthog_view][1]] + 0.5]
            )
            self.orthog_ax.plot([pos, pos], full_y, "r")
        else:
            full_x = (
                self.extent[orthog_view][:2]
                if self.scale_in_mm
                else [0.5, self.n_voxels[_plot_axes[orthog_view][0]] + 0.5]
            )
            self.orthog_ax.plot(full_x, [pos, pos], "r")

        if show:
            plt.tight_layout()
            plt.show()


class DeformationImage(Image):
    """Class for loading a plotting a deformation field."""

    def __init__(self, nii, spacing=_default_spacing, plot_type="grid", **kwargs):
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

        Image.__init__(self, nii, **kwargs)
        if not self.valid:
            return
        if self.data.ndim != 5:
            raise RuntimeError(
                f"Deformation field in {nii} must contain a " "five-dimensional array!"
            )
        self.data = self.data[:, :, :, 0, :]
        self.set_spacing(spacing)

    def set_spacing(self, spacing):
        """Assign grid spacing in each direction. If spacing in given in mm,
        convert it to number of voxels."""

        if spacing is None:
            return

        spacing = self.get_ax_dict(spacing, _default_spacing)
        if self.scale_in_mm:
            self.spacing = {
                ax: abs(round(sp / self.voxel_sizes[ax])) for ax, sp in spacing.items()
            }
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
        self.grid_kwargs = {"color": "green", "linewidth": 2}

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
        zoom_centre=None,
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
        self, view, sl, pos, ax, mpl_kwargs=None, zoom=None, zoom_centre=None
    ):
        """Draw a quiver plot on a set of axes."""

        # Get arrow positions and lengths
        self.set_ax(view, ax, zoom=zoom)
        x_ax, y_ax = _plot_axes[view]
        x, y, df_x, df_y = self.get_deformation_slice(view, sl, pos)
        arrows_x = df_x[:: self.spacing[y_ax], :: self.spacing[x_ax]]
        arrows_y = -df_y[:: self.spacing[y_ax], :: self.spacing[x_ax]]
        plot_x = x[:: self.spacing[y_ax], :: self.spacing[x_ax]]
        plot_y = y[:: self.spacing[y_ax], :: self.spacing[x_ax]]

        # Plot arrows
        if arrows_x.any() or arrows_y.any():
            M = np.hypot(arrows_x, arrows_y)
            ax.quiver(
                plot_x,
                plot_y,
                arrows_x,
                arrows_y,
                M,
                **self.get_kwargs(mpl_kwargs, self.quiver_kwargs),
            )
        else:
            # If arrow lengths are zero, plot dots
            ax.scatter(plot_x, plot_y, c="navy", marker=".")
        self.adjust_ax(view, zoom, zoom_centre)

    def plot_grid(
        self, view, sl, pos, ax, mpl_kwargs=None, zoom=None, zoom_centre=None
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
