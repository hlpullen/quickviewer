"""Classes for displaying interactive medical image plots."""
import os
import re
import itertools
import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from quickviewer.core import to_inches, check_settings_file
from quickviewer.image import MultiImage, OrthogonalImage, ChequerboardImage, \
        OverlayImage, DiffImage, standard_str
from quickviewer.image import _slider_axes, _df_plot_types, _orient, \
        _struct_plot_types, _orthog, _default_figsize, _plot_axes, _axes


_style = {"description_width": "initial"}


# Matplotlib settings
mpl.rcParams["figure.figsize"] = (7.4, 4.8)
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14.0


# Check settings file exists
check_settings_file()


class QuickViewer:
    """Display multiple ViewerImages and comparison images."""

    def __init__(
        self,
        nii,
        title=None,
        mask=None,
        dose=None,
        structs=None,
        jacobian=None,
        df=None,
        share_slider=True,
        orthog_view=False,
        plots_per_row=None,
        match_axes=None,
        scale_in_mm=True,
        show_cb=False,
        show_overlay=False,
        show_diff=False,
        comparison_only=False,
        cb_splits=2,
        overlay_opacity=0.5,
        overlay_legend=False,
        legend_loc="lower left",
        translation=False,
        translation_file_to_overwrite=None,
        suptitle=None,
        show=True,
        **kwargs
    ):
        """
        Display one or more interactive images.

        Parameters
        ----------

        nii : string/nifti/array/list
            Source of image data for each plot. If multiple plots are to be
            shown, this must be a list. Image sources can be any of:
            (a) The path to a NIfTI file;
            (b) A nibabel.nifti1.Nifti1Image object;
            (c) The path to a file containing a NumPy array;
            (d) A NumPy array.

        title : string or list of strings, default=None
            Custom title(s) to use for the image(s) to be displayed. If the 
            number of titles given, n, is less than the number of images, only
            the first n figures will be given custom titles. If any titles are
            None, the name of the image file will be used as the title.

        mask : string/nifti/array/list, default=None
            Source(s) of array(s) to with which to mask each plot (see valid
            image sources for <nii>).

        dose : string/nifti/array/list, default=None
            Source(s) of dose field array(s) to overlay on each plot (see valid
            image sources for <nii>).

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

        jacobian : string/nifti/array/list, default=None
            Source(s) of jacobian determinant array(s) to overlay on each plot 
            (see valid image sources for <nii>).

        df : string/nifti/array/list, default=None
            Source(s) of deformation field(s) to overlay on each plot 
            (see valid image sources for <nii>).

        share_slider : bool, default=True
            If True and all displayed images are in the same frame of 
            reference, a single slice slider will be shared between all plots.
            If plots have different frames of reference, this option will be 
            ignored.

        orthog_view : bool, default=False
            If True, an orthgonal view with an indicator line showing the 
            current slice will be displayed alongside each plot.

        plots_per_row : int, default=None
            Number of plots to display before starting a new row. If None,
            all plots will be shown on a single row.

        match_axes : int/str, default=None
            Method for adjusting axis limits. Can either be:
            - An integer n, where 0 < n < number of plots, or n is -1. The axes
              of all plots will be adjusted to be the same as those of plot n.
            - "all"/"both": axes for all plots will be adjusted to cover the 
              maximum range of all plots.
            - "overlap": axes for all plots will be adjusted to just show the
              overlapping region.
            - "x": same as "all", but only applied to the current x axis.
              whatever the x axis is in the current view.
            - "y": same as "all", but only applied to the current y axis.

        scale_in_mm : bool, default=True
            If True, the axis scales will be shown in mm instead of array
            indices.

        show_cb : bool, default=False
            If True, a chequerboard image will be displayed. This option will 
            only be applied if the number of images in <nii> is 2.

        show_overlay : bool, default=False
            If True, a blue/red transparent overlaid image will be displayed.
            This option will only be applied if the number of images in 
            <nii> is 2.

        show_diff : bool, default=False
            If True, a the difference between two images will be shown. This 
            option will only be applied if the number of images in <nii> 
            is 2.

        comparison_only : bool, False
            If True, only comparison images (overlay/chequerboard/difference)
            will be shown. If no comparison options are selected, this 
            parameter will be ignored.

        cb_splits : int, default=2
            Number of sections to show for chequerboard image. Minimum = 1
            (corresponding to no chequerboard). Can later be changed 
            interactively.

        overlay_opacity : float, default=0.5
            Initial opacity of overlay. Can later be changed interactively.

        overlay_legend : bool default=False
            If True, a legend will be displayed on the overlay plot.

        legend_loc : str, default='lower left'
            Location for any legends being displayed. Must be a valid 
            matplotlib legend location.

        translation : bool, default=False
            If True, widgets will be displayed allowing the user to apply a 
            translation to the image and write this to an elastix transform 
            file or plain text file.

        translation_file_to_overwrite : str, default=None
            If not None and the <translation> option is used, this parameter
            will be used to populate the "Original" and "Output" file fields in 
            the translation user interface.

        suptitle : string, default=None
            Global title for all subplots. If None, no suptitle will be added.

        show : bool, default=True
            If True, the plot will be displayed when the QuickViewer object is 
            created. Otherwise, the plot can be displayed later via
            QuickViewer.show().

        Keyword arguments
        -----------------

        init_view : string, default='x-y'
            Orientation at which to initially display the image(s).

        init_sl : integer, default=None
            Slice number in the initial orientation direction at which to 
            display the first image (can be changed interactively later). If
            None, the central slice will be displayed.

        init_pos : float, default=None
            Position in mm of the first slice to display. This will be rounded
            to the nearest slice. If <init_pos> and <init_idx> are both given,
            <init_pos> will override <init_idx> only if <scale_in_mm> is True.

        hu : float/tuple, default=(-300, 200)
            HU central value or range thresholds at which to display the image. 
            Can later be changed interactively. If a single value is given, the
            HU range will be centred at this value with width given by 
            <hu_width>. If a tuple is given, the HU range will be set to the
            two values in the tuple.

        hu_width : float, default=500
            Initial width of the HU window. Only used if <hu> is a single 
            value.

        hu_limits : tuple, default=(-2000, 2000)
            Full range to use for the HU slider. Can also set to "auto" to
            detect min and max HU in the image.

        figsize : float, default=5
            Height of the displayed figure in inches. If None, the value in
            _default_figsize is used.

        xlim : tuple, default=None
            Custom limits for the x axis. If one of the values in the tuple is 
            None, the default x limits will be used.

        ylim : tuple, default=None
            Custom limits for the y axis. If one of the values in the tuple is 
            None, the default y limits will be used.

        zlim : tuple, default=None
            Custom limits for the z axis. If one of the values in the tuple is 
            None, the default z limits will be used.

        colorbar : bool, default=False
            If True, colorbars will be displayed for HU, dose and Jacobian 
            determinant.

        mpl_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the main image.See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html 
            for options.
            Some useful keywords are:
                - "cmap": colormap (default="gray").
                - "interpolation": interpolation method (default="antialiased")

        dose_opacity : float, default=0.5
            Initial opacity of the overlaid dose field. Can later be changed
            interactively.

        dose_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the dose image. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html 
            for options.
            Some useful keywords are:
                - "cmap": colormap (default="jet").
                - "interpolation": interpolation method (default="antialiased")

        invert_mask : bool, default=False
            If True, any masks applied will be inverted.

        mask_color : matplotlib color, default="black"
            color in which to display masked areas.

        mask_threshold : float, default=0.5
            Threshold on mask array; voxels with values below this threshold
            will be masked (or values above, if <invert_mask> is True).

        jacobian_opacity : float, default=0.5
            Initial opacity of the overlaid jacobian determinant. Can later 
            be changed interactively.

        jacobian_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib.pyplot.imshow
            for the jacobian determinant. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
            for options.
            Some useful keywords are:
                - "cmap": colormap (default="seismic").
                - "interpolation": interpolation method (default="antialiased")

        df_plot_type : str, default="grid"
            Option for initial plotting of deformation field. Can be 'grid', 
            'quiver', or 'none'. Can later be changed interactively.

        df_spacing : int/tuple, default=30
            Spacing between arrows on the quiver plot/gridlines on the grid 
            plot of a deformation field. Can be a single value for spacing in 
            all directions, or a tuple with values for (x, y, z). Dimensions
            are mm if <scale_in_mm> is True, or voxels if <scale_in_mm> is 
            False.

        df_kwargs : dict, default=None
            Dictionary of keyword arguments to pass to matplotlib when plotting
            the deformation field.

            For grid plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
            Some useful keywords are:
                - "linewidth": default=2
                - "color": default="green"

            For quiver plotting options, see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html.

        struct_plot_type : str, default='contour'
            Option for initial plot of structures. Can be 'contour', 'mask', 
            'filled', or 'none'. Can later be changed interactively.

        struct_opacity : float, default=None
            Initial opacity of structures when plotted as masks or filled 
            contours. Can later be changed interactively. Default=1 for masks,
            0.3 for filled contours.

        struct_linewidth : float, default=2
            Initial linewidth of structures when plotted as contours. Can later 
            be changed interactively.

        struct_info : bool, default=True
            If True, the lengths and volumes of each structure will be 
            displayed below the plot.

        length_units : str, default=None
            Units in which to display the lengths of structures if 
            <struct_info> if True. If None, units will be voxels if 
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:
                (a) "mm"
                (b) "voxels"

        area_units : str, default=None
            Units in which to display the areas of structures if 
            <struct_info> if True. If None, units will be voxels if 
            <scale_in_mm> is False, or mm if <scale_in_mm> is True. Options:
                (a) "mm"
                (b) "voxels"

        vol_units : str, default=None
            Units in which to display the volumes of structures if 
            <struct_info> if True. If None, units will be voxels if 
            <scale_in_mm> is False, or mm^3 if <scale_in_mm> is True. Options:
                (a) "mm" for mm^3
                (b) "voxels" for voxels
                (c) "ml" for ml

        struct_legend : bool, default=False
            If True, a legend will be displayed for any plot with structures.

        init_struct : str, default=None
            If set to a structure name, the first slice to be displayed will
            be the central slice of that structure. This supercedes <init_pos>
            and <init_sl>.

        struct_colors : dict, default=None
            Map between structures and colors to use for that structure. The
            key can either be: 
                (a) The path corresponding to the structure NIfTI file;
                (b) The name of the structure (i.e. the name of the NIfTI file
                    without the extension, optionally with underscores replaced
                    by spaces);
                (c) A wildcard matching the name(s) of structure(s) to be 
                    colored, optionally with underscores replace by spaces;
                (d) A wildcard string matching the path(s) of the structure 
                    file(s) to be given the chosen color.
            Matching structures will be searched for in that order. If more 
            than one structure matches, all matching structures will have that
            color. (Note: structure names are case insensitive).

        structs_as_mask : bool, default=True
            If True, any loaded structures will be used to mask the image and
            dose field.

        many_structs_per_file : bool, default=False
            If True, multiple structures will be loaded from files containing
            multiple structure masks in a single array with different values.

        struct_names : list/dict, default=None
            If <many_structs_per_file>, this parameter will be used to name
            the structures. Can either be a list (i.e. the first structure in 
            the file will be given the first name in the list and so on), or a 
            dict of numbers and names (e.g. {1: "first structure"} etc).

        compare_structs : bool, default=False
            If True, structure comparisons will be performed.

        continuous_update : bool, default=False
            If True, sliders in the UI will continuously update the figure as 
            they are adjusted. Can cause lag.

        annotate_slice : str, default=None
            Color for annotation of slice number. If None, no annotation will 
            be added unless viewing outside jupyter, in which case the 
            annotation will be white by default.

        save_as : str, default=None
            File to which the figure will be saved upon creation. If not None,
            a text input and button will be added to the UI, allowing the 
            user to save the figure again at a later point.

        zoom : double/tuple, default=None
            Amount between by which to zoom in (e.g. zoom=2 would give a 
            2x zoom). Can be a single value for all directions or a tuple of
            values for the (x, y, z) directions.
            
        zoom_centre : tuple, default=None
            Centrepoint of zooming in order (x, y, z). If any of the values are
            None, the midpoint of the image will be used in that direction.

        zoom_ui : bool, default=False
            If True, a UI for zooming will be displayed.
            
        downsample : int/tuple, default=None
            Factor by which to downsample an image. Can be a single value for
            all directions or a tuple of values in the (x, y, z) directions.
            For no downsampling, set values to None or 1, e.g. to downsample
            in the z direction only: downsample=(1, 1, 3).

        affine : 4x4 array, default=None
            Affine matrix to be used if image source(s) are NumPy array(s). If 
            image sources are nifti file paths or nibabel objects, this 
            parameter is ignored. If None, the arguments <voxel_sizes> and 
            <origin> will be used to set the affine matrix.

        voxel_sizes : tuple, default=(1, 1, 1)
            Voxel sizes in mm, given in the order (y, x, z). Only used if
            image source is a numpy array and <affine> is None.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm, given in the order (y, x, z). Only used if
            image source is a numpy array and <affine> is None.

        """

        # Get image file inputs
        if not isinstance(nii, list) or isinstance(nii, tuple):
            self.nii = [nii]
        else:
            self.nii = nii
        self.n = len(self.nii)

        # Process other inputs
        self.title = self.get_input_list(title)
        self.dose = self.get_input_list(dose)
        self.mask = self.get_input_list(mask)
        self.structs = self.get_input_list(structs, allow_sublist=True)
        self.jacobian = self.get_input_list(jacobian)
        self.df = self.get_input_list(df)

        # Make individual viewers
        self.scale_in_mm = scale_in_mm
        self.viewer = []
        viewer_type = ImageViewer if not orthog_view else OrthogViewer
        kwargs = {key.replace("colour", "color"): val for key, val in 
                  kwargs.items()}
        for i in range(self.n):
            viewer = viewer_type(
                self.nii[i], title=self.title[i], dose=self.dose[i],
                mask=self.mask[i], structs=self.structs[i],
                jacobian=self.jacobian[i], df=self.df[i], standalone=False,
                scale_in_mm=scale_in_mm, legend_loc=legend_loc, **kwargs)
            if viewer.im.valid:
                self.viewer.append(viewer)
        self.n = len(self.viewer)
        if not self.n:
            print("No valid images found.")
            return

        # Load comparison images
        self.cb_splits = cb_splits
        self.overlay_opacity = overlay_opacity
        self.overlay_legend = overlay_legend
        self.legend_loc = legend_loc
        self.load_comparison(show_cb, show_overlay, show_diff)
        self.comparison_only = comparison_only
        self.translation = translation
        self.tfile = translation_file_to_overwrite

        # Settings needed for plotting
        self.figsize = to_inches(kwargs.get("figsize", _default_figsize))
        self.colorbar = kwargs.get("colorbar", False)
        self.zoom = kwargs.get("zoom", None)
        self.plots_per_row = plots_per_row
        self.suptitle = suptitle
        self.match_axes(match_axes)
        self.in_notebook = in_notebook()
        self.saved = False
        self.plotting = False

        # Make UI
        self.make_ui(share_slider)

        # Display
        self.show(show)

    def show(self, show=True):
        ImageViewer.show(self, show)

    def get_input_list(self, inp, allow_sublist=False):
        """Convert an input to a list with one item per image to be
        displayed."""

        if inp is None or len(inp) == 0:
            return [None for i in range(self.n)]

        # Convert arg to a list
        input_list = []
        if isinstance(inp, list) or isinstance(inp, tuple):
            if self.n == 1 and allow_sublist:
                input_list = [inp]
            else:
                input_list = inp
        else:
            input_list = [inp]
        return input_list + [None for i in range(self.n - len(input_list))]

    def any_attr(self, attr):
        """Check whether any of this object's viewers have a given attribute.
        """

        return any([getattr(v.im, "has_" + attr) for v in self.viewer])

    def load_comparison(self, show_cb, show_overlay, show_diff):
        """Create any comparison images."""

        self.comparison = []
        self.has_chequerboard = show_cb
        self.has_overlay = show_overlay
        self.has_diff = show_diff
        if not (show_cb or show_overlay or show_diff):
            return

        assert self.n > 1
        im1 = self.viewer[0].im
        im2 = self.viewer[1].im

        if show_cb:
            self.chequerboard = ChequerboardImage(
                im1, im2, title="Chequerboard", scale_in_mm=self.scale_in_mm)
            self.comparison.append(self.chequerboard)
        if show_overlay:
            self.overlay = OverlayImage(
                im1, im2, title="Overlay", scale_in_mm=self.scale_in_mm)
            self.comparison.append(self.overlay)
        if show_diff:
            self.diff = DiffImage(
                im1, im2, title="Difference", scale_in_mm=self.scale_in_mm)
            self.comparison.append(self.diff)

    def match_axes(self, match_axes):
        """Adjust axes of plots to match if the match_axes option is set."""

        if match_axes is None:
            return

        # Match axes in all orientations
        for view in _slider_axes:

            # Calculate limits using all plots
            ax_lims = []
            if match_axes in ["all", "both", "x", "y", "overlap"]:
                for i in range(2):
                    min_lims = [v.im.ax_lims[view][i][0] for v in self.viewer]
                    max_lims = [v.im.ax_lims[view][i][1] for v in self.viewer]
                    f1, f2 = min, max
                    if match_axes == "overlap":
                        f1, f2 = f2, f1
                    if min_lims[0] > max_lims[0]:
                        f1, f2 = f2, f1
                    ax_lims.append([f1(min_lims), f2(max_lims)])

            # Match axes to one plot
            else:
                try:
                    im = self.viewer[match_axes].im
                    ax_lims = im.ax_lims[view]

                except TypeError:
                    raise TypeError("Unrecognised option for <match_axes>",
                                    match_axes)

            # Set these limits for all plots
            all_ims = [v.im for v in self.viewer] \
                    + [c for c in self.comparison]
            for im in all_ims:
                if match_axes != "y":
                    im.ax_lims[view][0] = ax_lims[0]
                if match_axes != "x":
                    im.ax_lims[view][1] = ax_lims[1]

    def make_ui(self, share_slider):

        # Only allow share_slider if images have same frame of reference
        if share_slider:
            share_slider *= all([v.im.same_frame(self.viewer[0].im) for v in
                                 self.viewer])

        # Make UI for first image
        v0 = self.viewer[0]
        v0.make_ui()

        # Store needed UIs
        self.ui_view = v0.ui_view
        self.view = self.ui_view.value
        self.ui_struct_plot_type = v0.ui_struct_plot_type
        self.struct_plot_type = self.ui_struct_plot_type.value

        # Make main upper UI list (= view radio + single HU/slice slider)
        many_sliders = not share_slider and self.n > 1
        if not many_sliders:
            self.main_ui = v0.main_ui
            if v0.zoom_ui:
                v0.ui_zoom_reset.on_click(self.make_reset_zoom(v0))
        else:
            self.main_ui = [self.ui_view]

        # Make UI for other images
        for v in self.viewer[1:]:
            v.make_ui(vimage=v0, share_slider=share_slider)

        # Make UI for each image (= unique HU/slice sliders and struct jump)
        self.per_image_ui = []
        if many_sliders:
            for v in self.viewer:

                # Structure jumping
                sliders = []
                if v.im.has_structs:
                    sliders.append(v.ui_struct_jump)
                else:
                    if self.any_attr("structs"):
                        sliders.append(ipyw.Label())

                # HU sliders
                if v.hu_from_width:
                    sliders.extend([v.ui_hu_centre, v.ui_hu_width])
                else:
                    sliders.append(v.ui_hu)

                # Zoom sliders
                if v.zoom_ui:
                    sliders.extend(v.all_zoom_ui)
                    v.ui_zoom_reset.on_click(self.make_reset_zoom(v))

                # Slice slider
                sliders.append(v.ui_slice)
                self.per_image_ui.append(sliders)

        # Make extra UI list
        self.extra_ui = []
        for attr in ["mask", "dose", "df"]:
            if self.any_attr(attr):
                self.extra_ui.append(getattr(v0, "ui_" + attr))
        if self.any_attr("jacobian"):
            self.extra_ui.extend([v0.ui_jac_opacity, v0.ui_jac_range])
        if self.any_attr("structs"):
            self.extra_ui.extend([v0.ui_struct_plot_type, 
                                  v0.ui_struct_linewidth,
                                  v0.ui_struct_opacity])

        # Make extra UI elements
        self.make_lower_ui()
        self.make_comparison_ui()
        self.make_translation_ui()

        # Assemble UI boxes
        main_and_extra_box = ipyw.HBox([ipyw.VBox(self.main_ui),
                                        ipyw.VBox(self.extra_ui),
                                        ipyw.VBox(self.trans_ui),
                                        ipyw.VBox(self.comp_ui)])
        self.slider_boxes = [ipyw.VBox(ui) for ui in self.per_image_ui]
        self.set_slider_widths()
        self.upper_ui = [main_and_extra_box, ipyw.HBox(self.slider_boxes)]
        self.upper_ui_box = ipyw.VBox(self.upper_ui)
        self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)
        self.all_ui = (
            self.main_ui
            + self.extra_ui
            + list(itertools.chain.from_iterable(self.per_image_ui))
            + self.comp_ui
            + self.trans_ui
            + self.ui_struct_checkboxes
            + [self.trigger]
        )

    def make_lower_ui(self):
        """Make lower UI for structure checkboxes."""

        # Saving UI
        self.lower_ui = []
        if self.viewer[0].save_as is not None:
            self.lower_ui.extend([self.viewer[0].save_name,
                                  self.viewer[0].save_button])

        # Structure UI
        many_with_structs = sum([v.im.has_structs for v in self.viewer]) > 1
        self.ui_struct_checkboxes = []
        for v in self.viewer:

            # Add plot title to structure UI
            if many_with_structs and v.im.has_structs:
                title = f"<b>{v.im.title + ':'}</b>"
                if v.compare_structs:
                    v.ui_struct_comp[0].value = title
                elif v.struct_info:
                    v.ui_struct_checkboxes[0].value = title
                else:
                    self.lower_ui.append(ipyw.HTML(value=title))

            # Add to overall lower UI
            self.lower_ui.extend(v.lower_ui)
            self.ui_struct_checkboxes.extend(v.ui_struct_checkboxes)

    def make_comparison_ui(self):

        self.comp_ui = []

        if self.has_chequerboard:
            max_splits = max([10, self.cb_splits])
            self.ui_cb = ipyw.IntSlider(
                min=1, max=max_splits, value=self.cb_splits, step=1,
                continuous_update=self.viewer[0].continuous_update,
                description="Chequerboard splits",
                style=_style,
            )
            self.comp_ui.append(self.ui_cb)

        if self.has_overlay:
            self.ui_overlay = ipyw.FloatSlider(
                value=self.overlay_opacity, min=0, max=1, step=0.1,
                description="Overlay opacity",
                continuous_update=self.viewer[0].continuous_update,
                readout_format=".1f",
                style=_style,
            )
            self.comp_ui.append(self.ui_overlay)

        if len(self.comparison):
            self.ui_invert = ipyw.Checkbox(value=False,
                                           description="Invert comparison")
            self.comp_ui.append(self.ui_invert)

    def make_translation_ui(self):

        self.trans_ui = []
        if not self.translation:
            return

        self.trans_viewer = self.viewer[int(self.n > 1)]
        self.trans_ui.append(ipyw.HTML(value="<b>Translation:</b>"))

        # Get input/output filenames
        if self.tfile is None:
            tfile = self.find_translation_file(self.trans_viewer.im)
            self.has_translation_input = tfile is not None
            if self.has_translation_input:
                tfile_out = re.sub(".0.txt", "_custom.txt", tfile)
            else:
                tfile_out = "translation.txt"
        else:
            tfile = self.tfile
            tfile_out = self.tfile
            self.has_translation_input = True

        # Make translation file UI
        if self.has_translation_input:
            self.translation_input = ipyw.Text(description="Original:",
                                               value=tfile)
            self.trans_ui.append(self.translation_input)
        self.translation_output = ipyw.Text(description="Save as:",
                                            value=tfile_out)
        self.tbutton = ipyw.Button(description="Write translation")
        self.tbutton.on_click(self.write_translation_to_file)
        self.trans_ui.extend([self.translation_output, self.tbutton])

        # Make translation sliders
        self.tsliders = {}
        for ax in _axes:
            n = self.trans_viewer.im.n_voxels[ax]
            self.tsliders[ax] = ipyw.IntSlider(
                min=-n,
                max=n,
                value=0,
                description=f"{ax} (0 mm)",
                continuous_update=False,
                #  style=_style
            )
            self.trans_ui.append(self.tsliders[ax])
        self.current_trans = {ax: slider.value for ax, slider
                              in self.tsliders.items()}

    def find_translation_file(self, image):
        """Find an elastix translation file inside the directory of an image.
        """

        if not hasattr(image, "path"):
            return
        indir = os.path.dirname(image.path)
        tfile = indir + "/TransformParameters.0.txt"
        if os.path.isfile(tfile):
            return tfile

    def write_translation_to_file(self, _):
        """Write current translation to file."""

        input_file = self.translation_input.value \
            if self.has_translation_input else None
        if input_file == "":
            input_file = None
        output_file = self.translation_output.value
        translations = {f"d{ax}": self.tsliders[ax].value 
                        * abs(self.trans_viewer.im.voxel_sizes[ax])
                        for ax in self.tsliders}
        write_translation_to_file(output_file, input_file=input_file,
                                  **translations)

    def apply_translation(self):
        """Update the description of translation sliders to show translation
        in mm if the translation is changed."""

        new_trans = {ax: slider.value for ax, slider in self.tsliders.items()}
        if new_trans == self.current_trans:
            return

        # Set shift for image
        self.current_trans = new_trans
        self.trans_viewer.im.set_shift(self.current_trans["x"],
                                       self.current_trans["y"],
                                       self.current_trans["z"])

        # Adjust descriptions
        for ax, slider in self.tsliders.items():
            slider.description = "{} ({:.0f} mm)".format(
                ax, self.trans_viewer.im.shift_mm[ax])

    def set_slider_widths(self):
        """Adjust widths of slider UI."""

        if self.plots_per_row is not None and self.plots_per_row < self.n:
            return
        for i, slider in enumerate(self.slider_boxes[:-1]):
            width = self.figsize * self.viewer[i].im.get_relative_width(
                self.view, self.zoom, self.colorbar) \
                    * mpl.rcParams["figure.dpi"]
            slider.layout = ipyw.Layout(width=f"{width}px")

    def make_reset_zoom(self, viewer):
        """Make a reset zoom function for a given viewer that updates zoom 
        without plotting, then plots afterwards."""

        def reset_zoom(_):
            self.plotting = True
            viewer.reset_zoom(_)
            self.plotting = False
            self.trigger.value = not self.trigger.value

        return reset_zoom

    def make_fig(self):

        # Get relative width of each subplot
        width_ratios = [v.im.get_relative_width(self.view, self.zoom, 
                                                self.colorbar)
                        for v in self.viewer]
        width_ratios.extend([c.get_relative_width(self.view, self.zoom) for
                             c in self.comparison])

        # Get rows and columns
        n_plots = (not self.comparison_only) * self.n \
            + len(self.comparison)
        if self.comparison_only:
            width_ratios = width_ratios[self.n:]
        if self.plots_per_row is not None:
            n_cols = min([self.plots_per_row, n_plots])
            n_rows = int(np.ceil(n_plots / n_cols))
            width_ratios_padded = width_ratios + \
                [0 for i in range((n_rows * n_cols) - n_plots)]
            ratios_per_row = np.array(width_ratios_padded).reshape(n_rows,
                                                                   n_cols)
            width_ratios = np.amax(ratios_per_row, axis=0)
        else:
            n_cols = n_plots
            n_rows = 1

        # Calculate height and width
        height = self.figsize * n_rows
        width = self.figsize * sum(width_ratios)

        # Outside notebook, just resize figure
        if not self.in_notebook and hasattr(self, "fig"):
            self.fig.set_size_inches(width, height)
            return

        # Make new figure
        self.fig = plt.figure(figsize=(width, height))

        # Make gridspec
        gs = self.fig.add_gridspec(n_rows, n_cols, width_ratios=width_ratios)
        i = 0
        if not self.comparison_only:
            for v in self.viewer:
                v.gs = gs[i]
                i += 1
        for c in self.comparison:
            c.gs = gs[i]
            i += 1

        # Assign callbacks to figure
        if not self.in_notebook:
            self.set_callbacks()

    def set_callbacks(self):
        """Set callbacks for scrolls and keypresses."""

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_key(self, event):
        """Callbacks for key presses."""

        # Apply all callbacks to first viewer
        self.plotting = True
        self.viewer[0].on_key(event)

        # Extra callbacks for scrolling each plot
        if len(self.per_image_ui):
            for v in self.viewer[1:]:

                # Scrolling
                n_small = 1
                n_big = 5
                if event.key == "left":
                    v.decrease_slice(n_small)
                elif event.key == "right":
                    v.increase_slice(n_small)
                elif event.key == "down":
                    v.decrease_slice(n_big)
                elif event.key == "up":
                    v.increase_slice(n_big)

        # Press i to invert comparisons
        elif event.key == "i":
            if len(self.comparison):
                self.ui_invert.value = not self.ui_invert.value

        # Press o to change overlay opacity
        elif event.key == "o":
            if self.has_overlay:
                ops = [0.2, 0.35, 0.5, 0.65, 0.8]
                next_op = {ops[i]: ops[i + 1] if i + 1 < len(ops)
                             else ops[0] for i in range(len(ops))}
                diffs = [abs(op - self.ui_overlay.value) for op in ops]
                current = ops[diffs.index(min(diffs))]
                self.ui_overlay.value = next_op[current]

        # Remake plot
        self.plotting = False
        self.plot(tight_layout=(event.key == "v"))

    def on_scroll(self, event):
        """Callbacks for scroll events."""

        # First viewers
        self.plotting = True
        self.viewer[0].on_scroll(event)

        # Extra callbacks for scrolling each plot
        if len(self.per_image_ui):
            for v in self.viewer[1:]:
                if event.button == "up":
                    self.increase_slice()
                elif event.button == "down":
                    self.decrease_slice()

        # Remake plot
        self.plotting = False
        self.plot(tight_layout=False)

    def plot(self, tight_layout=True, **kwargs):
        """Plot all images."""

        if self.plotting:
            return
        self.plotting = True

        # Deal with view change
        if self.ui_view.value != self.view:
            self.view = self.ui_view.value
            for v in self.viewer:
                v.view = self.ui_view.value
                v.on_view_change()
            self.set_slider_widths()

        # Deal with structure plot type change
        if self.struct_plot_type != self.ui_struct_plot_type.value:
            self.struct_plot_type = self.ui_struct_plot_type.value
            self.viewer[0].update_struct_sliders()

        # Deal with structure jumps
        for v in self.viewer:
            if v.ui_struct_jump != "":
                v.jump_to_struct()

        # Apply any translations
        if self.translation:
            self.apply_translation()

        # Reset figure
        self.make_fig()

        # Plot all images
        for v in self.viewer:
            if self.comparison_only:
                v.set_slice_and_view()
                v.im.set_slice(self.view, v.slice[self.view])
            else:
                v.plot()

        # Plot all comparison images
        if len(self.comparison):

            invert = self.ui_invert.value

            # Plot chequerboard
            if self.has_chequerboard:
                ImageViewer.plot_image(self, self.chequerboard, invert=invert, 
                                       n_splits=self.ui_cb.value,
                                       zoom=self.viewer[0].zoom,
                                       zoom_centre=self.viewer[0].zoom_centre,
                                       mpl_kwargs=self.viewer[0].v_min_max)

            # Plot overlay
            if self.has_overlay:
                ImageViewer.plot_image(self, self.overlay, invert=invert,
                                       opacity=self.ui_overlay.value,
                                       mpl_kwargs=self.viewer[0].v_min_max,
                                       zoom=self.viewer[0].zoom,
                                       zoom_centre=self.viewer[0].zoom_centre,
                                       legend=self.overlay_legend,
                                       legend_loc=self.legend_loc)

            # Plot difference image
            if self.has_diff:
                ImageViewer.plot_image(self, self.diff, invert=invert,
                                       zoom=self.viewer[0].zoom,
                                       zoom_centre=self.viewer[0].zoom_centre,
                                       mpl_kwargs=self.viewer[0].v_min_max)

        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)

        if tight_layout:
            plt.tight_layout()
        self.plotting = False

        # Automatic saving on first plot
        if self.viewer[0].save_as is not None and not self.saved:
            self.viewer[0].save_fig()
            self.saved = True

        # Update figure
        if not self.in_notebook:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()


class ImageViewer():
    """Class for displaying a MultiImage with interactive elements."""

    def __init__(
        self,
        nii,
        init_view="x-y",
        init_sl=None,
        init_pos=None,
        hu=(-300, 200),
        hu_width=500,
        hu_limits=(-2000, 2000),
        figsize=_default_figsize,
        xlim=None,
        ylim=None,
        zlim=None,
        zoom=None,
        zoom_centre=None,
        zoom_ui=False,
        colorbar=False,
        mpl_kwargs=None,
        dose_opacity=0.5,
        dose_kwargs=None,
        invert_mask=False,
        mask_color="black",
        jacobian_opacity=0.5,
        jacobian_kwargs=None,
        df_plot_type="grid",
        df_spacing=30,
        df_kwargs=None,
        struct_plot_type="contour",
        struct_opacity=None,
        struct_linewidth=2,
        struct_info=False,
        length_units=None,
        area_units=None,
        vol_units=None,
        struct_legend=False,
        legend_loc="lower left",
        init_struct=None,
        standalone=True,
        continuous_update=False,
        annotate_slice=None,
        save_as=None,
        show=True,
        **kwargs
    ):

        self.im = self.make_image(nii, **kwargs)
        if not self.im.valid:
            return
        self.gs = None  # Gridspec in which to place plot axes

        # Set initial view
        view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
        if self.im.dim2:
            init_view = self.im.orientation
        if init_view in view_map:
            self.view = view_map[init_view]
        else:
            self.view = init_view

        # Set initial slice numbers
        self.slice = {view: np.ceil(self.im.n_voxels[z] / 2) for view, z
                      in _slider_axes.items()}
        if init_pos is not None and self.im.scale_in_mm:
            self.set_slice_from_pos(init_view, init_pos)
        else:
            self.set_slice(init_view, init_sl)

        # Assign plot settings
        # General settings
        self.in_notebook = in_notebook()
        self.mpl_kwargs = mpl_kwargs
        self.figsize = to_inches(figsize)
        self.continuous_update = continuous_update
        self.colorbar = colorbar
        self.annotate_slice = annotate_slice
        if self.annotate_slice is None and not self.in_notebook:
            self.annotate_slice = True
        self.save_as = save_as
        self.plotting = False
        self.callbacks_set = False
        self.standalone = standalone
        self.set_ax_lims(xlim, ylim, zlim)
        self.zoom = zoom
        self.zoom_centre = zoom_centre
        self.zoom_ui = zoom_ui

        # HU range settings
        self.hu = hu
        self.hu_width = hu_width
        self.hu_limits = hu_limits
        self.hu_from_width = isinstance(hu, float) or isinstance(hu, int)

        # Mask settings
        self.invert_mask = invert_mask
        self.mask_color = mask_color

        # Dose settings
        self.init_dose_opacity = dose_opacity
        self.dose_kwargs = dose_kwargs

        # Jacobian/deformation field settings
        self.init_jac_opacity = jacobian_opacity
        self.jacobian_kwargs = jacobian_kwargs
        self.df_plot_type = df_plot_type
        self.df_spacing = df_spacing
        self.df_kwargs = df_kwargs

        # Structure settings
        self.struct_plot_type = struct_plot_type
        self.struct_mask_opacity = 1
        self.struct_filled_opacity = 0.3
        if struct_opacity is not None:
            if struct_plot_type == "mask":
                self.struct_mask_opacity = struct_opacity
            elif struct_plot_type == "filled":
                self.struct_filled_opacity = struct_opacity
        self.struct_linewidth = struct_linewidth
        self.struct_legend = struct_legend
        self.legend_loc = legend_loc
        self.init_struct = init_struct

        # Structure info settings
        self.compare_structs = kwargs.get("compare_structs", False) \
                and len(self.im.struct_comparisons)
        self.struct_info = struct_info
        def get_units(units):
            if units is None:
                return "mm" if self.im.scale_in_mm else "voxels"
            return units
        self.area_units = get_units(area_units)
        self.length_units = get_units(length_units)
        self.vol_units = get_units(vol_units)

        # Display plot
        if standalone:
            self.make_ui()
            self.show(show)

    def make_image(self, *args, **kwargs):
        """Set up image object."""
        return MultiImage(*args, **kwargs)

    def set_ax_lims(self, xlim, ylim, zlim):
        """Set custom axis limits after image has been loaded."""

        axes = {"x": {"x-z": 1, "x-y": 0},
                "y": {"x-y": 1, "y-z": 1},
                "z": {"x-z": 0, "y-z": 0}}
        for ax, lims in zip(axes.keys(), [xlim, ylim, zlim]):
            if lims is None:
                continue
            for view, n in axes[ax].items():
                if lims[0] is not None:
                    self.im.ax_lims[view][n][0] = lims[0]
                if lims[1] is not None:
                    self.im.ax_lims[view][n][1] = lims[1]

    def set_slice(self, view, sl):
        """Set the current slice number in a specific view."""

        if sl is None:
            return
        max_slice = self.im.n_voxels[_slider_axes[view]]
        min_slice = 1
        if self.slice[view] < min_slice:
            self.slice[view] = min_slice
        elif self.slice[view] > max_slice:
            self.slice[view] = max_slice
        else:
            self.slice[view] = sl

    def set_slice_from_pos(self, view, pos):
        """Set the current slice number from a position in mm."""

        ax = _slider_axes[view]
        sl = self.im.pos_to_slice(pos, ax)
        self.set_slice(view, sl)

    def slider_to_sl(self, val, ax=None):
        """Convert a slider value to a slice number."""

        if ax is None:
            ax = _slider_axes[self.view]

        if self.im.scale_in_mm:
            return self.im.pos_to_slice(val, ax)
        else:
            return int(val)

    def slice_to_slider(self, sl, ax=None):
        """Convert a slice number to a slider value."""

        if ax is None:
            ax = _slider_axes[self.view]

        if self.im.scale_in_mm:
            return self.im.slice_to_pos(sl, ax)
        else:
            return sl

    def make_ui(self, vimage=None, share_slider=True):
        """Make Jupyter notebook UI. If qv_image contains another ImageViewer
        instance, the UI will be taken from that image. If share_slider is
        False, independent HU and slice sliders will be created."""

        shared_ui = isinstance(vimage, ImageViewer)
        self.main_ui = []

        # View radio buttons
        if not shared_ui:
            self.ui_view = ipyw.RadioButtons(
                options=["x-y", "y-z", "x-z"],
                value=self.view,
                description="Slice plane selection:",
                disabled=False,
                style=_style,
            )
            if not self.im.dim2:
                self.main_ui.append(self.ui_view)
        else:
            self.ui_view = vimage.ui_view
            self.view = self.ui_view.value

        # Structure jumping menu
        # Get list of structures
        self.structs_for_jump = {"": None, 
                                 **{s.name_unique: s for s in
                                    self.im.structs}}
        structs_standard = {standard_str(s): s for s in self.structs_for_jump}
        if standard_str(self.init_struct) in structs_standard:
            init_struct = structs_standard[standard_str(self.init_struct)]
        else:
            init_struct = ""
        self.ui_struct_jump = ipyw.Dropdown(
            options=self.structs_for_jump.keys(),
            value=init_struct,
            description="Jump to",
            style=_style,
        )
        if self.im.has_structs:
            self.main_ui.append(self.ui_struct_jump)

        # HU and slice sliders
        if not share_slider or not shared_ui:

            # Make HU slider
            if self.hu_limits == "auto":
                hu_limits = (self.im.data.min(), self.im.data.max())
            else:
                hu_limits = self.hu_limits
            
            # Single range slider
            if not self.hu_from_width:
                vmin = max([self.hu[0], hu_limits[0]])
                vmax = min([self.hu[1], hu_limits[1]])
                ui_hu_kwargs = {
                    "min": hu_limits[0],
                    "max": hu_limits[1],
                    "value": (vmin, vmax),
                    "description": "HU range",
                    "continuous_update": False,
                    "style": _style
                }
                if abs(hu_limits[1] - hu_limits[0]) < 1.01:
                    ui_hu_kwargs.update({"step": 0.1, "readout_format": ".1f"})
                    self.ui_hu = ipyw.FloatRangeSlider(**ui_hu_kwargs)
                else:
                    self.ui_hu = ipyw.IntRangeSlider(**ui_hu_kwargs)
                self.main_ui.append(self.ui_hu)

            # Centre and window sliders
            else:
                self.ui_hu_centre = ipyw.IntSlider(
                    min=hu_limits[0], max=hu_limits[1],
                    value=self.hu, description="HU centre",
                    continuous_update=False, style=_style)
                self.ui_hu_width = ipyw.IntSlider(
                    min=0, max=abs(hu_limits[1] - hu_limits[0]),
                    value=self.hu_width, description="HU width",
                    continuous_update=False, style=_style)
                self.ui_hu_list = [self.ui_hu_centre, self.ui_hu_width]
                self.main_ui.extend(self.ui_hu_list)
                self.ui_hu = ipyw.VBox(self.ui_hu_list)

            # Make zoom UI
            if self.zoom_ui:

                # Get initial zoom level for each view
                zoom = self.im.get_ax_dict(self.zoom)
                if zoom is None:
                    self.current_zoom = {view: 1 for view in _slider_axes}
                else:
                    self.current_zoom = {}
                    for view, (x, y) in _plot_axes.items():
                        zooms = [zoom[x], zoom[y]]
                        self.current_zoom[view] = max(zooms)

                # Zoom level slider
                self.ui_zoom = ipyw.FloatSlider(
                    min=1, max=5, step=0.1, description="Zoom",
                    readout_format=".1f", continuous_update=False, 
                    style=_style)

                # Get initial zoom centres
                zoom_centre = self.im.get_ax_dict(self.zoom_centre, 
                                                  default=None)
                self.default_centre = {view: self.im.get_image_centre(view) 
                                       for view in _slider_axes}
                self.current_centre = {}
                for view in _plot_axes:
                    self.current_centre[view] = []
                    for i, ax in enumerate(_plot_axes[view]):
                        if zoom_centre is None or zoom_centre[ax] is None:
                            self.current_centre[view].append(
                                self.default_centre[view][i])
                        else:
                            self.current_centre[view].append(zoom_centre[ax])

                # Make zoom centre sliders
                self.ui_zoom_centre_x = ipyw.FloatSlider(
                    continuous_update=self.continuous_update,
                    readout_format=".1f", step=1)
                self.ui_zoom_centre_y = ipyw.FloatSlider(
                    continuous_update=self.continuous_update,
                    readout_format=".1f", step=1)
                self.update_zoom_sliders()

                # Zoom reset button
                self.ui_zoom_reset = ipyw.Button(description="Reset zoom")
                if self.standalone:
                    self.ui_zoom_reset.on_click(self.reset_zoom)
                self.all_zoom_ui = [self.ui_zoom, self.ui_zoom_centre_x,
                                    self.ui_zoom_centre_y, self.ui_zoom_reset]
                self.main_ui.extend(self.all_zoom_ui)

            # Make slice slider
            readout = ".1f" if self.im.scale_in_mm else ".0f"
            self.ui_slice = ipyw.FloatSlider(
                continuous_update=self.continuous_update,
                style=_style,
                readout_format=readout
            )
            self.own_ui_slice = True
            self.update_slice_slider()
            if not self.im.dim2:
                self.main_ui.append(self.ui_slice)

        else:
            if self.hu_from_width:
                self.ui_hu_width = vimage.ui_hu_width
                self.ui_hu_centre = vimage.ui_hu_centre
            else:
                self.ui_hu = vimage.ui_hu
            self.ui_slice = vimage.ui_slice
            self.slice[self.view] = self.ui_slice.value
            if self.zoom_ui:
                self.ui_zoom = vimage.ui_zoom
                self.ui_zoom_centre_x = vimage.ui_zoom_centre_x
                self.ui_zoom_centre_y = vimage.ui_zoom_centre_y
                self.ui_zoom_reset = vimage.ui_zoom_reset
                self.current_zoom = vimage.current_zoom
                self.current_centre = vimage.current_centre
            self.own_ui_slice = False

        # Extra sliders
        self.extra_ui = []
        if not shared_ui:

            # Mask checkbox
            self.ui_mask = ipyw.Checkbox(value=self.im.has_mask,
                                         description="Apply mask")
            if self.im.has_mask:
                self.extra_ui.append(self.ui_mask)

            # Dose opacity
            self.ui_dose = ipyw.FloatSlider(
                value=self.init_dose_opacity, min=0, max=1, step=0.05,
                description="Dose opacity",
                continuous_update=self.continuous_update,
                readout_format=".2f", style=_style,
            )
            if self.im.has_dose:
                self.extra_ui.append(self.ui_dose)

            # Jacobian opacity and range
            self.ui_jac_opacity = ipyw.FloatSlider(
                value=self.init_jac_opacity, min=0, max=1, step=0.05,
                description="Jacobian opacity",
                continuous_update=self.continuous_update,
                readout_format=".2f", style=_style,
            )
            self.ui_jac_range = ipyw.FloatRangeSlider(
                min=-0.5, max=2.5, step=0.1, value=[0.8, 1.2],
                description="Jacobian range", continuous_update=False,
                style=_style, readout_format=".1f"
            )
            if self.im.has_jacobian:
                self.extra_ui.extend([self.ui_jac_opacity,
                                      self.ui_jac_range])

            # Deformation field plot type
            self.ui_df = ipyw.Dropdown(
                options=_df_plot_types,
                value=self.df_plot_type,
                description="Deformation field",
                style=_style,
            )
            if self.im.has_df:
                self.extra_ui.append(self.ui_df)

            # Structure UI
            # Structure plot type
            self.ui_struct_plot_type = ipyw.Dropdown(
                options=_struct_plot_types,
                value=self.struct_plot_type,
                description="Structure plotting",
                style=_style,
            )

            # Opacity/linewidth sliders
            self.ui_struct_linewidth = ipyw.IntSlider(
                min=1, max=8, step=1, value=self.struct_linewidth, 
                description="Linewidth", continuous_update=False, 
                style=_style)
            self.ui_struct_opacity = ipyw.FloatSlider(
                min=0, max=1, step=0.1, continuous_update=False, 
                description="Opacity", style=_style)
            self.update_struct_sliders()

            # Add all structure UIs
            if self.im.has_structs:
                self.extra_ui.extend([
                    self.ui_struct_plot_type,
                    self.ui_struct_linewidth,
                    self.ui_struct_opacity
                ])

        else:
            to_share = ["ui_mask", "ui_dose", "ui_jac_opacity", "ui_jac_range",
                        "ui_df", "ui_struct_plot_type", "ui_struct_linewidth",
                        "ui_struct_linewidth", "ui_struct_opacity"]
            for ts in to_share:
                setattr(self, ts, getattr(vimage, ts))

        # Make lower
        self.make_lower_ui()

        # Combine UI elements
        self.upper_ui = [ipyw.VBox(self.main_ui), ipyw.VBox(self.extra_ui)]
        self.upper_ui_box = ipyw.HBox(self.upper_ui)
        self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.trigger = ipyw.Checkbox(value=True)
        self.all_ui = (
            self.main_ui 
            + self.extra_ui 
            + self.ui_struct_checkboxes
            + [self.trigger]
        )

    def make_lower_ui(self):

        # Saving UI
        self.lower_ui = []
        if self.save_as is not None:
            self.save_name = ipyw.Text(description="Output file:",
                                       value=self.save_as)
            self.save_button = ipyw.Button(description="Save")
            self.save_button.on_click(self.save_fig)
            if self.standalone:
                self.lower_ui.extend([self.save_name, self.save_button])

        # Structure comparison display
        self.ui_struct_comp = []
        self.ui_struct_dice = []
        layout = ipyw.Layout(align_items="center", padding="0px 0px 0px 50px")
        if self.compare_structs:

            self.ui_struct_comp.append(ipyw.HTML(
                value="<b>Structure comparison:</b>"))
            self.ui_struct_dice.append(ipyw.HTML(
                value="<b>Dice score</b>"))

            for sc in self.im.struct_comparisons:
                self.ui_struct_comp.append(ipyw.Label(value=sc.name))
                self.ui_struct_dice.append(ipyw.Label())

            self.lower_ui.append(ipyw.HBox([
                ipyw.VBox(self.ui_struct_comp),
                ipyw.VBox(self.ui_struct_dice, layout=layout)
            ]))
            self.update_struct_comparisons()

        # Columns for structure info
        self.ui_struct_checkboxes = []
        self.ui_struct_vol = []
        self.ui_struct_area = []
        self.ui_struct_x = []
        self.ui_struct_y = []
        self.ui_struct_centre = []
        if self.struct_info:

            self.ui_struct_checkboxes.append(
                ipyw.HTML(value="<b>Structures:</b>"))

            vol_units = self.vol_units if self.vol_units != "mm" \
                else "mm<sup>3</sup>"
            self.ui_struct_vol.append(ipyw.HTML(
                value=f"<b>Volume ({vol_units})</b>"))
            area_units = self.area_units if self.area_units != "mm" \
                else "mm<sup>2</sup>"
            self.ui_struct_area.append(ipyw.HTML(
                value=f"<b>Area ({area_units})</b>"))

            self.ui_struct_x.append(ipyw.HTML())
            self.ui_struct_y.append(ipyw.HTML())

            centre_units = " (mm)" if self.im.scale_in_mm else ""
            self.ui_struct_centre.append(ipyw.HTML(
                value=f"<b>Centre{centre_units}</b>"))

        # Make checkbox for each structure
        for s in self.im.structs:
            s.checkbox = ipyw.Checkbox(value=True, description=s.name_unique)
            self.ui_struct_checkboxes.append(s.checkbox)
            if not self.struct_info:
                self.lower_ui.append(s.checkbox)
            else:
                vol_fmt = "{:.0f}" if self.vol_units == "ml" else "{:.0f}"
                self.ui_struct_vol.append(ipyw.Label(
                    value=vol_fmt.format(s.get_volume(self.vol_units))))
                s.ui_area = ipyw.Label()
                s.ui_x = ipyw.Label()
                s.ui_y = ipyw.Label()
                self.ui_struct_area.append(s.ui_area)
                self.ui_struct_x.append(s.ui_x)
                self.ui_struct_y.append(s.ui_y)
                s.ui_centre = ipyw.Label()
                self.ui_struct_centre.append(s.ui_centre)
        self.visible_structs = self.get_struct_visibility()

        if self.struct_info:
            self.update_struct_info()
            self.lower_ui.append(ipyw.HBox([
                ipyw.VBox(self.ui_struct_checkboxes),
                ipyw.VBox(self.ui_struct_vol,
                          layout=ipyw.Layout(align_items="center")),
                ipyw.VBox(self.ui_struct_area, layout=layout),
                ipyw.VBox(self.ui_struct_x, layout=layout),
                ipyw.VBox(self.ui_struct_y, layout=layout),
                ipyw.VBox(self.ui_struct_centre, layout=layout)
            ]))

    def get_struct_visibility(self):
        """Get list of currently visible structures from checkboxes."""

        return [s.description for s in self.ui_struct_checkboxes[1:] 
                if s.value]

    def update_struct_comparisons(self):
        """Update structure comparison metrics to reflect the current 
        view/slice."""

        if not self.compare_structs:
            return

        for i, sc in enumerate(self.im.struct_comparisons):
            dice = sc.dice_score(self.view, self.slice[self.view])
            dice_str = "{:.3f}".format(dice) if dice is not None else "—"
            self.ui_struct_dice[i + 1].value = dice_str
                
    def update_struct_info(self):
        """Update structure info UI to reflect current view/slice."""

        if not self.struct_info:
            return

        # Update column headers
        x, y = _plot_axes[self.view]
        self.ui_struct_x[0].value = f"<b>{x} length ({self.length_units})</b>"
        self.ui_struct_y[0].value = f"<b>{y} length ({self.length_units})</b>"

        # Update column values
        fmt = "{:.1f}" if self.length_units == "mm" else "{:.0f}"
        area_fmt = "{:.1f}" if self.area_units == "mm" else "{:.0f}"
        centre_fmt = "{:.1f}" if self.im.scale_in_mm else "{:.0f}"
        for s in self.im.structs:

            # Get area
            area = s.get_area(self.view, self.slice[self.view],
                              self.area_units)
            s.ui_area.value = "—" if area is None else area_fmt.format(area)

            # Get x/y lengths
            extents = s.get_extents(self.view, self.slice[self.view],
                                    self.length_units)
            extent_strs = []
            for ex in extents:
                if ex is None:
                    extent_strs.append("—")
                else:
                    extent_strs.append(fmt.format(ex))
            s.ui_x.value = extent_strs[0]
            s.ui_y.value = extent_strs[1]

            # Get centre
            centre = s.get_centre(self.view, self.slice[self.view])
            centre_str = f"({centre_fmt}, {centre_fmt})"
            s.ui_centre.value = centre_str.format(*centre) if centre[0] is \
                    not None else "—"

    def update_struct_sliders(self):
        """Update struct sliders depending on current plot type."""

        self.struct_plot_type = self.ui_struct_plot_type.value

        # Disable irrelevant sliders
        self.ui_struct_opacity.disabled = self.struct_plot_type \
                in ["contour", "none"]
        self.ui_struct_linewidth.disabled = self.struct_plot_type \
                in ["mask", "none"]

        # Set opacity of masked or filled structs
        if self.struct_plot_type == "mask":
            self.ui_struct_opacity.value = self.struct_mask_opacity
        elif self.struct_plot_type == "filled":
            self.ui_struct_opacity.value = self.struct_filled_opacity

    def set_callbacks(self):
        """Set up matplotlib callback functions for interactive plotting."""

        if not self.standalone or self.callbacks_set:
            return

        self.im.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.im.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.callbacks_set = True

    def on_key(self, event):
        """Events run on keypress outside jupyter notebook."""

        # Settings
        n_small = 1
        n_big = 5

        # Press v to change view
        if event.key == "v":
            next_view = {"x-y": "y-z", "y-z": "x-z", "x-z": "x-y"}
            self.ui_view.value = next_view[self.ui_view.value]

        # Press d to change dose opacity
        elif event.key == "d":
            if self.im.has_dose:
                doses = [0, 0.15, 0.35, 0.5, 1]
                next_dose = {doses[i]: doses[i + 1] if i + 1 < len(doses)
                             else doses[0] for i in range(len(doses))}
                diffs = [abs(d - self.ui_dose.value) for d in doses]
                current = doses[diffs.index(min(diffs))]
                self.ui_dose.value = next_dose[current]

        # Press m to switch mask on and off
        elif event.key == "m":
            if self.im.has_mask:
                self.ui_mask.value = not self.ui_mask.value

        # Press c to change structure plot type
        elif event.key == "c":
            if self.im.has_structs:
                next_type = {"mask": "contour", "contour": "filled",
                             "filled": "none", "none": "mask"}
                self.ui_struct_plot_type.value = \
                    next_type[self.ui_struct_plot_type.value]

        # Press j to jump between structures
        elif event.key == "j" and self.im.has_structs:
            structs = self.ui_struct_jump.options[1:]
            if not hasattr(self, "current_struct"):
                current_idx = 0
            else:
                current_idx = structs.index(self.current_struct)
            new_idx = current_idx + 1
            if new_idx == len(structs):
                new_idx = 0
            new_struct = structs[new_idx]
            self.ui_struct_jump.value = new_struct

        # Press arrow keys to scroll through many slices
        elif event.key == "left":
            self.decrease_slice(n_small)
        elif event.key == "right":
            self.increase_slice(n_small)
        elif event.key == "down":
            self.decrease_slice(n_big)
        elif event.key == "up":
            self.increase_slice(n_big)

        else:
            return
        
        # Remake plot
        if self.standalone:
            self.plot()

    def on_scroll(self, event):
        """Events run on scroll outside jupyter notebook."""

        if event.button == "up":
            self.increase_slice()
        elif event.button == "down":
            self.decrease_slice()
        else:
            return

        # Remake plot
        if self.standalone:
            self.plot()

    def increase_slice(self, n=1):
        """Increase slice slider value by n slices."""

        new_val = self.ui_slice.value + n * self.ui_slice.step
        if new_val <= self.ui_slice.max:
            self.ui_slice.value = new_val
        else:
            self.ui_slice.value = self.ui_slice.max

    def decrease_slice(self, n=1):
        """Decrease slice slider value by n slices."""

        new_val = self.ui_slice.value - n * self.ui_slice.step
        if new_val >= self.ui_slice.min:
            self.ui_slice.value = new_val
        else:
            self.ui_slice.value = self.ui_slice.min

    def on_view_change(self):
        """Deal with a view change."""

        self.update_slice_slider()
        if self.zoom_ui:
            self.update_zoom_sliders()

    def update_slice_slider(self):
        """Update the slice slider to show the axis corresponding to the
        current view, with value set to the last value used on that axis."""

        if not self.own_ui_slice:
            return

        # Get new min and max
        ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            new_min = min(self.im.lims[ax])
            new_max = max(self.im.lims[ax])
        else:
            new_min = 1
            new_max = self.im.n_voxels[ax]

        # Set slider values
        val = self.slice_to_slider(self.slice[self.view])
        self.update_slider(self.ui_slice, new_min, new_max, val)

        # Set step and description
        self.ui_slice.step = abs(self.im.voxel_sizes[ax]) if \
            self.im.scale_in_mm else 1
        self.update_slice_slider_desc()

    def update_slider(self, slider, new_min, new_max, val):
        """Update slider min, max, and value without causing errors due to
        values outside range."""

        # Set to widest range
        if new_min < slider.min:
            slider.min = new_min
        if new_max > slider.max:
            slider.max = new_max

        # Set new value
        slider.value = val

        # Set final limits
        if slider.min != new_min:
            slider.min = new_min
        if slider.max != new_max:
            slider.max = new_max

    def update_slice_slider_desc(self):
        """Update slice slider description to reflect current axis and
        position."""

        if not self.own_ui_slice:
            return
        ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            self.ui_slice.description = f"{ax} (mm)"
        else:
            pos = self.im.slice_to_pos(
                self.slider_to_sl(self.ui_slice.value), ax)
            self.ui_slice.description = f"{ax} ({pos:.1f} mm)"

    def update_zoom_sliders(self):
        """Update zoom sliders to reflect the current view."""

        if not self.zoom_ui:
            return

        self.ui_zoom.value = self.current_zoom[self.view]

        units = " (mm)" if self.im.scale_in_mm else ""
        for i, ui in enumerate([self.ui_zoom_centre_x, self.ui_zoom_centre_y]):

            # Set min, max, and value
            new_min = min(self.im.ax_lims[self.view][i])
            new_max = max(self.im.ax_lims[self.view][i])
            self.update_slider(ui, new_min, new_max, 
                               self.current_centre[self.view][i])

            # Update description
            ui.description = "{} centre {}".format(
                _plot_axes[self.view][i], units)

    def reset_zoom(self, _):
        """Reset zoom values to 1 and zoom centres to defaults."""

        self.current_zoom[self.view] = 1 
        self.current_centre[self.view] = self.im.get_image_centre(self.view)
        self.plotting = True
        self.update_zoom_sliders()
        self.plotting = False
        if self.standalone:
            self.trigger.value = not self.trigger.value

    def jump_to_struct(self):
        """Jump to the mid slice of a structure."""

        if self.ui_struct_jump.value == "":
            return

        self.current_struct = self.ui_struct_jump.value
        struct = self.structs_for_jump[self.current_struct]
        if not struct.empty:
            mid_slice = int(np.mean(list(struct.contours[self.view].keys())))
            self.ui_slice.value = self.slice_to_slider(
                mid_slice, _slider_axes[self.view])
            self.slice[self.view] = mid_slice
            self.centre_at_struct(struct)
        self.ui_struct_jump.value = ""

    def centre_at_struct(self, struct):
        """Set the current zoom centre to be the centre of a structure."""

        if not self.zoom_ui or self.ui_zoom.value == 1:
            return

        centre = struct.get_centre(self.view, self.slice[self.view])
        self.current_centre[self.view] = centre
        self.update_zoom_sliders()

    def show(self, show=True):
        """Display plot and UI."""

        if self.in_notebook:
            ImageViewer.show_in_notebook(self, show)
        else:
            self.plot()
            if show:
                plt.show()

    def show_in_notebook(self, show):
        """Display interactive output in a jupyter notebook."""

        from IPython.display import display
        ui_kw = {str(np.random.rand()): ui for ui in self.all_ui if
                 hasattr(ui, "value")}
        self.out = ipyw.interactive_output(self.plot, ui_kw)
        to_display = [self.upper_ui_box, self.out]
        if len(self.lower_ui):
            to_display.append(self.lower_ui_box)
        if show:
            display(*to_display)

    def set_slice_and_view(self):
        """Get the current slice and view to plot from the UI."""

        # Get view
        view = self.ui_view.value
        if self.view != view:
            self.view = view
            self.on_view_change()

        # Get slice
        self.jump_to_struct()
        self.slice[self.view] = self.slider_to_sl(self.ui_slice.value)
        if not self.im.scale_in_mm:
            self.update_slice_slider_desc()

        # Get HU range
        self.v_min_max = self.get_hu_range()

        # Get zoom settings
        if self.zoom_ui:
            self.zoom = self.ui_zoom.value
            self.current_zoom[self.view] = self.zoom
            self.zoom_centre = [1, 1, 1]
            for i, ui in enumerate([self.ui_zoom_centre_x, 
                                    self.ui_zoom_centre_y]):
                self.zoom_centre[_axes[_plot_axes[self.view][i]]] = ui.value
                self.current_centre[self.view][i] = ui.value

    def get_hu_range(self):
        """Get vmin and vmax from HU sliders."""

        if self.hu_from_width:
            w = self.ui_hu_width.value / 2
            centre = self.ui_hu_centre.value
            return {"vmin": centre - w, "vmax": centre + w}
        else:
            return {"vmin": self.ui_hu.value[0], "vmax": self.ui_hu.value[1]}

    def plot(self, **kwargs):
        """Plot a slice with current settings."""

        if self.plotting:
            return
        self.plotting = True
        self.set_slice_and_view()

        # Get main image settings
        mpl_kwargs = self.v_min_max
        if self.mpl_kwargs is not None:
            mpl_kwargs.update(self.mpl_kwargs)

        # Get dose settings
        dose_kwargs = {}
        if self.im.has_dose:
            dose_kwargs = {"alpha": self.ui_dose.value}
            if self.dose_kwargs is not None:
                dose_kwargs.update(self.dose_kwargs)

        # Get jacobian settings
        jacobian_kwargs = {}
        if self.im.has_jacobian:
            jacobian_kwargs = {"alpha": self.ui_jac_opacity.value,
                               "vmin": self.ui_jac_range.value[0],
                               "vmax": self.ui_jac_range.value[1]}
            if self.jacobian_kwargs is not None:
                jacobian_kwargs.update(jacobian_kwargs)

        # Get structure settings
        self.update_struct_info()
        self.update_struct_comparisons()
        struct_kwargs = {}
        if self.ui_struct_plot_type.value != self.struct_plot_type:
            self.update_struct_slider()
        if self.struct_plot_type in ["contour", "filled"]:
            self.struct_linewidth = self.ui_struct_linewidth.value
            struct_kwargs["linewidth"] = self.struct_linewidth
        if self.struct_plot_type == "mask":
            self.struct_mask_opacity = self.ui_struct_opacity.value
            struct_kwargs["alpha"] = self.struct_mask_opacity
        elif self.struct_plot_type == "filled":
            self.struct_filled_opacity = self.ui_struct_opacity.value
            struct_kwargs["alpha"] = self.struct_filled_opacity

        # Structure visibility
        for s in self.im.structs:
            s.visible = s.checkbox.value
        vis = self.get_struct_visibility()
        if vis != self.visible_structs:
            self.visible_structs = vis
            if self.im.structs_as_mask:
                self.im.set_masks()

        # Make plot
        self.plot_image(self.im,
                        view=self.view,
                        sl=self.slice[self.view],
                        gs=self.gs,
                        mpl_kwargs=mpl_kwargs,
                        figsize=self.figsize,
                        zoom=self.zoom,
                        zoom_centre=self.zoom_centre,
                        colorbar=self.colorbar,
                        masked=self.ui_mask.value,
                        invert_mask=self.invert_mask,
                        mask_color=self.mask_color,
                        dose_kwargs=dose_kwargs,
                        jacobian_kwargs=jacobian_kwargs,
                        df_plot_type=self.ui_df.value,
                        df_spacing=self.df_spacing,
                        df_kwargs=self.df_kwargs,
                        struct_plot_type=self.struct_plot_type,
                        struct_kwargs=struct_kwargs,
                        struct_legend=self.struct_legend,
                        legend_loc=self.legend_loc,
                        annotate_slice=self.annotate_slice,
                        show=False)
        self.plotting = False

        # Ensure callbacks are set if outside jupyter
        if not self.in_notebook:
            self.set_callbacks()

        # Update figure
        if not self.in_notebook:
            self.im.fig.canvas.draw_idle()
            self.im.fig.canvas.flush_events()

    def plot_image(self, im, **kwargs):
        """Plot a NiftiImage, reusing existing axes if outside a Jupyter 
        notebook."""

        # Get axes
        ax = None
        if not self.in_notebook and hasattr(im, "ax"):
            ax = getattr(im, "ax")
            ax.clear()

        # Plot image
        im.plot(ax=ax, **kwargs)

    def save_fig(self, _=None):
        """Save figure to a file."""

        self.im.fig.savefig(self.save_name.value)


class OrthogViewer(ImageViewer):
    """ImageViewer with an orthgonal view displayed."""

    def make_image(self, *args, **kwargs):
        """Set up image object."""
        return OrthogonalImage(*args, **kwargs)

    def jump_to_struct(self):
        """Jump to mid slice of a structure."""

        if self.ui_struct_jump.value == "":
            return

        struct = self.structs_for_jump[self.ui_struct_jump.value]
        ImageViewer.jump_to_struct(self)

        orthog_view = _orthog[self.view]
        mid_slice = int(np.mean(list(struct.contours[orthog_view].keys())))
        self.im.orthog_slices[_slider_axes[orthog_view]] = mid_slice


def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        else:
            return False
    except NameError:
        return False


def write_translation_to_file(
    output_file, dx=0, dy=0, dz=0, input_file=None, overwrite=False
):

    """Open an existing elastix transformation file and create a new
    version with the translation parameters either replaced or added to the
    current user-created translation in the displayed figure.

    Parameters
    ----------
    output_file : string
        Name of the output file to produce.

    input_file : string, default=None
        Path to an Elastix translation file to use as an input.

    dx, dy, dz : float, default=None
        Translations (in mm) to add to the initial translations in the
        input_file.

    overwrite : bool, default=False
        If True, the shifts will be overwritten. If False, they will be added.
    """

    # Make dictionary of shifts
    delta = {"x": dx, "y": dy, "z": dz}

    # Create elastix formatted text
    if input_file is not None:

        infile = open(input_file, "r")

        # Reverse directions of deltas for consistency with elastix
        delta = {ax: -d for ax, d in delta.items()}

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

    # Make simple text
    else:
        out_text = ''.join(f'{ax} {delta[ax]}\n' for ax in delta)

    # Write to output
    outfile = open(output_file, "w")
    outfile.write(out_text)
    outfile.close()
    print("Wrote translation to file:", output_file)
