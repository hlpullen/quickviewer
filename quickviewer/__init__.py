"""Classes for displaying interactive medical image plots."""

import os
import re
import itertools
import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from quickviewer.image import MultiImage, OrthogonalImage, ChequerboardImage, \
        OverlayImage, DiffImage
from quickviewer.image import _slider_axes, _df_plot_types, \
        _struct_plot_types, _orthog, _default_figsize, _plot_axes, _axes


_style = {"description_width": "initial",
          "value_width": "initial"}


# Matplotlib settings
mpl.rcParams["figure.figsize"] = (7.4, 4.8)
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 14.0


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
        translation=False,
        suptitle=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        match_axes : int/str, default=None
            Method for adjusting axis limits. Can either be:
            - An integer n, where 0 < n < number of plots, or n is -1. The axes
              of all plots will be adjusted to be the same as those of plot n.
            - "all": axes for all plots will be adjusted to cover the maximum
              range of all plots.
            - "overlap": axes for all plots will be adjusted to just show the
              overlapping region.
            - "x": axes will be adjusted to cover the same range across
              whatever the x axis is in the current view.
            - "x": axes will be adjusted to cover the same range across
              whatever the y axis is in the current view.
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
        self.structs = self.get_input_list(structs)
        self.jacobian = self.get_input_list(jacobian)
        self.df = self.get_input_list(df)

        # Make individual viewers
        self.scale_in_mm = scale_in_mm
        self.viewer = []
        viewer_type = ImageViewer if not orthog_view else OrthogViewer
        for i in range(self.n):
            viewer = viewer_type(
                self.nii[i], title=self.title[i], dose=self.dose[i],
                mask=self.mask[i], structs=self.structs[i],
                jacobian=self.jacobian[i], df=self.df[i], standalone=False,
                scale_in_mm=scale_in_mm, **kwargs)
            if viewer.im.valid:
                self.viewer.append(viewer)

        # Load comparison images
        self.load_comparison(show_cb, show_overlay, show_diff)
        self.comparison_only = comparison_only
        self.translation = translation

        # Settings needed for plotting
        self.figsize = kwargs.get("figsize", _default_figsize)
        self.colorbar = kwargs.get("colorbar", False)
        self.plots_per_row = plots_per_row
        self.suptitle = suptitle
        self.match_axes = match_axes
        if self.match_axes is not None and not self.scale_in_mm:
            self.match_axes = None
        self.plotting = False

        # Make UI
        self.make_ui(share_slider)

        # Display
        ImageViewer.show_in_notebook(self)

    def get_input_list(self, inp):
        """Convert an input to a list with one item per image to be
        displayed."""

        if inp is None or len(inp) == 0:
            return [None for i in range(self.n)]

        # Convert arg to a list
        input_list = []
        if isinstance(inp, list) or isinstance(inp, tuple):
            if self.n == 1:
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
        else:
            self.main_ui = [self.ui_view]

        # Make UI for other images
        for v in self.viewer[1:]:
            v.make_ui(vimage=v0, share_slider=share_slider)

        # Make UI for each image (= unique HU/slice sliders and struct jump)
        self.per_image_ui = []
        if many_sliders:
            for v in self.viewer:
                sliders = [v.ui_hu, v.ui_slice]
                if v.im.has_structs:
                    sliders.insert(0, v.ui_struct_jump)
                else:
                    if self.any_attr("structs"):
                        sliders.insert(0, ipyw.Label())
                self.per_image_ui.append(sliders)

        # Make extra UI list
        self.extra_ui = []
        for attr in ["mask", "dose", "df"]:
            if self.any_attr(attr):
                self.extra_ui.append(getattr(v0, "ui_" + attr))
        if self.any_attr("jacobian"):
            self.extra_ui.extend([v0.ui_jac_opacity, v0.ui_jac_range])
        if self.any_attr("structs"):
            self.extra_ui.extend([v0.ui_struct_plot_type, v0.ui_struct_slider])

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
        self.all_ui = (
            self.main_ui
            + self.extra_ui
            + list(itertools.chain.from_iterable(self.per_image_ui))
            + self.comp_ui
            + self.trans_ui
            + self.ui_struct_checkboxes
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
                if v.struct_info:
                    v.ui_struct_checkboxes[0].value = title
                else:
                    self.lower_ui.append(ipyw.HTML(value=title))

            # Add to overall lower UI
            self.lower_ui.extend(v.lower_ui)
            self.ui_struct_checkboxes.extend(v.ui_struct_checkboxes)

    def make_comparison_ui(self):

        self.comp_ui = []

        if self.has_chequerboard:
            self.ui_cb = ipyw.IntSlider(
                min=1, max=10, value=2, step=1,
                continuous_update=self.viewer[0].continuous_update,
                description="Chequerboard splits",
                style=_style,
            )
            self.comp_ui.append(self.ui_cb)

        if self.has_overlay:
            self.ui_overlay = ipyw.FloatSlider(
                value=0.5, min=0, max=1, step=0.1,
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
        self.translation *= self.n > 1
        if not self.translation:
            return

        self.trans_ui.append(ipyw.HTML(value="<b>Translation:</b>"))

        # Make input/output filename UI
        tfile = self.find_translation_file(self.viewer[1].im)
        self.has_translation_input = tfile is not None
        tfile_out = "translation.txt"
        if self.has_translation_input:
            self.translation_input = ipyw.Text(description="Original:",
                                               value=tfile)
            self.trans_ui.append(self.translation_input)
            tfile_out = re.sub(".0.txt", "_custom.txt", tfile)
        self.translation_output = ipyw.Text(description="Save as:",
                                            value=tfile_out)
        self.tbutton = ipyw.Button(description="Write translation")
        self.tbutton.on_click(self.write_translation_to_file)
        self.trans_ui.extend([self.translation_output, self.tbutton])

        # Make translation sliders
        self.tsliders = {}
        for ax in _axes:
            n = self.viewer[1].im.n_voxels[ax]
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
        translations = {f"d{ax}": self.tsliders[ax].value for ax in
                        self.tsliders}
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
        self.viewer[1].im.shift = self.current_trans

        # Adjust descriptions
        for ax, slider in self.tsliders.items():
            slider.description = "{} ({:.0f} mm)".format(
                ax, self.viewer[1].im.voxel_sizes[ax] * slider.value)

    def set_slider_widths(self):
        """Adjust widths of slider UI."""

        if self.plots_per_row is not None and self.plots_per_row < self.n:
            return
        for i, slider in enumerate(self.slider_boxes[:-1]):
            width = self.figsize * self.viewer[i].im.get_relative_width(
                self.view, self.colorbar) * mpl.rcParams["figure.dpi"]
            slider.layout = ipyw.Layout(width=f"{width}px")

    def make_fig(self):

        # Get relative width of each subplot
        if self.match_axes is None:
            width_ratios = [v.im.get_relative_width(self.view, self.colorbar)
                            for v in self.viewer]
            width_ratios.extend([c.get_relative_width(self.view) for
                                 c in self.comparison])
            self.xlim = None
            self.ylim = None

        # Set common x and y limits for all subplots if matching axes
        else:

            # Match all axes to one plot
            x, y = _plot_axes[self.view]
            if isinstance(self.match_axes, int):
                if self.match_axes >= self.n or self.match_axes == -1:
                    self.match_axes = self.n - 1
                extent = self.xlim = self.viewer[self.match_axes].im.extent[
                    self.view]
                self.xlim = extent[:2]
                self.ylim = extent[2:]

            # Match axes to either largest or smallest range
            else:

                # Get x and y limits of all plots
                x_lower = [min(v.im.lims[x]) for v in self.viewer]
                x_upper = [max(v.im.lims[x]) for v in self.viewer]
                y_lower = [max(v.im.lims[y]) for v in self.viewer]
                y_upper = [min(v.im.lims[y]) for v in self.viewer]

                # Set x and y limits
                min_func, max_func = min, max
                if self.match_axes == "overlap":
                    min_func, max_func = max_func, min_func
                self.xlim = min_func(x_lower), max_func(x_upper)
                self.ylim = max_func(y_lower), min_func(y_upper)

            # Recalculate width ratios
            if self.match_axes == "x":
                self.ylim = None
                x_range = abs(self.xlim[1] - self.xlim[0])
                width_ratios = [x_range / v.im.lengths[y] for v in self.viewer]
            elif self.match_axes == "y":
                self.xlim = None
                y_range = abs(self.ylim[1] - self.ylim[0])
                width_ratios = [v.im.lengths[x] / y_range for v in self.viewer]
            else:
                ratio = abs(self.xlim[1] - self.xlim[0]) \
                        / abs(self.ylim[1] - self.ylim[0])
                width_ratios = [ratio for i in range(self.n)]

        # Get rows and columns
        n_plots = (not self.comparison_only) * self.n \
            + len(self.comparison)
        if self.comparison_only:
            width_ratios = width_ratios[self.n:]
        if self.plots_per_row is not None:
            n_cols = min([self.plots_per_row, n_plots])
            n_rows = int(np.ceil(n_plots / n_cols))
            width_ratios_padded = width_ratios + \
                [0 for i in range(n_rows * n_cols - n_plots)]
            ratios_per_row = np.array(width_ratios_padded).reshape(n_rows, 
                                                                   n_cols)
            width_ratios = np.amax(ratios_per_row, axis=0)
        else:
            n_cols = n_plots
            n_rows = 1

        # Make figure
        height = self.figsize * n_rows
        width = self.figsize * sum(width_ratios)
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

    def adjust_axes(self, im):
        """Match the axis range of a view to the viewers whose indices are
        stored in self.match_viewers."""

        if self.xlim is not None:
            im.ax.set_xlim(self.xlim)
            im.apply_zoom(self.view, zoom_y=False)
        if self.ylim is not None:
            im.ax.set_ylim(self.ylim)
            im.apply_zoom(self.view, zoom_x=False)

    def plot(self, **kwargs):
        """Plot all images."""

        if self.plotting:
            return
        self.plotting = True

        # Deal with view change
        if self.ui_view.value != self.view:
            self.view = self.ui_view.value
            for v in self.viewer:
                v.view = self.ui_view.value
                v.update_slice_slider()
            self.set_slider_widths()

        # Deal with structure plot type change
        if self.struct_plot_type != self.ui_struct_plot_type.value:
            self.struct_plot_type = self.ui_struct_plot_type.value
            self.viewer[0].update_struct_slider()

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
                self.adjust_axes(v.im)

        # Plot all comparison images
        if len(self.comparison):
            invert = self.ui_invert.value
            if self.has_chequerboard:
                self.chequerboard.plot(invert=invert,
                                       n_splits=self.ui_cb.value,
                                       mpl_kwargs=self.viewer[0].v_min_max)
            if self.has_overlay:
                self.overlay.plot(invert=invert,
                                  opacity=self.ui_overlay.value,
                                  mpl_kwargs=self.viewer[0].v_min_max)
            if self.has_diff:
                self.diff.plot(invert=invert,
                               mpl_kwargs=self.viewer[0].v_min_max)
            for c in self.comparison:
                self.adjust_axes(c)

        if self.suptitle is not None:
            self.fig.suptitle(self.suptitle)

        plt.tight_layout()
        self.plotting = False


class ImageViewer():
    """Class for displaying a MultiImage with interactive elements."""

    def __init__(
        self,
        nii,
        figsize=_default_figsize,
        colorbar=False,
        dose_opacity=0.5,
        dose_cmap="jet",
        invert_mask=False,
        mask_colour="black",
        jacobian_opacity=0.5,
        jacobian_cmap="seismic",
        df_plot_type="grid",
        df_spacing=30,
        df_kwargs=None,
        struct_plot_type="contour",
        struct_opacity=1,
        struct_linewidth=2,
        struct_info=False,
        length_units=None,
        vol_units=None,
        struct_legend=True,
        legend_loc='lower left',
        standalone=True,
        init_view="x-y",
        init_idx=None,
        init_pos=None,
        v=(-300, 200),
        continuous_update=False,
        save_as=None,
        **kwargs
    ):

        self.im = self.make_image(nii, **kwargs)
        self.interactive = False  # Flag for creation of interactive elements
        self.gs = None  # Gridspec in which to place plot axes

        # Set initial view
        view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
        if init_view in view_map:
            self.view = view_map[init_view]
        else:
            self.view = init_view

        # Set initial slice numbers
        self.slice = {view: int(self.im.n_voxels[z] / 2) for view, z
                      in _slider_axes.items()}
        if init_pos is not None and self.im.scale_in_mm:
            self.set_slice_from_pos(init_view, init_pos)
        else:
            self.set_slice(init_view, init_idx)

        # Assign plot settings
        # General settings
        self.v = v
        self.figsize = figsize
        self.continuous_update = continuous_update
        self.colorbar = colorbar
        self.save_as = save_as
        self.plotting = False

        # Mask settings
        self.invert_mask = invert_mask
        self.mask_colour = mask_colour

        # Dose settings
        self.init_dose_opacity = dose_opacity
        self.dose_cmap = dose_cmap

        # Jacobian/deformation field settings
        self.init_jac_opacity = jacobian_opacity
        self.jacobian_cmap = jacobian_cmap
        self.df_plot_type = df_plot_type
        self.df_spacing = df_spacing
        self.df_kwargs = df_kwargs

        # Structure settings
        self.struct_plot_type = struct_plot_type
        self.struct_opacity = struct_opacity
        self.struct_linewidth = struct_linewidth
        self.struct_legend = struct_legend
        self.legend_loc = legend_loc
        self.struct_info = struct_info
        self.vol_units = vol_units
        self.length_units = length_units
        if self.vol_units is None:
            self.vol_units = "mm" if self.im.scale_in_mm else "voxels"
        if self.length_units is None:
            self.length_units = "mm" if self.im.scale_in_mm else "voxels"

        # Display plot
        self.standalone = standalone
        if standalone:
            self.show()

    def make_image(self, *args, **kwargs):
        """Set up image object."""
        return MultiImage(*args, **kwargs)

    def set_slice(self, view, sl):
        """Set the current slice number in a specific view."""

        if sl is None:
            return
        max_slice = self.im.n_voxels[_slider_axes[view]] - 1
        min_slice = 0
        if self.slice[view] < min_slice:
            self.slice[view] = min_slice
        elif self.slice[view] > max_slice:
            self.slice[view] = max_slice
        else:
            self.slice[view] = sl

    def set_slice_from_pos(self, view, pos):
        """Set the current slice number from a position in mm."""

        ax = _slider_axes[view]
        idx = self.im.pos_to_idx(pos, ax)
        self.set_slice(view, idx)

    def slider_to_idx(self, val, ax=None):
        """Convert a slider value to a slice index."""

        if ax is None:
            ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            return self.im.pos_to_idx(val, ax)
        else:
            if ax == "z":
                return int(self.im.n_voxels[ax] - val)
            else:
                return int(val - 1)

    def idx_to_slider(self, idx, ax=None):
        """Convert a slice index to a slider value."""

        if ax is None:
            ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            return self.im.idx_to_pos(idx, ax)
        else:
            if ax == "z":
                return self.im.n_voxels[ax] - idx
            else:
                return idx + 1

    def set_interactive(self):
        """Create interactive elements."""

        if in_notebook():
            self.make_ui()
        else:
            self.set_callbacks()

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
            self.main_ui.append(self.ui_view)
        else:
            self.ui_view = vimage.ui_view
            self.view = self.ui_view.value

        # Structure jumping menu
        self.structs_for_jump = {"": None, **{s.name_nice: s for s in
                                              self.im.structs}}
        self.ui_struct_jump = ipyw.Dropdown(
            options=self.structs_for_jump.keys(),
            value="",
            description="Jump to",
            style=_style,
        )
        if self.im.has_structs:
            self.main_ui.append(self.ui_struct_jump)

        # HU and slice sliders
        if not share_slider or not shared_ui:

            # Make HU slider
            self.ui_hu = ipyw.IntRangeSlider(
                min=-2000, max=2000,
                value=self.v,
                description="HU range",
                continuous_update=False,
                style=_style
            )
            self.main_ui.append(self.ui_hu)

            # Make slice slider
            self.ui_slice = ipyw.FloatSlider(
                continuous_update=self.continuous_update,
                style=_style,
                readout_format=".1f"
            )
            self.own_ui_slice = True
            self.update_slice_slider()
            self.main_ui.append(self.ui_slice)

        else:
            self.ui_hu = vimage.ui_hu
            self.ui_slice = vimage.ui_slice
            self.slice[self.view] = self.ui_slice.value
            self.own_ui_slice = False

        # Extra sliders
        self.extra_ui = []
        if not shared_ui:

            # Mask checkbox
            self.ui_mask = ipyw.Checkbox(value=self.im.has_mask,
                                         description="Apply mask",
                                         width=200)
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

            # Opacity/linewidth slider
            self.ui_struct_slider = ipyw.FloatSlider(continuous_update=False,
                                                     style=_style)
            self.update_struct_slider()

            # Add all structure UIs
            if self.im.has_structs:
                self.extra_ui.extend([
                    self.ui_struct_plot_type,
                    self.ui_struct_slider
                ])

        else:
            to_share = ["ui_mask", "ui_dose", "ui_jac_opacity", "ui_jac_range",
                        "ui_df", "ui_struct_plot_type", "ui_struct_slider"]
            for ts in to_share:
                setattr(self, ts, getattr(vimage, ts))

        # Make lower
        self.make_lower_ui()

        # Combine UI elements
        self.upper_ui = [ipyw.VBox(self.main_ui), ipyw.VBox(self.extra_ui)]
        self.upper_ui_box = ipyw.HBox(self.upper_ui)
        self.lower_ui_box = ipyw.VBox(self.lower_ui)
        self.all_ui = self.main_ui + self.extra_ui + self.ui_struct_checkboxes
        self.interactive = True

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

        # Columns for structure info
        self.ui_struct_checkboxes = []
        self.ui_struct_vol = []
        self.ui_struct_x = []
        self.ui_struct_y = []
        if self.struct_info:
            self.ui_struct_checkboxes.append(ipyw.HTML(value=""))
            vol_units = self.vol_units if self.vol_units != "mm" \
                else "mm<sup>3</sup>"
            self.ui_struct_vol.append(ipyw.HTML(
                value=f"<b>Volume ({vol_units})</b>"))
            self.ui_struct_x.append(ipyw.HTML())
            self.ui_struct_y.append(ipyw.HTML())

        # Make checkbox for each structure
        for s in self.im.structs:
            s.checkbox = ipyw.Checkbox(value=True, description=s.name_nice)
            self.ui_struct_checkboxes.append(s.checkbox)
            if not self.struct_info:
                self.lower_ui.append(s.checkbox)
            else:
                fmt = "{:.0f}" if self.vol_units == "ml" else "{:.0f}"
                self.ui_struct_vol.append(ipyw.Label(
                    value=fmt.format(s.volume[self.vol_units])))
                s.ui_x = ipyw.Label()
                s.ui_y = ipyw.Label()
                self.ui_struct_x.append(s.ui_x)
                self.ui_struct_y.append(s.ui_y)

        if self.struct_info:
            self.update_struct_info()
            layout = ipyw.Layout(align_items="center", 
                                 padding="0px 0px 0px 50px")
            self.lower_ui.append(ipyw.HBox([
                ipyw.VBox(self.ui_struct_checkboxes),
                ipyw.VBox(self.ui_struct_vol, 
                          layout=ipyw.Layout(align_items="center")),
                ipyw.VBox(self.ui_struct_x, layout=layout),
                ipyw.VBox(self.ui_struct_y, layout=layout)
            ]))

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
        for s in self.im.structs:
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

    def update_struct_slider(self):
        """Update range and description of structure slider."""

        self.struct_plot_type = self.ui_struct_plot_type.value
        if self.struct_plot_type == "mask":
            self.ui_struct_slider.disabled = False
            self.ui_struct_slider.min = 0
            self.ui_struct_slider.max = 1
            self.ui_struct_slider.step = 0.1
            self.ui_struct_slider.value = self.struct_opacity
            self.ui_struct_slider.description = 'Structure opacity'
            self.ui_struct_slider.readout_format = '.1f'
        elif self.struct_plot_type == "contour":
            self.ui_struct_slider.disabled = False
            self.ui_struct_slider.min = 1
            self.ui_struct_slider.max = 8
            self.ui_struct_slider.value = self.struct_linewidth
            self.ui_struct_slider.step = 1
            self.ui_struct_slider.description = 'Structure linewidth'
            self.ui_struct_slider.readout_format = '.0f'
        else:
            self.ui_struct_slider.disabled = True

    def set_callbacks(self):
        """Set up matplotlib callback functions for interactive plotting."""

        self.interactive = True

    def update_slice_slider(self):
        """Update the slice slider to show the axis corresponding to the
        current view, with value set to the last value used on that axis."""

        if not self.own_ui_slice:
            return

        # Get new min and max
        ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            new_min = min(self.im.lims[ax])
            new_max = max(self.im.lims[ax]) - self.im.voxel_sizes[ax]
        else:
            new_min = 1
            new_max = self.im.n_voxels[ax]

        # Set to widest range of new and old values
        if new_min < self.ui_slice.min:
            self.ui_slice.min = new_min
        if new_max > self.ui_slice.max:
            self.ui_slice.max = new_max

        # Set new value
        val = self.idx_to_slider(self.slice[self.view])
        self.ui_slice.value = val

        # Set to final axis limits
        if self.ui_slice.min != new_min:
            self.ui_slice.min = new_min
        if self.ui_slice.max != new_max:
            self.ui_slice.max = new_max

        # Set step and description
        self.ui_slice.step = abs(self.im.voxel_sizes[ax]) if \
            self.im.scale_in_mm else 1
        self.update_slice_slider_desc()

    def update_slice_slider_desc(self):
        """Update slice slider description to reflect current axis and
        position."""

        if not self.own_ui_slice:
            return
        ax = _slider_axes[self.view]
        if self.im.scale_in_mm:
            self.ui_slice.description = f"{ax} (mm)"
        else:
            pos = self.idx_to_pos(
                self.slider_to_idx(self.ui_slice.value), ax)
            self.ui_slice.description = f"{ax} ({pos:.1f} mm)"

    def jump_to_struct(self):
        """Jump to the mid slice of a structure."""

        if self.ui_struct_jump.value == "":
            return

        struct = self.structs_for_jump[self.ui_struct_jump.value]
        mid_slice = int(np.mean(list(struct.contours[self.view].keys())))
        self.ui_slice.value = self.idx_to_slider(
            mid_slice, _slider_axes[self.view])
        self.ui_struct_jump.value = ""

    def show(self):
        """Display plot and UI."""

        # Ensure interactive elements are created
        if not self.interactive:
            self.set_interactive()

        # Make output
        if in_notebook():
            self.show_in_notebook()
        else:
            self.plot()
            plt.show()

    def show_in_notebook(self):
        """Display interactive output in a jupyter notebook."""

        from IPython.display import display
        ui_kw = {str(np.random.rand()): ui for ui in self.all_ui if
                 hasattr(ui, "value")}
        self.out = ipyw.interactive_output(self.plot, ui_kw)
        to_display = [self.upper_ui_box, self.out]
        if len(self.lower_ui):
            to_display.append(self.lower_ui_box)
        display(*to_display)

    def set_slice_and_view(self):
        """Get the current slice and view to plot from the UI."""

        # Get view
        view = self.ui_view.value
        if self.view != view:
            self.view = view
            self.update_slice_slider()

        # Get slice
        self.jump_to_struct()
        self.slice[self.view] = self.slider_to_idx(self.ui_slice.value)
        if not self.im.scale_in_mm:
            self.update_slice_slider_desc()

        # Get HU range
        self.v_min_max = {"vmin": self.ui_hu.value[0],
                          "vmax": self.ui_hu.value[1]}

    def plot(self, **kwargs):
        """Plot a slice with current settings."""

        if self.plotting:
            return
        self.plotting = True
        self.set_slice_and_view()

        # Get dose settings
        dose_kwargs = {}
        if self.im.has_dose:
            dose_kwargs = {"alpha": self.ui_dose.value,
                           "cmap": self.dose_cmap}

        # Get jacobian settings
        jacobian_kwargs = {}
        if self.im.has_jacobian:
            jacobian_kwargs = {"alpha": self.ui_jac_opacity.value,
                               "cmap": self.jacobian_cmap,
                               "vmin": self.ui_jac_range.value[0],
                               "vmax": self.ui_jac_range.value[1]}

        # Get structure settings
        self.update_struct_info()
        if self.ui_struct_plot_type.value != self.struct_plot_type:
            self.update_struct_slider()
        if self.struct_plot_type == "contour":
            self.struct_linewidth = self.ui_struct_slider.value
            struct_kwargs = {"linewidth": self.struct_linewidth}
        elif self.struct_plot_type == "mask":
            self.struct_opacity = self.ui_struct_slider.value
            struct_kwargs = {"alpha": self.struct_opacity}
        else:
            struct_kwargs = {}
        for s in self.im.structs:
            s.visible = s.checkbox.value

        # Make plot
        self.im.plot(self.view,
                     self.slice[self.view],
                     gs=self.gs,
                     mpl_kwargs=self.v_min_max,
                     figsize=self.figsize,
                     colorbar=self.colorbar,
                     masked=self.ui_mask.value,
                     invert_mask=self.invert_mask,
                     mask_colour=self.mask_colour,
                     dose_kwargs=dose_kwargs,
                     jacobian_kwargs=jacobian_kwargs,
                     df_plot_type=self.ui_df.value,
                     df_spacing=self.df_spacing,
                     df_kwargs=self.df_kwargs,
                     struct_plot_type=self.struct_plot_type,
                     struct_kwargs=struct_kwargs,
                     struct_legend=self.struct_legend,
                     legend_loc=self.legend_loc,
                     show=False)
        self.plotting = False

    def save_fig(self, _):
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
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except NameError:
        return False
    return True


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
