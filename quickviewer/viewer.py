"""Classes for creating a UI and displaying interactive plots."""

import itertools
import ipywidgets as ipyw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from quickviewer.image import MultiImage, OrthogonalImage
from quickviewer.image import _slider_axes, _df_plot_types, \
        _struct_plot_types, _orthog, _default_figsize


_style = {"description_width": "initial",
          "value_width": "initial"}
_view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}


class ImageViewer():
    """Class for displaying a MultiImage with interactive elements."""

    def __init__(
        self,
        nii,
        title=None,
        scale_in_mm=True,
        figsize=_default_figsize,
        downsample=None,
        colorbar=False,
        dose=None,
        dose_opacity=0.5,
        dose_cmap="jet",
        mask=None,
        invert_mask=False,
        mask_colour="black",
        jacobian=None,
        jacobian_opacity=0.5,
        jacobian_cmap="seismic",
        df=None,
        df_plot_type="grid",
        df_spacing=30,
        df_kwargs=None,
        structs=None,
        struct_colours=None,
        struct_plot_type="contour",
        struct_opacity=1,
        struct_linewidth=2,
        struct_legend=True,
        legend_loc='lower left',
        structs_as_mask=False,
        standalone=True,
        init_view="x-y",
        init_idx=None,
        v=(-300, 200),
        continuous_update=False,
    ):

        self.im = self.make_image(
            nii, title, scale_in_mm, downsample, dose, mask, jacobian, df,
            structs, struct_colours, structs_as_mask)
        self.interactive = False  # Flag for creation of interactive elements
        self.gs = None  # Gridspec in which to place plot axes

        # Set initial view and slice numbers
        if init_view in _view_map:
            self.view = _view_map[init_view]
        else:
            self.view = init_view
        self.slice = {view: int(self.im.n_voxels[z] / 2) for view, z
                      in _slider_axes.items()}
        self.set_slice(init_view, init_idx)

        # Assign plot settings
        # General settings
        self.v = v
        self.figsize = figsize
        self.continuous_update = continuous_update
        self.colorbar = colorbar
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

        # Display plot
        if standalone:
            self.show()

    def make_image(self, *args):
        """Set up image object."""
        return MultiImage(*args)

    def set_slice(self, view, sl):
        """Set the current slice number in a specific view."""

        if sl is None:
            return
        max_slice = self.im.n_voxels[_slider_axes[view]] - 1
        min_slice = 0
        if self.slice < min_slice:
            self.slice[view] = min_slice
        elif self.slice > max_slice:
            self.slice[view] = max_slice
        else:
            self.slice[view] = sl

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

    def make_ui(self, vimage=None, share_sliders=True):
        """Make Jupyter notebook UI. If qv_image contains another ImageViewer
        instance, the UI will be taken from that image. If share_sliders is
        False, independent HU and slice sliders will be created."""

        shared_ui = isinstance(vimage, ImageViewer)
        self.main_ui = []
        self.main_sliders = []

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

        # HU and slice sliders
        if not share_sliders or not shared_ui:

            # HU slider
            self.ui_hu = ipyw.IntRangeSlider(
                min=-2000, max=2000,
                value=self.v,
                description="HU range",
                continuous_update=False,
                style=_style
            )
            self.main_ui.append(self.ui_hu)
            self.main_sliders.append(self.ui_hu)

            # Slice slider
            self.ui_slice = ipyw.FloatSlider(
                continuous_update=self.continuous_update,
                style=_style,
                readout_format=".1f"
            )
            self.own_ui_slice = True
            self.update_slice_slider()
            self.main_ui.append(self.ui_slice)
            self.main_sliders.append(self.ui_slice)

        else:
            self.ui_hu = vimage.ui_hu
            self.ui_slice = vimage.ui_slice
            self.slice[self.view] = self.ui_slice.value
            self.own_ui_slice = False

        # Extra sliders
        self.extra_ui = []
        self.lower_ui = []
        if not shared_ui:

            # Mask checkbox
            self.ui_mask = ipyw.Checkbox(value=self.im.has_mask,
                                         description="Apply mask",
                                         width=200)
            if self.im.has_mask:
                self.extra_ui.append(self.ui_mask)

            # Dose opacity
            self.ui_dose = ipyw.FloatSlider(
                value=self.init_dose_opacity, min=0, max=1, step=0.1,
                description="Dose opacity",
                continuous_update=self.continuous_update,
                readout_format=".1f", style=_style,
            )
            if self.im.has_dose:
                self.extra_ui.append(self.ui_dose)

            # Jacobian opacity and range
            self.ui_jac_opacity = ipyw.FloatSlider(
                value=self.init_jac_opacity, min=0, max=1, step=0.1,
                description="Jacobian opacity",
                continuous_update=self.continuous_update,
                readout_format=".1f", style=_style,
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

            # Structure jumping UI
            self.structs_for_jump = {"": None, **{s.name_nice: s for s in
                                                  self.im.structs}}
            self.ui_struct_jump = ipyw.Dropdown(
                options=self.structs_for_jump.keys(),
                value="",
                description="Jump to",
                style=_style,
            )

            # Add all structure UIs
            if self.im.has_structs:
                self.extra_ui.extend([
                    self.ui_struct_plot_type,
                    self.ui_struct_slider,
                    self.ui_struct_jump
                ])

        else:
            to_share = ["ui_mask", "ui_dose", "ui_jac_opacity", "ui_jac_range",
                        "ui_df", "ui_struct_plot_type", "ui_struct_slider", 
                        "ui_struct_jump"]
            for ts in to_share:
                setattr(self, ts, getattr(vimage, ts))

        # Make structure checkboxes
        for s in self.im.structs:
            s.checkbox = ipyw.Checkbox(value=True, description=s.name_nice)
            self.lower_ui.append(s.checkbox)

        # Combine UI elements
        self.main_ui_box = ipyw.VBox(self.main_ui)
        self.extra_ui_box = ipyw.VBox(self.extra_ui)
        self.ui = self.main_ui + self.extra_ui + self.lower_ui
        self.ui_box = ipyw.HBox([self.main_ui_box, self.extra_ui_box])
        self.ui_box_lower = ipyw.VBox(self.lower_ui)
        self.interactive = True

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
            ui_kw = {str(np.random.rand()): ui for ui in self.ui}
            self.out = ipyw.interactive_output(self.plot, ui_kw)
            from IPython.display import display
            to_display = [self.ui_box, self.out]
            if self.im.has_structs:
                to_display.append(self.ui_box_lower)
            display(*to_display)
        else:
            self.plot()
            plt.show()

    def plot(self, **kwargs):
        """Plot a slice with current settings."""

        if self.plotting:
            return
        self.plotting = True

        # Get view
        view = self.ui_view.value
        if self.view != view:
            self.view = view
            self.update_slice_slider()

        # Get HU range
        mpl_kwargs = {"vmin": self.ui_hu.value[0],
                      "vmax": self.ui_hu.value[1]}

        # Get slice
        self.jump_to_struct()
        self.slice[self.view] = self.slider_to_idx(self.ui_slice.value)
        if not self.im.scale_in_mm:
            self.update_slice_slider_desc()

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
                     mpl_kwargs=mpl_kwargs,
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


class OrthogViewer(ImageViewer):
    """ImageViewer with an orthgonal view displayed."""

    def make_image(self, *args):
        """Set up image object."""
        return OrthogonalImage(*args)

    def jump_to_struct(self):
        """Jump to mid slice of a structure."""

        if self.ui_struct_jump.value == "":
            return

        struct = self.structs_for_jump[self.ui_struct_jump.value]
        ImageViewer.jump_to_struct(self)

        orthog_view = _orthog[self.view]
        mid_slice = int(np.mean(list(struct.contours[orthog_view].keys())))
        self.im.orthog_slices[_slider_axes[orthog_view]] = mid_slice


class QuickViewer:
    """Display multiple images with UI."""

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
        **kwargs
    ):

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
        self.viewers = []
        viewer_type = ImageViewer if not orthog_view else OrthogViewer
        for i in range(self.n):
            viewer = viewer_type(
                self.nii[i], title=self.title[i], dose=self.dose[i],
                mask=self.mask[i], structs=self.structs[i], 
                jacobian=self.jacobian[i], df=self.df[i], standalone=False, 
                **kwargs)
            if viewer.im.valid:
                self.viewers.append(viewer)

        # Settings needed for plotting
        self.figsize = kwargs.get("figsize", _default_figsize)
        self.colorbar = kwargs.get("colorbar", False)
        self.plotting = False

        # Make UI
        self.make_ui(share_slider)

        # Display
        if True:
            self.out = ipyw.interactive_output(self.plot, self.ui_kw)
            from IPython.display import display
            to_display = [self.ui_box, self.out]
            if len(self.lower_ui):
                to_display.append(self.ui_box_lower)
            display(*to_display)

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

    def make_ui(self, share_slider):

        # Only share slider if images have same shape
        if share_slider:
            same_shape = [v.im.shape == self.viewers[0].im.shape
                          for v in self.viewers]
            share_slider *= all(same_shape)

        self.main_sliders = []
        self.lower_ui = []

        # Make UI for first image
        self.viewers[0].make_ui()
        self.main_sliders.append(self.viewers[0].main_sliders)
        self.extra_ui = self.viewers[0].extra_ui
        self.lower_ui.extend(self.viewers[0].lower_ui)
        many_with_structs = sum([v.im.has_structs for v in self.viewers]) > 1
        if many_with_structs:
            self.lower_ui.insert(
                0, ipyw.HTML(value=f"<b>{v.im.title + ':'}</b>"))

        # Store orientation UI
        self.ui_view = self.viewers[0].ui_view
        self.view = self.ui_view.value

        # Make UI for subsequent images
        for v in self.viewers[1:]:
            v.make_ui(vimage=self.viewers[0], share_sliders=share_slider)
            if not share_slider:
                self.main_sliders.append(v.main_sliders)
            if v.im.has_structs:
                if many_with_structs:
                    self.lower_ui.append(
                        ipyw.HTML(value=f"<b>{v.im.title + ':'}</b>"))
                self.lower_ui.extend(v.lower_ui)

        # Set UI as plotting kwargs
        ui_all = [self.ui_view] \
                + list(itertools.chain.from_iterable(self.main_sliders)) \
                + self.extra_ui \
                + list(itertools.chain.from_iterable(self.lower_ui))
        self.ui_kw = {str(np.random.rand()): ui for ui in ui_all}

        # Assemble UI boxes
        self.main_slider_boxes = [ipyw.VBox(ui) for ui in self.main_sliders]
        self.set_slider_widths()
        self.main_ui_box = ipyw.VBox([self.ui_view, 
                                      ipyw.HBox(self.main_slider_boxes)])
        self.extra_ui_box = ipyw.VBox(self.extra_ui)
        self.ui_box = ipyw.HBox([self.main_ui_box, self.extra_ui_box])
        self.ui_box_lower = ipyw.VBox(self.lower_ui)

    def set_slider_widths(self):
        """Adjust widths of slider UI."""

        for i, slider in enumerate(self.main_slider_boxes[:-1]):
            width = self.figsize * self.viewers[i].im.get_relative_width(
                self.view, self.colorbar) * mpl.rcParams["figure.dpi"]
            slider.layout = ipyw.Layout(width=f"{width}px", 
                                       justify_content="center")

    def make_fig(self):

        # Get width of each figure 
        width_ratios = [v.im.get_relative_width(self.view, self.colorbar) 
                        for v in self.viewers]
        height = self.figsize
        width = self.figsize * sum(width_ratios)

        # Make figure
        self.fig = plt.figure(figsize=(width, height))

        # Make gridspec
        gs = self.fig.add_gridspec(1, self.n, width_ratios=width_ratios)
        for i, v in enumerate(self.viewers):
            v.gs = gs[i]

    def plot(self, **kwargs):
        """Plot all images."""

        if self.plotting:
            return
        self.plotting = True

        # Deal with view change
        if self.ui_view.value != self.view:
            self.view = self.ui_view.value
            for v in self.viewers:
                v.view = self.ui_view.value
                v.update_slice_slider()
            self.set_slider_widths()

        # Reset figure
        self.make_fig()

        # Plot all images
        for v in self.viewers:
            v.plot()
        plt.tight_layout()
        self.plotting = False


def in_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except NameError:
        return False
    return True
