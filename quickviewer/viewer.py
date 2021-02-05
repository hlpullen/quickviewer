"""Classes for creating a UI and displaying interactive plots."""

import ipywidgets as ipyw
import numpy as np

from quickviewer.image import MultiImage
from quickviewer.image import _axes, _plot_axes, _slider_axes

_style = {"description_width": "initial"}
_view_map = {"y-x": "x-y", "z-x": "x-z", "z-y": "y-z"}
_orthog = {'x-y': 'y-z', 'y-z': 'x-z', 'x-z': 'y-z'}


class ViewerImage(MultiImage):
    """Class for displaying a MultiImage with interactive elements."""

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
        standalone=True,
        init_view="x-y",
        init_idx=None,
        v=(-300, 200),
        continuous_update=False
    ):

        MultiImage.__init__(self, nii, title, scale_in_mm, downsample, dose,
                            mask, jacobian, df, structs, struct_colours)
        self.interactive = False  # Flag for creation of interactive elements

        # Set initial view and slice numbers
        if init_view in _view_map:
            self.view = _view_map[init_view]
        else:
            self.view = init_view
        self.slice = {view: int(self.n_voxels[z] / 2) for view, z 
                      in _slider_axes.items()}
        self.set_slice(init_view, init_idx)

        # Assign plot settings
        self.v = v
        self.standalone = standalone
        self.continuous_update = continuous_update
        self.plotting = False

        # Display plot
        self.show()

    def set_slice(self, view, sl):
        """Set the current slice number in a specific view."""

        if sl is None:
            return
        max_slice = self.n_voxels[_slider_axes[init_view]] - 1
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
        if self.scale_in_mm:
            return self.pos_to_idx(val, ax)
        else:
            if ax == "z":
                return int(self.n_voxels[ax] - val)
            else:
                return int(val - 1)

    def idx_to_slider(self, idx, ax=None):
        """Convert a slice index to a slider value."""

        if ax is None:
            ax = _slider_axes[self.view]
        if self.scale_in_mm:
            return self.idx_to_pos(idx, ax)
        else:
            if ax == "z":
                return self.n_voxels[ax] - idx
            else:
                return idx + 1

    def set_interactive(self, ui_view=None, ui_hu=None, ui_slice=None):
        """Create interactive elements."""
        
        if in_notebook():
            self.make_ui(ui_view, ui_hu, ui_slice)
        else:
            self.set_callbacks()
        self.interactive = True

    def make_ui(self, ui_view=None, ui_hu=None, ui_slice=None):
        """Make Jupyter notebook UI."""

        # View radio buttons
        if ui_view is None:
            self.ui_view = ipyw.RadioButtons(
                options=["x-y", "y-z", "x-z"],
                value=self.view,
                description="Slice plane selection:",
                disabled=False,
                style=_style,
            )
        else:
            self.ui_view = ui_view
            self.view = self.ui_view.value

        # HU slider
        if ui_hu is None:
            self.ui_hu = ipyw.IntRangeSlider(
                min=self.data.min(),
                max=self.data.max(),
                value=self.v,
                description="HU range",
                continuous_update=False,
                style=_style
            )
        else:
            self.ui_hu = ui_hu

        # Slice slider
        if ui_slice is None:
            self.ui_slice = ipyw.FloatSlider(
                continuous_update=self.continuous_update,
                style=_style,
                readout_format=".1f"
            )
            self.own_ui_slice = True
            self.update_slice_slider()
        else:
            self.ui_slice = ui_slice
            self.slice[self.view] = self.ui_slice.value
            self.own_ui_slice = False

        # Combine UI elements
        self.ui = [self.ui_view, self.ui_hu, self.ui_slice]
        self.ui_box = ipyw.VBox(self.ui)

    def update_slice_slider(self):
        """Update the slice slider to show the axis corresponding to the 
        current view, with value set to the last value used on that axis."""

        if not self.own_ui_slice:
            return

        # Get new min and max
        ax = _slider_axes[self.view]
        if self.scale_in_mm:
            new_min = min(self.lims[ax])
            new_max = max(self.lims[ax]) - self.voxel_sizes[ax]
        else:
            new_min = 1
            new_max = self.n_voxels[ax]

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
        self.ui_slice.step = abs(self.voxel_sizes[ax]) if self.scale_in_mm \
                else 1
        self.update_slice_slider_desc()
    
    def update_slice_slider_desc(self):
        """Update slice slider description to reflect current axis and 
        position."""

        if not self.own_ui_slice:
            return
        ax = _slider_axes[self.view]
        if self.scale_in_mm:
            self.ui_slice.description = f"{ax} (mm)"
        else:
            pos = self.idx_to_pos(
                self.slider_to_idx(self.ui_slice.value), ax)
            self.ui_slice.description = f"{ax} ({pos:.1f} mm)"

    def set_callback(self):
        """Set up matplotlib callbacks for interactivity."""
        pass

    def show(self):
        """Display plot and UI."""

        # Ensure interactive elements are created
        if not self.interactive:
            self.set_interactive()

        # Make output
        if self.standalone:
            if in_notebook():
                ui_kw = {str(np.random.rand()): ui for ui in self.ui}
                self.out = ipyw.interactive_output(self.plot, ui_kw)
                from IPython.display import display
                display(self.ui_box, self.out)
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
        view_changed = self.view != view
        self.view = view

        # Get HU range
        mpl_kwargs = {"vmin": self.ui_hu.value[0],
                      "vmax": self.ui_hu.value[1]}

        # Get slice
        if view_changed:
            self.update_slice_slider()
        elif not self.scale_in_mm:
            self.update_slice_slider_desc()
        self.slice[self.view] = self.slider_to_idx(self.ui_slice.value)

        # Make plot
        MultiImage.plot(self, 
                        self.view, 
                        self.slice[self.view], 
                        mpl_kwargs=mpl_kwargs, 
                        show=False)
        self.plotting = False


def in_notebook():
    try:
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except NameError:
        return False
    return True

