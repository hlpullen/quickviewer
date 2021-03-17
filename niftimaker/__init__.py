"""Program for interactively creating synthetic NIfTI files containing simple
geometric shapes."""

import ipywidgets as ipyw
import numpy as np

from quickviewer import OrthogViewer, _style


class NiftiMaker(OrthogViewer):

    def __init__(self, shape, bg_hu=-1000, **kwargs):

        # Make empty numpy array
        self.background = np.ones(shape) * bg_hu
        OrthogViewer.__init__(self, self.background, show=False,
                             **kwargs)

        self.make_shape_ui()

        self.show()

    def make_shape_ui(self):
        """Make UI for adding and editing shapes."""

        # Adding shapes
        # To do: calculate some default length and centre from image properties
        self.ui_new_shape_title = ipyw.HTML(value="<b>Add shape:</b>")
        self.ui_new_shape_type = ipyw.Dropdown(
            options=["", "Sphere", "Cube"],
            value="", description="Shape", style=_style)
        self.ui_new_shape_name = ipyw.Text(description="Name")
        self.ui_new_shape_hu = ipyw.FloatText(description="HU", value=0)
        self.ui_new_shape_length = ipyw.FloatText(description="Length", value=1)
        self.ui_new_shape_centre_title = ipyw.Label(value="Centre")
        self.ui_new_shape_centre = [
            ipyw.FloatText(layout=ipyw.Layout(width="40px", value=0)) 
            for i in range(3)]
        self.ui_new_shape_centre_box = ipyw.HBox(
            [self.ui_new_shape_centre_title] + self.ui_new_shape_centre)
        self.ui_new_shape_button = ipyw.Button(description="Add")

        # Combine shape-adding UI
        layout = ipyw.Layout(align_items="stretch", padding="0px 0px 0px 50px")
        new_shape_box = ipyw.VBox([
            self.ui_new_shape_title,
            self.ui_new_shape_type,
            self.ui_new_shape_name,
            self.ui_new_shape_hu,
            self.ui_new_shape_length,
            self.ui_new_shape_centre_box,
            self.ui_new_shape_button
        ], layout=layout)

        # Editing existing shapes
        # To do: infer maximum shape from image properties
        self.ui_edit_shape_title = ipyw.HTML(value="<b>Edit shape:</b>")
        self.ui_shape_to_edit = ipyw.Dropdown(
            options=[""], value="", description="Shape", style=_style)
        self.ui_shape_hu = ipyw.FloatSlider(
            value=0, min=self.ui_hu.min, max=self.ui_hu.max, description="HU")
        self.ui_shape_length = ipyw.FloatSlider(
            value=1, min=0, max=10, description="Length")
        self.ui_shape_x = ipyw.FloatSlider(
            value=0, min=-10, max=10, description="x centre")
        self.ui_shape_y = ipyw.FloatSlider(
            value=0, min=-10, max=10, description="y centre")
        self.ui_delete_shape = ipyw.Button(description="Delete")

        # Combine shape-edited UI
        edit_shape_box = ipyw.VBox([
            self.ui_edit_shape_title,
            self.ui_shape_to_edit,
            self.ui_shape_hu,
            self.ui_shape_length,
            self.ui_shape_x,
            self.ui_shape_y,
            self.ui_delete_shape
        ])

        # Combine with existing UI
        self.upper_ui_orig = self.upper_ui_box
        view_box = ipyw.VBox([
            ipyw.HTML(value="<b>View:</b>"), self.upper_ui_orig])
        self.upper_ui_box = ipyw.HBox([
            view_box, 
            new_shape_box, 
            #  edit_shape_box
        ])

    def update_new_shape_ui(self):
        """Update new shape UI when a new shape option is selected."""

        pass

    def update_edit_shape_ui(self):
        """Update edit shape UI when a new shape option is selected."""

        pass

        # Note: also need to rerun plotting
        # Could save the image arrays (only remake them when necessary i.e. 
        # when a shape is edited or view/slice is changed

        #  from IPython.display import display
        #  display(self.ui)


        # View and slice slider


    #  def add_shape(self):

        #  # Occurs when "Add" button gets pressed

        #  self.shape_code.append()
        #  self.shapes.append()

    #  def plot(self):

        #  # Get slice and view

        #  # If add_structure is selected in dropdown, draw dotted outline of
        #  # the new structure

        #  # If existing_structure is selected, draw solid outline

        #  pass


#  class NiftiMakerShape:

    #  def __init__(self, kind, **kwargs):

        #  pass

    #  def get_slice(self, view, sl):

        #  pass
