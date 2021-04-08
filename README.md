# QuickViewer package

A package for interactively viewing medical image data.

## Installation

### Installing QuickViewer via pip

The easiest way to install QuickViewer is via [pip](https://pypi.org/project/pip/):
```
pip install git+https://github.com/hpullen/quickviewer.git
```
QuickViewer is continually being developed, so make sure to check for updates often! QuickViewer can be updated with the command:
```
pip install --upgrade git+https://github.com/hpullen/quickviewer.git
```
If you wish to uninstall quickviewer, simply run:
```
pip uninstall quickviewer
```
### Setting up Jupyter notebooks

QuickViewer works best inside a Jupyter notebook; see [here](https://jupyter.org/install.html) for installation instructions. Once you have Jupyter notebooks installed, you need to run the following command once to ensure that QuickViewer's widgets will work:
```
jupyter nbextension enable --py widgetsnbextension
```
Now you can launch a Jupyter server using
```
jupyter notebook
```
in which you can enjoy QuickViewer and its widgets!

## How to use QuickViewer

### Basic usage

QuickViewer can be used to view medical images in [NIfTI](https://nifti.nimh.nih.gov/) format or [NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html). DICOM files are currently not supported, but there are various tools for converting from dicom to NIfTI such as [dcm2niix](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage).

To use QuickViewer to view a NIfTI image, first import the QuickViewer class:
```
from quickviewer import QuickViewer
```
and then create a QuickViewer instance, giving the path to your NIfTI file:
```
QuickViewer("my_file.nii")
```
This will launch a viewer that looks something like this:

<img src="images/basic.png" alt="basic quickviewer usage" height="500"/>

The widgets can be used to change
- The image orientation (x-y = axial view; y-z = sagittal view; x-z = coronal view)
- The range of Hounsfield Units to cover;
- The position of the slice currently being displayed.

#### Saving a QuickViewer plot

QuickViewer plots (minus the widgets) can be saved to an image file by providing the extra argument `save_as` when QuickViewer is called:
```
QuickViewer("my_file.nii", save_as="output_image.pdf")
```
The file `output_image.pdf` will automatically be created when this command is run. When QuickViewer is run with the `save_as` option, an extra widget will appear below the image:

<img src="images/save_as.png" alt="quickviewer file saving widget" height="70"/>

This can be used to overwrite the original output file after some changes have been made via the other widgets, or to save the image to a different file.

#### Displaying multiple images

To show multiple images side-by-side, you can give QuickViewer a list of filepaths rather than a single path:
```
QuickViewer(["my_file.nii", "another_file.nii"])
```
The result will look something like this:

<img src="images/two_images.png" alt="two images in quickviewer" height="500"/>

If the two images are in the same frame of reference, they will automatically share their HU and position sliders, so that you can scroll through both simultaneously. This behaviour can be turned off with the `share_slider` argument:
```
QuickViewer(["my_file.nii", "another_file.nii"], share_slider=False)
```

#### Orthogonal view

#### More display settings

##### Initial QuickViewer config

By default, QuickViewer will show the central slice of the axial plane, with the HU range set to `-300 -- 200`. These initial settings can be changed via the arguments:
- `init_view`: set this to `x-y`, `y-z`, or `x-z` to set initial orientation (default = `x-y`);
- `init_pos`: initial slice position in mm (will be rounded to the nearest slice;
- `hu`: set this to a pair of values (e.g. `hu=[-500, 500]`) to set the initial HU range.

To show scales in terms of voxel numbers rather than mm, run QuickViewer with the argument `scale_in_mm=False`. The initial slice position can be set via the `init_sl` argument.

##### More Housfield Unit options

To define the HU range in terms of a central value and window width, rather than minimum and maximum values:
- Set the `hu` argument to a single value, which will be taken as the window centre;
- Optionally set the `hu_width` argument to change the initial window width (default = 500).
Both the HU centre and HU width can be change later via widgets.

The full range of available HU values can also be adjusted by changing the `hu_limits` argument (default = `[-2000, 2000]`); changing this might be useful if you're viewing an image with a much smaller range of intensities.

##### Custom titles

To give your plot a custom title, set the `title` argument:
```
QuickViewer("my_file.nii", title="A CT scan image")
```
When viewing multiple images, multiple titles can be set by providing a list:
```
QuickViewer(["file1.nii", "file2.nii"], title=["First image", "Second image"])
```

##### Zooming, panning and downsampling

If QuickViewer is run with the setting `zoom_ui=True`, extra widgets for zooming and panning will be displayed.

To set the initial zoom level and image centre (regardless of whether `zoom_ui` is set), use:
- `zoom`: set to either a single value for zoom in all directions, or to a list of 3 values for the zoom level in each orientation.
- `zoom_centre`: centrepoint of the image, if zooming is used. This should be a list containing the x, y, and z coordinates of the desired centre.

For large images, you may sometimes wish to downsample for speed. This can be done via the `downsample` parameter; similar to the `zoom` parameter, this can either be a single value setting the downsampling factor to be equal in all directions, or a list of three values for each direction.

##### Other display options

There are  several options for fine-tuning the appearance of the plot:
- `figsize`: sets the figure height in inches (default = 5 inches).
- `xlim`/`ylim`/`zlim`: set these to a pair of values to specify custom `x`, `y`, or `z` limits (i.e. to crop the plot).
- `plots_per_row`: maximum number of images to display per row, when using multiple images; this allows images to be spread over multiple rows.
- `major_ticks`: sets the interval between major axis tick marks.
- `minor_ticks`: sets the number of minor tick marks to display between each pair of major tickmarks (by default, no minor ticks are shown).
- `ticks_all_sides`: if set to True, tick marks will be shown at the top and right-hand side of plots as well as the bottom and left-hand side.

#### Viewing a NumPy array

QuickViewer can be used to view NumPy arrays as well as NIfTI images. This is done by giving the array in place of the NIfTI filepath:
```
import numpy
my_array = numpy.random.rand(512, 512, 40)
QuickViewer(my_array)
```
By default, QuickViewer will set the origin to `(0, 0, 0)` and the voxel sizes in all directions to 1mm. This can be changed either by providing the `voxel_sizes` and `origin` arguments (which should be lists in order `(x, y, z)`) or by giving an affine matrix via the `affine` argument, e.g.:
```
QuickViewer(my_array, voxel_sizes=[2.5, 2.5, 4], origin=[-200, -300, 0])
```

### From the command line:
1. A script for creating a quickviewer plot from the command line can be found in `quickviewer/bin/quick_view.py`. The basic usage for viewing a NIfTI file is:
```quick_view.py <filename>```.
2. To see the available input options for this script, run:
```quick_view.py -h```
3. Running this script will create a figure in a seperate window, which can be interacted with using the following commands:
    - **scroll wheel**: scroll through image one slice at a time
    - **left/right arrows**: scroll through image one slice at a time
    - **up/down arrows**: scroll through image five slices at a time
    - **v**: switch orientation
    - **d**: change dose field opacity
    - **m**: turn masks on and off
    - **c**: toggle structure plotting type between contours, masks, and none
    - **j**: jump between the structures on the image
    - **i**: invert any comparison images
    - **o**: change the opacity of overlay comparison image

### Inside a python script:
The `QuickViewer` class can be imported into a python script by adding
```from quickviewer import QuickViewer```
to the script. Creating a `QuickViewer` object inside the code will cause a window containing the plot to be opened when the code is run.
