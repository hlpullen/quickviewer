'''Prototype classes for core data functionality.'''

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import os
import pydicom


_axes = ['y', 'x', 'z']
_slice_axes = {
    'x-y': 2,
    'y-z': 1,
    'x-z': 0
}
_plot_axes = {
    'x-y': [0, 1],
    'y-z': [0, 2],
    'x-z': [1, 2]
}


class Image:
    '''Loads and stores a medical image and its geometrical properties, either 
    from a dicom/nifti file or a numpy array.'''

    def __init__(
        self,
        source,
        load=True,
        affine=None,
        voxel_size=(1, 1, 1),
        origin=(0, 0, 0),
    ):
        '''
        Initialise from a medical image source.

        Parameters
        ----------
        source : str/array/Nifti1Image
            Source of image data. Can be either:
                (a) A string containing the path to a dicom or nifti file;
                (b) A string containing the path to a numpy file containing a
                    2D or 3D array;
                (c) A 2D or 3D numpy array;
                (d) A nibabel.nifti1.Nifti1Image object.

        load : bool, default=True
            If True, the image data will be immediately loaded. Otherwise, it
            can be loaded later with the load_data() method.

        affine : 4x4 array, default=None
            Array containing the affine matrix to use if <source> is a numpy
            array or path to a numpy file. If not None, this takes precendence 
            over <voxel_size> and <origin>.

        voxel_size : tuple, default=(1, 1, 1)
            Voxel sizes in mm in order (y, x, z) to use if <source> is a numpy
            array or path to a numpy file and <affine> is not provided.

        origin : tuple, default=(0, 0, 0)
            Origin position in mm in order (y, x, z) to use if <source> is a 
            numpy array or path to a numpy file and <affine> is not provided.
        '''

        self.data = None
        self.source = source
        self.affine = affine
        self.voxel_size = voxel_size
        self.origin = origin
        if load:
            self.load_data()

        self.mask = None
        self.mask_per_view = None
        self.shift = None

    def get_data(self):
        '''Return image array.'''

        if self.data is None:
            self.load_data()
        return self.data

    def get_masked_data(self, view=None):
        '''Return image array with mask applied, if self.mask is not None.'''

        self.load_data()
        if self.mask:
            return self.data * self.mask
        elif self.mask_per_view and view in self.mask_per_view:
            return self.data * self.mask_per_view[view]
        return self.data

    def load_data(self, force=False):
        '''Load pixel array from image source.'''

        if self.data is not None and not force:
            return

        window_width = None
        window_centre = None

        # Load image array from source
        # Numpy array
        if isinstance(self.source, np.ndarray):
            self.data = self.source

        # Try loading from nifti file
        else:
            if not os.path.exists(self.source):
                raise RuntimeError(f'Image input {self.source} does not exist!')
            if os.path.isfile(self.source):
                self.data, self.affine = load_nifti(self.source)

        # Try loading from dicom file
        if self.data is None:
            self.data, self.affine, window_centre, window_width \
                    = load_dicom(self.source)

        # Try loading from numpy file
        if self.data is None:
            self.data = load_npy(self.source)

        # Ensure array is 3D
        if self.data.ndim == 2:
            self.data = self.data[..., np.newaxis]
    
        # Set affine matrix, voxel sizes, and origin
        if self.affine is None:
            self.affine = np.array([
                [self.voxel_size[0], 0, 0, self.origin[0]],
                [0, self.voxel_size[1], 0, self.origin[1]],
                [0, 0, self.voxel_size[2], self.origin[2]],
                [0, 0, 0, 1]
            ])
        else:
            self.voxel_size = list(np.diag(self.affine)[:-1])
            self.origin = list(self.affine[:-1, -1])

        # Set other geometric properties
        self.shape = self.data.shape
        self.lims = [
            (self.origin[i], 
             self.origin[i] + (self.shape[i] - 1) * self.voxel_size[i])
            for i in range(3)
        ]
        self.image_extent = [
            (self.lims[i][0] - self.voxel_size[i] / 2,
             self.lims[i][1] + self.voxel_size[i] / 2)
            for i in range(3)
        ]

        # Set default grayscale range
        if window_width and window_centre:
            self.default_window = [
                window_centre - window_width / 2,
                window_centre + window_width / 2
            ]
        else:
            self.default_window = [-300, 200] 

    def plot(
        self, 
        view='x-y', 
        sl=None, 
        idx=None, 
        pos=None,
        scale_in_mm=True,
    ):
        '''Plot a 2D slice of the image.

        Parameters
        ----------
        view : str, default='x-y'
            Orientation in which to plot the image. Can be any of 'x-y', 
            'y-z', and 'x-z'.

        sl : int, default=None
            Slice number to plot. Takes precedence over <idx> and <pos> if not
            None. If all of <sl>, <idx>, and <pos> are None, the central 
            slice will be plotted.

        idx : int, default=None
            Index of the slice in the array to plot. Takes precendence over 
            <pos>.

        pos : float, default=None
            Position in mm of the slice to plot. Will be rounded to the nearest
            slice. Only used if <sl> and <idx> are both None.

        scale_in_mm : bool, default=True
            If True, axis labels will be in mm; otherwise, they will be slice 
            numbers.
        '''

        # Get index of the slice to plot
        self.load_data()
        if sl is not None:
            idx = self.slice_to_idx(sl, _slice_axes[view])
        elif idx is None:
            if pos is not None:
                idx = self.pos_to_idx(pos, _slice_axes[view])
            else:
                centre_pos = self.get_image_centre()[_slice_axes[view]]
                idx = self.pos_to_idx(centre_pos, _slice_axes[view])

        # Get image slice
        transpose = list(_plot_axes[view]) + [_slice_axes[view]]
        image_slice = np.transpose(self.data, transpose)[:, :, idx]

        # Get image extent and aspect ratio
        x_ax = _plot_axes[view][1]
        y_ax = _plot_axes[view][0]
        if scale_in_mm:
            extent = self.image_extent[x_ax] + self.image_extent[y_ax][::-1]
            aspect = 1
        else:
            extent = [
                0.5, self.shape[x_ax] + 0.5, self.shape[y_ax] + 0.5, 0.5
            ]
            aspect = abs(self.voxel_size[y_ax] / self.voxel_size[x_ax])

        # Plot the slice
        plt.imshow(
            image_slice, 
            cmap='gray', 
            extent=extent,
            aspect=aspect,
            vmin=self.default_window[0],
            vmax=self.default_window[1]
        )

        # Set title and axis labels

    def idx_to_pos(self, idx, ax):
        '''Convert an array index to a position in mm along a given axis.'''

        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        return self.origin[i_ax] + (self.shape[i_ax] - 1 - idx) \
                * self.voxel_size[i_ax]

    def pos_to_idx(self, pos, ax, return_int=True):
        '''Convert a position in mm to an array index along a given axis.'''

        self.load_data()
        i_ax = ax if isinstance(ax, int) else _axes.index(ax)
        idx = self.shape[i_ax] - 1 + (self.origin[i_ax] - pos) \
                / self.voxel_size[i_ax]
        if return_int:
            return round(idx)
        else:
            return idx

    def idx_to_slice(self, idx, ax):
        '''Convert an array index to a slice number along a given axis.'''
        
        return idx + 1

    def slice_to_idx(self, sl, ax):
        '''Convert a slice number to an array index along a given axis.'''

        return sl - 1

    def pos_to_slice(self, pos, ax):
        '''Convert a position in mm to a slice number along a given axis.'''

        return idx_to_slice(pos_to_idx(pos, ax), ax)

    def slice_to_pos(self, sl, ax):
        '''Convert a slice number to a position in mm along a given axis.'''

        return idx_to_pos(slice_to_idx(sl, ax), ax)

    def get_image_centre(self):
        '''Get position in mm of the centre of the image.'''

        return [np.mean(self.lims[i]) for i in range(3)]

    def get_voxel_coords(self):
        '''Get arrays of voxel coordinates in each direction.'''

        return

    def get_plot_aspect_ratio(self, view, zoom=None, n_colorbars=0,
                              figsize=None):
        '''Estimate the ideal width/height ratio for a plot of this image 
        in a given orientation.'''

    def translate(self, dx=0, dy=0, dz=0):
        '''Apply a translation to the image data.'''

        return

    def rotate(self, yaw=0, pitch=0, roll=0):
        '''Apply a rotation to the image data.'''

        return

    def reset(self):
        '''Return image data to its original state before any translations or
        rotations.'''

        return

    def set_shift(self, nx, ny, nz):
        '''Set a shift amount in voxels to apply to the image before plotting.
        (Faster than translating the entire 3D image).
        '''

        return

    def set_mask(self, mask, threshold=0.5):
        '''Set a mask that can be applied to the image at plotting time.
        Can be either a single array, list of arrays to stack, or dict of 
        arrays for different orientations.'''
 
        # Get single mask or list of masks
        single_mask = None
        mask_per_view = None
        if isinstance(mask, np.ndarray):
            single_mask = mask
        elif is_list(mask):
            single_mask = mask[0]
            for m in mask[1:]:
                single_mask += m
        elif isinstance(mask, dict):
            mask_per_view = mask

        # Convert mask(s) to boolean
        if single_mask:
            single_mask = single_mask > threshold
        if mask_per_view:
            for view, m in mask_per_view.items():
                mask_per_view[view] = m > threshold

        # Assign to class properties
        self.mask = single_mask
        self.mask_per_view = mask_per_view

    def write(self, outname):
        '''Write image data to a file.'''

        # Detect output filetype based on outname

        # Write to file

        pass



def load_nifti(path):
    '''Load an image from a nifti file.'''

    try:
        nii = nibabel.load(path)
        data = nii.get_fdata()
        affine = nii.affine
        return data, affine

    except FileNotFoundError:
        print(f'Warning: file {path} not found! Could not load nifti.')
        return None, None

    except nibabel.filebasedimages.ImageFileError:
        return None, None


def load_dicom(path):
    '''Load a dicom image from one or more dicom files.'''

    # Try loading single dicom file
    paths = []
    if os.path.isfile(path):
        try:
            ds = pydicom.read_file(path)
            if ds.get('ImagesInAcquisition', None) == 1:
                paths = [path]
        except pydicom.errors.InvalidDicomError:
            return None, None

    # Case where there are multiple dicom files for this image
    if not paths:
        if os.path.isdir(path):
            dirname = path
        else:
            dirname = os.path.dirname(path)
        paths = sorted([os.path.join(dirname, p) for p in os.listdir(dirname)])

        # Ensure user-specified file is loaded first
        if path in paths:
            paths.insert(0, paths.pop(paths.index(path)))

    # Load image arrays from all files
    study_uid = None
    series_num = None
    modality = None
    slice_thickness = None
    pixel_size = None
    origin = None
    rescale_slope = None
    rescale_intercept = None
    window_centre = None
    window_width = None
    data_slices = {}
    for dcm in paths:
        try:
            
            # Load file and check it matches the others
            ds = pydicom.read_file(dcm)
            if study_uid is None:
                study_uid = ds.StudyInstanceUID
            if series_num is None:
                series_num = ds.SeriesNumber
            if modality is None:
                modality = ds.Modality
            if (ds.StudyInstanceUID != study_uid 
                or ds.SeriesNumber != series_num
                or ds.Modality != modality):
                continue

            # Get data 
            z = getattr(ds, 'ImagePositionPatient', [0, 0, 0])[2]
            data_slices[z] = ds.pixel_array

            # Get voxel spacings
            if pixel_size is None:
                for attr in ['PixelSpacing', 'ImagerPixelSpacing']:
                    pixel_size = getattr(ds, attr, None)
                    if pixel_size:
                        break
            if slice_thickness is None:
                slice_thickness = getattr(ds, 'SliceThickness', None)

            # Get origin
            if origin is None:
                origin = ds.ImagePositionPatient

            # Get rescale settings
            if rescale_slope is None:
                rescale_slope = getattr(ds, 'RescaleSlope', None)
            if rescale_intercept is None:
                rescale_intercept = getattr(ds, 'RescaleIntercept', None)

            # Get HU window defaults
            if window_centre is None:
                window_centre = getattr(ds, 'WindowCenter', None)
            if window_width is None:
                window_width = getattr(ds, 'WindowWidth', None)

        # Skip any invalid dicom files
        except pydicom.errors.InvalidDicomError:
            continue

    # Case where no data was found
    if not data_slices:
        print(f'Warning: no valid dicom files found in {path}')
        return None, None

    # Case with single image array
    if len(data_slices) == 1:
        data = list(data_slices.values())[0]

    # Combine arrays
    else:

        # Sort by slice position
        sorted_slices = sorted(list(data_slices.keys()))
        sorted_data = [data_slices[z] for z in sorted_slices]
        data = np.stack(sorted_data, axis=-1)

        # Recalculate slice thickness from spacing
        slice_thickness = sorted_slices[1] - sorted_slices[0]

    # Make affine matrix
    affine =  np.array([
        [pixel_size[0], 0, 0, origin[0]],
        [0, pixel_size[1], 0, origin[1]],
        [0, 0, slice_thickness, origin[2]],
        [0, 0, 0, 1]
    ])

    return data, affine, window_centre, window_width


def load_npy(path):
    '''Load a numpy array from a .npy file.'''

    try:
        data = np.load(path)
        return data

    except (IOError, ValueError):
        return


def is_list(var):
    '''Check whether a variable is a list, tuple, or array.'''

    is_a_list = False
    for t in [list, tuple, np.ndarray]:
        if isinstance(var, t):
            is_a_list = True
    return is_a_list
