'''Classes and functions for loading, plotting, and comparing structures.'''
import fnmatch
import matplotlib.cm
import numpy as np
import os
import pydicom
import re
import skimage.measure
from scipy.ndimage import morphology
from shapely import geometry

from quickviewer.prototype import Image
from quickviewer.prototype import (
    is_list, 
    to_three, 
    _axes,
    _slice_axes, 
    _plot_axes,
    _default_figsize
)


class StructDefaults:
    '''Singleton class for assigning default structure names and colours.'''

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __StructDefaults:

        def __init__(self):

            self.n_structs = 0
            self.n_structure_sets = 0
            self.n_colors_used = 0

            self.colors = (
                list(matplotlib.cm.Set1.colors)[:-1]
                + list(matplotlib.cm.Set2.colors)[:-1]
                + list(matplotlib.cm.Set3.colors)
                + list(matplotlib.cm.tab20.colors)
            )
            for i in [9, 10]:  # Remove greys
                del self.colors[i]

    def __init__(self, reset=False):
        '''Constructor of StructDefaults singleton class.'''

        if not StructDefaults.instance:
            StructDefaults.instance = StructDefaults.__StructDefaults()
        elif reset:
            StructDefaults.instance.__init__()

    def get_default_struct_name(self):
        '''Get a default name for a structure.'''

        StructDefaults.instance.n_structs += 1
        return f'Structure {StructDefaults.instance.n_structs}'

    def get_default_struct_color(self):
        '''Get a default structure color.'''

        color = StructDefaults.instance.colors[
            StructDefaults.instance.n_colors_used]
        StructDefaults.instance.n_colors_used += 1
        return color

    def get_default_structure_set_name(self):
        '''Get a default name for a structure set.'''

        StructDefaults.instance.n_structure_sets += 1
        return f'Structure Set {StructDefaults.instance.n_structure_sets}'


StructDefaults()


class Structure(Image):
    '''Single structure.'''

    def __init__(
        self,
        source=None,
        name=None,
        color=None,
        load=None,
        contours=None,
        image=None,
        shape=None,
        mask_level=0.25,
        **kwargs
    ):

        '''Load structure from mask or contour.

        Parameters
        ----------
        source : str/array/nifti, default=None
            Source of image data to load. Can be either:
                (a) The path to a nifti file containing a binary mask;
                (b) A numpy array containing a binary mask;
                (c) The path to a file containing a numpy array;
                (d) The path to a dicom structure set file.

            If <source> is not given, <contours> and <shape> must be given in
            order to load a structure directly from a contour.

        name : str, default=None
            Name of the structure. If <source> is a file and no name is given,
            the name will be inferred from the filename.

        color : matplotlib color, default=None
            Color in which this structure will be plotted. If None, a color
            will be assigned.

        load : bool, default=True
            If True, contours/mask will be created during initialisation; 
            otherwise they will be created on-demand.

        contours: dict, default=None
            Dictionary of contours in the x-y orienation, where the keys are 
            z positions in mm and values are the 3D contour points in order
            (x, y, z). Only used if <source> is None. These contours will be
            used to generate a binary mask.

        image : Image/str, default=None
            Associated image from which to extract shape and affine matrix.

        shape : list, default=None
            Number of voxels in the image to which the structure belongs, in 
            order (x, y, z). Needed if <contours> is used instead of <source>.

        kwargs : dict, default=None
            Extra arguments to pass to the initialisation of the parent
            Image object (e.g. affine matrix if loading from a numpy array).

        '''

        # Assign properties
        self.source = source
        self.custom_color = color is not None
        self.set_color(color)
        self.input_contours = contours
        self.image = image
        if image and not isinstance(image, Image):
            self.image = Image(image)
        self.shape = shape
        self.mask_level = mask_level
        self.kwargs = kwargs

        # Create name
        self.name = name
        if self.name is None:
            if isinstance(self.source, str):
                basename = os.path.basename(self.source).split('.')[0]
                name = re.sub(r'RTSTRUCT_[MVCT]+_\d+_\d+_\d+', '', basename)
                name = standard_str(name)
                self.name = name[0].upper() + name[1:]
            else:
                self.name = StructDefaults().get_default_struct_name()

        # Load structure data
        self.loaded = False
        self.loaded_contours = False
        self.loaded_mask = False
        if load:
            self.load()

    def load(self):
        '''Load structure from file.'''

        if self.loaded:
            return

        if self.image:
            self.image.load_data()

        # Try loading from dicom structure set
        structs = []
        if isinstance(self.source, str):
        
            structs = load_structs_dicom(self.source, names=self.name)
            if len(structs):

                # Check a shape or image was given
                if self.shape is None and self.image is None:
                    raise RuntimeError('Must provide an associated image or '
                                       'image shape if loading a structure '
                                       'from dicom!')

                # Get structure info
                struct = structs[list(structs.keys())[0]]
                self.name = struct['name']
                self.input_contours = struct['contours']
                if not self.custom_color:
                    self.set_color(struct['color'])

        # Load structure mask
        if not len(structs):
            Image.__init__(self, self.source, **self.kwargs)
            self.create_mask()

        # Deal with input from dicom
        if self.input_contours is not None:

            # Create Image object
            if self.image is not None:
                self.kwargs['voxel_size'] = self.image.voxel_size
                self.kwargs['origin'] = self.image.origin
                self.shape = self.image.data.shape
                Image.__init__(self, np.zeros(self.shape), **self.kwargs)

            # Set x-y contours with z indices as keys
            self.contours = {'x-y': {}}
            for z, contours in self.input_contours.items():
                iz = self.pos_to_idx(z, 'z')
                self.contours['x-y'][iz] = [
                    [tuple(p[:2]) for p in points] for points in contours
                ]


        self.loaded = True

    def get_contours(self, view='x-y'):
        '''Get dict of contours in a given orientation.'''

        self.create_contours()
        return self.contours[view]

    def get_mask(self):
        '''Get binary mask.'''

        self.create_mask()
        return self.data

    def create_contours(self):
        '''Create contours in all orientations.'''
        
        if self.loaded_contours:
            return
        if not self.loaded:
            self.load()

        if not hasattr(self, 'contours'):
            self.contours = {}
        self.create_mask()

        # Create contours in every orientation
        for view, z_ax in _slice_axes.items():
            if view in self.contours:
                continue

            # Make new contours from mask
            self.contours[view] = {}
            for iz in range(self.n_voxels[z_ax]):

                # Get slice of mask array
                mask_slice = self.get_slice(view, idx=iz).T
                if mask_slice.max() < 0.5:
                    continue 

                # Convert mask array to contour(s)
                contours = skimage.measure.find_contours(
                    mask_slice, 0.5, 'low', 'low')
                if not contours:
                    continue

                # Convert indices to positions in mm
                x_ax, y_ax = _plot_axes[view]
                points = []
                for contour in contours:
                    contour_points = []
                    for ix, iy in contour:
                        px = self.idx_to_pos(ix, x_ax)
                        py = self.idx_to_pos(iy, y_ax)
                        contour_points.append((px, py))
                    points.append(contour_points)
                self.contours[view][iz] = points

        self.loaded_contours = True

    def create_mask(self):
        '''Create binary mask.'''

        if self.loaded_mask:
            return

        # Create mask from x-y contours if needed
        if self.input_contours:

            # Check an image or shape was given
            if self.image is None and self.shape is None:
                raise RuntimeError('Must set structure.image or structure.shape'
                                   ' before creating mask!')
            if self.image is None:
                Image.__init__(self, np.zeros(self.shape), **self.kwargs)

            # Create mask on each z layer
            for z, contours in self.input_contours.items():

                # Convert z position to index
                iz = self.pos_to_idx(z, 'z')

                # Loop over each contour on the z slice
                pos_to_idx_vec = np.vectorize(self.pos_to_idx)
                for points in contours:

                    # Convert (x, y) positions to array indices
                    points_idx = np.zeros((points.shape[0], 2))
                    for i in range(2):
                        points_idx[:, i] = pos_to_idx_vec(points[:, i], i,
                                                          return_int=False)

                    # Create polygon
                    polygon = geometry.Polygon(points_idx)

                    # Get polygon's bounding box
                    ix1, iy1, ix2, iy2 = [int(xy) for xy in polygon.bounds]
                    ix1 = max(0, ix1)
                    ix2 = min(ix2 + 1, self.shape[1])
                    iy1 = max(0, iy1)
                    iy2 = min(iy2 + 1, self.shape[0])

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
                            self.data[iy, ix, int(iz)] += overlap
                            
            self.data = self.data > self.mask_level

        # Convert to boolean mask
        if hasattr(self, 'data'):
            if not self.data.dtype == 'bool':
                self.data = self.data > 0.5
            if not hasattr(self, 'empty'):
                self.empty = not np.any(self.data)
            self.loaded_mask = True

    def get_slice(self, *args, **kwargs):

        self.create_mask()
        return Image.get_slice(self, *args, **kwargs)

    def get_indices(self, view='x-y', slice_num=False):
        '''Get list of slice indices on which this structure exists. If 
        <slice_num> is True, slice numbers will be returned instead of 
        indices.'''

        if not hasattr(self, 'contours') or view not in self.contours:
            self.create_contours()
        indices = list(self.contours[view].keys())
        if slice_num:
            z_ax = _slice_axes[view]
            return [self.idx_to_slice(i, z_ax) for i in indices]
        else:
            return indices

    def get_mid_idx(self, view='x-y', slice_num=False):
        '''Get central slice index of this structure in a given orientation.'''
        
        return round(np.mean(self.get_indices(view, slice_num=slice_num)))

    def on_slice(self, view, sl=None, idx=None, pos=None):
        '''Check whether this structure exists on a given slice.'''

        idx = self.get_idx(view, sl, idx, pos)
        return idx in self.get_indices(view)

    def get_centroid(self, view=None, sl=None, idx=None, pos=None, units='mm',
                     standardise=True):
        '''Get centroid position in 2D or 3D.'''
        
        # Get 2D or 3D data from which to calculate centroid
        if view is not None:
            if sl is None and idx is None and pos is None:
                idx = self.get_mid_idx(view)
            if not self.on_slice(view, sl, idx, pos):
                print('not on slice!', view, sl, idx, pos)
                return [None, None]
            data = self.get_slice(view, sl, idx, pos)
            axes = _plot_axes[view]
        else:
            data = self.get_data(standardise)
            axes = _axes

        # Compute centroid
        non_zero = np.argwhere(data)
        centroid_rowcol = list(non_zero.mean(0))
        centroid = [centroid_rowcol[1], centroid_rowcol[0]] \
                + centroid_rowcol[2:] 
        
        # Convert to mm
        if units == 'mm':
            centroid = [self.idx_to_pos(c, axes[i]) for i, c in 
                        enumerate(centroid)]
        return centroid

    def get_centre(self, view=None, sl=None, idx=None, pos=None, units='mm',
                   standardise=True):
        '''Get centre position in 2D or 3D.'''

        # Get 2D or 3D data for which to calculate centre
        if view is None:
            data = self.get_data(standardise)
            axes = _axes
        else:
            if sl is None and idx is None and pos is None:
                idx = self.get_mid_idx(view)
            data = self.get_slice(view, sl, idx, pos)
            axes = _plot_axes[view]

        # Calculate mean of min and max positions
        non_zero = np.argwhere(data)
        if not len(non_zero):
            return [0 for i in axes]
        centre_rowcol = list((non_zero.max(0) + non_zero.min(0)) / 2)
        centre = [centre_rowcol[1], centre_rowcol[0]] + centre_rowcol[2:]
        
        # Convert to mm
        if units == 'mm':
            centre = [self.idx_to_pos(c, axes[i]) for i, c in enumerate(centre)]
        return centre

    def get_volume(self, units='mm'):
        '''Get structure volume.'''
        pass

    def get_area(self, view='x-y', sl=None, idx=None, pos=None, units='mm'):
        '''Get the area of the structure on a given slice.'''
        pass

    def get_length(self, units='mm'):
        '''Get total length of the structure.'''
        pass

    def set_color(self, color):
        '''Set plotting color.'''
        
        if color is not None and not matplotlib.colors.is_color_like(color):
            print(f'Warning: {color} not a valid color!')
            color = None
        if color is None:
            color = StructDefaults().get_default_struct_color()
        self.color = matplotlib.colors.to_rgba(color)

    def plot(
        self, 
        view='x-y',
        plot_type='contour',
        sl=None,
        idx=None,
        pos=None,
        opacity=None,
        linewidth=None,
        contour_kwargs=None,
        mask_kwargs=None,
        **kwargs
    ):
        '''Plot this structure as either a mask or a contour.'''

        show_centroid = 'centroid' in plot_type

        # Plot a mask
        if plot_type == 'mask':
            self.plot_mask(view, sl, idx, pos, mask_kwargs, opacity, **kwargs)

        # Plot a contour
        elif plot_type in ['contour', 'centroid']:
            self.plot_contour(view, sl, idx, pos, contour_kwargs, linewidth,
                              centroid=show_centroid, **kwargs)

        # Plot transparent mask + contour
        elif 'filled' in plot_type:
            if opacity is None:
                opacity = 0.3
            self.plot_mask(view, sl, idx, pos, mask_kwargs, opacity, **kwargs)
            kwargs['ax'] = self.ax
            kwargs['include_image'] = False
            self.plot_contour(view, sl, idx, pos, contour_kwargs, linewidth,
                              centroid=show_centroid, **kwargs)

        else:
            print('Unrecognised structure plotting option:', plot_type)

    def plot_mask(
        self,
        view='x-y',
        sl=None,
        idx=None,
        pos=None,
        mask_kwargs=None,
        opacity=None,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        **kwargs
    ):
        '''Plot the structure as a mask.'''

        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)
        self.create_mask()
        self.set_ax(view, ax, gs, figsize)
        mask_slice = self.get_slice(view, idx=idx)

        # Make colormap
        norm = matplotlib.colors.Normalize()
        cmap = matplotlib.cm.hsv
        s_colors = cmap(norm(mask_slice))
        s_colors[mask_slice > 0, :] = self.color
        s_colors[mask_slice == 0, :] = (0, 0,  0, 0)

        # Get plotting arguments
        if mask_kwargs is None:
            mask_kwargs = {}
        mask_kwargs.setdefault('alpha', opacity)
        mask_kwargs.setdefault('interpolation', 'none')

        # Make plot
        if include_image:
            self.image.plot(view, idx=idx, ax=self.ax, show=False)
        self.ax.imshow(s_colors, extent=self.plot_extent[view], **mask_kwargs)

        # Adjust axes
        self.label_ax(view, idx, **kwargs)
        self.zoom_ax(view, zoom, zoom_centre)

    def plot_contour(
        self,
        view='x-y',
        sl=None,
        idx=None,
        pos=None,
        contour_kwargs=None,
        linewidth=None,
        centroid=False,
        show=True,
        ax=None,
        gs=None,
        figsize=_default_figsize,
        include_image=False,
        zoom=None,
        zoom_centre=None,
        **kwargs
    ):
        '''Plot the structure as a contour.'''

        if not hasattr(self, 'contours') or view not in self.contours:
            self.create_contours()

        if sl is None and idx is None and pos is None:
            idx = self.get_mid_idx(view)
        else:
            idx = self.get_idx(view, sl, idx, pos)
        if not self.on_slice(view, idx=idx):
            return
        self.set_ax(view, ax, gs, figsize)

        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs.setdefault('color', self.color)
        contour_kwargs.setdefault('linewidth', linewidth)

        # Plot
        if include_image:
            self.image.plot(view, idx=idx, ax=self.ax, show=False)
        for points in self.contours[view][idx]:
            points_x = [p[0] for p in points]
            points_y = [p[1] for p in points]
            points_x.append(points_x[0])
            points_y.append(points_y[0])
            self.ax.plot(points_x, points_y, **contour_kwargs)
        if not include_image:
            self.ax.invert_yaxis()

        # Plot centroid point
        if centroid:
            self.ax.plot(*self.get_centroid(view, sl, idx, pos), '+',
                         **contour_kwargs)

        # Adjust axes
        self.ax.set_aspect('equal')
        self.label_ax(view, idx, **kwargs)
        self.zoom_ax(view, zoom, zoom_centre)

    def zoom_ax(self, view, zoom=None, zoom_centre=None):
        '''Zoom in on axes, using centre of structure as zoom centre if not
        otherwise specified.'''

        if not zoom:
            return
        zoom = to_three(zoom)
        x_ax, y_ax = _plot_axes[view]
        if zoom_centre is None:
            mid_x, mid_y = self.get_centre(view)
        else:
            mid_x, mid_y = zoom_centre[x_ax], zoom_centre[y_ax]

        # Calculate new axis limits
        init_xlim = self.plot_extent[view][:2]
        init_ylim = self.plot_extent[view][2:]
        xlim = [
            mid_x - (init_xlim[1] - init_xlim[0]) / (2 * zoom[x_ax]),
            mid_x + (init_xlim[1] - init_xlim[0]) / (2 * zoom[x_ax])
        ]
        ylim = [
            mid_y - (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax]),
            mid_y + (init_ylim[1] - init_ylim[0]) / (2 * zoom[y_ax])
        ]

        # Set axis limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)



class StructureSet:
    '''Structure set.'''

    def __init__(
        self,
        sources,
        name=None,
        image=None,
        load=True,
        names=None,
        to_keep=None,
        to_remove=None,
    ):
        '''Load structures from sources.'''

        self.name = name
        if name is None:
            self.name = StructDefaults().get_default_structure_set_name()
        self.sources = sources
        if not is_list(sources):
            self.sources = [sources]
        self.structs = []
        self.set_image(image)
        self.to_keep = to_keep
        self.to_remove = to_remove
        self.names = names
        self.loaded = False
        if load:
            self.load()

    def load(self, sources=None, force=False):
        '''Load structures from sources. If None, will load from own 
        self.sources.'''

        if self.loaded and not force and sources is None:
            return

        if sources is None:
            sources = self.sources
        elif not is_list(sources):
            sources = [sources]

        # Expand any directories
        sources_expanded = []
        for source in sources:
            if os.path.isdir(source):
                sources_expanded.extend(os.path.listdir(source))
            else:
                sources_expanded.append(source)

        for source in sources_expanded:

            # Attempt to load from dicom
            structs = load_structs_dicom(source)
            if len(structs):
                for struct in structs.values():
                    self.structs.append(Structure(
                        name=struct['name'],
                        color=struct['color'],
                        contours=struct['contours'],
                        image=self.image
                    ))

            # Load from struct mask
            else:
                try:
                    self.structs.append(Structure(
                        source, image=self.image
                    ))
                except RuntimeError:
                    continue

        self.rename_structs()
        self.filter_structs()
        self.loaded = True

    def reset(self):
        '''Reload structures from sources.'''

        self.structs = []
        self.loaded = False
        self.load(force=True)

    def set_image(self, image):
        '''Set image for self and all structures.'''

        if image and not isinstance(image, Image):
            image = Image(image)

        self.image = image
        for s in self.structs:
            s.image = image

    def rename_structs(self, names=None, first_match_only=True):
        '''Rename structures if a naming dictionary is given. If 
        <first_match_only> is True, only the first structure matching the 
        possible matches will be renamed.'''

        if names is None:
            names = self.names
        if not names:
            return

        # Loop through each new name
        already_renamed = []
        for name, matches in names.items():

            if not is_list(matches):
                matches = [matches]

            # Loop through all possible original names
            name_matched = False
            for m in matches:

                # Loop through structures and see if there's a match
                for i, s in enumerate(self.structs):

                    # Don't rename a structure more than once
                    if i in already_renamed:
                        continue

                    if fnmatch.fnmatch(standard_str(s.name), standard_str(m)):
                        s.name = name
                        name_matched = True
                        already_renamed.append(i)
                        if first_match_only:
                            break

                # If first_match_only, don't rename more than one structure
                # with this new name
                if name_matched and first_match_only:
                    break

    def filter_structs(self, to_keep=None, to_remove=None):
        '''Keep only structs in the to_keep list, and remove any in the 
        to_remove list.'''

        if to_keep is None:
            to_keep = self.to_keep
        elif not is_list(to_keep):
            to_keep = [to_keep]
        if to_remove is None:
            to_remove = self.to_remove
        elif not is_list(to_remove):
            to_remove = [to_remove]

        if to_keep is not None:
            keep = []
            for s in self.structs:
                if any([fnmatch.fnmatch(standard_str(s.name), standard_str(k))
                       for k in to_keep]):
                    keep.append(s)
            self.structs = keep

        if to_remove is not None:
            keep = []
            for s in self.structs:
                if not any([fnmatch.fnmatch(standard_str(s.name), 
                                            standard_str(r))
                           for r in to_remove]):
                    keep.append(s)
            self.structs = keep

    def add_structs(self, sources):
        '''Add additional structures from sources.'''

        if not is_list(sources):
            sources = [sources]
        self.sources.extend(sources)
        self.load_structs(sources)

    def copy(self, name=None, names=None, to_keep=None, to_remove=None):
        '''Create a copy of this structure set, with structures optionally
        renamed or filtered.'''

        if not hasattr(self, 'n_copies'):
            self.n_copies = 1
        else:
            self.n_copies += 1
        if name is None:
            name = f'{self.name} (copy {self.n_copies})'

        ss = StructureSet(self.sources, name=name, image=self.image, 
                          load=False, names=names, to_keep=to_keep, 
                          to_remove=to_remove)
        if self.loaded:
            ss.structs = self.structs
            ss.loaded = True
            ss.rename_structs(names)
            ss.filter_structs(to_keep, to_remove)

        return ss

    def get_structs(self):
        '''Get list of Structure objects.'''

        self.load()
        return self.structs

    def get_struct_names(self):
        '''Get list of names of structures.'''

        self.load()
        return [s.name for s in self.structs]

    def get_struct_dict(self):
        '''Get dict of structure names and Structure objects.'''

        self.load()
        return {s.name: s for s in self.structs}

    def print_structs(self):

        self.load()
        print('\n'.join(self.get_struct_names()))

    def __repr__(self):

        self.load()
        out_str = 'StructureSet\n{'
        out_str += '\n  name : ' + str(self.name)
        out_str += '\n  structs :\n    '
        out_str += '\n    '.join(self.get_struct_names())
        out_str += '\n}'
        return out_str


def load_structs_dicom(path, names=None):
    '''Load structure(s) from a dicom structure file. <name> can be a single
    name or list of names of structures to load.'''

    # Load dicom object
    try:
        ds = pydicom.read_file(path)
    except pydicom.errors.InvalidDicomError:
        return []
    if not (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.481.3'):
        print(f'Warning: {path} is not a DICOM structure set file!')
        return

    # Get struture names
    seq = get_dicom_sequence(ds, 'StructureSetROI')
    structs = {}
    for struct in seq:
        structs[int(struct.ROINumber)] = {'name': struct.ROIName}

    # Find structures matching requested names
    names_to_load = None
    if isinstance(names, str):
        names_to_load = [names]
    elif is_list(names):
        names_to_load = names
    if names_to_load:
        structs = {i: s for i, s in structs.items() if 
                   any([fnmatch.fnmatch(standard_str(s['name']), 
                                        standard_str(n))
                        for n in names_to_load]
                      )
                  }
        if not len(structs):
            print(f'Warning: no structures found matching name(s): {names}')
            return

    # Get structure details
    roi_seq = get_dicom_sequence(ds, 'ROIContour')
    for roi in roi_seq:

        number = roi.ReferencedROINumber
        if number not in structs:
            continue
        data = {'contours': {}}

        # Get structure colour
        if 'ROIDisplayColor' in roi:
            data['color'] = [int(c) / 255 for c in list(roi.ROIDisplayColor)]
        else:
            data['color'] = None

        # Get contours
        contour_seq = get_dicom_sequence(roi, 'Contour')
        if contour_seq:
            contour_data = {}
            for c in contour_seq:
                plane_data = [
                    [float(p) for p in c.ContourData[i * 3: i * 3 + 3]]
                    for i in range(c.NumberOfContourPoints)
                ]
                z = float(c.ContourData[2])
                if z not in data['contours']:
                    data['contours'][z] = []
                data['contours'][z].append(np.array(plane_data))

        structs[number].update(data)

    return structs


def get_dicom_sequence(ds=None, basename=''):

    sequence = []
    for suffix in ['Sequence', 's']:
        attribute = f'{basename}{suffix}'
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break
    return sequence


def standard_str(string):
    '''Convert a string to lowercase and replace all spaces with
    underscores.'''

    try:
        return str(string).lower().replace(' ', '_')
    except AttributeError:
        return

