"""Classes for loading, plotting and comparing images and structures."""

import copy
import datetime
import distutils.dir_util
import functools
import glob
import itertools
import json
import math
import numpy as np
import operator
import os
import pydicom
import random
import re
import shutil
import sys
import time

from quickviewer.prototype import Image
from quickviewer.prototype.struct import StructureSet, Structure


# File: quickviewer/data/__init__.py
# Based on voxtox/utility/__init__.py, created by K. Harrison on 130906


default_stations = {"0210167": "LA3", "0210292": "LA4"}

default_opts = {}
default_opts['print_depth'] = 0


class Defaults:
    '''
    Singleton class for storing default values of parameters
    that may be used in object initialisation.

    Implementation of the singleton design pattern is based on:
    https://python-3-patterns-idioms-test.readthedocs.io
           /en/latest/Singleton.html
    '''

    # Define the single instance as a class attribute
    instance = None

    # Create single instance in inner class
    class __Defaults:

        # Define instance attributes based on opts dictionary
        def __init__(self, opts={}):
            for key, value in opts.items():
                setattr(self, key, value)

        # Allow for printing instance attributes
        def __repr__(self):
            out_list = []
            for key, value in sorted(self.__dict__.items()):
                out_list.append(f'{key}: {value}')
            out_string = '\n'.join(out_list)
            return out_string

    def __init__(self, opts={}, reset=False):
        '''
        Constructor of Defaults singleton class.

        Parameters
        ----------
        opts : dict, default={}
            Dictionary of attribute-value pairs.

        reset : bool, default=False
            If True, delete all pre-existing instance attributes before
            adding attributes and values from opts dictionary.
            If False, don't delete pre-existing instance attributes,
            but add to them, or modify values, from opts dictionary.
        '''

        if not Defaults.instance:
            Defaults.instance = Defaults.__Defaults(opts)
        else:
            if reset:
                Defaults.instance.__dict__ = {}
            for key, value in opts.items():
                setattr(Defaults.instance, key, value)

    # Allow for getting instance attributes
    def __getattr__(self, name):
        return getattr(self.instance, name)

    # Allow for setting instance attributes
    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)

    # Allow for printing instance attributes
    def __repr__(self):
        return self.instance.__repr__()


Defaults(default_opts)


class DataObject:
    '''
    Base class for objects serving as data containers.
    An object has user-defined data attributes, which may include
    other DataObject objects and lists of DataObject objects.

    The class provides for printing attribute values recursively, to
    a chosen depth, and for obtaining nested dictionaries of
    attributes and values.
    '''

    def __init__(self, opts={}, **kwargs):
        '''
        Constructor of DataObject class, allowing initialisation of an 
        arbitrary set of attributes.

        Parameters
        ----------
        opts : dict, default={}
            Dictionary to be used in setting instance attributes
            (dictionary keys) and their initial values.

        **kwargs
            Keyword-value pairs to be used in setting instance attributes
            and their initial values.
        '''

        for key, value in opts.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        return None

    def __repr__(self, depth=None):
        '''
        Create string recursively listing attributes and values.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, depth is set to the value
            of the object's print_depth property, if defined,
            or otherwise to the value of Defaults().print_depth.
        '''

        if depth is None:
            depth = self.get_print_depth()

        out_list = [f'\n{self.__class__.__name__}', '{']

        # Loop over attributes, with different treatment
        # depending on whether attribute value is a list.
        # Where an attribute value of list item is
        # an instance of DataObject or a subclass
        # it's string representation is obtained by calling
        # the instance's __repr__() method with depth decreased
        # by 1, or (depth less than 1) is the class representation.
        for key in sorted(self.__dict__):
            item = self.__dict__[key]
            if isinstance(item, list):
                items = item
                n = len(items)
                if n:
                    if depth > 0:
                        value_string = '['
                        for i, item in enumerate(items):
                            item_string = item.__repr__(depth=(depth - 1))
                            comma = "," if (i + 1 < n) else ''
                            value_string = \
                                f'{value_string} {item_string}{comma}'
                        value_string = f'{value_string}]'
                    else:
                        value_string = f'[{n} * {item[0].__class__}]'
                else:
                    value_string = '[]'
            else:
                if issubclass(item.__class__, DataObject):
                    if depth > 0:
                        value_string = item.__repr__(depth=(depth - 1))
                    else:
                        value_string = f'{item.__class__}'
                else:
                    value_string = item.__repr__()
            out_list.append(f'  {key} : {value_string} ')
        out_list.append('}')
        out_string = '\n'.join(out_list)
        return out_string

    def get_dict(self):
        '''
        Return a nested dictionary of object attributes (dictionary keys)
        and their values.
        '''

        objects = {}
        for key in self.__dict__:
            try:
                objects[key] = self.__dict__[key].get_dict()
            except AttributeError:
                objects[key] = self.__dict__[key]

        return objects

    def get_print_depth(self):
        '''
        Retrieve the value of the object's print depth,
        setting an initial value if not previously defined.
        '''

        if not hasattr(self, 'print_depth'):
            self.set_print_depth()
        return self.print_depth

    def print(self, depth=None):
        '''
        Convenience method for recursively printing
        object attributes and values, with recursion
        to a specified depth.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, depth is set in the
            __repr__() method.
        '''

        print(self.__repr__(depth))
        return None

    def set_print_depth(self, depth=None):
        '''
        Set the object's print depth.

        Parameters
        ----------

        depth : integer/None, default=None
            Depth to which recursion is performed.
            If the value is None, the object's print depth is
            set to the value of Defaults().print_depth.
        '''

        if depth is None:
            depth = Defaults().print_depth
        self.print_depth = depth
        return None


class PathObject(DataObject):
    """DataObject with and associated directory; has the ability to 
    extract a list of dated objects from within this directory."""

    def __init__(self, path=""):
        self.path = fullpath(path)
        self.subdir = ""

    def get_dated_objects(self, dtype="DatedObject", subdir=""):
        """Create list of objects of a given type, <dtype>, inside own 
        directory, or inside own directory + <subdir> if given."""

        # Create object for each file in the subdir
        objs = []
        path = os.path.join(self.path, subdir)
        if os.path.isdir(path):
            filenames = os.listdir(path)
            for filename in filenames:
                if is_timestamp(filename):
                    filepath = os.path.join(path, filename)
                    objs.append(globals()[dtype](path=filepath))

        # Sort and assign subdir to the created objects
        objs.sort()
        if subdir:
            for obj in objs:
                obj.subdir = subdir

        return objs


@functools.total_ordering
class DatedObject(PathObject):
    """PathObject with an associated date and time, which can be used for 
    sorting multiple DatedObjects."""

    def __init__(self, path=""):

        PathObject.__init__(self, path)

        # Assign date and time
        timestamp = os.path.basename(self.path)
        self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(os.path.dirname(self.path))
            self.date, self.time = get_time_and_date(timestamp)
        if (self.date is None) and (self.time is None):
            timestamp = os.path.basename(self.path)
            try:
                self.date, self.time = timestamp.split("_")
            except ValueError:
                self.date, self.time = (None, None)

        self.timestamp = f"{self.date}_{self.time}"

    def in_date_interval(self, min_date=None, max_date=None):
        """Check whether own date falls within an interval."""

        if min_date:
            if self.date < min_date:
                return False
        if max_date:
            if self.date > max_date:
                return False
        return True

    def __eq__(self, other):
        return self.date == other.date and self.time == other.time

    def __ne__(self, other):
        return self.date == other.date or self.time == other.time

    def __lt__(self, other):
        if self.date == other.date:
            return self.time < other.time
        return self.date < other.date

    def __gt__(self, other):
        if self.date == other.date:
            return self.time > other.time
        return self.date > other.date

    def __le__(self, other):
        return self
        if self.date == other.date:
            return self.time < other.time
        return self.date < other.date


class MachineObject(DatedObject):
    """Dated object with an associated machine name."""

    def __init__(self, path=""):
        DatedObject.__init__(self, path)
        self.machine = os.path.basename(os.path.dirname(path))


class ArchiveObject(DatedObject):
    """Dated object associated with multiple files."""

    def __init__(self, path=""):

        DatedObject.__init__(self, path)
        self.files = []
        try:
            filenames = os.listdir(self.path)
        except OSError:
            filenames = []
        for filename in filenames:

            # Disregard hidden files
            if not filename.startswith("."):
                filepath = os.path.join(self.path, filename)
                if not os.path.isdir(filepath):
                    self.files.append(File(path=filepath))

        self.files.sort()


class Scan(ArchiveObject, Image):
    """Object associated with a list of dicom files that are combined into 
    a single scan image."""

    def __init__(self, path=""):

        ArchiveObject.__init__(self, path)
        scan_path = ""
        if path:
            files = glob.glob(f"{path}/*.dcm")
            if files:
                scan_path = files[0]
                Image.__init__(self, scan_path, load=False)

        self.couch_translation, self.couch_rotation = get_couch_shift(scan_path)
        self.scan_position = self.get_scan_position()
        self.voxel_size = get_voxel_size(scan_path)
        self.slice_thickness = self.voxel_size[2]

        self.transform_ijk_to_xyz = get_transform_ijk_to_xyz(self)
        self.image_stack = None
        self.structs = []
        self.machine = self.get_machine()

    def clone(self, image_stack=None):
        """Create a copy of this Scan object with the image stack replaced
        with <image_stack>."""

        clone = Scan()
        clone.imageStack = image_stack
        clone.scan_position = tuple(self.scan_position)
        clone.voxel_size = tuple(self.voxel_size)
        clone.transform_ijk_to_xyz = get_transform_ijk_to_xyz(ct)
        return clone

    def get_machine(self, stations=default_stations):

        machine = None

        if self.files:
            ds = pydicom.read_file(self.files[0].path, force=True)
            try:
                station = ds.StationName
            except BaseException:
                station = None
            if station in stations:
                machine = stations[station]

        return machine

    def get_scan_centre(self):

        self.image_stack = self.get_image_stack()
        nx, ny, nz = self.image_stack.shape

        ijk1 = (0, 0, 0)
        ijk2 = (nx - 1, ny - 1, nz - 1)
        xyz1 = point_ijk_to_xyz(ijk1, self.transform_ijk_to_xyz)
        xyz2 = point_ijk_to_xyz(ijk2, self.transform_ijk_to_xyz)
        xyzc = tuple(0.5 * (xyz1[i] + xyz2[i]) for i in range(3))

        return xyzc

    def get_scan_position(self):

        x0, y0, z0 = (0.0, 0.0, 0.0)
        zs = []
        for file in self.files:
            ds = pydicom.read_file(fp=file.path, force=True)
            try:
                x0, y0, z = [float(xyz) for xyz in ds.ImagePositionPatient]
            except AttributeError:
                continue
            zs.append(z)

        if zs:
            z0 = min(zs)

        return (x0, y0, z0)

    def get_scan_range(self):
        z1, z2 = (0.0, 0.0)
        zs = []
        for fileItem in self.files:
            ds = pydicom.read_file(fp=fileItem.path, force=True)
            try:
                x0, y0, z = [float(xyz) for xyz in ds.ImagePositionPatient]
            except AttributeError:
                continue
            zs.append(z)

        if zs:
            z1 = min(zs)
            z2 = max(zs)

        return (z1, z2)

    def get_image_stack(self, rescale=True, renew=False, reset_voxel_size=True):

        if not (self.image_stack is None or renew):
            return self.image_stack

        # Load pixel array from each file
        files1 = {}
        files2 = {}
        for file in self.files:
            ds = pydicom.read_file(file.path, force=True)

            x0, y0, z0 = self.scan_position
            dx, dy, dz = self.voxel_size
            x1, y1, z1 = [float(v) for v in ds.ImagePositionPatient]
            if rescale and hasattr(ds, "RescaleSlope"):
                files1[z1] = ds.pixel_array * float(ds.RescaleSlope) + float(
                    ds.RescaleIntercept
                )
            else:
                files1[z1] = ds.pixel_array
            if hasattr(ds, "InstanceNumber"):
                z2 = x0 + ds.InstanceNumber * dz
                files2[z2] = files1[z1]

        if len(files1.keys()) == len(self.files):
            files = files1
        elif len(files2.keys()) == len(self.files):
            files = files2
        else:
            print(f"Problem scan: {self.path}")
            print(f"Number of file items: {len(self.files)}")
            print(
                f"Number of values for ImagePositionPatient:" f"{len(files1.keys())}"
            )
            print(f"Number of values for InstanceNumber:" f"{len(files2.keys())}")

        # Find the shape of each slice
        shapes = {}
        for z, data in files.items():
            shape = data.shape
            if shape not in shapes:
                shapes[shape] = []
            shapes[shape].append(z)

        # Ensure that all slices considered have the same shape
        # Find the shape with the most slices
        if len(shapes) > 1:
            max_len = 0
            n1 = len(files)
            zs_to_use = []
            for shape, z_list in shapes.items:
                current_len = len(z_list)
                if current_len > max_len:
                    max_len = current_len
                    zs_to_use = z_list

            zs_to_use.sort()
            n2 = len(zs_to_use)
            print(f"WARNING: '{self.path}' - dropping {n1 - n2} slices")

        # Reset voxel size based on difference between slice positions
        if reset_voxel_size:
            n_slices = len(zs_to_use)
            if n_slices > 1:
                dx, dy, dz = self.voxel_size
                dz = abs(zs_to_use[-1] - zs_to_use[0]) / float(n_slices - 1)
                self.voxel_size = (dx, dy, dz)
                self.transform_ijk_to_xyz = get_transform_ijk_to_xyz(self)

        # Combine arrays for each slice
        arrays = [files[z] for z in zs_to_use]
        if arrays:
            image = np.dstack(arrays)
        else:
            image = None

        return image

    def get_banded_image_stack(
        self,
        rescale=True,
        renew=False,
        reset_voxel_size=True,
        bands={-100: -1024, 100: 0, 1e10: 1024},
    ):
        """Return image array with voxel values quantised based on user-defined
        bands."""

        image = self.get_image_stack(rescale, renew, reset_voxel_size)
        vals = sorted(bands.keys())
        v_band = bands[vals[0]]  # Value for lowest band
        banded_array = np.ones(image.shape) * v_band
        n_bands = len(bands)
        for i in range(1, n_bands):
            v1 = vals[i - 1]
            v2 = vals[i]
            v_band = bands[v2]
            banded_array[(image > v1) & (image <= v2)] = v_band

        return banded_array

    def set_image_stack(self, rescale=True):
        """Set own image_stack parameter by loading the image stack."""

        self.image_stack = self.get_image_stack(rescale, renew=True)

        return None

    def write(self, outDir=".", rescale=True):
        """Write to a dicom file."""

        image = self.get_image_stack()[:, :, ::-1]

        # Get list of the dicom files used to load this scan
        dcm_list = glob.glob(f"{self.path}/*.dcm")
        dcm_list.sort(key=alphanumeric)

        # Create empty output directory
        outdir = fullpath(outdir)
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        distutils.dir_util.mkpath(outdir)

        # Write each slice to a dicom file
        for i, dcm in enumerate(dcm_list):
            ds = pydicom.read_file(dcm)
            pixel_array = image[:, :, i]
            if rescale:
                pixel_array = (pixel_array - float(ds.RescaleIntercept)) \
                        / float(ds.RescaleSlope)
            ds.PixelData = pixel_array.astype(np.uint16).byteswap().tobytes()
            out_path = os.path.join(outdir, os.path.basename(dcm))
            ds.save_as(out_path)

        return


class File(DatedObject):
    """File with an associated date. Files can be sorted based on their 
    filenames."""

    def __init__(self, path=""):
        DatedObject.__init__(self, path)

    def __cmp__(self, other):

        result = DatedObject.__cmp__(self, other)
        if not result:
            self_basename = os.path.basename(self.path)
            other_basename = os.path.basename(other.path)
            basenames = [self_basename, other_basename]
            basenames.sort(key=alphanumeric)
            if basenames[0] == self_basename:
                result = -1
            else:
                result = 1
        return result

    def __eq__(self, other):
        return self.path == other.path

    def __ne__(self, other):
        return self.path != other.path

    def __lt__(self, other):

        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = eval(self_name) < eval(other_name)
        except (NameError, SyntaxError):
            result = self.path < other.path
        return result

    def __gt__(self, other):

        self_name = os.path.splitext(os.path.basename(self.path))[0]
        other_name = os.path.splitext(os.path.basename(other.path))[0]
        try:
            result = eval(self_name) > eval(other_name)
        except (NameError, SyntaxError):
            result = self.path > other.path
        return result


class RtDose(MachineObject):

    def __init__(self, path=""):

        MachineObject.__init__(self, path)

        if not os.path.exists(path):
            return

        ds = pydicom.read_file(path, force=True)

        # Get dose summation type
        try:
            self.summation_type = ds.DoseSummationType
        except AttributeError:
            self.summation_type = None

        # Get slice thickness
        if ds.SliceThickness:
            slice_thickness = float(ds.SliceThickness)
        else:
            slice_thickness = None

        # Get scan position and voxel sizes
        if ds.GridFrameOffsetVector[-1] > ds.GridFrameOffsetVector[0]:
            self.reverse = False
            self.scan_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[0]),
            )
        else:
            self.reverse = True
            self.scan_position = (
                float(ds.ImagePositionPatient[0]),
                float(ds.ImagePositionPatient[1]),
                float(ds.ImagePositionPatient[2] + ds.GridFrameOffsetVector[-1]),
            )
        self.voxel_size = (
            float(ds.PixelSpacing[0]),
            float(ds.PixelSpacing[1]),
            slice_thickness,
        )
        self.transform_ijk_to_xyz = get_transform_ijk_to_xyz(self)
        self.image_stack = None

    def get_image_stack(self, rescale=True, renew=False):

        if self.image_stack is not None and not renew:
            return self.image_stack

        # Load dose array from dicom
        ds = pydicom.read_file(self.path, force=True)
        self.image_stack = np.transpose(ds.pixel_array, (1, 2, 0))

        # Rescale voxel values
        if rescale:
            try:
                rescale_intercept = ds.RescaleIntercept
            except AttributeError:
                rescale_intercept = 0
            self.image_stack = self.image_stack * float(ds.DoseGridScaling) \
                    + float(rescale_intercept)

        if self.reverse:
            self.image_stack[:, :, :] = self.image_stack[:, :, ::-1]

        return self.image_stack


class RtPlan(MachineObject):

    def __init__(self, path=""):

        MachineObject.__init__(self, path)

        ds = pydicom.read_file(path, force=True)

        try:
            self.approval_status = ds.ApprovalStatus
        except AttributeError:
            self.approval_status = None

        try:
            self.n_fraction_group = len(ds.FractionGroupSequence)
        except AttributeError:
            self.n_fraction_group = None

        try:
            self.n_beam_seq = len(ds.BeamSequence)
        except AttributeError:
            self.n_beam_seq = None

        self.n_fraction = None
        self.target_dose = None
        if self.n_fraction_group is not None:
            self.n_fraction = 0
            for fraction in ds.FractionGroupSequence:
                self.n_fraction += fraction.NumberOfFractionsPlanned
                if hasattr(fraction, "ReferencedDoseReferenceSequence"):
                    if self.target_dose is None:
                        self.target_dose = 0.0
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        self.target_dose += dose.TargetPrescriptionDose


class RtStruct(ArchiveObject):
    """Class for loading a structure set associated with a specific scan."""

    def __init__(self, path="", scans=[]):

        ArchiveObject.__init__(self, path)
        self.structure_sets = []

        self.set_scan(scans=scans)
        self.structs = {}
        self.file_index = -1
        self.structure_set_label = ""
        self.series_description = ""

        # Initialise structure sets
        self.structure_sets = []
        for i, f in enumerate(self.files):
            self.structure_sets.append(StructureSet(
                f.path,
                image=self.scan,
                name=f'{f.timestamp}_StructureSet{i + 1}',
                load=False
            ))

    def get_structs(self, file_index=None):

        renew = False
        structs = {}

        if not self.structs:
            renew = True

        if file_index is not None:
            if self.file_index != file_index:
                self.file_index = file_index
                renew = True
        if renew:
            try:
                filename = self.files[self.file_index].path
            except IndexError:
                filename = ""
                structs = self.structs
            if filename:
                structs = get_structs(filename)
                self.structs = structs
                ds = pydicom.read_file(fp=filename, force=True)
                self.structure_set_label = ds.StructureSetLabel
                if hasattr(ds, "SeriesDescription"):
                    self.series_description = ds.SeriesDescription
                else:
                    self.series_description = ""
        else:
            structs = self.structs

        return structs

    def set_scan(self, scans=[]):

        self.scan = Scan()
        structs_scan_dir = os.path.basename(self.path)
        scan_set = False

        # Try matching on path
        for scan in scans:
            if structs_scan_dir == os.path.basename(scan.path):
                self.scan = scan
                add_struct = True
                for struct in scan.structs:
                    if self.path == struct.path:
                        add_struct = False
                if add_struct:
                    scan.structs.append(self)
                scan_set = True
                break

        # If no path match, try matching on timestamp
        if not scan_set:
            for scan in scans:
                if (self.date == scan.date) and (self.time == scan.time):
                    self.scan = scan
                    add_struct = True
                    for struct in scan.structs:
                        if self.path == struct.path:
                            add_struct = False
                    if add_struct:
                        scan.structs.append(self)
                    break

        # Set scans for StructureSet objects
        for ss in self.structure_sets:
            ss.set_image(self.scan) 


class Patient(PathObject):
    """Object associated with a top-level directory whose name corresponds to
    a patient ID, and whose subdirectories contain studies."""

    def __init__(self, path=None, exclude=["logfiles"]):

        start = time.time()

        # Set path and patient ID
        if path is None:
            path = os.getcwd()
        self.path = fullpath(path)
        self.id = os.path.basename(self.path)

        # Find studies
        self.studies = self.get_dated_objects(dtype="Study")
        if not self.studies:
            if os.path.isdir(self.path):
                if os.access(self.path, os.R_OK):
                    subdirs = sorted(os.listdir(self.path))
                    for subdir in subdirs:
                        if subdir not in exclude:
                            self.studies.extend(
                                self.get_dated_objects(
                                    dtype="Study", subdir=subdir
                                )
                            )

        # Get patient demographics
        self.birth_date, self.age, self.sex = self.get_demographics()

    def combined_files(self, dtype, min_date=None, max_date=None):
        """Get list of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified."""

        files = []
        for study in self.studies:
            objs = getattr(study, dtype)
            for obj in objs:
                for file in obj.files:
                    if file.in_date_interval(min_date, max_date):
                        files.append(file)
        files.sort()
        return files

    def combined_files_by_dir(self, dtype, min_date=None, max_date=None):
        """Get dict of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified. The dict keys 
        will be the directories that the files are in."""

        files = {}
        for study in self.studies:
            objs = getattr(study, dtype)
            for object in objs:
                for file in object.files:
                    if file.in_date_interval(min_date, max_date):
                        folder = os.path.dirname(fullpath(file.path))
                        if folder not in files:
                            files[folder] = []
                        files[folder].append(file)

        for folder in files:
            files[folder].sort()

        return files

    def combined_objs(self, dtype):
        """Get list of all objects of a given data type <dtype> associated
        with this patient."""

        all_objs = []
        for study in self.studies:
            objs = getattr(study, dtype)
            if objs:
                all_objs.extend(objs)
        all_objs.sort()
        return all_objs

    def first_struct_in_interval(self, dtype=None, min_date=None, 
                                 max_date=None):
        """Get the first structure set in a given date interval for a 
        given data type <dtype>."""

        structs = self.combined_objs(dtype)
        first = None
        for struct in structs:
            for file in struct.files:
                if file.in_date_interval(min_date, max_date):
                    if first:
                        if file < first.files[0]:
                            first.files = [file]
                            first.scan = struct.scan
                    else:
                        first = RtStruct(path=struct.path)
                        first.files = [file]
                        first.scan = struct.scan
        return first

    def get_demographics(self):
        """Return patient's birth date, age, and sex."""

        info = {"BirthDate": None, "Age": None, "Sex": None}

        # Find an object from which to extract the info
        obj = None
        if self.studies:
            study = self.studies[-1]
            if study.ct_scans:
                im = study.ct_scans[-1]
            elif study.mvct_scans:
                obj = study.mvct_scans[-1]
            elif study.ct_scans:
                obj = study.ct_scans[-1]
            elif study.mvct_scans:
                obj = study.mvct_scans[-1]

        # Read demographic info from the object
        if obj and obj.files:
            ds = pydicom.read_file(fp=obj.files[-1].path, force=True)
            for key in info:
                for prefix in ["Patient", "Patients"]:
                    attr = f"{prefix}{key[0].upper()}{key[1:]}"
                    if hasattr(ds, attr):
                        info[key] = getattr(ds, attr)
                        break

        # Ensure sex is uppercase and single character
        if info["Sex"]:
            info["Sex"] = info["Sex"][0].upper()

        return info["BirthDate"], info["Age"], info["Sex"]

    def getSubdirStudyList(self, subdir=""):

        subdirStudyList = []
        for study in self.studyList:
            if subdir == study.subdir:
                subdirStudyList.append(study)

        subdirStudyList.sort()

        return subdirStudyList

    def last_in_interval(self, dtype=None, min_date=None, max_date=None):
        """Get the last object of a given data type <dtype> in a given
        date interval."""

        files = self.combined_files(dtype)
        last = None
        files.reverse()
        for file in files:
            if file.in_date_interval(min_date, max_date):
                last = file
                break
        return last

    def last_struct_in_interval(self, dtype=None, min_date=None, max_date=None):
        """Get the last structure set in a given date interval for a 
        given data type <dtype>."""

        structs = self.combined_objs(dtype)
        last = None
        for struct in structs:
            for file in struct.files:
                if file.in_date_interval(min_date, max_date):
                    if last:
                        if file > last.files[0]:
                            last.files = [file]
                            last.scan = struct.scan
                    else:
                        last = RtStruct(path=struct.path)
                        last.files = [file]
                        last.scan = struct.scan
        return last


class Study(DatedObject):

    def __init__(self, path=""):

        DatedObject.__init__(self, path)

        # Load RT plans, CT and MR scans, all doses, and CT structure sets
        self.plans = self.get_plan_data(dtype="RtPlan", subdir="RTPLAN")
        self.ct_scans = self.get_dated_objects(dtype="Scan", subdir="CT")
        self.mr_scans = self.get_dated_objects(dtype="Scan", subdir="MR")
        self.doses = self.get_plan_data(
            dtype="RtDose",
            subdir="RTDOSE",
            exclude=["MVCT", "CT"],
            scans=self.ct_scans
        )
        self.ct_structs = self.get_structs(subdir="RTSTRUCT/CT", 
                                           scans=self.ct_scans)

        # Look for HD CT scans and add to CT list
        ct_hd = self.get_dated_objects(dtype="CT", subdir="CT_HD")
        ct_hd_structs = self.get_structs(subdir="RTSTRUCT/CT_HD", scans=ct_hd)
        if ct_hd:
            self.ct_scans.extend(ct_hd)
            self.ct_scans.sort()
        if ct_hd_structs:
            self.ct_structs.extend(ct_hd_structs)
            self.ct_structs.sort()

        # Load CT-specific RT doses
        self.ct_doses = self.get_plan_data(
            dtype="RtDose", subdir="RTDOSE/CT", scans=self.ct_scans
        )
        self.ct_doses = self.correct_dose_scan_position(self.ct_doses)

        # Load MVCT images, doses, and structs
        self.mvct_scans = self.get_dated_objects(dtype="Scan", subdir="MVCT")
        self.mvct_doses = self.get_plan_data(
            dtype="RtDose", subdir="RTDOSE/MVCT", scans=self.mvct_scans
        )
        self.mvct_doses = self.correct_dose_scan_position(self.mvct_doses)
        self.mvct_structs = self.get_structs(
            subdir="RTSTRUCT/MVCT", scans=self.mvct_scans
        )

        # Set description
        self.description = self.get_description()

    def correct_dose_scan_position(self, doses=[]):
        """Correct for scan positions from CheckTomo being offset by one slice
        relative to scan positions."""

        for dose in doses:
            dx, dy, dz = dose.voxel_size
            x0, y0, z0 = dose.scan_position
            dose.scan_position = (x0, y0, z0 + dz)
        return doses

    def get_machine_sublist(self, dtype="", machine="", ignore_case=True):
        """Get list of doses or treatment plans corresponding to a specific
        machine."""

        sublist = []
        if dtype.lower() in ["plan", "rtplan"]:
            objs = self.plans
        elif dtype.lower() in ["dose", "rtdose"]:
            objs = self.doses
        else:
            objs = []

        if ignore_case:
            for obj in objs:
                if objs.machine.lower() == machine.lower():
                    sublist.append(obj)
        else:
            for obj in objs:
                if objs.machine == machine:
                    sublist.append(object)
        return sublist

    def get_mvct_selection(self, mvct_dict={}, min_delta_hours=0.0):
        """Get a selection of MVCT scans which were taken at least 
        <min_delta_hours> apart. <mvct_dict> is a dict where the keys are 
        patient IDs, and the paths are directory paths from which to load scans
        for that patient."""

        # Find scans meeting the time separation requirement
        if min_delta_hours > 0:
            mvct_scans = get_time_separated_objects(
                self.mvct_scans, min_delta_hours)
        else:
            mvct_scans = self.mvct_scans

        # Find scans matching the directory requirement
        selected = []
        patient_id = self.get_patient_id()
        if patient_id in mvct_dict:

            # Get all valid directories for this patient
            valid_dirs = [fullpath(path) for path in mvct_dict[patient_id]]

            # Check for scans matching that directory requirement
            for mvct in mvct_scans:
                mvct_dir = os.path.dirname(mvct.files[-1].path)
                if fullpath(mvct_dir) in valid_dirs:
                    selected.append(mvct)

        # Otherwise, just return all scans for this patient
        else:
            selection = mvct_scans

        return selection

    def get_patient_id(self):
        patient_id = os.path.basename(os.path.dirname(self.path))
        return patient_id

    def get_plan_data(
        self, dtype="RtPlan", subdir="RTPLAN", exclude=[], scans=[]
    ):
        """Get list of RT dose or plan objects specified by dtype="RtDose" or 
        "RtPlan" <dtype>, respectively) by searching within a given directory, 
        <subdir> (or within the top level directory of this Study, if 
        <subdir> is not provided).

        Subdirectories with names in <exclude> will be ignored.

        Each dose-like object will be matched by timestamp to one of the scans 
        in <scans> (which should be a list of DatedObjects), if provided."""

        doses = []

        # Get initial path to search
        if subdir:
            path1 = os.path.join(self.path, subdir)
        else:
            path1 = self.path

        # Look for subdirs up to two levels deep from initial dir
        subdirs = []
        if os.path.isdir(path1):

            # Search top level of dir
            path1_subdirs = os.listdir(path1)
            for item1 in path1_subdirs:

                if item1 in exclude:
                    continue
                path2 = os.path.join(path1, item1)
                n_sub_subdirs = 0

                # Search any directories in the top level dir
                if os.path.isdir(path2):
                    path2_subdirs = os.listdir(path2)
                    for item2 in path2_subdirs:
                        path3 = os.path.join(path2, item2)

                        # Search another level (subdir/item1/item2/*)
                        if os.path.isdir(path3):
                            n_sub_subdirs += 1
                            if subdir:
                                subdirs.append(os.path.join(
                                    subdir, item1, item2))
                            else:
                                subdirs.append(item1, item2)

                if not n_sub_subdirs:
                    if subdir:
                        subdirs = [os.path.join(subdir, item1)]
                    else:
                        subdirs = [item1]

                for subdir_item in subdirs:
                    doses.extend(
                        self.get_dated_objects(
                            dtype=dtype, subdir=subdir_item
                        )
                    )

        # Assign dose-specific properties
        if dtype == "RtDose":
            new_doses = []
            for dose in doses:

                # Search for scans with matching timestamp
                timestamp = os.path.basename(os.path.dirname(dose.path))
                if scans:
                    try:
                        dose.date, dose.time = timestamp.split("_")
                        scan = get_dated_obj(scans, dose)
                        dose.machine = scan.machine
                    except BaseException:
                        scan = scans[-1]
                        dose.date = scan.date
                        dose.time = scan.time

                    dose.timestamp = f"{dose.date}_{dose.time}"
                    dose.scan = scan

                dose.couch_translation, dose.couch_rotation \
                        = get_couch_shift(dose.path)
                # WARNING!
                #     Couch translation third component (y) inverted with
                #     respect to CT scan
                # WARNING!
                new_doses.append(dose)
            doses = new_doses

        doses.sort()
        return doses

    def get_plan_dose(self):

        plan_dose = None
        dose_dict = {}

        # Group doses by summation type
        for dose in self.doses:
            if dose.summationType not in dose_dict:
                dose_dict[dose.summationType] = []
            dose_dict[dose.summationType].append(dose)
        for st in dose_dict:
            dose_dict[st].sort()

        # "PLAN" summation type: just take the newest entry
        if "PLAN" in dose_dict:
            plan_dose = dose_dict["PLAN"][-1]
            plan_dose.imageStack = plan_dose.getImageStack()

        else:
            
            # Get fraction froup and beam sequence
            if self.plans:
                n_frac_group = self.plans[-1].nFractionGroup
                n_beam_seq = self.plans[-1].nBeamSequence
            else:
                n_frac_group = None
                n_beam_seq = None

            # Sum over fractions
            if "FRACTION" in dose_dict:
                if len(dose_dict["FRACTION"]) == n_frac_group:
                    
                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = doseDict["FRACTION"][0]

                    # Sum fractions
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, "FRACTION")

            # Sum over beams
            elif "BEAM" in sum_type:
                if len(dose_dict["BEAM"]) == n_beam_seq:

                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = dose_dict["BEAM"][0]

                    # Sum beams
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, "BEAM")

        return plan_dose

    def get_structs(self, subdir="", scans=[]):
        """Make list of RtStruct objects found within a given subdir, and
        set their associated scan objects."""

        structs = self.get_dated_objects(dtype="RtStruct", subdir=subdir)
        for struct in structs:
            struct.set_scan(scans=scans)
        return structs

    def get_description(self):
        """Load a study description."""

        # Find an object from which to extract description
        obj = None
        if self.ct_scans:
            obj = self.ct_scans[-1]
        elif self.mvct_scans:
            obj = self.mvct_scans[-1]
        elif self.ct_structs:
            obj = self.ct_structs[-1]
        elif self.mvct_structs:
            obj = self.mvct_structs[-1]

        description = ""
        if obj:
            if obj.files:
                scan_path = obj.files[-1].path
                ds = pydicom.read_file(fp=scan_path, force=True)
                if hasattr(ds, "StudyDescription"):
                    description = ds.StudyDescription

        return description

    def sum_dose_plans(self, dose_dict={}, sum_type=""):
        """Sum over doses using a given summation type."""

        plan_dose = None
        if sum_type in dose_dict:
            dose = dose_dict[sum_type].pop()
            plan_dose = RtDose()
            plan_dose.machine = dose.machine
            plan_dose.path = dose.path
            plan_dose.subdir = dose.subdir
            plan_dose.date = dose.date
            plan_dose.time = dose.time
            plan_dose.timestamp = dose.timestamp
            plan_dose.summationType = "PLAN"
            plan_dose.scanPosition = dose.scanPosition
            plan_dose.reverse = dose.reverse
            plan_dose.voxelSize = dose.voxelSize
            plan_dose.transform_ijk_to_xyz = dose.transform_ijk_to_xyz
            plan_dose.imageStack = dose.getImageStack()
            for dose in dose_dict[sum_type]:
                plan_dose.imageStack += dose.getImageStack()

        return plan_dose


def alphanumeric(in_str=""):
    """Function that can be passed as value for list sort() method
    to have alphanumeric (natural) sorting"""

    import re

    elements = []
    for substr in re.split("(-*[0-9]+)", in_str):
        try:
            element = int(substr)
        except BaseException:
            element = substr
        elements.append(element)
    return elements


def applyCouchShifts(xyz=None, translation=None, rotation=None, reverse=False):

    if xyz is None:
        return (None, None, None)

    xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    if reverse:
        x, y, z = applyRotation(xyz, rotation, True)
        x, y, z = applyTranslation((x, y, z), translation, True)
    else:
        x, y, z = applyTranslation(xyz, translation)
        x, y, z = applyRotation((x, y, z), rotation)

    return (x, y, z)


def applyRotation(xyz=None, rotation=None, reverse=False):

    if xyz is None:
        return (None, None, None)

    x, y, z = xyz

    try:
        if None in rotation:
            rotation = None
    except BaseException:
        rotation = None

    if rotation is not None:
        pitch, yaw, roll = rotation
        x0 = float(x)
        y0 = float(y)
        theta = math.pi * roll / 180.0
        if reverse:
            theta = -theta
        x = x0 * math.cos(theta) - y0 * math.sin(theta)
        y = x0 * math.sin(theta) + y0 * math.cos(theta)

    return (x, y, z)


def applyTranslation(xyz=None, translation=None, reverse=False):

    if xyz is None:
        return (None, None, None)

    x, y, z = xyz

    try:
        if None in translation:
            translation = None
    except BaseException:
        translation = None

    if translation is not None:
        v1, v2, v3 = translation
        if reverse:
            v1, v2, v3 = (-v1, -v2, -v3)
        x = x + v1
        y = y - v3
        z = z + v2

    return (x, y, z)


def checkScanStructDict(scanStructDict={}, mode="one-to-one"):

    logger = getLogger("checkScanStructDict")
    checkModeList = ["one-to-one", "one-or-more"]

    failList = []

    if mode not in checkModeList:
        logger.warning(f"Check mode '{mode}' not known")
        logger.warning(f"Check mode must be in '{checkModeList}'")

    for scanDate in scanStructDict.keys():
        nStruct = len(scanStructDict[scanDate]["structList"])
        if "one-to-one" == mode:
            if 1 != nStruct:
                failList.append(scanDate)
        if "one-or-more" == mode:
            if 0 == nStruct:
                failList.append(scanDate)

    failList.sort()

    return failList


def convertNPixel(ct=None, ctRef=None, nRef=0, ixyz=0, outFloat=False):

    if nRef > 0:
        n = float(nRef) * ctRef.voxelSize[ixyz] / ct.voxelSize[ixyz]
    else:
        n = 0.0

    if not outFloat:
        n = int(n) + 1

    return n


def fitCircle(pointList=[]):
    """
    Fit circle to points specified as list of ( x, y ) tuples
    Code addapted from implementation 2b of:
    http://wiki.scipy.org/Cookbook/Least_Squares_Circle
    """

    from scipy import optimize

    # Convert input list of (x, y) to two numpy arrays
    xList = []
    yList = []
    for point in pointList:
        xList.append(float(point[0]))
        yList.append(float(point[1]))

    xArray = np.r_[xList]
    yArray = np.r_[yList]

    # Determine mean ( x, y ) values to provide first approximation to
    # circle centre
    xyMean = (np.mean(xArray), np.mean(yArray))

    def calc_R(xc, yc):
        """Calculate point distances from estimated circle
        centre (xc,yc)"""
        return np.sqrt((xArray - xc) ** 2 + (yArray - yc) ** 2)

    def f_2b(c):
        """Calculate point residuals about estimated circle radius"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """
         Jacobian of f_2b
        The axis corresponding to derivatives must be
        coherent with the col_deriv option of leastsq
        """
        xc, yc = c
        df2b_dc = np.empty((len(c), xArray.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - xArray) / Ri  # dR/dxc
        df2b_dc[1] = (yc - yArray) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center_2b, ier = optimize.leastsq(f_2b, xyMean, Dfun=Df_2b, col_deriv=True)
    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(xc_2b, yc_2b)
    R_2b = Ri_2b.mean()

    return (xc_2b, yc_2b, R_2b)


def fitSphere(pointList=[], p0=[0.0, 0.0, 0.0, 1.0]):
    """
    Fit sphere to points specified as list of ( x, y, z ) tuples
    Code addapted from top answer at:
    http://stackoverflow.com/questions/15785428/how-do-i-fit-3d-data
    """

    from scipy import optimize

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords.T
        return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    def errfunc(p, x):
        return fitfunc(p, x) - p[3]

    coords = np.array(pointList)
    p1, flag = optimize.leastsq(errfunc, p0, args=(coords,))

    return p1


def frange(*args):

    if 0 == len(args):
        raise TypeError("frange expected at least 1 arguments, got 0")
    elif 3 < len(args):
        raise TypeError(f"frange expected at most 3 arguments, got {len(args)}")
    else:
        start = 0.0
        stop = args[0]
        step = 1.0
        if len(args) >= 2:
            start = args[0]
            stop = args[1]
        if len(args) == 3:
            step = args[2]

        r = start
        while r < stop:
            yield r
            r = r + step


def fullpath(path=""):
    """Evaluate full path, expanding '~', environment variables, and 
    symbolic links."""

    expanded = ""
    if path:
        tmp = os.path.expandvars(path.strip())
        tmp = os.path.abspath(os.path.expanduser(tmp))
        expanded = os.path.realpath(tmp)
    return expanded


def getCoordinateArrays(ct=None):

    x, y, z = ct.scanPosition
    dx, dy, dz = ct.voxelSize

    if 2 == len(ct.getImageStack().shape):
        ny, nx = ct.getImageStack().shape
        ct.imageStack = ct.getImageStack().reshape(ny, nx, 1)

    ny, nx, nz = ct.getImageStack().shape

    try:
        xArray = np.linspace(x, x + (nx - 1) * dx, nx)
    except TypeError:
        xArray = None
    try:
        yArray = np.linspace(y, y + (ny - 1) * dy, ny)
    except TypeError:
        yArray = None
    try:
        zArray = np.linspace(z, z + (nz - 1) * dz, nz)
    except TypeError:
        zArray = None

    return (xArray, yArray, zArray)


def get_couch_shift(path=""):
    """Extract couch translation and rotation vectors from a dicom file."""

    translation = None
    rotation = None

    # Extract translation and rotation strings from dicom
    if os.path.exists(path):
        ds = pydicom.read_file(path, force=True)
        stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            translation = ds[0x0099, 0x1011]
        except KeyError:
            pass
        try:
            rotation = ds[0x0099, 0x1012]
        except KeyError:
            pass
        sys.stdout.close()
        sys.stdout = stdout

    # Parse translation string
    translation_list = [None, None, None]
    if translation:

        # Split string into list
        if isinstance(translation.value, str):
            translation_list = translation.value.split("\\")
        elif isinstance(translation.value, bytes):
            translation_list = translation.value.decode().split("\\")
        else:
            translation_list = list(translation.value)

        # Convert to floats
        if len(translation_list) == 3:
            for i in range(len(translation_list)):
                try:
                    translation_list[i] = float(translation_list[i])
                except ValueError:
                    translation_list[i] = None
                    break
        else:
            translation_list = [None, None, None]

    # Parse rotation string
    rotation_list = [None, None, None]
    if rotation:

        # Split string into list
        if isinstance(rotation.value, str):
            rotation_list = rotation.value.split("\\")
        elif isinstance(rotation.value, bytes):
            rotation_list = rotation.value.decode().split("\\")
        else:
            rotation_list = list(rotation.value)

        # Convert to floats
        if len(rotation_list) == 3:
            for i in range(len(rotation_list)):
                try:
                    rotation_list[i] = float(rotation_list[i])
                except ValueError:
                    rotation_list[i] = None
                    break
        else:
            rotation_list = [None, None, None]

    return (tuple(translation_list), tuple(translation_list))


def get_dated_object(objs=[], timestamp=""):
    """For a given list of objects <obj>, find the first object that matches
    a given timestamp <timestamp> (which can either be a string in format
    date_time, or any object with date and time attributes)."""

    # Convert timestamp to consistent format
    if hasattr(timestamp, "date") and hasattr(timestamp, "time"):
        timestamp = f"{timestamp.date}_{timestamp.time}"

    # Find object with matching timestamp
    dated_obj = None
    if objs and timestamp:
        for obj in objs:
            if f"{obj.date}_{obj.time}" == timestamp:
                dated_obj = obj
                break

    return dated_obj


def getDatedObjectList(objectList=[], timestamp=""):

    datedObjectList = []

    if hasattr(timestamp, "date") and hasattr(timestamp, "time"):
        timestamp = f"{timestamp.date}_{timestamp.time}"

    if objectList and timestamp:
        for tmpObject in objectList:
            tmpTimestamp = f"{tmpObject.date}_{tmpObject.time}"
            if tmpTimestamp == timestamp:
                datedObjectList.append(tmpObject)

    return datedObjectList


def get_dicom_sequence(ds=None, basename=""):

    sequence = []

    for suffix in ["Sequence", "s"]:
        attribute = f"{basename}{suffix}"
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break

    return sequence


def getDirList(topDir="", timestamp=False):
    """Obtain list of all subdirectories below a given top-level director"""
    dirList = []
    if os.path.isdir(topDir):
        tmpList = os.listdir(topDir)
        for fileNow in tmpList:
            if timestamp:
                if not is_timestamp(fileNow):
                    continue
            pathNow = os.path.join(topDir, fileNow)
            if os.path.isdir(pathNow):
                dirList.append(pathNow)
    dirList.sort()
    return dirList


def getJsonDict(pathList=[]):

    jsonDict = {}
    for jsonPath in pathList:
        jsonFile = open(jsonPath)
        jsonData = json.load(jsonFile)
        if isinstance(jsonData, type([])):
            for jsonItem in jsonData:
                jsonDict.update(jsonItem)
        else:
            jsonDict.update(jsonData)

    return jsonDict


def getLabel(path=""):

    label = pydicom.read_file(path, force=True).StructureSetLabel.strip()
    if 1 + label.find("JES"):
        if 1 + label.find("contour"):
            label = "Jessica2"
        else:
            label = "Jessica1"

    if 1 + label.find("ProSoma"):
        label = "Gill"

    if 1 + label.find("Automated"):
        label = "robot"

    return label


def getLogger(name=""):
    import logging

    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def getLongestSequenceBounds(valueList=[]):

    valueList.sort()
    seqListList = []
    valueLast = valueList[0] - 2
    iseq = -1

    for value in valueList:
        if value != valueLast + 1:
            iseq = iseq + 1
            seqListList.append([])
        seqListList[iseq].append(value)
        valueLast = value

    seqMinAndMax = (None, None)
    if seqListList:
        seqMaxList = max(seqListList, key=len)
        seqMinAndMax = (seqMaxList[0], seqMaxList[-1])

    return seqMinAndMax


def getRoiObserver(roi="", structDict={}):

    roiObserver = None
    for roiKey in structDict.keys():
        if re.search(roi, structDict[roiKey]["name"], re.I):
            roiObserver = structDict[roiKey]["name"]
            break

    return roiObserver


def getScanStructDict(
    topdir="",
    patient="",
    study="",
    scan="CT",
    struct="RTSTRUCT",
    structSuffixList=[".dcm"],
):
    import os

    logger = getLogger(name="getScanStructDict")
    studyDir = os.path.join(topdir, patient, study)
    scanDir = os.path.join(studyDir, scan)
    structDir = os.path.join(studyDir, struct, scan)

    scanStructDict = {}

    scanDirOK = os.path.isdir(scanDir)
    structDirOK = os.path.isdir(structDir)

    if not (scanDirOK):
        logger.warning(f"Scan directory '{scanDir}' not found")
    if not (structDirOK):
        logger.warning(f"Structure-set directory '{structDir}' not found")

    if not (scanDirOK or structDirOK):
        return scanStructDict

    if scanDirOK:
        scanDateList = os.listdir(scanDir)
    elif structDirOK:
        scanDateList = os.listdir(structDir)
    scanDateList.sort()
    for scanDate in scanDateList:
        structDate = "_".join(scanDate.split("_")[0:2])
        scanDateDir = os.path.join(scanDir, scanDate)
        structDateDir = os.path.join(structDir, structDate)
        if os.path.isdir(scanDateDir) or os.path.isdir(structDateDir):
            scanStructDict[scanDate] = {}

            scanStructDict[scanDate]["scanDir"] = None
            if os.path.isdir(scanDateDir):
                scanStructDict[scanDate]["scanDir"] = scanDateDir

            scanStructDict[scanDate]["structList"] = []
            if os.path.isdir(structDateDir):
                structFileList = sorted(os.listdir(structDateDir))
                for structFile in structFileList:
                    suffix = os.path.splitext(structFile)[1]
                    if suffix in structSuffixList:
                        structPath = os.path.join(structDateDir, structFile)
                        scanStructDict[scanDate]["structList"].append(structPath)

    return scanStructDict


def get_separated_timestamp_dict(timestamp_dict={}, min_delta_hours=4.0):
    """Take a dict where the keys are timestamps, and reduce down to only
    elements where the timestamps are at least <min_delta_hours> apart."""

    timestamps = timestamp_dict.keys()
    separated_timestamps = get_separated_timestamp_list(
        timestamps, min_delta_hours)
    return {ts: timestamp_dict[ts] for ts in separated_timestamps}


def get_separated_timestamp_list(timestamps=[], min_delta_hours=4.0):
    """Take a list of timestamps and reduce down to only elements where the 
    timestamps are at least <min_delta_hours> apart."""

    # Find all timestamps containing a valid date and time
    checked = []
    for timestamp in timestamps:
        date, time = get_time_and_date(os.path.basename(timestamp))
        if not (date is None or time is None):
            checked.append("_".join([date, time]))

    # Find timestamps separated by <min_delta_hours>
    separated = []
    if checked:
        checked.sort()
        for i in range(len(checked) - 1):
            timestamp1 = checked[i]
            timestamp2 = checked[i + 1]
            delta_seconds = get_timestamp_difference_seconds(
                timestamp1, timestamp2)
            delta_hours = seconds / (60.0 * 60.0)
            if delta_hours > min_delta_hours:
                separated.append(timestamp1)
        separated.append(checked[-1])

    return separated


def get_study_data_dict(study=None, requirements=[]):
    """
    Retreive a study's non-empty lists of data objects.

    Arguments:
        study        -- voxtox.utility.Study object
        requirements -- list of required attributes for data objects;
                        for example: requirements=["imageStack"] will select
                        objects with image data,
                        requirements=["structureSetLabel"]
                        will select structure-set objects
    """

    # If requirements passed as a string, convert to single-item list.
    if isinstance(requirements, str):
        requirements = [requirements]

    # Examine all attributes of the Study object, and identify the data lists.
    data_dict1 = {}
    for attribute in dir(study):
        if "list" in attribute.lower():
            item_list = getattr(study, attribute)
            if isinstance(item_list, list) and item_list:
                data_type = attribute.lower().split("list")[0].lower()
                data_dict1[data_type] = item_list

    # Filter on requirements.
    data_dict2 = {}
    for key in data_dict1:
        item_list = data_dict1[key]
        item_list2 = []
        if requirements:
            for item in item_list:
                add_item = True
                for requirement in requirements:
                    if not hasattr(item, requirement):
                        add_item = False
                        break
                if add_item:
                    item_list2.append(item)

        if item_list2:
            data_dict2[key] = item_list2

    return data_dict2


def get_time_and_date(timestamp=""):

    timeAndDate = (None, None)
    if is_timestamp(timestamp):
        valueList = os.path.splitext(timestamp)[0].split("_")
        valueList = [value.strip() for value in valueList]
        if valueList[0].isalpha():
            timeAndDate = tuple(valueList[1:3])
        else:
            timeAndDate = tuple(valueList[0:2])
    else:
        i1 = timestamp.find("_")
        i2 = timestamp.rfind(".")
        if (-1 != i1) and (-1 != i2):
            bitstamp = timestamp[i1 + 1 : i2]
            if is_timestamp(bitstamp):
                timeAndDate = tuple(bitstamp.split("_"))

    return timeAndDate


def get_time_separated_objects(in_list=[], min_delta_hours=4.0):
    """Parse a list of objects and return only those with time separations 
    greater than min_delta_hours."""

    timestamps = {obj.timestamp: obj for obj in in_list}
    separated_timestamps = get_separated_timestamp_dict(timestamps, 
                                                        min_delta_hours)
    return sorted(separated_timestamps.values())


def get_transform_ijk_to_xyz(obj=None):
    """Get a matrix for transforming a set of indices (ijk) to a spatial 
    position (xyz) for a given object that has a scan position and voxel 
    size."""

    x0, y0, z0 = obj.scan_position
    dx, dy, dz = obj.voxel_size
    if not any([var is None for var in [x0, y0, z0, dx, dy, dz]]):
        transform = np.array(
            [
                [dx, 0.0, 0.0, x0],
                [0.0, dy, 0.0, y0],
                [0.0, 0.0, dz, z0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        transform = None

    return transform


def getTransformixPointDict(filepath=""):
    def getCoordinateTuple(inString=""):
        coordinateTuple = tuple([eval(x) for x in inString.split()[3:6]])
        return coordinateTuple

    inFile = open(filepath)
    lineList = inFile.readlines()
    inFile.close()
    pointDict = {}
    for line in lineList:
        elementList = line.split(";")
        point = eval(elementList[0].split()[1])
        pointDict[point] = {}
        pointDict[point]["InputIndex"] = getCoordinateTuple(elementList[1])
        pointDict[point]["InputPoint"] = getCoordinateTuple(elementList[2])
        pointDict[point]["OutputIndexFixed"] = getCoordinateTuple(elementList[3])
        pointDict[point]["OutputPoint"] = getCoordinateTuple(elementList[4])
        pointDict[point]["Deformation"] = getCoordinateTuple(elementList[5])

    return pointDict


def get_voxel_size(in_data=""):
    """Find voxel size from a path/list of paths to dicom file(s)."""

    dx, dy, dz = (None, None, None)
    if isinstance(in_data, list):
        path = in_data[0]
    else:
        path = in_data
    if os.path.exists(path):
        ds = pydicom.read_file(path, force=True)
        if hasattr(ds, "PixelSpacing"):
            dx, dy = ds.PixelSpacing
            dx = float(dx)
            dy = float(dy)
        if hasattr(ds, "SliceThickness"):
            try:
                dz = float(ds.SliceThickness)
            except TypeError:
                dz = None
    return (dx, dy, dz)


def groupConsecutiveIntegers(inList=[]):

    # Based on recipe at:
    # https://docs.python.org/2.6/library/itertools.html#examples
    outList = []
    inList.sort()
    for k, g in itertools.groupby(enumerate(inList), lambda i_x: i_x[0] - i_x[1]):
        outList.append(map(operator.itemgetter(1), g))

    return outList


def iround(value=0.0):

    floatValue = float(value)
    if floatValue > 0.0:
        intValue = int(floatValue + 0.5)
    else:
        intValue = int(floatValue - 0.5)

    return intValue


def is_timestamp(testString=""):
    timestamp = True
    valueList = os.path.splitext(testString)[0].split("_")
    valueList = [value.strip() for value in valueList]
    """
  if len( valueList ) in [ 3, 4 ]:
    if valueList[ 0 ].isalpha() and valueList[ 1 ].isdigit() \
      and valueList[ 2 ].isdigit():
      valueList = valueList[ 1 : ]
    elif valueList[ 0 ].isdigit() and valueList[ 1 ].isdigit() \
      and valueList[ 2 ].isalpha():
      valueList = valueList[ : 2 ]
    elif valueList[ 0 ].isdigit() and valueList[ 1 ].isdigit() \
      and valueList[ 2 ].isalnum():
      valueList = valueList[ : 2 ]
  """
    if len(valueList) > 2:
        if valueList[0].isalpha() and valueList[1].isdigit() and valueList[2].isdigit():
            valueList = valueList[1:3]
        elif valueList[0].isdigit() and valueList[1].isdigit():
            valueList = valueList[:2]
        elif valueList[0].isdigit() and valueList[1].isdigit():
            valueList = valueList[:2]
    if len(valueList) != 2:
        timestamp = False
    else:
        for value in valueList:
            if not value.isdigit():
                timestamp = False
                break
    return timestamp


def get_structs(filename="", rois=None, first_only=False):
    """Returns a dictionary of structures (ROIs)."""

    # Load contents of input DICOM file
    ds = pydicom.read_file(filename, force=True)

    structures = {}

    # Adapted from dicompyler:
    # https://github.com/dicompyler/dicompyler-core
    # Determine whether this is RT Structure Set file
    if not (ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"):
        return structures

    # Locate the name and number of each ROI
    structure_set_roi_sequence = get_dicom_sequence(ds, "StructureSetROI")
    for item in structure_set_roi_sequence:
        data = {}
        number = int(item.ROINumber)
        data["id"] = number
        data["name"] = item.ROIName
        structures[number] = data

    # Determine the type of each structure (PTV, organ, external, etc)
    rt_roi_observations_sequence = get_dicom_sequence(ds, "RTROIObservations")
    for item in rt_roi_observations_sequence:
        number = item.ReferencedROINumber
        if number in structures:
            structures[number]["type"] = item.RTROIInterpretedType

    # Coordinate data of each ROI is stored within ROIContourSequence
    roi_contour_sequence = get_dicom_sequence(ds, "ROIContour")
    for roi in roi_contour_sequence:
        number = roi.ReferencedROINumber

        # Generate a random color for the current ROI
        structures[number]["color"] = np.array(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            dtype=int,
        )
        # Get the RGB color triplet for the current ROI if it exists
        if "ROIDisplayColor" in roi:
            # Make sure the color is not none
            if not (roi.ROIDisplayColor is None):
                color = roi.ROIDisplayColor
            # Otherwise decode values separated by forward slashes
            else:
                value = roi[0x3006, 0x002A].repval
                color = value.strip("'").split("/")
            # Try to convert the detected value to a color triplet
            try:
                structures[number]["color"] = np.array(color, dtype=int)
            # Otherwise fail and fallback on the random color
            except ValueError:
                pass

        # Determine whether the ROI has any contours present
        contour_sequence = get_dicom_sequence(roi, "Contour")
        if contour_sequence:
            structures[number]["empty"] = False

            # Locate the contour sequence for each referenced ROI
            planes = {}
            for c in contour_sequence:
                # For each plane, initialize a new plane dict
                plane = dict()

                # Determine all the plane properties
                plane["type"] = c.ContourGeometricType
                try:
                    n_contour_points = c.NumberOfContourPoints
                except AttributeError:
                    n_contour_points = c.NumberofContourPoints
                plane["num_points"] = int(n_contour_points)
                plane["data"] = [
                    c.ContourData[i : i + 3] for i in range(0, len(c.ContourData), 3)
                ]

                if "ContourImageSequence" in c:
                    plane["UID"] = c.ContourImageSequence[0].ReferencedSOPInstanceUID

                # Add each plane to the planes dict
                # of the current ROI
                # z = str(round(plane["data"][0][2], 2)) + '0'
                z = f"{plane['data'][0][2]:.2f}".replace("-0", "0")
                if z not in planes:
                    planes[z] = []
                planes[z].append(plane)

            # Add contour data to the structure
            structures[number]["planes"] = planes

            # Calculate slice thickness
            planes_list = []
            for z in planes.keys():
                planes_list.append(float(z))
                planes_list.sort()

            # Determine the thickness
            thickness = 10000
            for n in range(0, len(planes_list)):
                if n > 0:
                    new_thickness = planes_list[n] - planes_list[n - 1]
                    if new_thickness < thickness:
                        thickness = new_thickness

            # If the thickness was not detected, set it to 0
            if thickness == 10000:
                thickness = 0

            # Add thickness to structure info
            structures[number]["thickness"] = thickness

        else:
            structures[number]["empty"] = True

    if rois is not None:
        structures = get_struct_dict_selection(rois, structures, first_only)

    return structures


def get_struct_dict_selection(struct_names="", structs={}, first_only=False):
    """Returns structures in a list of ROIs."""

    selection = {}

    # Process structure names into lowercase list
    if isinstance(struct_names, str):
        struct_names = [struct_names]
    else:
        struct_names = list(struct_names)
    struct_names  = [s.lower() for s in struct_names]

    i_first = 1 + len(struct_names)
    key_first = None
    if structs:
        for key, struct in structs.items():
            actual_name = struct["name"]
            if actual_name.lower() in struct_names:
                if first_only:
                    j_first = struct_names.index(actual_name.lower())
                    if j_first < i_first:
                        i_first = j_first
                        key_first = key
                else:
                    selection[key] = structs[key]

    if key_first is not None:
        selection[key_first] = structs[key_first]

    return selection


def get_timestamp_difference_days(timestamp1="", timestamp2=""):
    """Get difference between two timestamps in days."""

    delta_seconds = get_timestamp_difference_seconds(timestamp1, timestamp2)
    delta_days = None
    if delta_seconds is not None:
        delta_days = delta_seconds / (24.0 * 60.0 * 60.0)
    return delta_days


def get_timestamp_difference_seconds(timestamp1="", timestamp2=""):
    """Get difference between two timestamps in seconds."""

    delta_seconds = None
    if is_timestamp(timestamp1) and is_timestamp(timestamp2):

        datetime1 = datetime.datetime.strptime(timestamp1, "%Y%m%d_%H%M%S")
        datetime2 = datetime.datetime.strptime(timestamp2, "%Y%m%d_%H%M%S")
        td = datetime2 - datetime1
        delta_seconds = (
            td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6
        ) / 10 ** 6

    return delta_seconds


def get_timestamp_difference_years(timestamp1="", timestamp2=""):
    """Get difference between two timestamps in years, approximating to
    365.2425 days per year."""

    delta_days = get_timestamp_difference_days(timestamp1, timestamp2)
    if delta_days is None:
        return
    else:
        return delta_days / 365.2425


def matchSize(ct1=None, ct2=None, fillValue=-1024):

    ct1Array = ct1.getImageStack()
    ct2Array = ct2.getImageStack()

    if (None in ct1.voxelSize[0:2]) or (None in ct2.voxelSize):
        return None

    match = (
        (ct1Array.shape == ct2Array.shape)
        and (ct1.scanPosition == ct2.scanPosition)
        and (ct1.voxelSize == ct2.voxelSize)
    )

    ct3 = copy.deepcopy(ct1)

    if not match:
        # If slice thickness not specified for ct1, guess that it is
        # the same as for ct2
        if ct1.voxelSize[2] is None:
            x1, y1, z1 = ct1.voxelSize
            x2, y2, z2 = ct2.voxelSize
            ct1.voxelSize = (x1, y1, z2)
        print(f"interpolation start time: {time.strftime('%c')}")
        x1Array, y1Array, z1Array = getCoordinateArrays(ct1)
        if not (x1Array is None or y1Array is None or z1Array is None):
            interpolant = scipy.interpolate.RegularGridInterpolator(
                (y1Array, x1Array, z1Array),
                ct1.getImageStack(),
                method="linear",
                bounds_error=False,
                fill_value=fillValue,
            )

            x2Array, y2Array, z2Array = getCoordinateArrays(ct2)

            """
      pointArray = numpy.zeros( ct2Array.shape, dtype = ( float, 3 ) )
      ny, nx, nz = ct2Array.shape
      for iy in range( ny ):
        for ix in range( nx ):
          for iz in range( nz ):
            pointArray[ iy, ix, iz ] = \
              ( y2Array[ iy ], x2Array[ ix ], z2Array[ iz ] )
      """

            ny, nx, nz = ct2Array.shape
            meshgrid = np.meshgrid(y2Array, x2Array, z2Array, indexing="ij")
            vstack = np.vstack(meshgrid)
            pointArray = vstack.reshape(3, -1).T.reshape(ny, nx, nz, 3)

            ct3.imageStack = interpolant(pointArray)
            ct3.voxelSize = ct2.voxelSize
            ct3.scanPosition = ct2.scanPosition
            ct3.transform_ijk_to_xyz = ct2.transform_ijk_to_xyz
        else:
            ct3 = None
        print(f"interpolation end time: {time.strftime('%c')}")

    return ct3


def matchSlices(ct1=None, ct2=None, fill_value=-1024):

    if None in ct1.voxelSize[0:2]:
        return None

    # Create copy of ct2, then substitute slices with same
    # xy-dimensions as ct1
    ct1_image_stack = ct1.getImageStack()
    ny1, nx1, nz1 = ct1_image_stack.shape
    dx1, dy1, dz1 = ct1.voxelSize

    ct2_image_stack = ct2.getImageStack()
    ny2, nx2, nz2 = ct2_image_stack.shape
    dx2, dy2, dz2 = ct2.voxelSize
    x02, y02, z02 = ct2.scanPosition
    xc2, yc2, zc2 = ct2.getScanCentre()

    ref_ct = copy.deepcopy(ct1)
    ref_ct.imageStack = np.zeros((ny1, nx1, nz2), ct1_image_stack.dtype)
    ref_ct.voxelSize = (dx1, dy1, dz2)
    x0 = xc2 + 0.5 * nx1 * dx1
    y0 = yc2 - 0.5 * ny1 * dy1
    ref_ct.scanPosition = (x0, y0, z02)
    ref_ct.transform_ijk_to_xyz = get_transform_ijk_to_xyz(ref_ct)
    ref_ct = matchSize(ct1, ref_ct, fill_value)

    return ref_ct


def matchSliceThickness(ct=None, ref_dz=None, fill_value=-1024):

    if ct is None or ref_dz is None:
        return ct

    dx, dy, dz = ct.voxelSize
    imageStack = ct.getImageStack()
    nx, ny, nz = imageStack.shape
    ddz = nz * dz
    ref_nz = int(ddz / ref_dz)
    if int(nz) == ref_nz:
        return ct

    x0, y0, z0 = ct.scanPosition
    # ref_z0 = z0 + 0.5 * (ref_dz - dz)
    # Define extremes so that new slice centres
    # are all between centres of original slices
    # ref_z0 = z0 + 0.5 * ref_dz
    # ref_z1 = z0 + (nz - 1) * dz - 0.5 * ref_dz
    # ref_nz = int((ref_z1 - ref_z0) / ref_dz)
    # Centre new slices on original slices
    ref_z0 = z0 - 0.5 * ((ref_nz - 1) * ref_dz - (nz - 1) * dz)
    ref_ct = copy.deepcopy(ct)

    ref_ct.imageStack = np.zeros((nx, ny, ref_nz))
    ref_ct.scanPosition = (x0, y0, ref_z0)
    ref_ct.voxelSize = (dx, dy, ref_dz)
    ref_ct.transform_ijk_to_xyz = get_transform_ijk_to_xyz(ref_ct)
    ref_ct = matchSize(ct, ref_ct, fill_value)

    return ref_ct


def point_ijk_to_xyz(ijk=None, affine=None, translation=None, 
                     rotation=None, reverse=False):
    """Convert a set of indices (ijk) to a spatial position (xyz) using a 
    given affine tranform matrix. Can additionally apply a translation,
    rotation, and reverse direction."""

    if (ijk is None) or (affine is None):
        xyz = (None, None, None)
    else:
        i, j, k = ijk
        ijk_vec = np.array([i, j, k, 1])
        xyz_vec = affine.dot(ijk_vec)
        x, y, z, u = xyz_vec[0], xyz_vec[1], xyz_vec[2], xyz_vec[3]
        xyz = apply_couch_shifts((x, y, z), translation, rotation, reverse)

    return xyz


def point_xyz_to_ijk(xyz=None, affine=None, translation=None, rotation=None,
                     reverse=False, outFloat=False):
    """Convert a spatial position (xyz) to a set of indices (ijk) using a 
    given affine tranform matrix. Can additionally apply a translation,
    rotation, and reverse direction. If out_float is True, the output will
    not be rounded to an integer."""

    if xyz is None or affine is None:
        return (None, None, None)

    x, y, z = apply_couch_shifts(xyz, translation, rotation, reverse)

    affine_inv = np.linalg.inv(affine)
    xyz_vec = np.array([x, y, z, 1.0])
    ijk_vec = affine_inv.dot(xyz_vec)
    if out_float:
        i, j, k, ll = (m for m in ijk_vec)
    else:
        i, j, k, ll = (iround(m) for m in ijk_vec)

    return (i, j, k)


def pointXyzToImageIjk(
    xyz=None,
    affineTransform=None,
    translation=None,
    rotation=None,
    reverse=False,
    outFloat=False,
):

    if xyz is None or affineTransform is None:
        return (None, None, None)

    x, y, z = applyCouchShifts(xyz, translation, rotation, reverse)

    affineTransformInverse = np.linalg.inv(affineTransform)
    affineTransformInverse[0, 0] = -affineTransformInverse[0, 0]
    affineTransformInverse[1, 1] = -affineTransformInverse[1, 1]
    xyzVector = np.matrix([[x], [y], [z], [1.0]])
    ijkVector = affineTransformInverse * xyzVector
    if outFloat:
        i, j, k, ll = (m for m in ijkVector.flatten().tolist()[0])
    else:
        i, j, k, ll = (iround(m) for m in ijkVector.flatten().tolist()[0])

    return (i, j, k)


def randomDerangement(inList=[], seed=1):

    # Algorithm for random derangement based on early refusal,
    # as described in:
    # https://www.semanticscholar.org/paper/AN-ANALYSIS-OF-A-SIMPLE-ALGORITHM-FOR-RANDOM-Merlini-Sprugnoli/45b71814f65ec7939762adea5bee02807b0ec499
    # and as implemented at:
    # https://stackoverflow.com/questions/25200220/generate-a-random-derangement-of-a-list

    random.seed(seed)

    n = len(inList)
    iList = range(n)

    allOK = False

    while not allOK:
        for i in range(n - 1, -1, -1):
            j = random.randint(0, 1)
            if iList[j] == i:
                break
            else:
                iList[i], iList[j] = iList[j], iList[i]
        if iList[0] != 0:
            allOK = True

    outList = list(inList)
    for i in range(n):
        outList[iList[i]] = inList[i]

    return outList


def reflect(x=0.0, x0=0.0):

    dx = x - x0
    xr = x0 - dx

    return xr


def writeJson(dataDict={}, jsonFileName=None):

    if jsonFileName is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        jsonFileName = f"{timestamp}_data.json"
    jsonFile = open(jsonFileName, "w")
    json.dump(dataDict, jsonFile, indent=1, separators=(",", ":"), sort_keys=True)
    jsonFile.close()
    return None


def writeLineList(lineList=[], fileName=None):

    if fileName is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fileName = f"{timestamp}_data.txt"
    lineString = "\n".join(lineList)
    outFile = open(fileName, "w")
    outFile.write(lineString)
    outFile.close()
    return None
