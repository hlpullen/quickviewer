'''Classes for loading, plotting and comparing images and structures.'''

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

from quickviewer.prototype.core import DatedObject, MachineObject
from quickviewer.prototype import Image, RtStruct, ROI


# File: quickviewer/data/__init__.py
# Based on voxtox/utility/__init__.py, created by K. Harrison on 130906


class RtDose(MachineObject):

    def __init__(self, path=''):

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

    def __init__(self, path=''):

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
                if hasattr(fraction, 'ReferencedDoseReferenceSequence'):
                    if self.target_dose is None:
                        self.target_dose = 0.0
                    for dose in fraction.ReferencedDoseReferenceSequence:
                        self.target_dose += dose.TargetPrescriptionDose


    '''Object associated with a top-level directory whose name corresponds to
    a patient ID, and whose subdirectories contain studies.'''

    def __init__(self, path=None, exclude=['logfiles']):

        start = time.time()

        # Set path and patient ID
        if path is None:
            path = os.getcwd()
        self.path = fullpath(path)
        self.id = os.path.basename(self.path)

        # Find studies
        self.studies = self.get_dated_objects(dtype='Study')
        if not self.studies:
            if os.path.isdir(self.path):
                if os.access(self.path, os.R_OK):
                    subdirs = sorted(os.listdir(self.path))
                    for subdir in subdirs:
                        if subdir not in exclude:
                            self.studies.extend(
                                self.get_dated_objects(
                                    dtype='Study', subdir=subdir
                                )
                            )

        # Get patient demographics
        self.birth_date, self.age, self.sex = self.get_demographics()

    def combined_files(self, dtype, min_date=None, max_date=None):
        '''Get list of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified.'''

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
        '''Get dict of all files of a given data type <dtype> associated with 
        this patient, within a given date range if specified. The dict keys 
        will be the directories that the files are in.'''

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
        '''Get list of all objects of a given data type <dtype> associated
        with this patient.'''

        all_objs = []
        for study in self.studies:
            objs = getattr(study, dtype)
            if objs:
                all_objs.extend(objs)
        all_objs.sort()
        return all_objs

    def get_demographics(self):
        '''Return patient's birth date, age, and sex.'''

        info = {'BirthDate': None, 'Age': None, 'Sex': None}

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
                for prefix in ['Patient', 'Patients']:
                    attr = f'{prefix}{key[0].upper()}{key[1:]}'
                    if hasattr(ds, attr):
                        info[key] = getattr(ds, attr)
                        break

        # Ensure sex is uppercase and single character
        if info['Sex']:
            info['Sex'] = info['Sex'][0].upper()

        return info['BirthDate'], info['Age'], info['Sex']

    def get_subdir_study_list(self, subdir=''):

        subdir_studies = []
        for study in self.studies:
            if subdir == study.subdir:
                subdir_studies.append(study)

        subdir_studies.sort()

        return subdir_studies

    def last_in_interval(self, dtype=None, min_date=None, max_date=None):
        '''Get the last object of a given data type <dtype> in a given
        date interval.'''

        files = self.combined_files(dtype)
        last = None
        files.reverse()
        for file in files:
            if file.in_date_interval(min_date, max_date):
                last = file
                break
        return last


class Study(DatedObject):

    def __init__(self, path=''):

        DatedObject.__init__(self, path)

        # Load RT plans, CT and MR scans, all doses, and CT structure sets
        self.plans = self.get_plan_data(dtype='RtPlan', subdir='RTPLAN')
        self.ct_scans = self.get_dated_objects(dtype='Image', subdir='CT')
        self.mr_scans = self.get_dated_objects(dtype='Image', subdir='MR')
        self.doses = self.get_plan_data(
            dtype='RtDose',
            subdir='RTDOSE',
            exclude=['MVCT', 'CT'],
            scans=self.ct_scans
        )
        self.ct_structs = self.get_structs(subdir='RTSTRUCT/CT', 
                                           images=self.ct_scans)

        # Look for HD CT scans and add to CT list
        ct_hd = self.get_dated_objects(dtype='CT', subdir='CT_HD')
        ct_hd_structs = self.get_structs(subdir='RTSTRUCT/CT_HD', scans=ct_hd)
        if ct_hd:
            self.ct_scans.extend(ct_hd)
            self.ct_scans.sort()
        if ct_hd_structs:
            self.ct_structs.extend(ct_hd_structs)
            self.ct_structs.sort()

        # Load CT-specific RT doses
        self.ct_doses = self.get_plan_data(
            dtype='RtDose', subdir='RTDOSE/CT', scans=self.ct_scans
        )
        self.ct_doses = self.correct_dose_scan_position(self.ct_doses)

        # Load MVCT images, doses, and structs
        self.mvct_scans = self.get_dated_objects(dtype='Image', subdir='MVCT')
        self.mvct_doses = self.get_plan_data(
            dtype='RtDose', subdir='RTDOSE/MVCT', scans=self.mvct_scans
        )
        self.mvct_doses = self.correct_dose_scan_position(self.mvct_doses)
        self.mvct_structs = self.get_structs(
            subdir='RTSTRUCT/MVCT', scans=self.mvct_scans
        )

        # Set description
        self.description = self.get_description()

    def correct_dose_scan_position(self, doses=[]):
        '''Correct for scan positions from CheckTomo being offset by one slice
        relative to scan positions.'''

        for dose in doses:
            dx, dy, dz = dose.voxel_size
            x0, y0, z0 = dose.scan_position
            dose.scan_position = (x0, y0, z0 + dz)
        return doses

    def get_machine_sublist(self, dtype='', machine='', ignore_case=True):
        '''Get list of doses or treatment plans corresponding to a specific
        machine.'''

        sublist = []
        if dtype.lower() in ['plan', 'rtplan']:
            objs = self.plans
        elif dtype.lower() in ['dose', 'rtdose']:
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
        '''Get a selection of MVCT scans which were taken at least 
        <min_delta_hours> apart. <mvct_dict> is a dict where the keys are 
        patient IDs, and the paths are directory paths from which to load scans
        for that patient.'''

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
        self, dtype='RtPlan', subdir='RTPLAN', exclude=[], scans=[]
    ):
        '''Get list of RT dose or plan objects specified by dtype='RtDose' or 
        'RtPlan' <dtype>, respectively) by searching within a given directory, 
        <subdir> (or within the top level directory of this Study, if 
        <subdir> is not provided).

        Subdirectories with names in <exclude> will be ignored.

        Each dose-like object will be matched by timestamp to one of the scans 
        in <scans> (which should be a list of DatedObjects), if provided.'''

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
        if dtype == 'RtDose':
            new_doses = []
            for dose in doses:

                # Search for scans with matching timestamp
                timestamp = os.path.basename(os.path.dirname(dose.path))
                if scans:
                    try:
                        dose.date, dose.time = timestamp.split('_')
                        scan = get_dated_obj(scans, dose)
                        dose.machine = scan.machine
                    except BaseException:
                        scan = scans[-1]
                        dose.date = scan.date
                        dose.time = scan.time

                    dose.timestamp = f'{dose.date}_{dose.time}'
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

        # 'PLAN' summation type: just take the newest entry
        if 'PLAN' in dose_dict:
            plan_dose = dose_dict['PLAN'][-1]
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
            if 'FRACTION' in dose_dict:
                if len(dose_dict['FRACTION']) == n_frac_group:
                    
                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = doseDict['FRACTION'][0]

                    # Sum fractions
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, 'FRACTION')

            # Sum over beams
            elif 'BEAM' in sum_type:
                if len(dose_dict['BEAM']) == n_beam_seq:

                    # Single fraction
                    if n_frac_group == 1:
                        plan_dose = dose_dict['BEAM'][0]

                    # Sum beams
                    else:
                        plan_dose = self.sum_dose_plans(dose_dict, 'BEAM')

        return plan_dose

    def get_structs(self, subdir='', images=[]):
        '''Make list of RtStruct objects found within a given subdir, and
        set their associated scan objects.'''

        # Find RtStruct directories associated with each scan
        groups = self.get_dated_objects(dtype='ArchiveObject', subdir=subdir)

        # Load RtStruct files for each
        structs = []
        for group in groups:

            # Find the matching Image for this group
            image = Image()
            image_dir = os.path.basename(group.path)
            image_found = False

            # Try matching on path
            for im in images:
                if image_dir == os.path.basename(im.path):
                    image = im
                    image_found = True
                    break

            # If no path match, try matching on timestamp
            if not image_found:
                for im in images:
                    if (group.date == im.date) and (group.time == im.time):
                        image = im
                        break

            # Find all RtStruct files inside the dir
            for file in struct_dir.files:

                # Create RtStruct
                rt_struct = RtStruct(file, image=im)

                # Add to Image
                image.add_structs(rt_struct)

                # Add to list of all structure sets
                structs.append(rt_struct)

        return structs

    def get_description(self):
        '''Load a study description.'''

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

        description = ''
        if obj:
            if obj.files:
                scan_path = obj.files[-1].path
                ds = pydicom.read_file(fp=scan_path, force=True)
                if hasattr(ds, 'StudyDescription'):
                    description = ds.StudyDescription

        return description

    def sum_dose_plans(self, dose_dict={}, sum_type=''):
        '''Sum over doses using a given summation type.'''

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
            plan_dose.summationType = 'PLAN'
            plan_dose.scanPosition = dose.scanPosition
            plan_dose.reverse = dose.reverse
            plan_dose.voxelSize = dose.voxelSize
            plan_dose.transform_ijk_to_xyz = dose.transform_ijk_to_xyz
            plan_dose.imageStack = dose.getImageStack()
            for dose in dose_dict[sum_type]:
                plan_dose.imageStack += dose.getImageStack()

        return plan_dose


def alphanumeric(in_str=''):
    '''Function that can be passed as value for list sort() method
    to have alphanumeric (natural) sorting'''

    import re

    elements = []
    for substr in re.split('(-*[0-9]+)', in_str):
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


def convertNPixel(ct=None, ctRef=None, nRef=0, ixyz=0, outFloat=False):

    if nRef > 0:
        n = float(nRef) * ctRef.voxelSize[ixyz] / ct.voxelSize[ixyz]
    else:
        n = 0.0

    if not outFloat:
        n = int(n) + 1

    return n


def fitCircle(pointList=[]):
    '''
    Fit circle to points specified as list of ( x, y ) tuples
    Code addapted from implementation 2b of:
    http://wiki.scipy.org/Cookbook/Least_Squares_Circle
    '''

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
        '''Calculate point distances from estimated circle
        centre (xc,yc)'''
        return np.sqrt((xArray - xc) ** 2 + (yArray - yc) ** 2)

    def f_2b(c):
        '''Calculate point residuals about estimated circle radius'''
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        '''
         Jacobian of f_2b
        The axis corresponding to derivatives must be
        coherent with the col_deriv option of leastsq
        '''
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
    '''
    Fit sphere to points specified as list of ( x, y, z ) tuples
    Code addapted from top answer at:
    http://stackoverflow.com/questions/15785428/how-do-i-fit-3d-data
    '''

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
        raise TypeError('frange expected at least 1 arguments, got 0')
    elif 3 < len(args):
        raise TypeError(f'frange expected at most 3 arguments, got {len(args)}')
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


def fullpath(path=''):
    '''Evaluate full path, expanding '~', environment variables, and 
    symbolic links.'''

    expanded = ''
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


def get_couch_shift(path=''):
    '''Extract couch translation and rotation vectors from a dicom file.'''

    translation = None
    rotation = None

    # Extract translation and rotation strings from dicom
    if os.path.exists(path):
        ds = pydicom.read_file(path, force=True)
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
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
            translation_list = translation.value.split('\\')
        elif isinstance(translation.value, bytes):
            translation_list = translation.value.decode().split('\\')
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
            rotation_list = rotation.value.split('\\')
        elif isinstance(rotation.value, bytes):
            rotation_list = rotation.value.decode().split('\\')
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


def get_dated_object(objs=[], timestamp=''):
    '''For a given list of objects <obj>, find the first object that matches
    a given timestamp <timestamp> (which can either be a string in format
    date_time, or any object with date and time attributes).'''

    # Convert timestamp to consistent format
    if hasattr(timestamp, 'date') and hasattr(timestamp, 'time'):
        timestamp = f'{timestamp.date}_{timestamp.time}'

    # Find object with matching timestamp
    dated_obj = None
    if objs and timestamp:
        for obj in objs:
            if f'{obj.date}_{obj.time}' == timestamp:
                dated_obj = obj
                break

    return dated_obj


def getDatedObjectList(objectList=[], timestamp=''):

    datedObjectList = []

    if hasattr(timestamp, 'date') and hasattr(timestamp, 'time'):
        timestamp = f'{timestamp.date}_{timestamp.time}'

    if objectList and timestamp:
        for tmpObject in objectList:
            tmpTimestamp = f'{tmpObject.date}_{tmpObject.time}'
            if tmpTimestamp == timestamp:
                datedObjectList.append(tmpObject)

    return datedObjectList


def get_dicom_sequence(ds=None, basename=''):

    sequence = []

    for suffix in ['Sequence', 's']:
        attribute = f'{basename}{suffix}'
        if hasattr(ds, attribute):
            sequence = getattr(ds, attribute)
            break

    return sequence


def getDirList(topDir='', timestamp=False):
    '''Obtain list of all subdirectories below a given top-level director'''
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


def getLabel(path=''):

    label = pydicom.read_file(path, force=True).StructureSetLabel.strip()
    if 1 + label.find('JES'):
        if 1 + label.find('contour'):
            label = 'Jessica2'
        else:
            label = 'Jessica1'

    if 1 + label.find('ProSoma'):
        label = 'Gill'

    if 1 + label.find('Automated'):
        label = 'robot'

    return label


def getLogger(name=''):
    import logging

    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
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


def getRoiObserver(roi='', structDict={}):

    roiObserver = None
    for roiKey in structDict.keys():
        if re.search(roi, structDict[roiKey]['name'], re.I):
            roiObserver = structDict[roiKey]['name']
            break

    return roiObserver


def get_separated_timestamp_dict(timestamp_dict={}, min_delta_hours=4.0):
    '''Take a dict where the keys are timestamps, and reduce down to only
    elements where the timestamps are at least <min_delta_hours> apart.'''

    timestamps = timestamp_dict.keys()
    separated_timestamps = get_separated_timestamp_list(
        timestamps, min_delta_hours)
    return {ts: timestamp_dict[ts] for ts in separated_timestamps}


def get_separated_timestamp_list(timestamps=[], min_delta_hours=4.0):
    '''Take a list of timestamps and reduce down to only elements where the 
    timestamps are at least <min_delta_hours> apart.'''

    # Find all timestamps containing a valid date and time
    checked = []
    for timestamp in timestamps:
        date, time = get_time_and_date(os.path.basename(timestamp))
        if not (date is None or time is None):
            checked.append('_'.join([date, time]))

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
    '''
    Retreive a study's non-empty lists of data objects.

    Arguments:
        study        -- voxtox.utility.Study object
        requirements -- list of required attributes for data objects;
                        for example: requirements=['imageStack'] will select
                        objects with image data,
                        requirements=['structureSetLabel']
                        will select structure-set objects
    '''

    # If requirements passed as a string, convert to single-item list.
    if isinstance(requirements, str):
        requirements = [requirements]

    # Examine all attributes of the Study object, and identify the data lists.
    data_dict1 = {}
    for attribute in dir(study):
        if 'list' in attribute.lower():
            item_list = getattr(study, attribute)
            if isinstance(item_list, list) and item_list:
                data_type = attribute.lower().split('list')[0].lower()
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


def get_time_and_date(timestamp=''):

    timeAndDate = (None, None)
    if is_timestamp(timestamp):
        valueList = os.path.splitext(timestamp)[0].split('_')
        valueList = [value.strip() for value in valueList]
        if valueList[0].isalpha():
            timeAndDate = tuple(valueList[1:3])
        else:
            timeAndDate = tuple(valueList[0:2])
    else:
        i1 = timestamp.find('_')
        i2 = timestamp.rfind('.')
        if (-1 != i1) and (-1 != i2):
            bitstamp = timestamp[i1 + 1 : i2]
            if is_timestamp(bitstamp):
                timeAndDate = tuple(bitstamp.split('_'))

    return timeAndDate


def get_time_separated_objects(in_list=[], min_delta_hours=4.0):
    '''Parse a list of objects and return only those with time separations 
    greater than min_delta_hours.'''

    timestamps = {obj.timestamp: obj for obj in in_list}
    separated_timestamps = get_separated_timestamp_dict(timestamps, 
                                                        min_delta_hours)
    return sorted(separated_timestamps.values())


def get_transform_ijk_to_xyz(obj=None):
    '''Get a matrix for transforming a set of indices (ijk) to a spatial 
    position (xyz) for a given object that has a scan position and voxel 
    size.'''

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


def getTransformixPointDict(filepath=''):
    def getCoordinateTuple(inString=''):
        coordinateTuple = tuple([eval(x) for x in inString.split()[3:6]])
        return coordinateTuple

    inFile = open(filepath)
    lineList = inFile.readlines()
    inFile.close()
    pointDict = {}
    for line in lineList:
        elementList = line.split(';')
        point = eval(elementList[0].split()[1])
        pointDict[point] = {}
        pointDict[point]['InputIndex'] = getCoordinateTuple(elementList[1])
        pointDict[point]['InputPoint'] = getCoordinateTuple(elementList[2])
        pointDict[point]['OutputIndexFixed'] = getCoordinateTuple(elementList[3])
        pointDict[point]['OutputPoint'] = getCoordinateTuple(elementList[4])
        pointDict[point]['Deformation'] = getCoordinateTuple(elementList[5])

    return pointDict


def get_voxel_size(in_data=''):
    '''Find voxel size from a path/list of paths to dicom file(s).'''

    dx, dy, dz = (None, None, None)
    if isinstance(in_data, list):
        path = in_data[0]
    else:
        path = in_data
    if os.path.exists(path):
        ds = pydicom.read_file(path, force=True)
        if hasattr(ds, 'PixelSpacing'):
            dx, dy = ds.PixelSpacing
            dx = float(dx)
            dy = float(dy)
        if hasattr(ds, 'SliceThickness'):
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


def is_timestamp(testString=''):
    timestamp = True
    valueList = os.path.splitext(testString)[0].split('_')
    valueList = [value.strip() for value in valueList]
    '''
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
  '''
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


def get_timestamp_difference_days(timestamp1='', timestamp2=''):
    '''Get difference between two timestamps in days.'''

    delta_seconds = get_timestamp_difference_seconds(timestamp1, timestamp2)
    delta_days = None
    if delta_seconds is not None:
        delta_days = delta_seconds / (24.0 * 60.0 * 60.0)
    return delta_days


def get_timestamp_difference_seconds(timestamp1='', timestamp2=''):
    '''Get difference between two timestamps in seconds.'''

    delta_seconds = None
    if is_timestamp(timestamp1) and is_timestamp(timestamp2):

        datetime1 = datetime.datetime.strptime(timestamp1, '%Y%m%d_%H%M%S')
        datetime2 = datetime.datetime.strptime(timestamp2, '%Y%m%d_%H%M%S')
        td = datetime2 - datetime1
        delta_seconds = (
            td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6
        ) / 10 ** 6

    return delta_seconds


def get_timestamp_difference_years(timestamp1='', timestamp2=''):
    '''Get difference between two timestamps in years, approximating to
    365.2425 days per year.'''

    delta_days = get_timestamp_difference_days(timestamp1, timestamp2)
    if delta_days is None:
        return
    else:
        return delta_days / 365.2425
