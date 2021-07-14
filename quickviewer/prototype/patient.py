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

from quickviewer.prototype.core import DatedObject, MachineObject, PathObject
from quickviewer.prototype.core import fullpath
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


class Patient(PathObject):
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
