"""Tests to check that the Patient class works as expected."""

from quickviewer.data import Patient


p = Patient("~/Work/HeadAndNeck/dicom/VT1_H_03F693K1")

def test_combined_files():
    files = p.combined_files("ct_scans")
    assert len(files)

def test_combined_files_by_dir():
    files = p.combined_files_by_dir("ct_scans")
    assert len(files)

def test_combined_objs():
    objs = p.combined_objs("ct_scans")
    assert len(objs)

def test_demographics():
    demog = p.get_demographics()
    assert len(demog) == 3

def test_struct_in_interval():
    p.first_struct_in_interval("ct_structs")
    p.last_struct_in_interval("ct_structs")
    p.last_in_interval("ct_structs")
