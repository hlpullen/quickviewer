"""Test the StructImage class."""

from pytest import approx
from quickviewer.data.structures import Struct

sm = Struct("data/structs/"
                "RTSTRUCT_CT_20140715_113632_002_oral_cavity.nii.gz")

def test_struct_mask():
    assert len(sm.voxel_sizes) == 3
    assert len(sm.origin) == 3
    assert len(sm.n_voxels) == 3
    assert sm.get_volume("voxels") > 0
    assert sm.get_volume("ml") > 0
    assert sm.get_volume("mm") == approx(sm.get_volume("ml") * 1e3)
    assert len(sm.get_length("mm")) == 3
    assert sm.get_length("voxels")[0]

def test_contours():
    assert len(sm.contours["x-y"])
    assert len(sm.contours["y-z"]) <= sm.n_voxels["x"]

