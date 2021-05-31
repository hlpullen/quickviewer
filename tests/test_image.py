"""Test the classes in quickviewer.data.image"""

import numpy as np
from quickviewer.data.image import Image

# Make random numpy array
shape = []
for i in range(3):
    shape.append(np.random.randint(50, 500))
arr = np.random.rand(*shape)
arr_file = "data/array.npy"
np.save(arr_file, arr)

# Test Image class
# Loading from NumPy array
def test_nifti_image_from_array():
    t = "test title"
    im  = Image(arr, title=t) 
    assert im.valid
    assert im.n_voxels["x"] == shape[0]
    assert im.title == t

# Loading from NumPy file
def test_nifti_image_from_file():
    im  = Image(arr_file) 
    assert im.valid
    assert im.n_voxels["y"] == shape[1]
    assert im.title == "array.npy"

def test_nifti_image_downsample():
    ds = 10
    im = Image(arr, downsample=ds)
    assert im.valid
    assert im.n_voxels["z"] <= shape[2] / (ds - 1)

def test_nifti_image_plot():
    im = Image(arr)
    im.plot("x-y", 25, show=False, colorbar=True)

def test_nifti_image_zoom():
    im = Image(arr)
    im.plot(zoom=(1, 2, 3), show=False)
    assert im.valid
