'''Test prototype image class.'''

import os
import numpy as np
import shutil

from quickviewer.prototype import Image


# Create fake data
data = np.random.rand(40, 50, 20)
voxel_size = (1, 2, 3)
origin = (-40, -50, 20)
im = Image(data, voxel_size=voxel_size, origin=origin)

# Make temporary test dir
if os.path.exists('tmp'):
    shutil.rmtree('tmp')
os.mkdir('tmp')

# Create a test nifti file
nii_file = 'tmp/test.nii'
im.write(nii_file)
im_nii = Image(nii_file)

# Create a test dicom file
dcm_file = 'tmp/test_dcm'
im.write(dcm_file)
im_dcm = Image(dcm_file)


############################
# Test reading and writing #
############################
def test_array_reading():
    '''Check voxel sizes, origin, and affine matrix are correctly set for a 
    numpy array.'''

    assert all([voxel_size[i] == im.voxel_size[i] for i in range(3)])
    assert all([origin[i] == im.origin[i] for i in range(3)])
    assert all([voxel_size[i] == im.affine[i, i] for i in range(3)])
    assert all([origin[i] == im.affine[i, 3] for i in range(3)])
    assert im.data.shape == data.shape
    assert im.n_voxels[0] == data.shape[1]
    assert im.n_voxels[1] == data.shape[0]
    assert im.n_voxels[2] == data.shape[2]

def test_dicom_array():
    '''Check dicom array is the same as the input array.'''

    ddata, daffine = im.get_dicom_array_and_affine()
    assert np.all(ddata == data)
    assert np.all(daffine.astype(int) == im.affine)

def test_nifti_array():
    '''Check nifti array is correctly transposed wrt input array.'''

    ndata, naffine = im.get_nifti_array_and_affine()
    assert ndata.shape[1] == data.shape[0]
    assert ndata.shape[0] == data.shape[1]
    assert ndata.shape[2] == data.shape[2]
    assert naffine[1, 3] != origin[1]

def test_array_to_npy():
    '''Check numpy array can be saved to a .npy file and read in correctly.'''

    outname = 'tmp/test.npy'
    im.write(outname)
    im_npy = Image(outname, affine=im.affine)
    assert np.all(im_npy.data == im.data)
    assert np.all(im_npy.affine == im.affine)
    assert os.path.exists(outname.replace('npy', 'txt'))

def test_array_to_nifti_npy():
    '''Check numpy array is correctly saved in nifti-style.'''

    outname = 'tmp/test_nii.npy'
    im.write(outname, nifti_array=True)
    im_npy_nifti = Image(outname)
    ndata, naffine = im.get_nifti_array_and_affine()
    assert np.all(ndata == im_npy_nifti.data)

def test_array_to_nifti():
    '''Check numpy array is correctly saved to nifti.'''

    # Data and affine should be the same as nifti data and affine in 
    # original image
    ndata, naffine = im.get_nifti_array_and_affine()
    assert np.all(ndata == im_nii.data)
    assert np.all(naffine == im_nii.affine)

    # Standarised data and affine should be the same for both
    im.standardise_data()
    im_nii.standardise_data()
    assert np.all(im.sdata == im_nii.sdata)
    assert np.all(im.saffine == im_nii.saffine)

def test_nifti_to_nifti():
    '''Check a nifti file can be written and read correctly.'''

    # Write nifti image to second nifti file
    nii2 = 'tmp/test2.nii'
    im_nii.write(nii2)
    im_nii2 = Image(nii2)

    # Check data and affine matrix is the same
    assert np.all(im_nii.data == im_nii2.data)
    assert np.all(im_nii.affine == im_nii2.affine)

def test_nifti_to_npy():
    '''Check that a nifti file is correctly written to a numpy file.'''

    # Write to numpy file
    npy = 'tmp/test2.npy'
    im_nii.write(npy)
    affine_dcm = im_nii.get_dicom_array_and_affine()[1]
    im_npy = Image(npy, affine=affine_dcm)

    # Check nifti array matches
    ndata, naffine = im_npy.get_nifti_array_and_affine()
    assert np.all(naffine == im_nii.affine)
    assert np.all(ndata == im_nii.data)

    # Check standardised data matches
    im_nii.standardise_data()
    im_npy.standardise_data()
    assert np.all(im_nii.saffine == im_npy.saffine)
    assert np.all(im_nii.sdata == im_npy.sdata)

def test_dcm_to_nifti():
    '''Check that a dicom file is correctly written to nifti.'''

    # Write dicom to nifti
    nii = 'tmp/test_dcm2nii.nii'
    im_dcm.write(nii)
    im_dcm2nii = Image(nii)

    # Check nifti array is the same
    ndata, naffine = im_dcm.get_nifti_array_and_affine()
    assert np.all(naffine == im_dcm2nii.affine)
    assert np.all(ndata == im_dcm2nii.data)

    # Check standardised data is the same
    im_dcm.standardise_data()
    im_dcm2nii.standardise_data()
    print('saffine dcm:', im_dcm.saffine)
    print('saffine dcm2nii:', im_dcm2nii.saffine)
    assert np.all(im_dcm.saffine == im_dcm2nii.saffine)
    assert np.all(im_dcm.sdata == im_dcm2nii.sdata)

def test_dcm_to_dcm():
    '''Check that a dicom file is correctly written to dicom.'''

    # Write to dicom
    dcm = 'tmp/test_dcm2'
    im_dcm.write(dcm)
    im_dcm2 = Image(dcm)

    assert im_dcm.data.shape == im_dcm2.data.shape
    assert np.all(im_dcm.affine == im_dcm2.affine)
    assert np.all(im_dcm.data == im_dcm2.data)

def test_dcm_to_nifti_to_dcm():
    '''Check that a nifti file can be written to dicom using the header of 
    the dicom that was used to create that nifti file.'''

    pass

#  def test_array_to_dicom():
    #  '''Check numpy array is correctly saved to dicom.'''

    #  outname = 'tmp/test_dcm'
    #  im.write(outname)
    #  im_dcm = Image(outname)

    #  # Data and affine should be same for both
    #  assert im.data.shape == im_dcm.data.shape
