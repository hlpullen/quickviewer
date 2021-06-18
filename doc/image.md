# The Image class

Images can be loaded from dicom files, nifti files, or numpy arrays, and will be put into a consistent format. This format is:

- The `Image.data` property contains a numpy array, which stores (x, y, z) in (row, column, slice) respectively. Note that numpy elements are indexed in order (column, row, slice); so if you did `Image.data[i, j, k]`, `i` would correspond to y index, `j` would correspond to x index, `k` would correspond to z index.
- The `Image.affine` property contains a 4x4 matric that can convert a (row, column, slice) index to an (x, y, z) position. This will always be diagonal, so (0, 0) contains x voxel size etc, (0, 3) contains x origin.
- The `voxel_size` and `origin` properties are the diagonal and third column, respectively; they give voxel sizes and origin position in order (x, y, z).
- The `n_voxels` property containins the number of voxels in the (x, y, z) directions (same as `Image.data.shape`, but with 0 and 1 swapped).

In the standard dicom-style configuration (Left, Posterior, Superior):
- The x-axis increases along each row, and points towards the patient's left (i.e. towards the heart, away from the liver).
- The y-axis increase down each column, and points from the patient's front to back (posterior).
- The z-axis increases along the slice index, and points from the patient's feet to head (superior).

A canonical nifti-style array and affine can be obtained by running `Image.get_nifti_array_and_affine()`. By convention, this points in the same z direction but has x and y axes reversed (Right, Anterior, Superior). In the affine matrix, the x and y origins are therefore defined as being at the opposite end of the scale.

Note that positions can also be specified in terms of slice number:
- For x and y, slice number is just numpy array index + 1 (slice number ranges from 1 - n_voxels, whereas array index ranges from 0 - n_voxels-1)
- For z, by convention the slice numbers increases from 1 at the head to n_voxels at the feet, so it is in the opposite direction to the array index (convert as n_voxels[2] - idx).
