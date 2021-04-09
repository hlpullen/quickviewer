"""
Script for creating regular grids in each orientation in the frame of reference
of a given image, then transforming those grids using a given elastix 
transformation.
"""

import re
import os
import nibabel
import subprocess

from imagetools.simulation import GeometricNifti


def replace_elastix_setting(infile, outfile, setting, val):
    """Create a copy of infile with the value of a setting replaced with val,
    and save to outfile."""

    assert outfile != infile
    if os.path.exists(outfile):
        os.remove(outfile)
    with open(infile, "r") as source:
        with open(outfile, "w") as out:
            lines = source.readlines()
            for line in lines:
                if setting not in line:
                    out.write(line)
                else:
                    new = re.sub(f'^\({setting}.*', f'({setting} "{val}")',
                                 line)
                    out.write(new)


def deform_grids(mpath, outdir, tp, thickness=3, nx=20):
    """
    Parameters
    ----------
    mpath : str
        Path to a NIfTI file containing the moving image from which to take the 
        frame of reference for the grids.

    outdir : str
        Directory in while the original and transformed grids will be saved.

    tp : str
        Elastix transform parameter file.

    thickness : int, default=3
        Thickness of the gridlines in mm.

    nx : int, default=20
        Number of squares to produce in the x-direction. The number of squares
        in other directions will be set automatically in order to produce a
        uniform grid.
    """

    # Get info from moving image
    nii = nibabel.load(mpath)
    affine = nii.affine
    origin = [affine[i, 3] for i in range(3)]
    voxel_sizes = [affine[i, i] for i in range(3)]
    shape = nii.get_fdata().shape
    grid_spacing = shape[1] / nx
    print(f"Making grids with spacing: {grid_spacing} mm")

    # Make outdir if needed
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Make new version of transform file with nearest neighbour interpolation
    new_tp = os.path.join(outdir, os.path.basename(tp))
    replace_elastix_setting(tp, new_tp, "ResampleInterpolator", 
                            "FinalNearestNeighborInterpolator")

    # Make deformed grid for each view
    for i, name in enumerate(["x-z", "y-z", "x-y"]):

        # Make regular grid image
        geo = GeometricNifti(shape, origin=origin, voxel_sizes=voxel_sizes)
        geo.add_grid(grid_spacing, thickness=thickness, axis=i)
        grid_file = os.path.join(outdir, f"grid_{name}.nii")
        geo.write(grid_file)

        # Transform the grid
        cmd = f"transformix -out {outdir} -tp {new_tp} -in {grid_file}"
        subprocess.run(cmd.split())
        os.rename(os.path.join(outdir, "result.nii"),
                  os.path.join(outdir, f"grid_transformed_{name}.nii"))


if __name__ == "__main__":

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mpath", type=str, help="Moving image path")
    parser.add_argument("outdir", type=str, help="Output directory")
    parser.add_argument("tp", type=str, help="Transform parameter file")
    parser.add_argument("--thickness", "-t", type=int, default=2, 
                        help="Gridline thickness in voxels")
    parser.add_argument("--thickness_xy", "-txy", type=int, 
                        help="Gridline thickness in the x/y directions in "
                             "voxels")
    parser.add_argument("--thickness_z", "-tz", type=int, 
                        help="Gridline thickness in the z direction in voxels")
    parser.add_argument("--nx", type=int, default=20, 
                        help="Number of squares in x direction")
    args = parser.parse_args()

    # Get thickness
    thickness = [args.thickness, args.thickness, args.thickness]
    if args.thickness_xy is not None:
        thickness[0] = args.thickness_xy
        thickness[1] = args.thickness_xy
    if args.thickness_z is not None:
        thickness[2] = args.thickness_z

    # Make deformed grids
    deform_grids(args.mpath, args.outdir, args.tp, thickness, args.nx)
