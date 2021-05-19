"""Code to make a synthetic patient."""

import os
import numpy as np
from quickviewer.simulation import GeometricNifti


def make_patient(outdir):
    """Create a random patient image."""

    ##################
    # Image settings #
    ##################
    n_voxels = [256, 256, round(np.random.uniform(50, 80))]
    origin = [-n_voxels[0] / 2, -n_voxels[1] / 2, 0]
    voxel_sizes = [1, 1, 3]
    length = [n_voxels[i] * voxel_sizes[i] for i in range(3)]
    noise_std = 10

    ####################################
    # Randomised anatomical parameters #
    ####################################
    # Head
    head_height = np.random.uniform(100, 170)
    head_radius = np.random.uniform(40, 80)

    # Ears
    ear_offset_x = np.random.uniform(-5, 5)
    ear_offset_y = np.random.uniform(-5, 5)
    ear_offset_z = np.random.uniform(-5, 10)
    ear_size_x = np.random.uniform(20, 60)
    ear_size_y = np.random.uniform(5, 20)
    ear_size_z = np.random.uniform(40, 60)

    # Eyes
    eye_angle = np.random.uniform(15, 40)
    eye_radius = np.random.uniform(5, 15)
    eye_offset_z = np.random.uniform(0, 10)

    # Teeth
    teeth_bottom_row_from_chin = np.random.uniform(10, 20)
    teeth_row_spacing = np.random.uniform(5, 15)
    teeth_angle_spacing = np.random.uniform(5, 10)
    teeth_size = np.random.uniform(5, 10)
    teeth_radius_frac = np.random.uniform(0.7, 0.9)


    ##############
    # Make image #
    ##############
    # Background and head
    nii = GeometricNifti(n_voxels, voxel_sizes=voxel_sizes, origin=origin, noise_std=noise_std)
    centre = nii.get_image_centre()
    nii.add_cylinder(radius=head_radius, length=head_height, group='head')

    # Add ears
    for i in [-1, 1]:
        nii.add_cuboid(
            side_length=[ear_size_x, ear_size_y, ear_size_z],
            centre=[
                centre[0] + i * (head_radius + ear_offset_x),
                centre[1] + ear_offset_y,
                centre[2] + ear_offset_z
            ],
            group='head'
        )

    # Add eyes
    for i, name in zip([-1, 1], ['right', 'left']):
        nii.add_sphere(
            radius=eye_radius,
            centre=[
                centre[0] + head_radius * np.sin(np.radians(eye_angle)) * i,
                centre[1] + head_radius * np.cos(np.radians(eye_angle)),
                centre[2] + eye_offset_z
            ],
            intensity=40,
            name='eye_' + name
        )

    # Add teeth
    for i in [-2, -1, 1, 2]:
        radius = head_radius * teeth_radius_frac
        angle = np.sign(i) * np.radians((abs(i) - 0.5) * teeth_angle_spacing)
        z = centre[2] - head_height / 2 + teeth_bottom_row_from_chin
        nii.add_cube(
            side_length=teeth_size,
            centre=[
                centre[0] + np.sin(angle) * radius,
                centre[1] + np.cos(angle) * radius,
                z
            ],
            intensity=100,
            group='teeth'
        )
        nii.add_cube(
            side_length=teeth_size,
            centre=[
                centre[0] + np.sin(angle) * radius,
                centre[1] + np.cos(angle) * radius,
                z + teeth_row_spacing
            ],
            intensity=100,
            group='teeth'
        )

    # Rotate/translation
    translation = np.random.uniform(-20, 20, 3)
    rotation = np.random.uniform(-10, 10, 3)
    nii.translate(*translation)
    nii.rotate(*rotation)

    # Save image
    nii.write(outdir)


def make_many_patients(n, outdir="."):
    """Make n random patients and save to directory."""

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i in range(1, n + 1):
        patient_dir = f"{outdir}/{i}"
        make_patient(patient_dir)


if __name__ == "__main__":

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="Number of patients to create")
    parser.add_argument("-d", "--dir", type=str, help="Output directory", default=".")
    args = parser.parse_args()

    # Make patients
    make_many_patients(args.n, args.dir)
