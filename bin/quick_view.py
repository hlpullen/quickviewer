"""Script for running QuickViewer from the command line."""

import argparse

from quickviewer import QuickViewer


parser = argparse.ArgumentParser()

# Get image paths
parser.add_argument("nii", nargs="+", type=str,
                    help="Path(s) to image file(s)")

#  Get initial viewing options
parser.add_argument("--init_sl", "-is", type=float, metavar="POS",
                    help="Initial slice position to display")
parser.add_argument("--init_idx", "-ii", type=int, metavar="IDX",
                    help="Initial slice index to display")
parser.add_argument("--init_view", "-iv", type=str, metavar="VIEW",
                    choices=["x-y", "y-z", "x-z"],
                    help="Initial view (x-y, y-z, or x-z)")
parser.add_argument("-v", nargs=2, type=float, metavar="V",
                    help="HU min and max")
parser.add_argument("--figsize", "-f", type=float, metavar="N",
                    help="Figure size")
parser.add_argument("--continuous_update", "-cu", action="store_true",
                    help="Allow continuous update")
parser.add_argument("--title", "-t", type=str, nargs="+",
                    help="Title(s)")
parser.add_argument("--cmap", "-cm", type=str, help="Colormap")
parser.add_argument("--no_share_slider", "-nss", action="store_false",
                    help="Turn off slider sharing for images with same shape")
parser.add_argument("--zoom", "-z", type=float, metavar="N",
                    help="Level of zoom")
parser.add_argument("--downsample", "-ds", nargs="+", type=float,
                    metavar="N",
                    help="Downsampling")
parser.add_argument("--suptitle", "-su", type=str, help="Suptitle")
parser.add_argument("--save_as", "-s", type=str, metavar="NAME",
                    help="Output file name")
parser.add_argument("--no_mm", "-nmm", action="store_false",
                    help="Display scale in voxels rather than mm")
parser.add_argument("--match_axes", "-ma", type=str,
                    metavar="METHOD", choices=["largest", "smallest"],
                    help="Method in which to match axes scales (smallest/largest)"
                    "for multiple plots")
parser.add_argument("--colorbar", "-c", action="store_true",
                    help="Turn on colorbars")
parser.add_argument("--interpolation", "-in", type=str, metavar="INTERP",
                    help="Interpolation method")
parser.add_argument("--orthog_view", "-ov", action="store_true",
                    help="Display sagittal view next to axial view")
parser.add_argument("--no_show", "-n", action="store_false",
                    help="Don't show figure in matplotlib window")

# Get mask options
parser.add_argument("--mask", "-m", nargs="+",
                    metavar="FILE",
                    help="Path(s) to mask file(s)")
parser.add_argument("--invert_mask", "-mi", action="store_true", 
                    help="Invert mask")
parser.add_argument("--mask_colour", "-mc", type=str,
                    help="Colour in which to display masked areas")

# Get comparison options
parser.add_argument("--show_cb", "-cb", action="store_true",
                    help="Show chequerboard")
parser.add_argument("--cb_splits", "-cbs", type=int, metavar="N",
                    help="Number of splits in the chequerboard")
parser.add_argument("--show_overlay", "-o", action="store_true",
                    help="Show overlay")
parser.add_argument("--overlay_opacity", "-oo", type=float,
                    metavar="OPACITY",
                    help="Show overlay")
parser.add_argument("--overlay_legend", "-ol", action="store_true",
                    help="Turn on overlay legend")
parser.add_argument("--show_diff", "-di", action="store_true",
                    help="Show difference")
parser.add_argument("--comparison_only", "-co", action="store_true",
                    help="Only show comparison images")

# Get dose options
parser.add_argument("--dose", "-d", nargs="+", type=str,
                    help="Path(s) to dose file(s)")
parser.add_argument("--dose_opacity", "-do", type=float, metavar="OPACITY",
                    help="Dose opacity")
parser.add_argument("--dose_cmap", "-dc", type=str, help="Dose colormap", 
                    metavar="CMAP")

# Get structure options
parser.add_argument("--structs", "-st", nargs="+", type=str, metavar="FILE",
                    help="Path(s) to structure file(s) (can be wildcard)")
parser.add_argument("--struct_plot_type", "-sp", type=str,
                    choices=["contour", "mask"], metavar="TYPE",
                    help="Structure plot type (contour/mask)")
parser.add_argument("--struct_linewidth", "-sl", type=float, metavar="WIDTH",
                    help="Structure contour linewidth")
parser.add_argument("--struct_opacity", "-so", type=float, metavar="OPACITY",
                    help="Structure mask opacity")
parser.add_argument("--no_struct_legend", "-nsl", action="store_false",
                    help="Turn off structure legend")
parser.add_argument("--struct_colours", "-sc", nargs="+",
                    metavar="NAME COLOUR",
                    help="Pairs of structure names and corresponding colours")
parser.add_argument("--legend_loc", "-l", type=str, metavar="LOC", help="Legend location")
parser.add_argument("--structs_as_mask", "-sm", action="store_true",
                    help="Mask image with structures")

# Deformation field options
parser.add_argument("--jacobian", "-j", nargs="+", type=str, metavar="FILE",
                    help="Path(s) to Jacobian determinant file(s)")
parser.add_argument("--jacobian_cmap", "-jc", type=str, metavar="CMAP",
                    help="Jacobian determinant colormap")
parser.add_argument("--jacobian_opacity", "-jo", type=float, metavar="OPACITY",
                    help="Jacobian determinant opacity")
parser.add_argument("--df", "-df", type=str, nargs="+", metavar="FILE",
                    help="Path(s) to deformation field file(s)")
parser.add_argument("--df_spacing", "-dfs", type=int, nargs="*", metavar="N",
                    help="Deformation grid spacing")
parser.add_argument("--df_plot_type", "-dfp", type=str, metavar="TYPE",
                    choices=["quiver", "grid"],
                    help="Deformation field plotm type (quiver/grid)")
parser.add_argument("--df_linespec", "-dfl", type=str, metavar="SPEC",
                    help="Linespec for deformation grid")

# Parse arguments
kwargs = vars(parser.parse_args())
kwargs = {k: v for k, v in kwargs.items() if v is not None}

# Swap negative booleans for positive
swaps = {
    "no_share_slider": "share_slider",
    "no_struct_legend": "struct_legend",
    "no_mm": "scale_in_mm",
    "no_show": "show"
}
for sw1, sw2 in swaps.items():
    if sw1 in kwargs:
        kwargs[sw2] = kwargs[sw1]
        del kwargs[sw1]

# Parse structure colours
if "struct_colours" in kwargs:
    sc = kwargs["struct_colours"]
    if len(sc) % 2 != 0:
        raise SyntaxError("Please provide a colour for every structure name!")
    new_sc = {sc[2 * i]: sc[2 * i + 1] for i in range(0, int(len(sc) / 2))}
    kwargs["struct_colours"] = new_sc

# Set downsample and zoom to be single values
for opt in ["zoom", "downsample"]:
    if opt in kwargs:
        if len(kwargs[opt]) == 1:
            kwargs[opt] = kwargs[opt][0]

# Launch QuickViewer
QuickViewer(**kwargs)
