"""Test QuickViewer."""

import glob
import pytest
import os
from quickviewer import QuickViewer
from quickviewer.viewer import OrthogViewer
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def close_after(func):
    def do_then_close():
        func()
        plt.close("all")
    return do_then_close


@close_after
def test_single_image():
    qv = QuickViewer("data/ct.nii", show=False)
    assert len(qv.viewer) == 1


@close_after
def test_invalid_image():
    fake_im = "fake.nii"
    if os.path.exists(fake_im):
        os.remove(fake_im)
    qv = QuickViewer(["data/ct.nii", fake_im], show=False)
    assert len(qv.viewer) == 1
    qv = QuickViewer(fake_im, show=False)
    assert not len(qv.viewer)


@close_after
def test_duplicate_image():
    qv = QuickViewer(["data/ct.nii", "data/ct.nii"], show=False)
    assert len(qv.viewer) == 2
    assert not len(qv.slider_boxes)


@close_after
def test_multiple_images():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                      'data/MI_Translation/result.0.nii'], show=False)
    assert len(qv.viewer) == 2
    assert not len(qv.slider_boxes)


@close_after
def test_different_size_images():
    qv = QuickViewer(['data/MI_Translation/ct_planning.nii',
                      'data/MI_Translation/result.0.nii'], show=False)
    assert len(qv.viewer) == 2
    assert len(qv.slider_boxes) == 2


@close_after
def test_init_idx():
    init_sl = 50
    qv = QuickViewer("data/ct.nii", init_sl=init_sl, scale_in_mm=False,
                     show=False)
    assert qv.viewer[0].ui_slice.value == init_sl


@close_after
def test_custom_hu():
    v = (-500, 100)
    qv = QuickViewer("data/ct.nii", hu=v, show=False)
    assert qv.viewer[0].ui_hu.value == v


@close_after
def test_figsize():
    QuickViewer("data/ct.nii", figsize=10, show=False)


@close_after
def test_init_views():
    QuickViewer("data/ct.nii", init_view="x-y", show=False)
    QuickViewer("data/ct.nii", init_view="x-z", show=False)
    QuickViewer("data/ct.nii", init_view="y-z", show=False)


@close_after
def test_suptitle():
    title = "test"
    qv = QuickViewer("data/ct.nii", suptitle=title, show=False)
    assert qv.suptitle == title


@close_after
def test_translation():
    QuickViewer(['data/MI_Translation/ct_relapse.nii',
                 'data/MI_Translation/result.0.nii'],
                translation=True, show=False)


@close_after
def test_cb():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                      'data/MI_Translation/result.0.nii'],
                     show_cb=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_overlay():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                      'data/MI_Translation/result.0.nii'],
                     show_overlay=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_diff():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                 'data/MI_Translation/result.0.nii'],
                show_diff=True, show=False)
    assert len(qv.comparison) == 1


@close_after
def test_comparison_only():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                      'data/MI_Translation/result.0.nii'],
                     show_cb=True, show_diff=True, show_overlay=True,
                     comparison_only=True, show=False)
    assert len(qv.comparison) == 3


@close_after
def test_titles():
    title = ["test1", "test2"]
    qv = QuickViewer(["data/ct.nii", "data/ct.nii"], title=title, show=False)
    assert qv.viewer[0].im.title == title[0]
    assert qv.viewer[1].im.title == title[1]


@close_after
def test_mask():
    QuickViewer("data/ct.nii", mask=("data/structs/RTSTRUCT_CT_20140715_113632"
                                     "_002_alterio_pcs.nii.gz"), show=False)


@close_after
def test_dose():
    opacity = 0.3
    cmap = "gray"
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                     dose="data/MI_BSpline30/spatialJacobian.nii",
                     dose_kwargs={"cmap": cmap},
                     dose_opacity=opacity, show=False)
    assert qv.viewer[0].ui_dose.value == opacity


@close_after
def test_share_slider():
    qv = QuickViewer(['data/MI_Translation/ct_relapse.nii',
                      'data/MI_Translation/result.0.nii'], share_slider=False,
                      show=False)
    assert len(qv.viewer) == 2
    assert len(qv.slider_boxes) == 2


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_structs():

    # Test directory
    qv = QuickViewer("data/ct.nii", structs="data/structs", show=False)
    assert len(qv.viewer[0].im.structs) == len(os.listdir("data/structs")) - 1

    # Test list of files
    qv = QuickViewer(
        "data/ct.nii",
        structs=[
            "data/structs/RTSTRUCT_CT_20140715_113632_002_mpc.nii.gz",
            "data/structs/RTSTRUCT_CT_20140715_113632_002_right_smg.nii.gz"
        ], show=False)
    assert len(qv.viewer[0].im.structs) == 2

    # Test wildcard directory
    qv = QuickViewer("data/ct.nii", structs="data/str*", show=False)
    assert len(qv.viewer[0].im.structs) == len(os.listdir("data/structs")) - 1

    # Test wildcard files
    qv = QuickViewer("data/ct.nii", show=False,
                     structs=["data/structs/*parotid*", "data/structs/*mpc*"])
    assert len(qv.viewer[0].im.structs) == 3


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_struct_colours():

    colour = 'cyan'

    # Test with wildcard filename
    qv = QuickViewer("data/ct.nii", structs="data/structs/*parotid*",
                     struct_colours={"*parotid*": colour},
                     show=False)
    assert len(qv.viewer[0].im.structs) == 2
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)

    # Test with structure name
    qv = QuickViewer("data/ct.nii", structs="data/structs/*right_parotid*",
                     struct_colours={"right parotid": colour},
                     show=False)
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)

    # Test with wildcard structure name
    qv = QuickViewer("data/ct.nii", structs="data/structs/*right_parotid*",
                     struct_colours={"*parotid": colour}, show=False)
    assert qv.viewer[0].im.structs[0].color == to_rgba(colour)


@close_after
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_struct_mask():
    opacity = 0.6
    qv = QuickViewer("data/ct.nii", structs="data/structs/*mpc*",
                     struct_plot_type="mask", struct_opacity=opacity,
                     show=False)
    assert qv.viewer[0].ui_struct_opacity.value == opacity


@close_after
def test_zoom():
    QuickViewer("data/ct.nii", zoom=2, show=False)


@close_after
def test_downsample():
    QuickViewer("data/ct.ni", downsample=(5, 4, 2), show=False)


@close_after
def test_jacobian():
    opacity = 0.2
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                jacobian="data/MI_BSpline30/spatialJacobian.nii",
                jacobian_opacity=opacity, show=False)
    assert qv.viewer[0].ui_jac_opacity.value == opacity


@close_after
def test_df_grid():
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                df="data/MI_BSpline30/deformationField.nii", show=False)
    assert qv.viewer[0].ui_df.value == "grid"


@close_after
def test_df_quiver():
    qv = QuickViewer("data/MI_BSpline30/result.0.nii",
                df="data/MI_BSpline30/deformationField.nii",
                df_plot_type="quiver", show=False)
    assert qv.viewer[0].ui_df.value == "quiver"


@close_after
def test_save():
    output = "data/test_march2.pdf"
    #  if os.path.isfile(output):
        #  os.remove(output)
    QuickViewer("data/ct.nii", save_as=output, show=False)
    #  assert os.path.isfile(output)


@close_after
def test_orthog_view():
    qv = QuickViewer("data/ct.nii", orthog_view=True, show=False)
    assert isinstance(qv.viewer[0], OrthogViewer)


@close_after
def test_plots_per_row():
    qv = QuickViewer(["data/ct.nii", "data/ct.nii"], plots_per_row=1,
                     show=False)
