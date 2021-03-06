"""Class for comparing structures."""

import numpy as np
import matplotlib
import matplotlib
import matplotlib.pyplot as plt

from quickviewer.image import StructImage


class StructComparison:
    """Class for computing comparison metrics for two structures and plotting
    the structures together."""

    def __init__(self, struct1, struct2, **kwargs):
        """Initialise from a pair of StructImages, or load new StructImages.
        """

        for i, s in enumerate([struct1, struct2]):
            struct = s if isinstance(s, StructImage) \
                else StructImage(s, **kwargs)
            setattr(self, f"s{i + 1}", s)

        # Check both structres are valid and in same reference frame
        self.valid = self.s1.valid and self.s2.valid
        if not self.valid:
            return
        if not self.s1.same_frame(self.s2):
            raise TypeError(f"Comparison structures {self.s1.name} and "
                            f"{self.s2.name} are not in the same reference "
                            "frame!")

        # Ensure unique names are set
        if not hasattr(self.s1, "unique_name"):
            self.s1.set_unique_name([self.s2])
        if not hasattr(self.s2, "unique_name"):
            self.s2.set_unique_name([self.s1])

    def plot(
        self, 
        view, 
        sl=None, 
        pos=None, 
        ax=None, 
        mpl_kwargs=None, 
        plot_type="contour",
        zoom=None,
        zoom_centre=None,
        show=False
    ):
        """Plot comparison structures."""

        if not self.valid:
            return
        if mpl_kwargs is None:
            mpl_kwargs = {}

        # Make plot
        if plot_type == "contour":
            self.s1.plot_contour(view, sl, pos, ax, mpl_kwargs, zoom, 
                                 zoom_centre)
            self.s2.plot_contour(view, sl, pos, self.s1.ax, mpl_kwargs, zoom, 
                                 zoom_centre)
        elif plot_type == "mask":
            self.plot_mask(view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre)
        elif plot_type == "filled":
            mask_kwargs = {"alpha": mpl_kwargs.get("alpha", 0.3)}
            self.plot_mask(view, sl, pos, ax, mask_kwargs, zoom, zoom_centre)
            contour_kwargs = {"linewidth": mpl_kwargs.get("linewidth", 2)}
            self.s1.plot_contour(view, sl, pos, self.s1.ax, mpl_kwargs, zoom, 
                                 zoom_centre)
            self.s2.plot_contour(view, sl, pos, self.s1.ax, mpl_kwargs, zoom, 
                                 zoom_centre)

        if show:
            plt.show()

    def plot_mask(self, view, sl, pos, ax, mpl_kwargs, zoom, zoom_centre):
        """Plot two masks, with intersection in different colour."""

        # Set slice for both images
        self.s1.set_ax(view, ax, zoom=zoom)
        self.s1.set_slice(view, sl, pos)
        self.s2.set_slice(view, sl, pos)

        # Get differences and overlap
        diff1 = self.s1.current_slice & ~self.s2.current_slice
        diff2 = self.s2.current_slice & ~self.s1.current_slice
        overlap = self.s1.current_slice & self.s2.current_slice
        mean_col = np.array([np.array(self.s1.color), 
                             np.array(self.s2.color)]).mean(0)
        to_plot = [
            (diff1, self.s1.color),
            (diff2, self.s2.color),
            (overlap, mean_col)
        ]

        for im, color in to_plot:

            # Make colormap
            norm = matplotlib.colors.Normalize()
            cmap = matplotlib.cm.hsv
            s_colors = cmap(norm(im))
            s_colors[im > 0, :] = color
            s_colors[im == 0, :] = (0, 0, 0, 0)

            # Display mask
            self.s1.ax.imshow(
                s_colors,
                extent=self.s1.extent[view],
                aspect=self.s1.aspect[view],
                **self.s1.get_kwargs(mpl_kwargs, default=self.s1.mask_kwargs)
            )

        self.s1.adjust_ax(view, zoom, zoom_centre)

    def dice_score(self, view, sl):
        """Get dice score on a given slice."""

        if not self.s1.on_slice(view, sl) or not self.s2.on_slice(view, sl):
            return

        self.s1.set_slice(view, sl)
        self.s2.set_slice(view, sl)
        slice1 = self.s1.current_slice
        slice2 = self.s2.current_slice
        return (slice1 & slice2).sum() / np.mean([slice1.sum(), slice2.sum()])
