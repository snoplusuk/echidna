""" Contains functions to view and interrogate chi-squared minimisation

Attributes:
  MAIN_FONT (dict): style properties for the main font to use in plot labels

"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colorbar import make_axes_gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy

import echidna.calc.decay as decay

MAIN_FONT = {"size": 22}
BOLD_FONT = {"size": 22, "weight": "bold"}


def chi_squared_vs_signal(signal_config, converter=None, fig_num=1,
                          n_atoms=None, peanalty=None, effective_mass=False,
                          half_life=False, save_as=None, show=False, **kwargs):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): Signal
        config class, where chi squareds have been stored.
      converter (:class:`echidna.calc.decay.DBIsotope`, optional): Converter
        used to convert between counts and half-life/effective mass.
      fig_num (int): Fig number. When creating multiple plots in the
        same script, ensures matplotlib doesn't overwrite them.
      n_atoms (float): Number of atoms for converter to use in
        calculations of half life or effective mass.
      penalty (:class:`echidna.limit.limit_config.LimitConfig`, optional):
        config for signal with penalty term.
      effective_mass (bool, optional): if True, plot the x-axis as the
        signal contribution effective mass.
      half_life (bool, optional): if True, plot the x-axis as the signal
        contribution half life.
      save_as (string, optional): Name of plot to save. All plots are
        saved in .png format.
      show (bool, optional): Display the plot to screen. Default is False.
      \**kwargs: Keyword arguments to pass to converter methods.

    Raises:
      TypeError: If 'half_life' or 'effective_mass' keyword arguments
        are used without :class:`echidna.calc.decay.DBIsotope` object
        to use as converter.

    Returns:
      matplotlib.pyplot.figure: Plotted figure.
    """
    if (converter is None and half_life or effective_mass):
        raise TypeError("converter is None. Cannot use 'half_life' or "
                        "'effective_mass' keywords without converter")
    # Fig. 1 (axes generated automatically)
    fig = plt.figure(fig_num, figsize=(10, 10))

    # X axis values
    if effective_mass:
        x = numpy.zeros(shape=(signal_config.get_chi_squareds()[2].shape))
        for i_bin, count in enumerate(signal_config.get_chi_squareds()[2]):
            effective_mass = converter.counts_to_eff_mass(count, **kwargs)
            x[i_bin] = effective_mass
        plt.xlabel(r"$m_{\beta\beta}$", **BOLD_FONT)
    elif half_life:
        x = numpy.zeros(shape=(signal_config.get_chi_squareds()[2].shape))
        for i_bin, count in enumerate(signal_config.get_chi_squareds()[2]):
            x.append(1./converter.counts_to_half_life(count, **kwargs))
        plt.xlabel(r"$1/T_{1/2}^{0\nu}$", **BOLD_FONT)
    else:
        x = signal_config.get_chi_squareds()[2]
        plt.xlabel("Signal counts", **BOLD_FONT)
    # Y axis values
    y_1 = signal_config.get_chi_squareds()[0]
    plt.ylabel(r"$\chi^{2}$", **BOLD_FONT)
    if penalty:
        y_2 = penalty.get_chi_squareds()[0]
        plt.plot(x, y_1, "bo-", label="no systematic uncertainties")
        # lines and dots
        plt.plot(x, y_2, "ro-", label="systematic uncertainties")
        plt.legend(loc="upper left")
    else:
        plt.plot(x, y_1, "o-")  # lines and dots

    # Set the tick labels, via Axes instance
    ax = fig.gca()  # Get current Axes instance
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # Set other properties here e.g. colour, rotation
        label.set_fontsize(MAIN_FONT.get("size"))

    if save_as:
        plt.savefig(save_as + ".png", dpi=400)
    if show:
        plt.show()
    return fig


def chi_squared_map(syst_analyser, fig_num=1, preferred_values=True,
                    minima=True, contours=False, save_as=None):
    """ Plot chi squared surface for systematic vs. signal counts

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): A
        systematic analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from hdf5
      fig_num (int): Fig number. When creating multiple plots in the
        same script, ensures matplotlib doesn't overwrite them.
      preferred_values (bool, optional): if False "preferred values" curve
        is not overlayed on colour map. Default is True.
      minima (bool, optional): if False "minima" are not overlayed on
        colour map. Default is True.
      contours (bool, optional): if True produces a contour plot of chi
        squared surface. Default is False.
      save_as (string, optional): Name of plot to save. All plots are
        saved with in .png format.

      Default is to produce a colour map, with "preferred values" curve
      and "minima" overlayed.

    Returns:
      matplotlib.pyplot.figure: Plotted figure.
    """
    # Set x and y axes
    x = syst_analyser.get_actual_counts()
    y = syst_analyser.get_syst_values()

    # Set chi squared map values
    data = numpy.average(syst_analyser.get_chi_squareds(), axis=1)
    data = numpy.transpose(data)  # transpose it so that axes are correct

    # Set preferred value values
    y_2 = numpy.average(syst_analyser.get_preferred_values(), axis=1)

    # Set minima values
    x_3 = syst_analyser.get_minima()[0]
    y_3 = syst_analyser.get_minima()[1]

    # Create meshgrid
    X, Y = numpy.meshgrid(x, y)

    # Set sensible levels, pick the desired colormap and define normalization
    color_map = plt.get_cmap('hot_r')

    linear = numpy.linspace(numpy.sqrt(data.min()), numpy.sqrt(data.max()),
                            num=100)
    locator = FixedLocator(linear**2)
    levels = locator.tick_values(data.min(), data.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)

    if contours:
        fig = plt.figure(fig_num, figsize=(16, 10))  # Fig. 2
        fig.text(0.1, 0.9, syst_analyser._name, **BOLD_FONT)
        ax = Axes3D(fig)
        ax.view_init(elev=17.0, azim=-136.0)  # set intial viewing position

        # Plot surface
        surf = ax.plot_surface(X, Y, data, rstride=1, cstride=1,
                               cmap=color_map, norm=norm, linewidth=0,
                               antialiased=False)
        ax.zaxis.set_minor_locator(locator)
        ax.ticklabel_format(style="scientific", scilimits=(3, 4))

        # Set axis labels
        ax.set_xlabel("\nSignal counts", **BOLD_FONT)
        ax.set_ylabel("\nValue of systematic", **BOLD_FONT)
        for label in (ax.get_xticklabels() +
                      ax.get_yticklabels() +
                      ax.get_zticklabels()):
            label.set_fontsize(MAIN_FONT.get("size"))  # tick label size

        ax.dist = 11  # Ensures tick labels are not cut off
        ax.margins(0.05, 0.05, 0.05)  # Adjusts tick margins

        # Draw colorbar
        color_bar = fig.colorbar(surf, ax=ax, orientation="vertical",
                                 fraction=0.2, shrink=0.5, aspect=10)
        # kwargs here control axes that the colorbar is drawn in
        color_bar.set_label(r"$\chi^2$", size=MAIN_FONT.get("size"))
        color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))

        plt.show()
        if save_as:
            fig.savefig(save_as + "_contour.png", dpi=300)
    else:
        fig = plt.figure(fig_num, figsize=(12, 10))  # Fig. 2
        fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
        ax = fig.add_subplot(1, 1, 1)

        # Set labels
        ax.set_xlabel("Signal counts", **BOLD_FONT)
        ax.set_ylabel("Value of systematic", **BOLD_FONT)

        # Plot color map
        color_map = ax.pcolormesh(X, Y, data, cmap=color_map, norm=norm)
        color_bar = fig.colorbar(color_map)
        color_bar.set_label("$\chi^2$", size=MAIN_FONT.get("size"))
        # tick label size
        color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))

        # Set axes limits
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])

        if preferred_values:
            ax.plot(x, y_2, "bo-", label="Preferred values")
        if minima:
            ax.plot(x_3, y_3, "ko", label="Minima")

        # Set axes tick label size
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(MAIN_FONT.get("size"))

        ax.legend(loc="upper left")
        if save_as:
            fig.savefig(save_as + "_color_map.png", dpi=300)
    return fig


def penalty_vs_systematic(syst_analyser, fig_num=1, save_as=None):
    """ Plot penalty_value vs. systematic

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): A
        systematic analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from hdf5
      fig_num (int, optional): Fig number. When creating multiple plots in the
        same script, ensures matplotlib doesn't overwrite them.
      save_as (string, optional): Name of plot to save. All plots are
        saved with in .png format.

    Returns:
      matplotlib.pyplot.figure: Plotted figure.
    """
    fig = plt.figure(fig_num, figsize=(9, 7))  # Fig. 3
    fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
    ax = fig.add_subplot(1, 1, 1)

    x = syst_analyser._penalty_values[0]
    y = syst_analyser._penalty_values[1]
    plt.xlabel("Value of systematic", **BOLD_FONT)
    plt.ylabel("Value of penalty term", **BOLD_FONT)
    plt.plot(x, y, "bo")

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        # Set other properties here e.g. colour, rotation
        label.set_fontsize(MAIN_FONT.get("size"))

    if save_as:
        plt.savefig(kwagrs.get("save_as") + ".png")
    return fig


def turn_on(syst_analyser, signal_config, fig=1, save_as=None):
    """ Plot deviation from chi-squared with no floated systematics.

    When does the effect of floating the systematic "turn on"?

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): A
        systematic analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from hdf5.
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): Signal
        config class, where chi squareds have been stored.
      fig_num (int): Fig number. When creating multiple plots in the
        same script, ensures matplotlib doesn't overwrite them.
      save_as (string, optional): Name of plot to save. All plots are
        saved with in .png format.

    Returns:
      matplotlib.pyplot.figure: Plotted figure.
    """
    # Set x and y axes
    x = syst_analyser.get_actual_counts()
    y = syst_analyser.get_syst_values()

    # Set chi squared map values
    data = numpy.average(syst_analyser.get_chi_squareds(), axis=1)
    data = numpy.transpose(data)  # transpose it so that axes are correct

    # Create meshgrid
    X, Y = numpy.meshgrid(x, y)

    # Define an array of \chi_0 values - chi squared without
    # floating systematics

    chi_squareds = signal_config.get_chi_squareds()[0]
    data_np = numpy.zeros(data.shape)  # zeroed array the same size as data
    for y in range(len(data_np)):
        for x, chi_squared in enumerate(chi_squareds):
            data_np[y][x] = chi_squared
    #if numpy.any((numpy.average(data_np, axis=0) != chi_squareds)):
    #    raise AssertionError("Incorrect chi squareds (no floating) array.")

    # Make an array of the offsets
    offsets = data - data_np

    # Set sensible levels, pick the desired colormap and define normalization
    color_map = plt.get_cmap('coolwarm')

    positives = numpy.linspace(numpy.log10(offsets.max())*-1.,
                               numpy.log10(offsets.max()), num=50)
    # linear array in log space
    if offsets.min() < 0.:
        negatives = numpy.linspace(offsets.min(), 0.0, num=51)
    else:
        negatives = numpy.zeros((51))

    # Add the positive part to the negative part
    full_scale = numpy.append(negatives, numpy.power(10, positives))
    locator = FixedLocator(full_scale)
    levels = locator.tick_values(offsets.min(), offsets.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)
    fig = plt.figure(fig, figsize=(12, 10))  # Fig. 4
    fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
    ax = fig.add_subplot(1, 1, 1)

    # Set labels
    ax.set_xlabel("Signal counts", **BOLD_FONT)
    ax.set_ylabel("Value of systematic", **BOLD_FONT)

    # Plot color map
    color_map = ax.pcolormesh(X, Y, offsets, cmap=color_map, norm=norm)
    color_bar = fig.colorbar(color_map)
    color_bar.set_label("$\chi^2 - \chi_0^2$", size=MAIN_FONT.get("size"))
    # tick label size
    color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))

    # Set axes limits
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])

    # Set axes tick label size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))

    ax.legend(loc="upper left")
    if save_as:
        fig.savefig(save_as + "_turn_on.png", dpi=300)
    return fig


def push_pull(syst_analyser, fig=1, save_as=None):
    """ Plot penalty value - poisson likelihood chi squared.

    When does minimising chi squared, which wants to "pull" the away
    from the data/prior value dominate and when does the penalty term,
    which wants to "pull" towards the data/prior, constraining the fit
    dominate?

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): A
        systematic analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from hdf5
      fig_num (int): Fig number. When creating multiple plots in the
        same script, ensures matplotlib doesn't overwrite them.
      save_as (string, optional): Name of plot to save. All plots are
        saved with in .png format.

    Returns:
      matplotlib.pyplot.figure: Plotted figure.
    """
    # Set x and y axes
    x = syst_analyser.get_actual_counts()
    y = syst_analyser.get_syst_values()

    # Set chi squared map values
    data = numpy.average(syst_analyser.get_chi_squareds(), axis=1)
    data = numpy.transpose(data)  # transpose it so that axes are correct

    # Create meshgrid
    X, Y = numpy.meshgrid(x, y)

    # Define an array penalty values
    penalty_values = syst_analyser._penalty_values[1, 0:len(y)]
    # zeroed array the same size as data
    penalty_array = numpy.zeros(data.shape)
    for y, penalty_value in enumerate(penalty_values):
        for x in range(len(penalty_array[y])):
            penalty_array[y][x] = penalty_value

    # Define the push pull array penalty term - chi_squared
    # --> push_pull > 0 when penalty_value > chi_squared
    # --> push_pull < 1 when penalty_value < chi_squared
    push_pull = (2.*penalty_array) - data

    # Set sensible levels, pick the desired colormap and define normalization
    color_map = plt.get_cmap('coolwarm')

    if push_pull.min() < 0.:
        negatives = numpy.linspace(push_pull.min(), 0.,
                                   num=50, endpoint=False)
    else:
        negatives = numpy.zeros((50))
    if push_pull.max() > 0.:
        positives = numpy.linspace(0., push_pull.max(), num=51)
    else:
        positives = numpy.zeros((51))

    # Add the pull part to the push part
    full_scale = numpy.append(negatives, positives)
    locator = FixedLocator(full_scale)
    levels = locator.tick_values(push_pull.min(), push_pull.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)
    fig = plt.figure(fig, figsize=(12, 10))  # Fig. 4
    fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
    ax = fig.add_subplot(1, 1, 1)

    # Set labels
    ax.set_xlabel("Signal counts", **BOLD_FONT)
    ax.set_ylabel("Value of systematic", **BOLD_FONT)

    # Plot color map
    color_map = ax.pcolormesh(X, Y, push_pull, cmap=color_map, norm=norm)
    color_bar = fig.colorbar(color_map)
    color_bar.set_label("$s-\chi^{2}_{\lambda,p}$", size=MAIN_FONT.get("size"))
    # tick label size
    color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))

    # Set axes limits
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])

    # Set axes tick label size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))

    ax.legend(loc="upper left")
    if save_as:
        fig.savefig(save_as + "_push_pull.png", dpi=300)
    return fig


def main(args):
    """ Script to produce chi squared plots for a given systematic.

    .. note:: Produces

      * Plot of chi squared vs. signal counts
      * Plot of systematic vs. signal chi squared surface, either
        contour plot or color map
      * Plot of systematic value vs. penalty term value

    Args:
      args (dict): command line arguments from argparse.

    """
    # Load signal config from hdf5
    signal_config = LimitConfig(0, [0])
    signal_config = store.load_ndarray(args.signal_config, signal_config)
    if args.penalty_config is not None:
        penalty_config = LimitConfig(0, [0])
        penalty_config = store.load_ndarray(args.penalty_config,
                                            penalty_config)
    else:
        penalty_config = None

    # Loaf systematic analyser from hdf5
    syst_analyser = SystAnalyser("", numpy.zeros((1)), numpy.zeros((1)))
    syst_analyser = store.load_ndarray(args.syst_analyser, syst_analyser)

    # Produce plots
    # Currently not possible to produce chi squared vs signal plot with half
    # life or effective mass on x-axis, from outside of limit setting code.
    # Just produce with signal counts on x-axis here.
    fig_1 = chi_squared_vs_signal(signal_config, fig_num=1,
                                  penalty=penalty_config,
                                  save_as=args.image_name)
    fig_2 = penalty_vs_systematic(syst_analyser, 2)
    fig_3 = turn_on(syst_analyser, signal_config, 3, save_as=args.image_name)
    fig_4 = push_pull(syst_analyser, 4, save_as=args.image_name)
    fig_5 = chi_squared_map(syst_analyser, 5,
                            contours=args.contours,
                            save_as=args.image_name)

    plt.show()
    raw_input("RETURN to exit")


if __name__ == "__main__":
    import echidna.output.store as store
    from echidna.limit.limit_config import LimitConfig
    from echidna.limit.limit_setting import SystAnalyser
    from echidna.scripts.zero_nu_limit import ReadableDir

    import argparse

    parser = argparse.ArgumentParser(description="Produce chi squared plots "
                                     "for a systematic")
    parser.add_argument("-s", "--signal_config", action=ReadableDir,
                        help="Supply location of signal config hdf5 file")
    parser.add_argument("-p", "--penalty_config", action=ReadableDir,
                        help="Supply location of signal config with "
                        "penalty term")
    parser.add_argument("-a", "--syst_analyser", action=ReadableDir,
                        help="Supply location of syst analyser hdf5 file")
    parser.add_argument("-i", "--image_name", type=str, default="output",
                        help="Supply an image name")
    parser.add_argument("-c", "--contours", action="store_true",
                        help="If true produces a contour plot, "
                        "defualt is colour map")
    args = parser.parse_args()

    main(args)
