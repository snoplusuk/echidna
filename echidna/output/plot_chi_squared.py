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


def chi_squared_vs_signal(signal_config, fig=1, **kwargs):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
        config class, where chi squareds have been stored.

    .. note::

      Keyword arguments include:

        * penalty (:class:`echidna.limit.limit_config.LimitConfig`): config
          for signal with penalty term.
		* effective_mass (*bool*): if True, plot the x-axis as the
		  signal contribution effective mass.
		* half_life (*bool*): if True, plot the x-axis as the signal
          contribution half life.
    """
    fig = plt.figure(fig, figsize=(10, 10))  # Fig. 1 (axes generated automatically)

    # X axis values
    if kwargs.get("effective_mass"):
        effective_masses = numpy.zeros(shape=(signal_config.get_chi_squareds()[2].shape))
        te130_converter = decay.DBIsotope("Te130", 0.003, 129.906229, 127.6,
                                          0.3408, 3.69e-14, 4.03)
        n_atoms = te130_converter.get_n_atoms()
        for i_bin, count in enumerate(signal_config.get_chi_squareds()[2]):
            effective_mass = te130_converter.counts_to_mass(count, n_atoms,
                                                            5., roi_cut=True)
            effective_masses[i_bin] = effective_mass
        x = effective_masses  # Set x-axis
        plt.xlabel(r"$m_{\beta\beta}$", **BOLD_FONT)
    elif kwargs.get("half_life"):
        x = []
        # Decay variables
        Te130_atm_weight = 129.906229  # SNO+-doc-1728v2
        TeNat_atm_weight = 127.6  # SNO+-doc-1728v2
        Te130_abundance = 0.3408  # SNO+-doc-1728v2
        phase_space = 3.69e-14  # PRC 85, 034316 (2012)
        matrix_element = 4.03  # IBM-2 PRC 87, 014315 (2013)

        converter = decay.DBIsotope("Te130", Te130_atm_weight,
                                    TeNat_atm_weight, Te130_abundance,
                                    phase_space, matrix_element)
        for num_decays in signal_config._chi_squareds[1]:
            x.append(1./converter.counts_to_half_life(num_decays/0.554))
        pylab.xlabel(r"$1/T_{1/2}^{0\nu}$")
        pylab.ylabel(r"$\chi^{2}$")
    else:
        x = signal_config.get_chi_squareds()[2]
        plt.xlabel("Signal counts", **BOLD_FONT)
    # Y axis values
    y_1 = signal_config.get_chi_squareds()[0]
    plt.ylabel(r"$\chi^{2}$", **BOLD_FONT)
    if kwargs.get("penalty") is not None:
        y_2 = kwargs.get("penalty")._chi_squareds[0]
        axis.plot(x, y_1, "bo-", label="no systematic uncertainties")
        axis.plot(x, y_2, "ro-", label="systematic uncertainties")  # lines and dots
        axis.legend(loc="upper left")
    else:
        plt.plot(x, y_1, "o-")  # lines and dots

    # Set the tick labels, via Axes instance
    ax = fig.gca()  # Get current Axes instance
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))  # Set other properties here e.g. colour, rotation

    if kwargs.get("save_as") is not None:
        plt.savefig(kwargs.get("save_as") + ".png", dpi=400)
    return fig


def chi_squared_map(syst_analyser, fig=1, **kwargs):
    """ Plot chi squared surface for systematic vs. signal counts

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): systematic
        analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from
        hdf5

    .. note::

      Keyword arguments include:

        * contours (*bool*): if True produces a contour plot of chi
          squared surface. Default (*False*).
        * preferred_values (*bool*): if False "preferred values" curve
          is not overlayed on colour map. Default (*True*)
        * minima (*bool*): if False "minima" are not overlayed on
          colour map. Default (*True*)
        * save_as (*string*): supply file name to save image

      Default is to produce a colour map, with "preferred values" curve
      and "minima" overlayed.
    """
    # Set kwargs defaults
    if kwargs.get("preferred_values") is None:
        kwargs["preferred_values"] = True
    if kwargs.get("minima") is None:
        kwargs["minima"] = True

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

    if kwargs.get("contours"):
        fig = plt.figure(fig, figsize=(16, 10))  # Fig. 2
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
        if kwargs.get("save_as") is not None:
            fig.savefig(kwargs.get("save_as") + "_contour.png", dpi=300)
    else:
        fig = plt.figure(fig, figsize=(12, 10))  # Fig. 2
        fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
        ax = fig.add_subplot(1, 1, 1)

        # Set labels
        ax.set_xlabel("Signal counts", **BOLD_FONT)
        ax.set_ylabel("Value of systematic", **BOLD_FONT)

        # Plot color map
        color_map = ax.pcolormesh(X, Y, data, cmap=color_map, norm=norm)
        color_bar = fig.colorbar(color_map)
        color_bar.set_label("$\chi^2$", size=MAIN_FONT.get("size"))
        color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))  # tick label size

        # Set axes limits
        ax.set_xlim([X.min(), X.max()])
        ax.set_ylim([Y.min(), Y.max()])

        if kwargs.get("preferred_values"):
            ax.plot(x, y_2, "bo-", label="Preferred values")
        if kwargs.get("minima"):
            ax.plot(x_3, y_3, "ko", label="Minima")

        # Set axes tick label size
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(MAIN_FONT.get("size"))

        ax.legend(loc="upper left")
        if kwargs.get("save_as") is not None:
            fig.savefig(kwargs.get("save_as") + "_color_map.png", dpi=300)
    return fig


def penalty_vs_systematic(syst_analyser, fig=1, **kwargs):
    """ Plot penalty_value vs. systematic

    Args:
      syst_analyser (:class:`echidna.limit.limit_setting.SystAnalyser`): systematic
        analyser object, created during limit setting. Can be used
        during limit setting setting or can load an instance from
        hdf5

    .. note::

      Keyword arguments include:

        * save_as (*string*): supply file name to save image
    """
    fig = plt.figure(fig, figsize=(9, 7))  # Fig. 3
    fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
    ax = fig.add_subplot(1, 1, 1)

    x = syst_analyser._penalty_values[0]
    y = syst_analyser._penalty_values[1]
    plt.xlabel("Value of systematic", **BOLD_FONT)
    plt.ylabel("Value of penalty term", **BOLD_FONT)
    plt.plot(x, y, "bo")

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))  # Set other properties here e.g. colour, rotation

    if kwargs.get("save_as") is not None:
        plt.savefig(kwagrs.get("save_as"))
    return fig


def turn_on(syst_analyser, signal_config, fig=1, **kwargs):
    """
    """
    # Set x and y axes
    x = syst_analyser.get_actual_counts()
    y = syst_analyser.get_syst_values()

    # Set chi squared map values
    data = numpy.average(syst_analyser.get_chi_squareds(), axis=1)
    data = numpy.transpose(data)  # transpose it so that axes are correct

    # Create meshgrid
    X, Y = numpy.meshgrid(x, y)

    # Define an array of \chi_0 values - chi squared without floating systematics
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

    positives = numpy.linspace(numpy.log10(offsets.max())*-1,
                               numpy.log10(offsets.max()), num=50)
    # linear array in log space
    if offsets.min() < 0:
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
    ax.set_ylabel("$\chi^{2} - \chi_{0}^{2}$", **BOLD_FONT)

    # Plot color map
    color_map = ax.pcolormesh(X, Y, offsets, cmap=color_map, norm=norm)
    color_bar = fig.colorbar(color_map)
    color_bar.set_label("$\chi^2 - \chi_0^2$", size=MAIN_FONT.get("size"))
    color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))  # tick label size

    # Set axes limits
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])

    # Set axes tick label size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))

    ax.legend(loc="upper left")
    if kwargs.get("save_as") is not None:
        fig.savefig(kwargs.get("save_as") + "_turn_on.png", dpi=300)
    return fig


def push_pull(syst_analyser, fig=1, **kwargs):
    """
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
    penalty_array = numpy.zeros(data.shape)  # zeroed array the same size as data
    for y, penalty_value in enumerate(penalty_values):
        for x in range(len(penalty_array[y])):
            penalty_array[y][x] = penalty_value

    # Define the push pull array
    # --> push_pull > 1 when penalty_value > chi_squared
    # --> push_pull < 1 when penalty_value < chi_squared
    push_pull = penalty_array / (data - penalty_array)

    # Set sensible levels, pick the desired colormap and define normalization
    color_map = plt.get_cmap('coolwarm')

    if push_pull.min() < 1.0:
        push = numpy.linspace(numpy.log10(push_pull.min()), numpy.log10(1.0),
                              num=50, endpoint=False)
    else:  # start at 1.0
        push = numpy.zeros((50))  # log10(1.0) = 0
    if push_pull.max() >= 1.0:
        pull = numpy.linspace(numpy.log10(1.0), numpy.log10(push_pull.max()),
                              num=51)
    else:  # stop at 1.0
        pull = numpy.zeros((51))  # log10(1.0) = 0

    # Add the pull part to the push part
    full_scale = numpy.append(numpy.power(10, push), numpy.power(10, pull))
    locator = FixedLocator(full_scale)
    levels = locator.tick_values(push_pull.min(), push_pull.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)
    fig = plt.figure(fig, figsize=(12, 10))  # Fig. 4
    fig.text(0.1, 0.95, syst_analyser._name, **BOLD_FONT)
    ax = fig.add_subplot(1, 1, 1)

    # Set labels
    ax.set_xlabel("Signal counts", **BOLD_FONT)
    ax.set_ylabel("$s/\chi^{2}_{\lambda,p}$", **BOLD_FONT)

    # Plot color map
    color_map = ax.pcolormesh(X, Y, push_pull, cmap=color_map, norm=norm)
    color_bar = fig.colorbar(color_map)
    color_bar.set_label("$s/\chi^{2}_{\lambda,p}$", size=MAIN_FONT.get("size"))
    color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))  # tick label size

    # Set axes limits
    ax.set_xlim([X.min(), X.max()])
    ax.set_ylim([Y.min(), Y.max()])

    # Set axes tick label size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))

    ax.legend(loc="upper left")
    if kwargs.get("save_as") is not None:
        fig.savefig(kwargs.get("save_as") + "_push_pull.png", dpi=300)
    return fig


def main(args):
    """ Script to produce chi squared plots for a given systematic.

    .. Produces::

      * Plot of chi squared vs. signal counts
      * Plot of systematic vs. signal chi squared surface, either
        contour plot or color map
      * Plot of systematic value vs. penalty term value

    Args:
      args (dict): command line arguments from argparse.
    """
    signal_config = LimitConfig(0, [0])
    signal_config = store.load_ndarray(args.signal_config, signal_config)
    if args.penalty_config is not None:
        penalty_config = LimitConfig(0, [0])
        penalty_config = store.load_ndarray(args.penalty_config,
                                            penalty_config)
    else:
        penalty_config = None
    syst_analyser = SystAnalyser("", numpy.zeros((1)), numpy.zeros((1)))
    syst_analyser = store.load_ndarray(args.syst_analyser, syst_analyser)
    fig_1 = chi_squared_vs_signal(signal_config, penalty=penalty_config,
                                  save_as=args.image_name)
    fig_2 = penalty_vs_systematic(syst_analyser, 2)
    fig_3 = turn_on(syst_analyser, signal_config, 3, save_as=args.image_name)
    fig_4 = push_pull(syst_analyser, 4, save_as=args.image_name)
    fig_5 = chi_squared_map(syst_analyser, 5, contours=args.contours, save_as=args.image_name)

    plt.show()
    raw_input("RETURN to exit")


if __name__ == "__main__":
    import echidna.output.store as store
    from echidna.limit.limit_config import LimitConfig
    from echidna.limit.limit_setting import SystAnalyser
    from echidna.scripts.zero_nu_limit import ReadableDir

    import argparse


    parser = argparse.ArgumentParser(description="Produce chi squared plots for a systematic")
    parser.add_argument("-s", "--signal_config", action=ReadableDir,
                        help="Supply location of signal config hdf5 file")
    parser.add_argument("-p", "--penalty_config", action=ReadableDir,
                        help="Supply location of signal config with penalty term")
    parser.add_argument("-a", "--syst_analyser", action=ReadableDir,
                        help="Supply location of syst analyser hdf5 file")
    parser.add_argument("-i", "--image_name", type=str, default="output",
                        help="Supply an image name")
    parser.add_argument("-c", "--contours", action="store_true",
                        help="If true produces a contour plot, defualt is colour map")
    args = parser.parse_args()

    main(args)
