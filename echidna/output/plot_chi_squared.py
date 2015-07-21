""" Contains functions to view and interrogate chi-squared minimisation

Attributes:
  MAIN_FONT (dict): style properties for the main font to use in plot labels
"""
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy

import echidna.calc.decay as decay

MAIN_FONT = {'family': 'normal',
             'weight': 'bold',
             'size': 22}


def chi_squared_vs_signal(signal_config, **kwargs):
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
    figure = pylab.figure(figsize=(10, 10))
    axis = figure.add_subplot(1, 1, 1)
    if kwargs.get("effective_mass"):
        effective_masses = numpy.zeros(shape=(signal_config.get_chi_squareds()[2].shape))
        te130_converter = decay.DBIsotope("Te130", 0.003, 129.906229, 127.6,
                                          0.3408, 3.69e-14, 4.03)
        n_atoms = te130_converter.get_n_atoms()
        for i_bin, count in enumerate(signal_config.get_chi_squareds()[2]):
            effective_mass = te130_converter.counts_to_mass(count, n_atoms,
                                                            5., roi_cut=True)
            effective_masses[i_bin] = effective_mass
        x = effective_masses
        pylab.xlabel(r"$m_{\beta\beta}$", **MAIN_FONT)
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
        pylab.xlabel("Signal counts", **MAIN_FONT)
        y_1 = signal_config.get_chi_squareds()[0]
        pylab.ylabel(r"$\chi^{2}$", **MAIN_FONT)
    if kwargs.get("penalty") is not None:
        y_2 = kwargs.get("penalty")._chi_squareds[0]
        axis.plot(x, y_1, "bo-", label="no systematic uncertainties")
        axis.plot(x, y_2, "ro-", label="systematic uncertainties")  # lines and dots
        axis.legend(loc="upper left")
    else:
        axis.plot(x, y_1, "o-")  # lines and dots
    for label in (axis.get_xticklabels() +
                  axis.get_yticklabels()):
        label.set_fontsize(MAIN_FONT.get("size"))
    pylab.show()
    if kwargs.get("save_as") is not None:
        figure.savefig(kwargs.get("save_as") + ".png", dpi=400)


def chi_squared_map(syst_analyser, **kwargs):
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
    color_map = pylab.get_cmap('hot_r')

    linear = numpy.linspace(numpy.sqrt(data.min()), numpy.sqrt(data.max()),
                            num=100)
    locator = FixedLocator(linear**2)
    levels = locator.tick_values(data.min(), data.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)

    if kwargs.get("contours"):
        fig = plt.figure(figsize=(15, 10))
        gridspec.GridSpec(2, 1).update(wspace=0.0, hspace=0.0)  # spacing between axes.
        axes = Axes3D(fig)
        axes.view_init(elev=17.0, azim=-136.0)  # set intial viewing position

        # Plot surface
        surf = axes.plot_surface(X, Y, data, rstride=1, cstride=1,
                                 cmap=color_map, norm=norm, linewidth=0,
                                 antialiased=False)
        axes.zaxis.set_minor_locator(locator)
        axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Set axis labels
        pylab.ylabel(syst_analyser.get_name(), **MAIN_FONT)
        pylab.xlabel("Signal counts", **MAIN_FONT)
        for label in (axes.get_xticklabels() +
                      axes.get_yticklabels() +
                      axes.get_zticklabels()):
            label.set_fontsize(MAIN_FONT.get("size"))  # tick label size
        color_bar = fig.colorbar(surf, shrink=0.5, aspect=5)
        color_bar.set_label(r"$\chi^2$", size=MAIN_FONT.get("size"))
        color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))

        fig.show()
        if kwargs.get("save_as") is not None:
            fig.savefig(kwargs.get("save_as") + "_contour.png", dpi=300)
    else:
        fig = plt.figure(figsize=(12, 10))
        axes = fig.add_subplot(1, 1, 1)

        # Set labels
        pylab.xlabel("Signal counts", **MAIN_FONT)
        pylab.ylabel(syst_analyser.get_name(), **MAIN_FONT)

        # Plot color map
        color_map = axes.pcolormesh(X, Y, data, cmap=color_map, norm=norm)
        color_bar = fig.colorbar(color_map)
        color_bar.set_label(r"$\chi^2$", size=MAIN_FONT.get("size"))
        color_bar.ax.tick_params(labelsize=MAIN_FONT.get("size"))  # tick label size

        # Set axes limits
        pylab.xlim([X.min(), X.max()])
        pylab.ylim([Y.min(), Y.max()])

        if kwargs.get("preferred_values"):
            axes.plot(x, y_2, "bo-", label="Preferred values")
        if kwargs.get("minima"):
            axes.plot(x_3, y_3, "ko", label="Minima")

        # Set axes tick label size
        for label in (axes.get_xticklabels() + axes.get_yticklabels()):
            label.set_fontsize(MAIN_FONT.get("size"))

        axes.legend(loc="upper left")
        pylab.show()
        if kwargs.get("save_as") is not None:
            fig.savefig(kwargs.get("save_as") + "_color_map.png", dpi=300)


def penalty_vs_systematic(syst_analyser):
    plt.subplot(1, 1, 1)
    x = syst_analyser._penalty_values[0]
    y = syst_analyser._penalty_values[1]
    plt.xlabel(syst_analyser.get_name())
    plt.ylabel("Value of penalty term")
    plt.plot(x, y, "bo")
    plt.show()


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
    signal_config = LimitConfig(0, [0])
    signal_config = store.load_ndarray(args.signal_config, signal_config)
    penalty_config = LimitConfig(0, [0])
    penalty_config = store.load_ndarray(args.penalty_config, penalty_config)
    syst_analyser = SystAnalyser("", numpy.zeros((1)), numpy.zeros((1)))
    syst_analyser = store.load_ndarray(args.syst_analyser, syst_analyser)
    chi_squared_vs_signal(signal_config, penalty=penalty_config)
    chi_squared_map(syst_analyser, contours=True, save_as=args.image_name)
    penalty_vs_systematic(syst_analyser)


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
    args = parser.parse_args()

    main(args)
