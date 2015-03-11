from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FixedLocator
import numpy


def chi_squared_vs_signal(signal_config, **kwargs):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
        config class, where chi squareds have been stored.

    .. note::

      Keyword arguments include:

        * penalty (:class:`echidna.limit.limit_config.LimitConfig`): config
          for signal with penalty term.
    """
    figure = pylab.figure()
    axis = figure.add_subplot(1, 1, 1)
    x = signal_config._chi_squareds[2]
    y_1 = signal_config._chi_squareds[0]
    pylab.xlabel("Signal counts")
    pylab.ylabel(r"$\chi^{2}$")
    if kwargs.get("penalty") is not None:
        y_2 = kwargs.get("penalty")._chi_squareds[0]
        axis.plot(x, y_1, "bo-", label="no penalty term")
        axis.plot(x, y_2, "ro-", label="penalty term")  # lines and dots
        axis.legend(loc="upper left")
    else:
        axis.plot(x, y_1, "o-")  # lines and dots
    pylab.show()


def chi_squared_map(syst_analyser):
    plt.subplot(1, 1, 1)

    # Set x and y axes
    x = syst_analyser._actual_counts
    y = syst_analyser._syst_values

    # Set chi squared map values
    data = numpy.average(syst_analyser._chi_squareds, axis=1)
    data = numpy.transpose(data)  # transpose it so that axes are correct
    data = data[:-1, :-1]  # remove the last values from z array

    # Set preferred value values
    y_2 = numpy.average(syst_analyser._preferred_values, axis=1)

    # Set minima values
    x_3 = syst_analyser._minima[0]
    y_3 = syst_analyser._minima[1]

    # Create meshgrid
    X, Y = numpy.meshgrid(x, y)

    # Set sensible levels, pick the desired colormap and define normalization
    linear = numpy.linspace(numpy.sqrt(data.min()), 
                            numpy.sqrt(data.max()), num=100)
    squared = linear**2
    levels = FixedLocator(squared).tick_values(data.min(), data.max())
    color_map = pylab.get_cmap('hot_r')
    norm = BoundaryNorm(levels, ncolors=color_map.N)

    # Set labels
    plt.xlabel("Signal counts")
    plt.ylabel("Value of systematic")

    image = plt.pcolormesh(X, Y, data, cmap=color_map, norm=norm)
    plt.colorbar()
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.plot(x, y_2, "bo-", label="Preferred values")
    plt.plot(x_3, y_3, "ko", label="Minima")
    plt.legend(loc="upper left")
    pylab.show()


def penalty_values(syst_analyser):
    plt.subplot(1, 1, 1)
    x = syst_analyser._penalty_values[0]
    y = syst_analyser._penalty_values[1]
    plt.xlabel("Value of systematic")
    plt.ylabel("Value of penalty term")
    plt.plot(x, y, "o-")
    plt.show()


if __name__ == "__main__":
    import echidna
    import echidna.output.store as store
    from echidna.limit.limit_setting import SystAnalyser


    Te130_2n2b_filename = "/Te130_2n2b_mc200.0_light_yield100.0_position_resolution_counts.hdf5"

    Te130_2n2b_analyser = SystAnalyser("", numpy.zeros((1)), numpy.zeros((1)))
    Te130_2n2b_analyser = store.load_ndarray(echidna.__echidna_home__ + Te130_2n2b_filename,
                                             Te130_2n2b_analyser)
    chi_squared_map(Te130_2n2b_analyser)
    penalty_values(Te130_2n2b_analyser)

    B8_Solar_filename = "/B8_Solar_mc200.0_light_yield100.0_position_resolution_counts.hdf5"

    B8_Solar_analyser = SystAnalyser("", numpy.zeros((1)), numpy.zeros((1)))
    B8_Solar_analyser = store.load_ndarray(echidna.__echidna_home__ + B8_Solar_filename,
                                           B8_Solar_analyser)
    chi_squared_map(B8_Solar_analyser)
    penalty_values(B8_Solar_analyser)
    
    
