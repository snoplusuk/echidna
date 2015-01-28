from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy


def _produce_axis(low, high, bins):
    """ This method produces an array that represents the axis between low and
    high with bins.

    Args:
      low (float): Low edge of the axis
      high (float): High edge of the axis
      bins (int): Number of bins
    """
    return [low + x * (high - low) / bins for x in range(bins)]


def plot_projection(spectra, dimension):
    """ Plot the spectra as projected onto the dimension.
    For example dimension == 0 will plot the spectra as projected onto the
    energy dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project the spectra onto.
    """
    figure = pylab.figure()
    axis = figure.add_subplot(1, 1, 1)
    if dimension == 0:
        x = _produce_axis(spectra._energy_low, spectra._energy_high, spectra._energy_bins)
        width = spectra._energy_width
        pylab.xlabel("Energy [MeV]")
    elif dimension == 1:
        x = _produce_axis(spectra._radial_low, spectra._radial_high, spectra._radial_bins)
        width = spectra._radial_width
        pylab.xlabel("Radius [mm]")
    elif dimension == 2:
        x = _produce_axis(spectra._time_low, spectra._time_high, spectra._time_bins)
        width = spectra._time_width
        pylab.xlabel("Time [yr]")
    pylab.ylabel("Count per %f bin" % width)
    data = spectra.project(dimension)
    axis.bar(x, data, width=width)
    pylab.show()


def plot_surface(spectra, dimension):
    """ Plot the spectra with the dimension projected out.
    For example dimension == 0 will plot the spectra as projected onto the
    radial and time dimensions i.e. not energy.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project out.
    """
    figure = pylab.figure()
    axis = figure.add_subplot(111, projection='3d')
    if dimension == 0:
        x = _produce_axis(spectra._radial_low, spectra._radial_high, spectra._radial_bins)
        y = _produce_axis(spectra._energy_low, spectra._energy_high, spectra._energy_bins)
        data = spectra.surface(2)
        axis.set_xlabel("Radius [mm]")
        axis.set_ylabel("Energy [MeV]")
    elif dimension == 1:
        x = _produce_axis(spectra._time_low, spectra._time_high, spectra._time_bins)
        y = _produce_axis(spectra._energy_low, spectra._energy_high, spectra._energy_bins)
        data = spectra.surface(1)
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel("Energy [MeV]")
    elif dimension == 2:
        x = _produce_axis(spectra._time_low, spectra._time_high, spectra._time_bins)
        y = _produce_axis(spectra._radial_low, spectra._radial_high, spectra._radial_bins)
        data = spectra.surface(0)
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel("Radius [mm]")
    axis.set_zlabel("Count per bin")
    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    axis.plot_surface(X, Y, data)
    pylab.show()
