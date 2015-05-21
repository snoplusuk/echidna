from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy


def _produce_axis(spectra, dimension):
    """ This method produces an array that represents the axis between low and
    high with bins.

    Args:
      low (float): Low edge of the axis
      high (float): High edge of the axis
      bins (int): Number of bins
    """
    parameter = spectra.get_config().getpar(dimension)
    return [parameter.low + x * (parameter.high - parameter.low) / parameter.bins 
            for x in range(parameter.bins)]


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
    x = _produce_axis(spectra, dimension)
    width = spectra.get_config().getpar(dimension).get_width()
    pylab.xlabel("FIXME: add helper function to get labels")
    pylab.ylabel("Count per %f bin" % width)
    data = spectra.project(dimension)
    axis.bar(x, data, width=width)
    pylab.show()


def plot_surface(spectra, dimension1, dimension2):
    """ Plot the spectra with the dimension projected out.
    For example dimension == 0 will plot the spectra as projected onto the
    radial and time dimensions i.e. not energy.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project out.
    """
    figure = pylab.figure()
    axis = figure.add_subplot(111, projection='3d')
    x = _produce_axis(spectra, dimension1)
    y = _produce_axis(spectra, dimension2)
    # FIXME: if the index of x is higher than y then the surface may be returned the wrong way around
    data = spectra.surface(dimension1, dimension2)
    axis.set_xlabel("FIXME: add helper function to get labels")
    axis.set_ylabel("FIXME: add helper function to get labels")
    axis.set_zlabel("Count per bin")
    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    axis.plot_surface(X, Y, data)
    pylab.show()
