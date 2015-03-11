from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy

import echidna.calc.decay as decay


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
    effective_masses = numpy.zeros(shape=(signal_config._chi_squareds[2].shape))
    for i_bin, count in enumerate(signal_config._chi_squareds[2]):
        effective_mass = decay.
        effective_masses[i_bin] = effective_mass
    x = effective_masses
    pylab.xlabel("Signal counts")
    pylab.ylabel(r"$\chi^{2}$")
    if kwargs.get("penalty") is not None:
        y_1 = signal_config._chi_squareds[0]
        y_2 = kwargs.get("penalty")._chi_squareds[0]
        axis.plot(x, y_1, "bo-", label="no penalty term")
        axis.plot(x, y_2, "ro-", label="penalty term")  # lines and dots
        axis.legend(loc="upper left")
    else:
        axis.plot(x, y, "o-")  # lines and dots
    pylab.show()
