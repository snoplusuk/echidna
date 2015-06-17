from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy
from echidna.calc import decay

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
    x = []
    converter = decay.DBIsotope("Te130", 0.003, 129.9062244, 127.603, 0.3408, 3.69e-14, 4.03)
    for num_decays in signal_config._chi_squareds[1]:
        x.append(1./converter.counts_to_half_life(num_decays/0.554))
    pylab.xlabel(r"$1/T_{1/2}^{0\nu}$")
    pylab.ylabel(r"$\chi^{2}$")
    if kwargs.get("penalty") is not None:
        y_1 = signal_config._chi_squareds[0]
        y_2 = kwargs.get("penalty")._chi_squareds[0]
        axis.plot(x, y_1, "bo-", label="no systematic uncertainties")
        axis.plot(x, y_2, "ro-", label="systematic uncertainties")  # lines and dots
        axis.legend(loc="upper left")
    else:
        axis.plot(x, y, "o-")  # lines and dots
    pylab.show()
