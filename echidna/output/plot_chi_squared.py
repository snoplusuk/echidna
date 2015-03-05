from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy

from matplotlib import rc

def chi_squared_vs_signal(signal_config, **kwargs):
    figure = pylab.figure()
    axis = figure.add_subplot(1, 1, 1)
    x = signal_config._chi_squareds[2]
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
