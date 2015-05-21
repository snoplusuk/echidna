from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.pyplot as plt
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

def spectral_plot(spectra_dict, dimension=0, **kwargs):
    """
    """
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    # All spectra should have same width
    first_spectra = True
    if dimension == 0:
        for value in spectra_dict.values():
            spectra = value.get("spectra")
            if first_spectra:
                energy_low = spectra._energy_low
                energy_high = spectra._energy_high
                energy_bins = spectra._energy_bins
                width = spectra._energy_width
                shape = (energy_bins)  # Shape for summed arrays
                first_spectra = False
            else:
                if spectra._energy_low != energy_low:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect energy lower limit")
                if spectra._energy_high != energy_high:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect energy upper limit")
                if spectra._energy_bins != energy_bins:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect energy upper limit")
        x = _produce_axis(energy_low, energy_high, energy_bins)
        plt.xlabel("Energy [MeV]")
    elif dimension == 1:
        for value in spectra_dict.values:
            spectra = value.get("spectra")
            if first_spectra:
                radial_low = spectra._radial_low
                radial_high = spectra._radial_high
                radial_bins = spectra._radial_bins
                width = spectra._radial_width
                shape = (radial_bins)
                first_spectra = False
            else:
                if spectra._radial_low != radial_low:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect radial lower limit")
                if spectra._radial_high != radial_high:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect radial upper limit")
                if spectra._radial_bins != radial_bins:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect radial upper limit")
        x = _produce_axis(radial_low, radial_high, radial_bins)
        plt.xlabel("Radius [mm]")
    elif dimension == 2:
        for value in spectra_dict.values:
            spectra = value.get("spectra")
            if first_spectra:
                time_low = spectra._time_low
                time_high = spectra._time_high
                time_bins = spectra._time_bins
                width = spectra._time_width
                shape = (time_bins)
                first_spectra = False
            else:
                if spectra._time_low != time_low:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect time lower limit")
                if spectra._time_high != time_high:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect time upper limit")
                if spectra._time_bins != time_bins:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect time upper limit")
        x = _produce_axis(spectra._time_low, spectra._time_high, spectra._time_bins)
        plt.xlabel("Time [yr]")
    summed_background = numpy.zeros(shape=shape)
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        ax.plot(x, spectra.project(dimension), value.get("style"),
                label=value.get("label"))
        if value.get("type") is "background":
            summed_background = summed_background + spectra.project(dimension)
    ax.plot(x, summed_background, "k--", label="Summed background")
    plt.ylabel("Count per %f bin" % width)
    if kwargs.get("log_y") is True:
        ax.set_yscale("log")
    plt.legend(loc="upper right")
    plt.ylim(ymin=0.1)
    plt.show()
    return fig

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
    print len(x), len(y), data.shape
    X, Y = numpy.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    print X.shape, Y.shape
    axis.plot_surface(X, Y, data)
    pylab.show()


if __name__ == "__main__":
    import echidna
    import echidna.output.store as store


    filename = "/data/Te130_0n2b_mc_smeared.hdf5"
    spectre = store.load(echidna.__echidna_home__ + filename)
    plot_surface(spectre, 2)
