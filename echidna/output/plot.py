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

def spectral_plot(spectra_dict, dimension=0, fig_num=1, **kwargs):
    """
    """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(3, 1, (1, 2))
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
    summed_total = numpy.zeros(shape=shape)
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        ax.hist(x, bins=x.shape[0], weights=spectra.project(dimension),
                histtype="step", label=value.get("label"))
        if value.get("type") is "background":
            summed_background = summed_background + spectra.project(dimension)
        else:
            summed_total = summed_total + spectra.project(dimension)
    ax.plot(x, summed_background, "k--", label="Summed background")
    summed_total = summed_total + summed_background
    ax.plot(x, summed_total, "k-", label="Sum")
    kev_width = width * 1.0e3
    plt.ylabel("Count per %.1f keV bin" % kev_width)
    # if kwargs.get("log_y") is True:
        # ax.set_yscale("log")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.25), fontsize="10")

    plt.ylim(ymin=0.1)

    # Plot chi squared per bin, if required
    if kwargs.get("per_bin") is not None:
        calculator = kwargs.get("per_bin")
        ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        chi_squared_per_bin = calculator.get_chi_squared_per_bin()
        ax2.scatter(x, chi_squared_per_bin, marker="_")  # same x axis as above
        plt.ylabel("$\chi^2$")

    if kwargs.get("title") is not None:
        plt.figtext(0.05, 0.95, kwargs.get("title"))
    if kwargs.get("text") is not None:
        for index, entry in enumerate(kwargs.get("text")):
            plt.figtext(0.05, 0.90-(index*0.05), entry)
    # plt.show()
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
