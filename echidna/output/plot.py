from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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
    return [parameter.low + x * (parameter.high - parameter.low) /
            parameter.bins
            for x in range(parameter.bins)]


def plot_projection(spectra, dimension, fig_num=1, show_plot=True):
    """ Plot the spectra as projected onto the dimension.

    For example dimension == 0 will plot the spectra as projected onto the
    energy dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project the spectra onto.
    """
    fig = pylab.figure()
    axis = fig.add_subplot(1, 1, 1)
    x = _produce_axis(spectra, dimension)
    width = spectra.get_config().getpar(dimension).get_width()
    pylab.xlabel("FIXME: add helper function to get labels")
    pylab.ylabel("Count per %f bin" % width)
    data = spectra.project(dimension)
    ax.bar(x, data, width=width)
    if show_plot:
        plt.show()
    return fig


def spectral_plot(spectra_dict, dimension, fig_num=1,
                  show_plot=True, **kwargs):
    """ Produce spectral plot.

    For a given signal, produce a plot showing the signal and relevant
    backgrounds. Backgrounds are automatically summed to create the
    spectrum 'Summed background' and all spectra passed in
    :obj:`spectra_dict` will be summed to produce the "Sum" spectrum.

    Args:
      spectra_dict (dict): Dictionary containing each spectrum you wish
        to plot, and the relevant parameters required to plot them.
      dimension (int, optional): The dimension or axis along which the
        spectra should be plotted.
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.

    Example:

      An example :obj:`spectra_dict` is as follows::

        {Te130_0n2b._name: {'spectra': Te130_0n2b, 'label': 'signal',
                            'style': {'color': 'blue'}, 'type': 'signal'},
         Te130_2n2b._name: {'spectra': Te130_2n2b, 'label': r'$2\\nu2\\beta',
                            'style': {'color': 'red'}, 'type': 'background'},
         B8_Solar._name: {'spectra': B8_Solar, 'label': 'solar',
                          'style': {'color': 'green'}, 'type': 'background'}}

    .. note::

      Keyword arguments include:

        * log_y (*bool*): Use log scale on y-axis.
        * per_bin (*bool*): Include chi-squared per bin histogram.
        * limit (:class:`spectra.Spectra`): Include a spectrum showing
          a current or target limit.
    """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(3, 1, (1, 2))
    # All spectra should have same width
    first_spectra = True
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        if first_spectra:
            width = spectra.get_config().getpar(dimension).get_width()
            low = spectra.get_config().getpar(dimension).low
            high = spectra.get_config().getpar(dimension).high
            bins = spectra.get_config().getpar(dimension).bins
            shape = (bins)  # Shape for summed arrays
            first_spectra = False
        else:
            if spectra.get_config().getpar(dimension).low != low:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s lower limit"
                                     % dimension)
            if spectra.get_config().getpar(dimension).high != high:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s higher limit"
                                     % dimension)
            if spectra.get_config().getpar(dimension).bins != bins:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s bins"
                                     % dimension)
        x = _produce_axis(spectra, dimension)
        x_label = "FIXME: add helper function to get labels"
    summed_background = numpy.zeros(shape=shape)
    summed_total = numpy.zeros(shape=shape)
    hist_range = (x[0]-0.5*width, x[-1]+0.5*width)
    if kwargs.get("log_y") is True:
        log = True
    else:
        log = False
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        ax.hist(x, bins=len(x), weights=spectra.project(dimension),
                range=hist_range, histtype="step", label=value.get("label"),
                color=spectra.get_style().get("color"), log=log)
        if value.get("type") is "background":
            summed_background = summed_background + spectra.project(dimension)
        else:
            summed_total = summed_total + spectra.project(dimension)
    ax.hist(x, bins=len(x), weights=summed_background, range=hist_range,
            histtype="step", color="DarkSlateGray", linestyle="dashed",
            label="Summed background", log=log)
    y = summed_background
    yerr = numpy.sqrt(y)
    ax.fill_between(x, y-yerr, y+yerr, facecolor="DarkSlateGray", alpha=0.5,
                    label="Summed background, standard error")
    summed_total = summed_total + summed_background
    ax.hist(x, bins=len(x), weights=summed_total, range=hist_range,
            histtype="step", color="black", label="Sum", log=log)
    kev_width = width * 1.0e3

    # Plot limit
    if kwargs.get("limit") is not None:
        limit = kwargs.get("limit")
        ax.hist(x, bins=len(x), weights=limit.project(dimension),
                range=hist_range, histtype="step", color="LightGrey",
                label="KamLAND-Zen limit", log=log)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.25), fontsize="8")
    # FIXME: why keV by default
    plt.ylabel("Count per %.1f keV bin" % kev_width)
    plt.ylim(ymin=0.1)

    # Plot chi squared per bin, if required
    if kwargs.get("per_bin") is not None:
        calculator = kwargs.get("per_bin")
        ax2 = fig.add_subplot(3, 1, 3, sharex=ax)
        chi_squared_per_bin = calculator.get_chi_squared_per_bin()
        ax2.hist(x, bins=len(x), weights=chi_squared_per_bin,
                 range=hist_range, histtype="step")  # same x axis as above
        plt.ylabel("$\chi^2$ per %.1f keV bin" % kev_width)

    plt.xlabel(x_label)
    plt.xlim(xmin=x[0], xmax=x[-1])
    if kwargs.get("title") is not None:
        plt.figtext(0.05, 0.95, kwargs.get("title"))
    if kwargs.get("text") is not None:
        for index, entry in enumerate(kwargs.get("text")):
            plt.figtext(0.05, 0.90-(index*0.05), entry)
    if show_plot:
        plt.show()
    return fig


def plot_surface(spectra, dimension1, dimension2):
    """ Plot the spectra with the dimension projected out.
    For example dimension == 0 will plot the spectra as projected onto the
    radial and time dimensions i.e. not energy.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project out.
    """
    fig = pylab.figure()
    axis = fig.add_subplot(111, projection='3d')
    x = _produce_axis(spectra, dimension1)
    y = _produce_axis(spectra, dimension2)
    # FIXME: if the index of x is higher than y then the surface may be
    # returned the wrong way around
    data = spectra.surface(dimension1, dimension2)
    axis.set_xlabel("FIXME: add helper function to get labels")
    axis.set_ylabel("FIXME: add helper function to get labels")
    axis.set_zlabel("Count per bin")
    print len(x), len(y), data.shape
    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = numpy.meshgrid(x, y)
    print X.shape, Y.shape
    axis.plot_surface(X, Y, data)
    if show_plot:
        plt.show()
    return fig


if __name__ == "__main__":
    import echidna
    import echidna.output.store as store

    filename = "/data/Te130_0n2b_mc_smeared.hdf5"
    spectre = store.load(echidna.__echidna_home__ + filename)
    plot_surface(spectre, 2)
