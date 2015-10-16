import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import FixedLocator
from matplotlib.colors import BoundaryNorm


def _produce_axis(spectra, dimension):
    """ This method produces an array that represents the axis.

    The array is formed of the value at the bin-centre for each bin
    along the specified dimension.

    Args:
      spectra (:class:`echidna.core.spectra.Spectra`): The spectra you
        wish to produce the axis from.
      dimension (string): The dimension you wish to produce the axis for.

    Returns:
      :class:`numpy.array`: The values for the axis.
    """
    par = spectra.get_config().get_par(dimension)
    width = par.get_width()
    return numpy.arange(par._low, par._high, width) + 0.5*width


def _produce_bins(spectra, dimension):
    """ This method produces an array that represents the bin boundaries.

    The array is formed of the value at each bin boundary from par._low
    to par._high along the specified dimension. The length of the array
    of bin boundaries is one greater than the length of the axis array.

    Args:
      spectra (:class:`echidna.core.spectra.Spectra`): The spectra you
        wish to produce the bin boundaries from.
      dimension (string): The dimension you wish to produce the bin
        boundaries for.

    Returns:
      :class:`numpy.array`: The values for the axis.
    """
    par = spectra.get_config().get_par(dimension)
    return numpy.linspace(par._low, par._high, par._bins+1)


def plot_projection(spectra, dimension, fig_num=1, show_plot=True):
    """ Plot the spectra as projected onto the dimension.

    For example dimension == 0 will plot the spectra as projected onto the
    energy dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (string): The dimension to project the spectra onto.
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      show_plot (bool, optional): Displays the plot if true. Default is True.

    Returns:
      matplotlib.pyplot.figure: Plot of the projection.
    """
    fig = plt.figure(fig_num)
    axis = fig.add_subplot(1, 1, 1)

    par = spectra.get_config().get_par(dimension)
    width = par.get_width()
    # Define array of bin boundaries (1 more than number of bins)
    bins = _produce_bins(spectra, dimension)
    # Define array of bin centres
    x = _produce_axis(spectra, dimension)

    data = spectra.project(dimension)
    axis.hist(x, bins, weights=data, histtype="step")
    plt.xlabel("%s (%s)" % (dimension, par.get_unit()))
    plt.ylabel("Count per %.2g %s bin" % (width, par.get_unit()))
    if show_plot:
        plt.show()
    return fig


def spectral_plot(spectra_dict, dimension, fig_num=1, show_plot=True,
                  log_y=False, per_bin=False, limit=None, title=None,
                  text=None, calculator=None):
    """ Produce spectral plot.

    For a given signal, produce a plot showing the signal and relevant
    backgrounds. Backgrounds are automatically summed to create the
    spectrum 'Summed background' and all spectra passed in
    :obj:`spectra_dict` will be summed to produce the "Sum" spectrum.

    Args:
      spectra_dict (dict): Dictionary containing each spectrum you wish
        to plot, and the relevant parameters required to plot them.
      dimension (string): The dimension  you wish to plot.
      fig_num (int, optional): The number of the figure. If you are
        producing multiple spectral plots automatically, this can be
        useful to ensure pyplot treats each plot as a separate figure.
        Default is 1.
      show_plot (bool, optional): Displays the plot if true. Default is True.
      log_y (bool, optional): Use log scale on y-axis.
      calculator (:class:`echidna.limit.chi_squared.ChiSquared`, optional):
        Calculator for including chi-squared per bin histogram.
      limit (:class:`spectra.Spectra`, optional): Include a spectrum showing
        a current or target limit.
      title (string, optional): Title of the plot.
      text (list, optional): Text to be written on top of plot.

    Example:

      An example :obj:`spectra_dict` is as follows::

        {Te130_0n2b._name: {'spectra': Te130_0n2b, 'label': 'signal',
                            'style': {'color': 'blue'}, 'type': 'signal'},
         Te130_2n2b._name: {'spectra': Te130_2n2b, 'label': r'$2\\nu2\\beta',
                            'style': {'color': 'red'}, 'type': 'background'},
         B8_Solar._name: {'spectra': B8_Solar, 'label': 'solar',
                          'style': {'color': 'green'}, 'type': 'background'}}

    Returns:
      matplotlib.pyplot.figure: Plot of the signal and backgrounds.
    """
    fig = plt.figure(fig_num)
    axis = fig.add_subplot(3, 1, (1, 2))

    # All spectra should have same width
    first_spectra = True
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        par = spectra.get_config().get_par(dimension)
        if first_spectra:
            low = par._low
            high = par._high
            bins = par._bins
            shape = (bins)  # Shape for summed arrays
            first_spectra = False
        else:
            if par._low != low:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s lower limit"
                                     % dimension)
            if par._high != high:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s higher limit"
                                     % dimension)
            if par._bins != bins:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect dimension %s bins"
                                     % dimension)
    # Define x-axis
    x = _produce_axis(spectra, dimension)
    x_label = "%s (%s)" % (dimension, par.get_unit())

    # Define bins
    bins = _produce_bins(spectra, dimension)

    # Define empty arrays for summed background and summed total
    summed_background = numpy.zeros(shape=shape)
    summed_total = numpy.zeros(shape=shape)

    for value in spectra_dict.values():
        spectra = value.get("spectra")
        data = spectra.project(dimension)
        # Set label
        if value.get("label"):
            label = value.get("label")
        elif spectra.get_style().get("label"):
            label = spectra.get_style().get("label")
        else:  # Use spectrum name
            label = spectra._name
        # Set color
        if value.get("color"):
            color = value.get("color")
        elif spectra.get_style().get("color"):
            color = spectra.get_style().get("color")
        else:  # None uses default line color sequence
            color = None
        axis.hist(x, bins=bins, weights=data, histtype="step",
                  label=label, color=color, log=log_y)
        if value.get("type") is "background":
            summed_background = summed_background + spectra.project(dimension)
        summed_total = summed_total + spectra.project(dimension)
    axis.hist(x, bins=bins, weights=summed_background,
              histtype="step", color="DarkSlateGray",
              linestyle="dashed", label="Summed background", log=log_y)
    y = summed_background
    yerr = numpy.sqrt(y)
    axis.fill_between(x, y-yerr, y+yerr, facecolor="DarkSlateGray",
                      alpha=0.5, label="Summed background, standard error")
    axis.hist(x, bins=bins, weights=summed_total, histtype="step",
              color="black", label="Sum", log=log_y)

    # Plot limit
    if limit:
        # Set color
        if limit.get_style().get("color"):
            color = limit.get_style().get("color")
        else:
            color = "LightGrey"
        # Set label
        if limit.get_style().get("label"):
            label = limit.get_style().get("label")
        else:
            label = "limit"
        axis.hist(x, bins=bins, weights=limit.project(dimension),
                  histtype="step", color=color, label=label, log=log_y)

    # Shrink current axis by 20%
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width, box.height*0.8])
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.25), fontsize="8")
    plt.ylabel("Count per %.2g %s bin" % (par.get_width(), par.get_unit()))
    plt.ylim(ymin=0.1)

    # Plot chi squared per bin, if required
    if calculator:
        axis2 = fig.add_subplot(3, 1, 3, sharex=axis)
        chi_squared_per_bin = calculator.get_chi_squared_per_bin()
        axis2.hist(x, bins=bins, weights=chi_squared_per_bin,
                   histtype="step")  # same x axis as above
        plt.ylabel("$\chi^2$ per %f %s bin"
                   % (par.get_width(), par.get_unit()))

    plt.xlabel(x_label)
    plt.xlim(xmin=x[0], xmax=x[-1])
    if title:
        plt.figtext(0.05, 0.95, title)
    if text:
        for index, entry in enumerate(text):
            plt.figtext(0.05, 0.90-(index*0.05), entry)
    if show_plot:
        plt.show()
    return fig


def plot_surface(spectra, dimension1, dimension2,
                 fig_num=1, show_plot=True, **kwargs):
    """ Plot the two dimensions from spectra as a 2D histogram

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension1 (string): The name of the dimension you want to plot.
      dimension2 (string): The name of the dimension you want to plot.

    Returns:
      matplotlib.pyplot.figure: Plot of the surface of the two dimensions.
    """
    fig = plt.figure(num=fig_num)
    axis = fig.add_subplot(111)
    index1 = spectra.get_config().get_index(dimension1)
    index2 = spectra.get_config().get_index(dimension2)
    if index1 < index2:
        x = _produce_axis(spectra, dimension2)
        y = _produce_axis(spectra, dimension1)
    else:
        x = _produce_axis(spectra, dimension1)
        y = _produce_axis(spectra, dimension2)
    data = spectra.surface(dimension1, dimension2)
    par1 = spectra.get_config().get_par(dimension1)
    par2 = spectra.get_config().get_par(dimension2)
    if index1 < index2:
        axis.set_xlabel("%s (%s)" % (dimension2, par2.get_unit()))
        axis.set_ylabel("%s (%s)" % (dimension1, par1.get_unit()))
    else:
        axis.set_xlabel("%s (%s)" % (dimension1, par1.get_unit()))
        axis.set_ylabel("%s (%s)" % (dimension2, par2.get_unit()))

    # `plot_surface` expects `x` and `y` data to be 2D
    X, Y = numpy.meshgrid(x, y)

    # Set sensible levels, pick the desired colormap and define normalization
    if kwargs.get("color_scheme") is None:
        color_scheme = "hot_r"  # default
    else:
        color_scheme = kwargs.get("color_scheme")
    color_map = plt.get_cmap(color_scheme)
    linear = numpy.linspace(numpy.sqrt(data.min()),
                            numpy.sqrt(data.max()), num=100)
    locator = FixedLocator(linear**2)
    levels = locator.tick_values(data.min(), data.max())
    norm = BoundaryNorm(levels, ncolors=color_map.N)

    # Plot color map
    color_map = axis.pcolormesh(X, Y, data, cmap=color_map, norm=norm)
    color_bar = fig.colorbar(color_map)
    color_bar.set_label("Counts per bin")

    if show_plot:
        plt.show()
    return fig


if __name__ == "__main__":
    import echidna
    import echidna.output.store as store

    filename = "/data/Te130_0n2b_mc_smeared.hdf5"
    spectre = store.load(echidna.__echidna_home__ + filename)
    plot_surface(spectre, 2)
