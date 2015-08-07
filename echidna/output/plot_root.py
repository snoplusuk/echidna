from echidna.util import root_help
from ROOT import TH1D, TH2D


def plot_projection(spectra, dimension, graphical=True):
    """ Plot the spectra as projected onto the dimension.
    For example dimension == 0 will plot the spectra as projected onto the
    energy dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project the spectra onto.
      graphical (bool): Shows plot and waits for user input when true.

    Returns:
      (:class:`ROOT.TH1D`): plot.
    """
    plot = TH1D(dimension, "%s;Count per bin" % dimension,
                int(spectra.get_config().get_par(dimension)._bins),
                spectra.get_config().get_par(dimension)._low,
                spectra.get_config().get_par(dimension)._high)
    data = spectra.project(dimension)
    for index, datum in enumerate(data):
        plot.SetBinContent(index, datum)
    if graphical:
        plot.Draw()
        raw_input("Return to quit")
    return plot


def plot_surface(spectra, dimension1, dimension2, graphical=True):
    """ Plot the spectra with the dimension projected out.
    For example dimension == 0 will plot the spectra as projected onto the
    radial and time dimensions i.e. not energy.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (int): The dimension to project out.
      graphical (bool): Shows plot and waits for user input when true.

    Returns:
      (:class:`ROOT.TH2D`): plot.
    """
    plot = TH2D("%s:%s" % (dimension1, dimension2),
                "%s;%s;Count per bin" % (dimension1, dimension2),
                spectra.get_config().get_par(dimension1)._bins,
                spectra.get_config().get_par(dimension1)._low,
                spectra.get_config().get_par(dimension1)._high,
                spectra.get_config().get_par(dimension2)._bins,
                spectra.get_config().get_par(dimension2)._low,
                spectra.get_config().get_par(dimension2)._high)
    data = spectra.surface(dimension1, dimension2)
    for index_x, data_x in enumerate(data):
        for index_y, datum in enumerate(data_x):
            plot.SetBinContent(index_x, index_y, datum)
    if graphical:
        plot.Draw("COLZ")
        raw_input("Return to quit")
    return plot


# TO DO: Convert the rest of the functions below
def spectral_plot(spectra_dict, dimension=0, show_plot=False, **kwargs):
    """ Produce spectral plot.

    For a given signal, produce a plot showing the signal and relevant
    backgrounds. Backgrounds are automatically summed to create the
    spectrum "Summed background" and all spectra passed in
    :obj:`spectra_dict` will be summed to produce the "Sum" spectra

    Args:
      spectra_dict (dict): Dictionary containing each spectrum you wish
        to plot, and the relevant parameters required to plot them.
      dimension (int, optional): The dimension or axis along which the
        spectra should be plotted. Default is energy axis.

    Example:

      An example :obj:`spectra_dict` is as follows::

        {Te130_0n2b._name: {'spectra': Te130_0n2b,
                            'label': 'signal',
                            'style': {'color': ROOT.kBlue},
                            'type': 'signal'},
         Te130_2n2b._name: {'spectra': Te130_2n2b,
                            'label': r'$2\\nu2\\beta',
                            'style': {'color': ROOT.kRed},
                            'type': 'background'},
         B8_Solar._name: {'spectra': B8_Solar,
                          'label': 'solar',
                          'style': {'color': ROOT.kGreen},
                          'type': 'background'}}

    .. note::

      Keyword arguments include:

        * log_y (*bool*): Use log scale on y-axis.
        * limit (:class:`spectra.Spectra`): Include a spectrum showing
          a current or target limit.

    Returns:
      :class:`ROOT.TCanvas`: Canvas containing spectral plot.
    """
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
        summed_background = ROOT.TH1F("summed_background",
                                      "; Energy (MeV); Counts",
                                      spectra._energy_bins,
                                      spectra._energy_low,
                                      spectra._energy_high)
        summed_total = ROOT.TH1F("summed_total",
                                 "; Energy (MeV); Counts",
                                 spectra._energy_bins,
                                 spectra._energy_low,
                                 spectra._energy_high)
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
                                         "incorrect time lower limit")
                if spectra._radial_high != radial_high:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect time upper limit")
                if spectra._radial_bins != radial_bins:
                    raise AssertionError("Spectra " + spectra._name + " has "
                                         "incorrect time upper limit")
        summed_background = ROOT.TH1F("summed_background",
                                      "; Radius (mm); Counts",
                                      spectra._radial_bins,
                                      spectra._radial_low,
                                      spectra._radial_high)
        summed_total = ROOT.TH1F("summed_total",
                                 "; Radius (mm); Counts",
                                 spectra._radial_bins,
                                 spectra._radial_low,
                                 spectra._radial_high)
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
        summed_background = ROOT.TH1F("summed_background",
                                      "; Time (Yr); Counts",
                                      spectra._time_bins,
                                      spectra._time_low,
                                      spectra._time_high)
        summed_total = ROOT.TH1F("summed_total",
                                 "; Time (Yr); Counts",
                                 spectra._time_bins,
                                 spectra._time_low,
                                 spectra._time_high)
    summed_total = numpy.zeros(shape=shape)
    can = ROOT.TCanvas()
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    summed_background.SetLineStyle(7)
    summed_background.SetLineColor(ROOT.kRed)
    summed_total.SetLineStyle(7)
    summed_total.SetLineColor(ROOT.kBlack)
    leg.AddEntry(summed_total, "Background + Signal", "l")
    leg.AddEntry(summed_background, "Background", "l")
    hists = []
    if kwargs.get("log_y") is True:
        can.SetLogy()
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        hist = spectra.project(dimension, graphical=False)
        hist.SetLineColor(value.get("style")["color"])
        leg.AddEntry(hist, value.get("label"), 'l')
        hists.append(hist)
        if value.get("type") is "background":
            summed_background.Add(hist)
        else:
            summed_total.Add(hist)
    summed_total.Draw()
    summed_background.Draw("same")
    for hist in hists:
        hist.Draw("same")

    # Plot limit
    if kwargs.get("limit") is not None:
        limit = kwargs.get("limit")
        hist = limit.project(dimension, graphical=False)
        hist.SetLineColor(ROOT.kGray)
        leg.AddEntry(hist, "Kamland-Zen Limit", "l")
    leg.Draw("same")
    return can


def plot_chi_squared_per_bin(calculator, x_bins, x_low, x_high,
                             x_title=None, graphical=False):
    """ Produces a histogram of chi-squared per bin.

    Args:
      calculator (:class:`echidna.limit.chi_squared.ChiSquared`): Calculator
        containing the chi-squared values to plot.
      x_bins (int): Number of bins.
      x_low (float): Lower edge of first bin to plot.
      x_high (float): Upper edge of last bin to plot.
      x_title (string, optional): X Axis title.
      graphical (bool): Plots hist to screen if true.

    Returns:
      :class:`ROOT.TH1D`: Histogram of chi-squared per bin.
    """
    if x_title:
        hist_title = "; "+x_title+"; #chi^{2}"
    else:
        hist_title = "; Energy (MeV); #chi^{2}"
    hist = ROOT.TH1F("chi_sq_per_bin", hist_title, x_bins, x_low, x_high)
    bin = 1  # 0 is underflow
    for chi_sq in calculator.get_chi_squared_per_bin():
        hist.SetBinContent(bin, chi_sq)
        bin += 1
    if graphical:
        hist.Draw()
        raw_input("RET to quit")
    return hist
