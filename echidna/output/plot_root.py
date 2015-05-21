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
      (:class`ROOT.TH1D`:) plot.
    """
    plot = TH1D(dimension, "%s;Count per bin" % dimension,
                int(spectra.get_config().getpar(dimension).bins),
                spectra.get_config().getpar(dimension).low,
                spectra.get_config().getpar(dimension).high)
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
      (:class`ROOT.TH2D`:) plot.
    """
    plot = TH2D("%s:%s" % (dimension1, dimension2),
                "%s;%s;Count per bin" % (dimension1, dimension2),
                spectra.get_config().getpar(dimension1).bins,
                spectra.get_config().getpar(dimension1).low,
                spectra.get_config().getpar(dimension1).high,
                spectra.get_config().getpar(dimension2).bins,
                spectra.get_config().getpar(dimension2).low,
                spectra.get_config().getpar(dimension2).high)
    data = spectra.surface(dimension1, dimension2)
    for index_x, data_x in enumerate(data):
        for index_y, datum in enumerate(data_x):
            plot.SetBinContent(index_x, index_y, datum)
    if graphical:
        plot.Draw("COLZ")
        raw_input("Return to quit")
    return plot
