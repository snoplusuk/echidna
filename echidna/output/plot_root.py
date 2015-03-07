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
    if dimension == 0:
        plot = TH1D("Energy", ";Energy[MeV];Count per bin",
                    int(spectra._energy_bins),
                    spectra._energy_low,
                    spectra._energy_high)
    elif dimension == 1:
        plot = TH1D("Radial", ";Radius[mm];Count per bin",
                    int(spectra._radial_bins),
                    spectra._radial_low,
                    spectra._radial_high)
    elif dimension == 2:
        plot = TH1D("Time", ";Time[yr];Count per bin",
                    int(spectra._time_bins),
                    spectra._time_low,
                    spectra._time_high)
    data = spectra.project(dimension)
    for index, datum in enumerate(data):
        plot.SetBinContent(index, datum)
    if graphical == True:
        plot.Draw()
        raw_input("Return to quit")
    return plot


def plot_surface(spectra, dimension, graphical=True):
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
    if dimension == 0:
        plot = TH2D("EnergyRadial", ";Energy[MeV];Radius[mm];Count per bin",
                    int(spectra._energy_bins),
                    spectra._energy_low, spectra._energy_high,
                    int(spectra._radial_bins),
                    spectra._radial_low, spectra._radial_high)
        data = spectra.surface(2)
    elif dimension == 1:
        plot = TH2D("TimeEnergy", ";Time[yr];Energy[MeV];Count per bin",
                    int(spectra._time_bins),
                    spectra._time_low, spectra._time_high,
                    int(spectra._energy_bins),
                    spectra._energy_low, spectra._energy_high)
        data = spectra.surface(1)
    elif dimension == 2:
        plot = TH2D("TimeRadial", ";Time[yr];Radius[mm];Count per bin",
                    int(spectra._time_bins),
                    spectra._time_low, spectra._time_high,
                    int(spectra._radial_bins),
                    spectra._radial_low, spectra._radial_high)
        data = spectra.surface(0)
    for index_x, data_x in enumerate(data):
        for index_y, datum in enumerate(data_x):
            plot.SetBinContent(index_x, index_y, datum)
    if graphical == True:
        plot.Draw("COLZ")
        raw_input("Return to quit")
    return plot
