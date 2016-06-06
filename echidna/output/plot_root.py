from echidna.util import root_help, strings
from echidna.output import root_style

import ROOT
import numpy
import copy

def plot_projection(spectra, dimension, graphical=True, fig_num=1,
                    h_name=None):
    """ Plot the spectra as projected onto the dimension.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension (string): The dimension to project the spectra onto.
      graphical (bool, optional): Displays plot if True. Default is True.

    Returns:
      (:class:`ROOT.TH1D`): Root histogram of the projection onto
        the given dimension.
      (:class:`ROOT.TCanvas`): Root canvas object containing plot of
        histogram.
    """
    if not h_name:
        h_name = dimension + strings.id_generator()
    plot = ROOT.TH1D(h_name, "; %s; Count per bin" % dimension,
                     int(spectra.get_config().get_par(dimension)._bins),
                     spectra.get_config().get_par(dimension)._low,
                     spectra.get_config().get_par(dimension)._high)
    data = spectra.project(dimension)
    for index, datum in enumerate(data):
        plot.SetBinContent(index + 1, datum)
    if graphical:
        can = ROOT.TCanvas()
        can.cd()
        plot.Draw()
        raw_input("Return to quit")
        del can
    return plot


def plot_surface(spectra, dimension1, dimension2, graphical=True, h_name=None):
    """ Plots a 2D histogram of the dimensions in spectra.

    Args:
      spectra (:class:`echidna.core.spectra`): The spectra to plot.
      dimension1 (string): The dimension to plot.
      dimension2 (string): The dimension to plot.
      graphical (bool, optional): Displays plot if True. Default is True.

    Returns:
      (:class:`ROOT.TH2D`): Root 2D histogram of the spectra surface
        projection.
      (:class:`ROOT.TCanvas`): Root canvas object containing plot of
        histogram.
    """
    if not h_name:
        h_name = dimension1 + ":" + dimension2 + strings.id_generator()
    plot = ROOT.TH2D(h_name,
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
            plot.SetBinContent(index_x + 1, index_y + 1, datum)
    if graphical:
        can = ROOT.TCanvas("Figure", "Figure")
        can.cd()
        plot.Draw("COLZ")
        raw_input("Return to quit")
        del can
    return plot


def spectral_plot(spectra_dict, dimension="energy", show_plot=False,
                  log_y=False, limit=None):
    """ Produce spectral plot.

    For a given signal, produce a plot showing the signal and relevant
    backgrounds. Backgrounds are automatically summed to create the
    spectrum "Summed background" and all spectra passed in
    :obj:`spectra_dict` will be summed to produce the "Sum" spectra

    Args:
      spectra_dict (dict): Dictionary containing each spectrum you wish
        to plot, and the relevant parameters required to plot them.
      dimension (string, optional): The dimension or axis along which the
        spectra should be plotted. Default is energy.
      show_plot (bool, optional): Displays plot if True. Default is False.
      log_y (bool, optional): Use log scale on y-axis.
      limit (:class:`spectra.Spectra`): Include a spectrum showing
        a current or target limit.

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

    Returns:
      :class:`ROOT.TCanvas`: Canvas containing spectral plot.
    """
    root_style.root_style()
    first_spectra = True
    can = ROOT.TCanvas()
    root_style.set_ticks(can)
    leg = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    if log_y is True:
        can.SetLogy()
    hists = []
    for value in spectra_dict.values():
        spectra = value.get("spectra")
        if first_spectra:
            dim_type = spectra.get_config().get_dim_type(dimension)
            par = spectra.get_config().get_par(dimension+"_"+dim_type)
            low = par._low
            high = par._high
            bins = par._bins
            shape = (bins)  # Shape for summed arrays
            root_labels = "; %s (%s); Counts" % (dimension, par.get_unit())
            summed_background = ROOT.TH1F("summed_background", root_labels,
                                          bins, low, high)
            summed_total = ROOT.TH1F("summed_total", root_labels,
                                     bins, low, high)
            summed_total = numpy.zeros(shape=shape)
            summed_background.SetLineStyle(7)
            summed_background.SetLineColor(ROOT.kRed)
            summed_total.SetLineStyle(7)
            summed_total.SetLineColor(ROOT.kBlack)
            leg.AddEntry(summed_total, "Background + Signal", "l")
            leg.AddEntry(summed_background, "Background", "l")
            hist = spectra.project(dimension, graphical=False)
            hist.SetLineColor(value.get("style")["color"])
            leg.AddEntry(hist, value.get("label"), 'l')
            hists.append(hist)
            if value.get("type") is "background":
                summed_background.Add(hist)
            summed_total.Add(hist)
            first_spectra = False
        else:
            dim_type = spectra.get_config().get_dim_type("energy")
            par = spectra.get_config().get_par("energy_"+dim_type)
            if par._low != low:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect energy lower limit")
            if par._high != high:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect energy upper limit")
            if par._bins != bins:
                raise AssertionError("Spectra " + spectra._name + " has "
                                     "incorrect energy upper limit")
            hist = spectra.project(dimension, graphical=False)
            hist.SetLineColor(value.get("style")["color"])
            leg.AddEntry(hist, value.get("label"), 'l')
            hists.append(hist)
            if value.get("type") is "background":
                summed_background.Add(hist)
            summed_total.Add(hist)
    # Draw after making hists so they are drawn in the right order
    summed_total.Draw()
    summed_background.Draw("same")
    for hist in hists:
        hist.Draw("same")

    # Plot limit
    if limit:
        hist = limit.project(dimension, graphical=False)
        hist.SetLineColor(ROOT.kGray)
        leg.AddEntry(hist, "Kamland-Zen Limit", "l")
    leg.Draw("same")
    return can


def plot_chi_squared_per_bin(calculator, x_bins, x_low, x_high,
                             x_title=None, graphical=False, h_name=None):
    """ Produces a histogram of chi-squared per bin.

    Args:
      calculator (:class:`echidna.limit.chi_squared.ChiSquared`): Calculator
        containing the chi-squared values to plot.
      x_bins (int): Number of bins.
      x_low (float): Lower edge of first bin to plot.
      x_high (float): Upper edge of last bin to plot.
      x_title (string, optional): X Axis title.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TH1D`: Histogram of chi-squared per bin.
    """
    if not h_name:
        h_name = "chi_sq_per_bin" + strings.id_generator()
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


def plot_stats_vs_scale(limit_results, graphical=True):
    """ Plots the test statistics vs signal scales as a :class:`ROOT.TGraph`
      object.

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = limit_results.get_scales()
    g = ROOT.TGraph(len(scales), scales, limit_results.get_full_stats())
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle("Number of Signal Decays")
    g.GetYaxis().SetTitle("Test Statistic")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_raw_stats_vs_par_scale(fit_results, par, graphical=True, **kwargs):
    """ Plots the test statistics vs signal scales as a :class:`ROOT.TGraph`
      object. The penalty term has not been added to the test_statistic.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      par (string): The name of the parameter you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.
      kwargs (dict): Fit par names as keys and fit par values as values.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = fit_results.get_scales(par)
    g = ROOT.TGraph(len(scales), scales, fit_results.get_raw_stats(**kwargs))
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle(par)
    g.GetYaxis().SetTitle("Test Statistic")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_raw_stats_vs_pars(fit_results, par1, par2, graphical=True, **kwargs):
    """ Plots the test statistics vs signal scales as a :class:`ROOT.TGraph`
      object. The penalty term has not been added to the test_statistic.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      par1 (string): The name of the parameter you want to plot (x axis).
      par2 (string): The name of the parameter you want to plot (y axis).
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.
      kwargs (dict): Other fit par names as keys and fit par values as values.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    p1 = fit_results._fit_config.get_par(par1)
    p2 = fit_results._fit_config.get_par(par2)
    h = ROOT.TH2D("chi_sq"+strings.id_generator(), ";"+par1+";"+par2,
                  p1._bins, p1._low, p1._high, p2._bins, p2._low, p2._high)
    stats = fit_results.get_raw_stats(**kwargs)
    idx_x = fit_results._fit_config.get_index(par1)
    idx_y = fit_results._fit_config.get_index(par2)
    for idx_1, data_1 in enumerate(stats):
        for idx_2, datum in enumerate(data_1):
            if idx_x < idx_y:
                h.SetBinContent(idx_1 + 1, idx_2 + 1, datum)
            else:
                h.SetBinContent(idx_2 + 1, idx_1 + 1, datum)
    if graphical:
        can = ROOT.TCanvas()
        root_style.root_style()
        root_style.set_ticks(can)
        can.SetLogz()
        h.Draw("colz")
        can.Print("test.png")
        raw_input("RET to quit")
    return h


def plot_stats_vs_pars(fit_results, par1, par2, graphical=True, **kwargs):
    """ Plots the test statistics vs signal scales as a :class:`ROOT.TGraph`
      object. The penalty term has not been added to the test_statistic.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      par1 (string): The name of the parameter you want to plot (x axis).
      par2 (string): The name of the parameter you want to plot (y axis).
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.
      kwargs (dict): Other fit par names as keys and fit par values as values.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    p1 = fit_results._fit_config.get_par(par1)
    p2 = fit_results._fit_config.get_par(par2)
    h = ROOT.TH2D("chi_sq"+strings.id_generator(), ";"+par1+";"+par2,
                  p1._bins, p1._low, p1._high, p2._bins, p2._low, p2._high)
    stats = fit_results.get_raw_stats(**kwargs)
    idx_x = fit_results._fit_config.get_index(par1)
    idx_y = fit_results._fit_config.get_index(par2)
    for idx_1, data_1 in enumerate(stats):
        for idx_2, datum in enumerate(data_1):
            if idx_x < idx_y:
                kwargs[par1] = p1.get_value_at(idx_1)
                kwargs[par2] = p2.get_value_at(idx_2)
                datum += fit_results.get_penalty_at(**kwargs)
                h.SetBinContent(idx_1 + 1, idx_2 + 1, datum)
            else:
                kwargs[par1] = p1.get_value_at(idx_2)
                kwargs[par2] = p2.get_value_at(idx_1)
                datum += fit_results.get_penalty_at(**kwargs)
                h.SetBinContent(idx_2 + 1, idx_1 + 1, datum)
    if graphical:
        can = ROOT.TCanvas()
        can.SetLogz()
        root_style.root_style()
        root_style.set_ticks(can)
        h.Draw("colz")
        can.Print("test.png")
        raw_input("RET to quit")
    return h


def plot_stats_vs_par_scale(fit_results, par, graphical=True):
    """ Plots the test statistics vs signal scales as a :class:`ROOT.TGraph`
      object.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      par (string): The name of the parameter you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = fit_results.get_scales(par)
    g = ROOT.TGraph(len(scales), scales, fit_results.get_stats())
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle(par)
    g.GetYaxis().SetTitle("Test Statistic")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_best_fit_vs_scale(limit_results, par, graphical=True):
    """ Plots the best fit of the parameter vs signal scales as a
      :class:`ROOT.TGraph` object.

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data.
      par (string): Paramter name you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = limit_results.get_scales()
    g = ROOT.TGraph(len(scales), scales, limit_results.get_best_fits(par))

    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle("Number of Signal Decays")
    g.GetYaxis().SetTitle(par+" Best Fit")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_sigma_best_fit_vs_scale(limit_results, par, graphical=True):
    """ Plots the best fit in term of number of sigma away from the prior of
      the parameter vs signal scales as a :class:`ROOT.TGraph` object.

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data.
      par (string): Paramter name you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    best_fits = limit_results.get_best_fits(par)
    prior = limit_results._fit_config.get_par(par).get_prior()
    sigma = limit_results._fit_config.get_par(par).get_sigma()
    best_fits -= prior
    best_fits /= sigma
    scales = limit_results.get_scales()
    g = ROOT.TGraph(len(scales), scales, best_fits)
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle("Number of Signal Decays")
    g.GetYaxis().SetTitle(par+" Best fit (Num. #sigma)")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_per_bin(fit_results, min_chisq=True, graphical=True,
                 x_dim="energy_mc", y_dim=None, **kwargs):
    """ Plots the raw test statistics (no penalty) of the fitting space.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      min_chisq (bool, optional): Plots fitted minimum test statistic space.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.
      x_dim (string, optional): The name of the parameter you want to plot
        (x axis). Default is energy_mc.
      y_dim (string, optional): The name of the parameter you want to plot
        (y axis). Default is None.
      kwargs (dict): Other fit par names as keys and fit par values as values
        you want to plot the test statistic space at.

    Returns:
      :class:`ROOT.TH1D` or :class:`ROOT.TH2D`: The test statistic space
        histogram.
    """
    if min_chisq:
        best_fits = {}
        summary = fit_results.get_summary()
        for par in fit_results._fit_config.get_pars():
            best_fits[par] = fit_results._fit_config.get_par(par)._best_fit
        stats = fit_results.get_raw_stats_at(**best_fits)
    else:
        stats = fit_results.get_raw_stats_at(**kwargs)
    if y_dim:
        x_par = fit_results._spectra_config.get_par(x_dim)
        y_par = fit_results._spectra_config.get_par(y_dim)
        idx_x = fit_results._spectra_config.get_index(x_dim)
        idx_y = fit_results._spectra_config.get_index(x_dim)
        h = ROOT.TH2D("chi_sq" + strings.id_generator(),
                      ";" + x_dim + ";" + y_dim + ";#chi^{2}", x_par._bins,
                      x_par._low, x_par._high, y_par._bins, y_par._low,
                      y_par._high)
        for idx_1, data_1 in enumerate(stats):
            for idx_2, datum in enumerate(data_1):
                if idx_x < idx_y:
                    h.SetBinContent(idx_1 + 1, idx_2 + 1, datum)
                else:
                    h.SetBinContent(idx_2 + 1, idx_1 + 1, datum)
    else:
        x_par = fit_results._spectra_config.get_par(x_dim)
        h = ROOT.TH1D("chi_sq" + strings.id_generator(),
                      ";" + x_dim + "; #chi^{2}", x_par._bins, x_par._low,
                      x_par._high)
        for i, stat in enumerate(stats):
            h.SetBinContent(i+1, stat)
    if graphical:
        if y_dim:
            h.Draw("colz")
        else:
            h.Draw()
        raw_input("RET to cont")
    return h


def plot_penalty_term_vs_par_scale(fit_results, par, graphical=True):
    """ Plots the contribution of the penalty term to the test statistic for
      the parameter vs signal scales as a :class:`ROOT.TGraph` object.

    Args:
      fit_results (:class:`echidna.fit.fit_results.FitResults`): The
        fit_results object which contains the data.
      par (string): Paramter name you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = fit_results.get_scales(par)
    g = ROOT.TGraph(len(scales), scales, fit_results.get_penalty_terms(par))
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle(par)
    g.GetYaxis().SetTitle("Test Statistic Penalty")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g


def plot_penalty_term_vs_scale(limit_results, par, graphical=True):
    """ Plots the contribution of the penalty term to the test statistic for
      the parameter vs signal scales as a :class:`ROOT.TGraph` object.

    Args:
      limit_results (:class:`echidna.fit.fit_results.LimitResults`): The
        limit_results object which contains the data.
      par (string): Paramter name you want to plot.
      graphical (bool, optionl): Plots hist to screen if True.
        Default is False.

    Returns:
      :class:`ROOT.TGraph`: The plot.
    """
    scales = limit_results.get_scales()
    g = ROOT.TGraph(len(scales), scales, limit_results.get_penalty_terms(par))
    g.SetMarkerStyle(3)
    g.GetXaxis().SetTitle("Number of Signal Decays")
    g.GetYaxis().SetTitle("Test Statistic Penalty")
    if graphical:
        g.Draw("AP")
        raw_input("RET to quit")
    return g
