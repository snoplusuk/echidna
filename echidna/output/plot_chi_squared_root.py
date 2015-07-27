import ROOT
import numpy

def chi_squared_vs_signal(signal_config):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
        config class, where chi squareds have been stored.

    .. note::

      Keyword arguments include:

        * penalty (:class:`echidna.limit.limit_config.LimitConfig`): config
          for signal with penalty term.
    """
    x = numpy.array(signal_config._chi_squareds[2])
    y = numpy.array(signal_config._chi_squareds[0])
    graph = ROOT.TGraph(len(x), x , y)
    graph.GetXaxis().SetTitle("Signal counts")
    graph.GetYaxis().SetTitle("$\chi^{2}$")
    return graph

def chi_squared_vs_half_life(signal_config, converter, scaling):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
        config class, where chi squareds have been stored.

    .. note::

      Keyword arguments include:

        * penalty (:class:`echidna.limit.limit_config.LimitConfig`): config
          for signal with penalty term.
    """
    x = numpy.array(signal_config._chi_squareds[2])
    y = numpy.array(signal_config._chi_squareds[0])
    graph = ROOT.TGraph()
    for i in range(len(x)):
        half_life = converter.counts_to_half_life(x[i]/scaling)
        graph.SetPoint(i, 1./half_life, y[i])
    graph.GetXaxis().SetTitle("1/T_{1/2}")
    graph.GetYaxis().SetTitle("$\chi^{2}$")
    return graph


def chi_squared_vs_mass(signal_config, converter, scaling):
    """ Plot the chi squared as a function of signal counts

    Args:
      signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
        config class, where chi squareds have been stored.

    .. note::

      Keyword arguments include:

        * penalty (:class:`echidna.limit.limit_config.LimitConfig`): config
          for signal with penalty term.
    """
    x = numpy.array(signal_config._chi_squareds[2])
    y = numpy.array(signal_config._chi_squareds[0])
    graph = ROOT.TGraph()
    for i in range(len(x)):
        mass = converter.counts_to_eff_mass(x[i]/scaling)
        graph.SetPoint(i, mass, y[i])
    graph.GetXaxis().SetTitle("$m_{#beta#beta}$")
    graph.GetYaxis().SetTitle("$\chi^{2}$")
    return graph
