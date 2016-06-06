""" Fit results module, containing ``FitResults`` class.
"""
import numpy
import copy
import itertools
from collections import OrderedDict


class FitResults(object):
    """ Base class for handling results of the fit.

    Args:
      fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): The
        configuration for fit. This should be a direct copy of the
        ``FitConfig`` in :class:`echidna.limit.fit.Fit`.
      spectra_config (:class:`echidna.core.spectra.SpectraConfig`): The
        for spectra configuration. The recommended spectrum config to
        include here is the one from the data spectrum, to which you
        are fitting.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".

    Attributes:
      _fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): The
        configuration for fit. This should be a direct copy of the
        ``FitConfig`` in :class:`echidna.limit.fit.Fit`.
      _spectra_config (:class:`echidna.core.spectra.SpectraConfig`): The
        for spectra configuration. The recommended spectrum config to
        include here is the one from the data spectrum, to which you
        are fitting.
      _name (string): Name of this :class:`FitResults` class instance.

    Examples:

        >>> fit_results = FitResults(fitter.get_config(), data.get_config())
    """
    def __init__(self, fit_config, spectra_config, name=None):
        self._fit_config = fit_config
        self._spectra_config = spectra_config
        if name is None:
            name = fit_config.get_name() + "_results"
        self._name = name

    def get_fit_config(self):
        """
        Returns:
          (:class:`echidna.core.config.GlobalFitConfig`): Configuration
            of fit.
        """
        return self._fit_config

    def get_name(self):
        """
        Returns:
          string: Name of fit results object.
        """
        return self._name

    def get_spectra_config(self):
        """
        Returns:
          (:class:`echidna.core.config.SpectraConfig`): Configuration
            spectrum.
        """
        return self._spectra_config

    def get_summary(self):
        """ Get a summary of the fit parameters.

        Returns:
          dict: Results of fit. Dictionary with fit parameter names as
            keys and a nested dictionary as values containing the keys
            best_fit and penalty_term with the corresponding values for the
            parameter.
        """
        fit_results = OrderedDict({})
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            fit_results[par] = {"best_fit": parameter.get_best_fit(),
                                "penalty_term": parameter.get_penalty_term()}
        return fit_results

    def set_fit_config(self, fit_config):
        """ Set the fit config.

        .. warning:: This will automatically call :meth:`reset_grid`
          to update the grid based on the new fit config

        Args:
          fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): The
            configuration for fit. This should be a direct copy of the
            :class:`echidna.core.spectra.GlobalFitConfig` object in
            :class:`echidna.limit.fit.Fit`.
        """
        self._fit_config = fit_config
        self.reset_grid()

    def set_spectra_config(self, spectra_config):
        """ Set the spectra config.

        .. warning:: This will automatically call :meth:`reset_grid`
          to update the grid based on the new fit config

        Args:
          spectra_config (:class:`echidna.core.spectra.SpectraConfig`): The
            configuration for the spectra. This should usually be a
            direct copy of the :class:`echidna.core.spectra.SpectraConfig`
            in the data spectrum.
        """
        self._spectra_config = spectra_config
        self.reset_grid()


class LimitResults(FitResults):
    """ Base class for handling results of limit setting.

    Args:
      fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): The
        configuration for fit. This should be a direct copy of the
        ``FitConfig`` in :class:`echidna.limit.fit.Fit`.
      signal_config (:class:`echidna.core.spectra.SpectraConfig`): The
        spectra configuration. The recommended spectrum config to
        include here is the one from the data spectrum, to which you
        are fitting.
      limit_config (:class:`echidna.core.spectra.SpectraFitConfig`): The
        ``FitConfig`` of the signal you are setting a limit with.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".

    Attributes:
      _limit_config (:class:`echidna.core.spectra.SpectraFitConfig`): The
        ``FitConfig`` of the signal you are setting a limit with.
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during limit setting.
      _penalty_terms (:class:`numpy.ndarray`): Array of values of the
        penalty terms calculated during limit setting.
      _best_fits (:class:`numpy.ndarray`): Array of values of the
        best fits calculated during limit setting.
      _fit_results (:class:`echidna.fit.fit_results.FitResults`): An array of
        fit results for each signal scaling.
    """
    def __init__(self, fit_config, signal_config, limit_config, name=None):
        if name is None:
            name = signal_config.get_name() + "_limit_results"
        super(LimitResults, self).__init__(fit_config, signal_config, name)
        self._limit_config = limit_config
        scales = limit_config.get_par("rate").get_bins()
        shape = (scales, len(fit_config.get_pars()))
        self._penalty_terms = numpy.zeros(shape)
        self._best_fits = numpy.zeros(shape)
        self._stats = numpy.zeros(scales)
        self._fit_results = numpy.empty(scales, dtype=object)

    def get_best_fit(self, i, par):
        """Gets the best fit at scale index i.

        Args:
          i (int): Index of scaling.
          par (string): Name of parameter.

        Returns:
          numpy.ndarray: The best fits array.
        """
        par_idx = self._fit_config.get_index(par)
        return self._best_fits[i][par_idx]

    def get_best_fits(self, par):
        """Gets the best fit array.

        Args:
          par (string): Name of parameter.

        Returns:
          numpy.ndarray: The best fits array.
        """
        par_idx = self._fit_config.get_index(par)
        return self._best_fits[:, par_idx]

    def get_full_stat(self, i):
        """Gets the test statistic with penalty terms added for the signal
          scale at index i.

        Args:
          i (int): Index of stat.

        Returns:
          float: The test statistic with penalty terms addded.
        """
        stat = numpy.sum(self._stats[i])
        for penalty in self._penalty_terms[i]:
            stat += penalty
        return stat

    def get_full_stats(self):
        """Gets the test statistics with penalty terms added for each signal
          scale.

        Returns:
          numpy.ndarray: The test statistics array with penalty terms addded.
        """
        stats = numpy.zeros(self._stats.shape)
        for i, stat in enumerate(self._stats):
            stat = numpy.sum(stat)
            for penalty in self._penalty_terms[i]:
                stat += penalty
            stats[i] = stat
        return stats

    def get_limit_stat(self, i):
        """Gets the test statistic with penalty terms added for the signal
          scale at index i.

        Args:
          i (int): Index of stat.

        Returns:
          float: The test statistic without the penalty contribution.
        """
        return self._limit_stat[i]

    def get_stats(self):
        """Gets the test statistics without the penalty contribution.

        Returns:
          numpy.ndarray: The test statistics array without the
            penalty contribution.
        """
        return self._stats

    def get_penalty_terms_at_scale(self, i):
        """Get the penalty terms for signal scaling at position i.

        Args:
          i (int): Array index of signal scale.

        Returns:
          numpy.ndarray: The penalty terms
        """
        return self._penalty_terms[i]

    def get_penalty_terms(self, par):
        """Get the set of penalty terms for a parameter for all scales.

        Args:
          par (string): The name of the parameter.

        Returns:
          numpy.ndarray: Set of penalty terms for all signal scales.
        """
        i = self._fit_config.get_index(par)
        return self._penalty_terms[:, i]

    def get_penalty_term(self, scale_idx, par):
        """Get the set of penalty terms for a parameter for all scales.

        Args:
          scale_idx (int): Index of signal scaling.
          par (string): The name of the parameter.

        Returns:
          numpy.ndarray: Set of penalty terms for all signal scales.
        """
        i = self._fit_config.get_index(par)
        return self._penalty_terms[scale_idx, i]

    def get_scales(self):
        """Gets the signal scales used in limit setting

        Returns:
          numpy.ndarray: Signal scales.
        """
        return self._limit_config.get_par("rate").get_values()

    def set_best_fit(self, scale_idx, best_fit, par):
        """ Sets the best fit for parameter with index par_idx and scale
        with index scale_idx.

        Args:
          best_fit (float): Best fit.
          scale_idx (int): Scale index.
          par_idx (int): Fit parameter index.
        """
        par_idx = self._fit_config.get_index(par)
        self._best_fits[scale_idx][par_idx] = best_fit

    def set_penalty_term(self, scale_idx, penalty_term, par):
        """ Sets the penalty term for parameter with index par_idx and scale
        with index scale_idx.

        Args:
          penalty_term (float): Penalty term.
          scale_idx (int): Scale index.
          par (str): Fit parameter name.
        """
        par_idx = self._fit_config.get_index(par)
        self._penalty_terms[scale_idx][par_idx] = penalty_term

    def set_limit_stat(self, scale_idx, stat):
        """ Set the test statistic at scale with index scale_idx

        Args:
          stat (float): Test statitstic.
          scale_idx (int): Scale index.
       """
        self._stats[scale_idx] = stat

    def set_fit_result(self, scale_idx, fit_result):
        """ Sets the fit results object for a given signal scaling.

        Args:
          scale_idx (int): Scale index.
          fit_result (:class:`echidna.fit.fit_results.FitResults`): Results
            from fitting a given signal scale.
        """
        self._fit_results[scale_idx] = fit_result
