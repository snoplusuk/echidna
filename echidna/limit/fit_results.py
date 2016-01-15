""" Fit results module, containing ``FitResults`` class.
"""
import numpy

import copy


class FitResults(object):
    """ Base class for handling results of the fit.

    Args:
      fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".
      bins (tuple, optional): Number of bins (in each dimension) used
        in fit. This is used to accommodate per-bin reporting of
        test-statisitc.
      num_bins (int, optional): Number of bins used in fit. This is used to
        accommodate per-bin reporting of test-statisitc.

    Attributes:
      _fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      _name (string): Name of this :class:`FitResults` class instance.
      _bins (tuple): Number of bins (in each dimension) used in fit.
        This is used to accommodate per-bin reporting of test-statisitc.
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _resets (int): Number of times the grid has been reset.

    Examples:

        >>> fit_results = FitResults(fitter.get_config())
    """
    def __init__(self, fit_config, name=None, bins=None):
        self._fit_config = fit_config
        if name is None:
            name = fit_config.get_name() + "_results"
        self._name = name
        self._bins = bins
        self._stats = numpy.zeros(self.get_shape())
        self._penalties = numpy.zeros(self.get_basic_shape())
        self._resets = 0

    def get_basic_shape(self):
        """ Determine the basic shape of the grid of parameter values.

        Excludes shape of the spectrum data, purely the shape the fit
        parameter grid.

        Returns:
          tuple: Shape of parameter grid.
        """
        shape = []
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            shape.append(len(parameter.get_values()))
        return tuple(shape)

    def get_fit_config(self):
        """
        Returns:
          (:class:`echidna.limit.fit.FitConfig`): Configuration of fit.
        """
        return self._fit_config

    def get_fit_data(self):
        """ Combines the test-statistic array (collapsed to the parameter
        values grid - i.e. summed over spectral bins) with the penalty
        term grid of the same shape.

        Returns:
          (:class:`numpy.ndarray`): Array combining the values of the
            test statistic calculated during the fit and the penalty
            term values.
        """
        combined = copy.copy(self._stats)

        # Collapse by summing over per-bin dimensions
        spectral_dims = len(self.get_shape()) - len(self.get_basic_shape())
        for dim in range(spectral_dims):
            combined = numpy.sum(combined, axis=-1)  # always last axis

        # Add penalties
        combined = combined + self._penalties

        return combined

    def get_name(self):
        """
        Returns:
          string: Name of fit results object.
        """
        return self._name

    def get_penalties(self):
        """
        Returns:
          (:class:`numpy.ndarray`): Array stored in :attr:`_penalties`.
            Values of the penalty term calculated during the fit.
        """
        return self._penalties

    def get_per_bin(self, indices):
        """ Get per-bin values of the test statistic at a given set of
        indices.

        Returns:
          :class:`numpy.ndarray`: Per-bin array of the test statistic
            values.

        .. warning:: Since the penalty term is only calculated for the
          sum over all bins, it's effect is not included here. The
          array values returned here are raw test statistic values.
        """
        return self._stats[indices]

    def get_shape(self):
        """ Determine the shape of the grid of parameter values.

        Returns:
          tuple: Shape of parameter grid.
        """
        shape = []
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            shape.append(len(parameter.get_values()))

        # If using per-bin monitoring need to add number of bins
        if self._bins is not None:
            for dimension in self._bins:
                shape.append(dimension)
        return tuple(shape)

    def get_stats(self):
        """
        Returns:
          (:class:`numpy.ndarray`): Array store in :attr:`_stats`.
            Values of the test statistic calculated during the fit.
        """
        return self._stats

    def get_summary(self):
        """ Get a summary of the fit parameters.

        Returns:
          dict: Results of fit. Dictionary with fit parameter names as
            keys and a nested dictionary as values containing the keys
            best_fit and penalty_term with the corresponding values for the
            parameter.
        """
        fit_results = {}
        for par in self._fit_config.get_pars():
            parameter = self._fit_config.get_par(par)
            fit_results[par] = {"best_fit": parameter.get_best_fit(),
                                "penalty_term": parameter.get_penalty_term()}
        return fit_results

    def reset_grids(self):
        """ Resets the grids stored in :attr:`_stats` and
          :attr:`_penalties`, including shape.

        .. warning:: If fit parameters have been added/removed, calling
          this method will increase/decrease the dimensions of the grid
          to compensate for this change.
        """
        if self._resets == 0:
            self._resets = 1
            self._name += "_%d" % self._resets
        else:
            self._resets += 1
            self._name = self._name[:-1] + str(self._resets)
        self._stats = numpy.zeros(self.get_shape())
        self._penalties = numpy.zeros(self.get_basic_shape())

    def set_fit_config(self, fit_config):
        """ Set the fit config.

        Args:
          fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
            for fit. This should be a direct copy of the
            :class:`echidna.limit.fit.FitConfig` object in
            :class:`echidna.limit.fit.Fit`.
        """
        self._fit_config = fit_config
        self.reset_grid()
