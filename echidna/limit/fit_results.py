""" Fit results module, containing ``FitResults`` class.
"""
import numpy

import copy
import itertools


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
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _penalty_terms (:class:`numpy.ndarray`): Array of values of the
        penalty terms calculated during the fit.
      _minimum_value (float): Minimum value of the array returned by
        :meth:`get_fit_data`.
      _minimum_position (tuple): Position of the test statistic minimum
        value. The tuple contains the indices along each fit parameter
        (dimension), acting as coordinates of the position of the
        minimum.
      _resets (int): Number of times the grid has been reset.

    Examples:

        >>> fit_results = FitResults(fitter.get_config(), data.get_config())
    """
    def __init__(self, fit_config, spectra_config, name=None):
        self._fit_config = fit_config
        self._spectra_config = spectra_config
        if name is None:
            name = fit_config.get_name() + "_results"
        self._name = name
        stats_shape = fit_config.get_shape() + spectra_config.get_shape()
        self._stats = numpy.zeros(stats_shape)
        self._penalty_terms = numpy.zeros(fit_config.get_shape())
        self._minimum_value = None
        self._minimum_position = None
        self._resets = 0

    def get_fit_config(self):
        """
        Returns:
          (:class:`echidna.limit.fit.FitConfig`): Configuration of fit.
        """
        return self._fit_config

    def get_minimum_position(self):
        """
        Returns:
          (float): Position of the minimum value in the array returned
            by :meth:`get_fit_data`, stored in :attr:`_minimum_position`.
        """
        return self._minimum_position

    def get_minimum_value(self):
        """
        Returns:
          (float): Minimum value of the array returned by
            :meth:`get_fit_data`, stored in :attr:`_minimum_value`.
        """
        return self._minimum_value

    def get_name(self):
        """
        Returns:
          string: Name of fit results object.
        """
        return self._name

    def get_resets(self):
        """
        Returns:
          int: Number of times the grid has been reset (:attr:`_resets`).
        """
        return self._resets

    def get_penalty_term(self, indices):
        """ Gets the array of penalty terms.

        .. note:: Unlike the :class:`echidna.fit.summary.Summary` class
          individual penalty contributions from each fit parameter are
          not stored here, only the total penalty term value.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the total
            penalty term value.

        Returns:
          (:class:`numpy.ndarray`): Array stored in :attr:`_penalty_terms`.
            Values of the penalty term calculated during the fit.

        Raises:
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        return self._penalty_terms[indices]

    def get_penalty_terms(self):
        """ Gets the array of penalty terms.

        .. note:: Unlike the :class:`echidna.fit.summary.Summary` class
          individual penalty contributions from each fit parameter are
          not stored here, only the total penalty term value.

        Returns:
          (:class:`numpy.ndarray`): Array stored in :attr:`_penalty_terms`.
            Values of the penalty term calculated during the fit.
        """
        return self._penalty_terms

    def get_raw_stat(self, indices):
        """ Gets the raw test statistic(s) from array at the given indices.

        .. warning:: This has no penalty term contributions added.

        .. note:: Unlike :meth:`get_stat`, here you can specify indices
          for any number of fit parameters dimensions, so to geta slice
          of the raw array.

        Args:
          indices (tuple): Index along each fit parameter (dimension)
            specifiying the coordinates in the array.

        Returns:
          (float or :class:`numpy.ndarray`): The raw test statistic(s)
            at the given indices.

        Raises:
          TypeError: If the indices supplied are not at tuple
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        return self._stats[indices]

    def get_raw_stats(self):
        """ Gets the raw test statistics array.

        .. warning:: This has no penalty term contributions added.

        Returns:
          :class:`numpy.array`: The raw test statistics values at each
            combination of fit parameter values.
        """
        return self._stats

    def get_stat(self, indices):
        """ Combines the test-statistic array (collapsed to the parameter
        values grid - i.e. summed over spectral bins) with the penalty
        term grid of the same shape, for a single bin, specified by
        indices.

        .. warning:: Penalty term contributions **are** included here.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the test
            statistic value.

        Returns:
          (float): Combination of the value of the test statistic
            calculated during the fit and the penalty term value.

        Raises:
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        combined = copy.copy(self._stats[indices])

        # Collapse by summing over spectral dimensions
        for dim_size in self._spectra_config.get_shape():
            combined = numpy.sum(combined, axis=-1)  # always last axis

        # Add penalties
        combined = combined + self._penalty_terms[indices]
        return combined

    def get_stats(self):
        """ Combines the test-statistic array (collapsed to the parameter
        values grid - i.e. summed over spectral bins) with the penalty
        term grid of the same shape.

        .. warning:: Penalty term contributions **are** included here.

        Returns:
          (:class:`numpy.ndarray`): Array combining the values of the
            test statistic calculated during the fit and the penalty
            term values.
        """
        combined = copy.copy(self._stats)

        # Collapse by summing over spectral dimensions
        for dim_size in self._spectra_config.get_shape():
            combined = numpy.sum(combined, axis=-1)  # always last axis

        # Add penalties
        combined = combined + self._penalty_terms
        return combined

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

    def nd_project_stat(self, indices, *parameters):
        """ Projects the test statistic values, at given the given
        indices, onto the axes specified by fit and spectral parameters.

        .. note:: If only **fit** parameters are specified all spectral
          dimensions are collapsed and penalty term contributions
          **are** included. If any **spectral** parameters are provided
          penalty term contributions **are not** included.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the test
            statistic value.
          parameters (string): Names of a valid fit or spectral
            parameters onto which to project the test statistic values.

        Raises:
          TypeError: If the indices supplied are not at tuple
          IndexError: If the parameter names supplied do not match
            any of those stored in the fit or spectra configs.
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        for parameter in parameters:
            if parameter not in itertools.chain(
                    self._fit_config.get_pars(),
                    self._spectra_config.get_pars()):
                raise IndexError("Unknown parameter %s" % parameter)
        if parameters in self._fit_config.get_pars():
            # Can apply penalty term contributions
            projection = self.get_stat(indices)
            for axis, parameter in self._fit_config.get_pars():
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        else:  # No penalty terms, use raw stats
            projection = copy.copy(self.get_raw_stat(indices))
            for axis, parameter in enumerate(
                    itertools.chain(self._fit_config.get_pars(),
                                    self._spectra_config.get_pars())):
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        return projection

    def nd_project_stats(self, *parameters):
        """ Projects the test statistic values onto the axes specified
        by fit and spectral parameters.

        .. note:: If only **fit** parameters are specified all spectral
          dimensions are collapsed and penalty term contributions
          **are** included. If any **spectral** parameters are provided
          penalty term contributions **are not** included.

        Args:
          parameters (string): Names of a valid fit or spectral
            parameters onto which to project the test statistic values.

        Raises:
          IndexError: If the parameter names supplied do not match
            any of those stored in the fit or spectra configs.
        """
        for parameter in parameters:
            if parameter not in itertools.chain(
                    self._fit_config.get_pars(),
                    self._spectra_config.get_pars()):
                raise IndexError("Unknown parameter %s" % parameter)
        if parameters in self._fit_config.get_pars():
            # Can apply penalty term contributions
            projection = self.get_stats()
            for axis, parameter in self._fit_config.get_pars():
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        else:  # No penalty terms, use raw stats
            projection = copy.copy(self.get_raw_stats())
            for axis, parameter in enumerate(
                    itertools.chain(self._fit_config.get_pars(),
                                    self._spectra_config.get_pars())):
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        return projection

    def reset_grids(self):
        """ Resets the grids stored in :attr:`_stats` and
          :attr:`_penalty_terms`, including shape.

        .. warning:: If fit parameters have been added/removed, calling
          this method will increase/decrease the dimensions of the grid
          to compensate for this change.
        """
        if self._resets == 0:
            self._resets = 1
            self._name += "_%d" % self._resets
        else:
            new_name = self._name.split("_")[0]
            for part in self._name.split("_")[1:-1]:
                new_name += "_" + part
            self._resets += 1
            self._name = new_name + "_" + str(self._resets)
        stats_shape = (self._fit_config.get_shape() +
                       self._spectra_config.get_shape())
        self._stats = numpy.zeros(stats_shape)
        self._penalty_terms = numpy.zeros(self._fit_config.get_shape())

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

    def set_minimum_position(self, minimum_position):
        """
        Args:
          minimum_position (float): Position of the minimum value of
            the array returned by :meth:`get_fit_data`.
        """
        self._minimum_position = minimum_position

    def set_minimum_value(self, minimum_value):
        """
        Args:
          minimum_value (float): Minimum value of the array returned by
            :meth:`get_fit_data`.
        """
        self._minimum_value = minimum_value

    def set_penalty_terms(self, penalty_terms):
        """ Sets the array containing penalty term values.

        Args:
          penalty_terms (:class:`numpy.ndarray`): The array of penalty
            term values

        Raises:
          TypeError: If penalty_terms is not an :class:`numpy.ndarray`
          ValueError: If the penalty_terms array does not have the required
            shape.
        """
        if not isinstance(penalty_terms, numpy.ndarray):
            raise TypeError("penalty_terms must be a numpy array")
        if penalty_terms.shape != self._fit_config.get_shape():
            raise ValueError("penalty_terms array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(penalty_terms.shape),
                              str(self._fit_config.get_shape())))
        self._penalty_terms = penalty_terms

    def set_penalty_term(self, penalty_term, indices):
        """ Sets the total penalty term value at the point in the array
        specified by indices.

        Args:
          penalty_term (float): Best fit value of a fit parameter.
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to set the total
            penalty term value.

        Raises:
          TypeError: If penalty_term is not a float.
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if not isinstance(penalty_term, float):
            raise TypeError("penalty_term must be a float")
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        self._penalty_terms[indices] = penalty_term

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
        self._fit_config = spectra_config
        self.reset_grid()

    def set_stat(self, stat, indices):
        """ Sets the test statistic values in array at the point
        specified by indices

        Args:
          stat (:class:`numpy.ndarray`): Values of the test statistic.
          indices (tuple): Position in the array.

        Raises:
          TypeError: If the indices supplied are not at tuple
          IndexError: If the number of indices supplied does not match
            the dimensions of the fit
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
          TypeError: If stat is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(indices, tuple):
            raise TypeError("indices supplied must be a tuple of integers")
        if len(indices) != len(self._fit_config.get_shape()):
            raise IndexError("dimension mismatch, indices supplied contian "
                             "%d dimensions but fit contains %d dimensions "
                             "(parameters)" %
                             (len(indices), len(self._fit_config.get_shape())))
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        if not isinstance(stat, numpy.ndarray):
            raise TypeError("stat must be a numpy array")
        if stat.shape != self._stats[indices].shape:
            raise ValueError("stat array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stat.shape),
                              str(self._stats[indices].shape)))
        self._stats[indices] = stat

    def set_stats(self, stats):
        """ Sets the total test statistics array.

        Args:
          stats (:class:`numpy.ndarray`): The total test statistics array.

        Raises:
          TypeError: If stats is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(stats, numpy.ndarray):
            raise TypeError("stats must be a numpy array")
        if stats.shape != self._stats.shape:
            raise ValueError("stats array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stats.shape), str(self._stats.shape)))
        self._stats = stats
