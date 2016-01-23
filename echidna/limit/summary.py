import numpy

import copy


class Summary(object):
    """ This class contains the summary data for a given fit parameter when a
      limit has been set with a signal spectrum. It stores arrays of signal
      scales used in the limit setting and the corresponding best fits and
      penalty  term values for the given fit parameter as well as the prior
      and sigma values of the fit parameter used. The total test statistic
      values are also stored as an array.

    Args:
      name (string): Name of the summary object.
      num_scales (int): Number of signal scales used in the limit setting.
      spectra_config (:class:`echidna.core.spectra.SpectraConfig`): Config
        for the signal spectrum.
      fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): Fit
        config used during limit setting.

    Attributes:
      _name (string): Name of the summary object.
      _num_scales (int): Number of signal scales used in the limit setting.
      _spectra_config (:class:`echidna.core.spectra.SpectraConfig`): Config
        for the signal spectrum.
      _fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): Fit
        config used during limit setting.
      _best_fits (:class:`numpy.ndarray`): The best fit values of the fit
        parameter for each corresponding signal scale.
      _penalty_terms (:class:`numpy.ndarray`): The penalty term values of the
        fit parameter for each corresponding signal scale.
      _scales (:class:`numpy.ndarray`): The signal scales used in the limit
        setting.
      _stats (:class:`numpy.ndarray`): The total test statistic values for
        each corresponding signal scale.
      _priors (float): The prior values of each fit parameter
      _sigma (float): The systematic uncertainty value of the fit parameter
    """
    def __init__(self, name, num_scales, spectra_config, fit_config):
        """ Initialises the summary data container
        """
        self._name = name
        self._num_scales = num_scales
        self._spectra_config = spectra_config
        spectra_shape = self._spectra_config.get_shape()
        self._fit_config = fit_config
        pars_shape = (num_scales, len(self._fit_config.get_pars()))
        self._best_fits = numpy.zeros(shape=pars_shape, dtype=float)
        self._penalty_terms = numpy.zeros(shape=pars_shape, dtype=float)
        self._scales = numpy.zeros(shape=num_scales, dtype=float)
        stats_shape = tuple([num_scales]) + spectra_shape
        self._stats = numpy.zeros(shape=stats_shape, dtype=float)
        self._priors = numpy.zeros(shape=len(self._fit_config.get_pars()))
        self._sigmas = numpy.zeros(shape=len(self._fit_config.get_pars()))

    def get_best_fits(self, parameter=None):
        """ Gets the best_fits array.

        If a parameter name is specified, the array returned will
        contain only the best-fit value of this parameter, for each
        signal scale. Otherwise, the best-fit value for each fit
        parameter, at each signal scale, is returned.

        Args:
          parameter (string, optional): Name of a valid fit parameter
            for which to get best-fit values

        Returns:
          :class:`numpy.ndarray`: The best fit(s) at each signal scaling
            for the fit parameter(s).

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter is None:  # return full best_fits array
            return self._best_fits
        elif parameter in self._fit_config.get_pars():
            # Create a new numpy array to fill
            result = numpy.zeros(shape=self._num_scales)

            # Get parameter index
            par_num = self._fit_config.get_index(parameter)
            for idx, best_fits in enumerate(self._best_fits):
                result[idx] = best_fits[par_num]
            return result
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_best_fit(self, idx, parameter=None):
        """ Gets the best_fit from array at idx

        If a parameter name is specified, only the best-fit value of
        this parameter, at the given index, will be returned.
        Otherwise, the best-fit value for each fit parameter, at the
        given index, is returned.

        Args:
          idx (int): Index in the array.
          parameter (string, optional): Name of a valid fit parameter
            for which to get the best-fit value

        Returns:
          float: The best fit value of the fit parameter at idx in the array.
          :class:`numpy.ndarray`: If a parameter value is supplied, the
            best-fit values for each fit parameter are returned.

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter is None:  # return all best fit values at idx
            return self._best_fits[idx]
        elif parameter in self._fit_config.get_pars():
            # Get parameter index
            par_num = self._fit_config.get_index(parameter)
            return self._best_fits[idx][par_num]
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_name(self):
        """ Gets the name of the summary object

        Returns:
          string: Name of the summary object
        """
        return self._name

    def get_num_scales(self):
        """ Gets the number of scales in the summary object

        Returns:
          int: Number of scales
        """
        return self._num_scales

    def get_penalty_terms(self, parameter=None):
        """ Gets the penalty_terms array

        If a parameter name is specified, the array returned will
        contain only the penalty term value of this parameter, for each
        signal scale. Otherwise, the penalty term value for each fit
        parameter, at each signal scale, is returned.

        Args:
          parameter (string, optional): Name of a valid fit parameter
            for which to get the penalty term values

        Returns:
          :class:`numpy.array`: The values of the penalty terms of the fit
            parameter(s) at each signal scale.

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter is None:  # return full penalty_terms array
            return self._penalty_terms
        elif parameter in self._fit_config.get_pars():
            # Create a new numpy array to fill
            result = numpy.zeros(shape=self._num_scales)

            # Get parameter index
            par_num = self._fit_config.get_index(parameter)
            for idx, penalty_terms in enumerate(self._penalty_terms):
                result[idx] = penalty_terms[par_num]
            return result
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_penalty_term(self, idx, parameter=None):
        """ Gets the penalty_term from array at idx

        If a parameter name is specified, only the penalty term value of
        this parameter, at the given index, will be returned.
        Otherwise, the penalty term value for each fit parameter, at the
        given index, is returned.

        Args:
          idx (int): Index in the array.
          parameter (string, optional): Name of a valid fit parameter
            for which to get the penalty term value

        Returns:
          float: The penalty term value of the fit parameter at idx in
            the array.
          :class:`numpy.ndarray`: If a parameter value is supplied, the
            penalty term values for each fit parameter are returned.

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter is None:  # return all penalty term values at idx
            return self._penalty_terms[idx]
        elif parameter in self._fit_config.get_pars():
            # Get parameter index
            par_num = self._fit_config.get_index(parameter)
            return self._penalty_terms[idx][par_num]
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_priors(self):
        """ Get the priors array.

        Returns:
          :class:`numpy.ndarray`: The priors
        """
        return self._priors

    def get_prior(self, parameter):
        """ Get the prior for the given parameter.

        Args:
          parameter (string, optional): Name of a valid fit parameter
            for which to get the penalty term value

        Returns:
          float: The prior for the given parameter

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter in self._fit_config.get_pars():
            par_num = self._fit_config.get_index(parameter)
            return self._priors[par_num]
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_raw_stats(self):
        """ Gets the raw test statistics array.

        .. warning:: This has no penalty term contributions added.

        Returns:
          :class:`numpy.array`: The raw test statistics at the
            corresponding signal scales.
        """
        return self._stats

    def get_raw_stat(self, idx):
        """ Gets the raw test statistic(s) from array at idx.

        .. warning:: This has no penalty term contributions added.

        Args:
          idx (int): Index in the array.

        Returns:
          (float or :class:`numpy.ndarray`): The raw test statistic(s) at idx.
        """
        return self._stats[idx]

    def get_scales(self):
        """ Gets the signal scales array

        Returns:
          :class:`numpy.array`: The signal scales used in the limit setting.
        """
        return self._scales

    def get_scale(self, idx):
        """ Gets the signal scale from array at idx

        Args:
          idx (int): Index in the array.

        Returns:
          float: The signal scale at idx in the array.
        """
        return self._scales[idx]

    def get_sigmas(self):
        """ Get the sigmas array.

        Returns:
          float: The value of the systematic uncertainty of the fit parameter.
        """
        return self._sigmas

    def get_sigma(self, parameter):
        """ Get the sigma for the given parameter.

        Args:
          parameter (string, optional): Name of a valid fit parameter
            for which to get the penalty term value

        Returns:
          float: The sigma for the given parameter

        Raises:
          IndexError: If the parameter name supplied does not match
            any of those stored in the fit config.
        """
        if parameter in self._fit_config.get_pars():
            par_num = self._fit_config.get_index(parameter)
            return self._sigmas[par_num]
        else:  # unrecognised parameter
            raise IndexError("Unknown parameter %s" % parameter)

    def get_stats(self):
        """ Gets the total test statistics array.

        .. warning:: Penalty term contributions are added here.

        Returns:
          :class:`numpy.array`: The total test statistics at the corresponding
            signal scales.
        """
        total_penalties = numpy.sum(self._penalty_terms, axis=-1)
        total_stats = copy.copy(self._stats)
        for axis, parameter in self._spectra_config.get_pars():
            axis += 1  # first axis is always signal scale
            total_stats = numpy.sum(total_stats, axis=axis)
        return total_stats + total_penalties

    def get_stat(self, idx):
        """ Gets the total test statistic from array at idx

        .. warning:: Penalty term contributions are added here.

        Args:
          idx (int): Index in the array.

        Returns:
          float: The total stat at idx.
        """
        total_penalty = numpy.sum(self.get_penalty_term(idx))
        total_stat = copy.copy(self._stats[idx])
        for axis, parameter in self._spectra_config.get_pars():
            axis += 1  # first axis is always signal scale
            total_stat = numpy.sum(total_stat, axis=axis)
        return total_stat + total_penalty

    def nd_project_stats(self, *parameters):
        """ Projects the raw test statistic values (at each signal
        scale) onto the axes specified by the spectral parameters.

        This allows you to see the values of the test statistic per-bin
        in a given dimension.

        .. warning:: No penalty term contributions are included here.

        Args:
          parameters (string): Names of a valid spectal parameters onto
            which to project the test statistic values.
        """
        projection = copy.copy(self._stats)
        for axis, parameter in self._spectra_config.get_pars():
            axis += 1  # first axis is always signal scale
            if parameter not in parameters:
                projection = numpy.sum(projection, axis=axis)
        return projection

    def nd_project_stat(self, idx, *parameters):
        """ Projects the raw test statistic values at the given index
        onto the axes specified by the spectral parameters.

        This allows you to see the values of the test statistic per-bin
        in a given dimension.

        .. warning:: No penalty term contributions are included here.

        Args:
          idx (int): The index of the array.
          parameters (string): Names of a valid spectal parameters onto
            which to project the test statistic values.
        """
        projection = copy.copy(self._stats)[idx]
        for axis, parameter in self._spectra_config.get_pars():
            if parameter not in parameters:
                projection = numpy.sum(projection, axis=axis)
        return projection

    def set_best_fits(self, best_fits):
        """ Sets the array containing best fit values of the fit parameter.

        Args:
          best_fits (:class:`numpy.ndarray`): The best fits array.

        Raises:
          TypeError: If best_fits is not an :class:`numpy.ndarray`
          ValueError: If the best_fits array does not have the required
            shape.
        """
        if not isinstance(best_fits, numpy.ndarray):
            raise TypeError("best_fits must be a numpy array")
        if best_fits.shape != self._best_fits.shape:
            raise ValueError("best_fits array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(best_fits.shape),
                              str(self._best_fits.shape)))
        self._best_fits = best_fits

    def set_best_fit(self, best_fit, idx, parameter):
        """ Sets the best fit value of the fit parameter in array at idx

        Args:
          best_fit (float): Best fit value of a fit parameter.
          idx (int): Index in the array.
          parameter (string): Name of a valid fit parameter for which
            to set the best-fit value

        Raises:
          TypeError: If best_fit is not a float.
        """
        if not isinstance(best_fit, float):
            raise TypeError("best_fit must be a float")
        par_num = self._fit_config.get_index(parameter)
        self._best_fits[idx][par_num] = best_fit

    def set_name(self, name):
        """ Sets the name of the summary object

        Args:
          name (string): Name of the summary object

        Raises:
          TypeError: If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        self._name = name

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
        if penalty_terms.shape != self._penalty_terms.shape:
            raise ValueError("penalty_terms array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(penalty_terms.shape),
                              str(self._penalty_terms.shape)))
        self._penalty_terms = penalty_terms

    def set_penalty_term(self, penalty_term, idx, parameter):
        """ Sets the penalty term value of the fit parameter in array at idx

        Args:
          penalty_term (float): Best fit value of a fit parameter.
          idx (int): Index in the array.
          parameter (string): Name of a valid fit parameter for which
            to set the penalty term value

        Raises:
          TypeError: If penalty_term is not a float.
        """
        if not isinstance(penalty_term, float):
            raise TypeError("penalty_term must be a float")
        par_num = self._fit_config.get_index(parameter)
        self._penalty_terms[idx][par_num] = penalty_term

    def set_prior(self, prior, parameter):
        """ Set the prior value of fit parameter to store.

        Args:
          prior (float): Prior values for given fit parameter
          parameter (string): Name of a valid fit parameter for which
            to set the prior value

        Raises:
          TypeError: If prior is not a float.
        """
        if not isinstance(prior, float):
            raise TypeError("prior must be a float")
        par_num = self._fit_config.get_index(parameter)
        self._priors[par_num] = prior

    def set_priors(self, priors):
        """ Set the priors array.

        Args:
          priors (:class:`numpy.ndarray`): Array of prior values to set.

        Raises:
          TypeError: If priors is not a numpy array.
          ValueError: If the priors array does not have the required
            shape.
        """
        if not isinstance(priors, numpy.ndarray):
            raise TypeError("priors must be a numpy array")
        if priors.shape != self._priors.shape:
            raise ValueError("priors array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(priors.shape), str(self._priors.shape)))
        self._priors = priors

    def set_scales(self, scales):
        """ Sets the signal scales array

        Args:
          scales (:class:`numpy.ndarray`): The signal scales array.

        Raises:
          TypeError: If scales is not a :class:`numpy.ndarray`.
          ValueError: If length of scales array is not equal to the
            number of signal scales.
        """
        if not isinstance(scales, numpy.ndarray):
            raise TypeError("scales must be a numpy array")
        if scales.shape != self._scales.shape:
            raise ValueError("scales array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(scales.shape), str(self._scales.shape)))
        self._scales = scales

    def set_scale(self, scale, idx):
        """ Sets the signal scale in array at idx

        Args:
          scale (float): The signal scale.
          idx (int): Index in the array.

        Raises:
          TypeError: If scale is not a float.
        """
        if not isinstance(scale, float):
            raise TypeError("scale must be a float")
        self._scales[idx] = scale

    def set_sigma(self, sigma, parameter):
        """ Set the systematic uncertainty value of the fit parameter to store.

        Args:
          sigma (float): The systematic uncertainty value.

        Raises:
          TypeError: If sigma is not a float.
        """
        if not isinstance(sigma, float):
            raise TypeError("sigma must be a float")
        par_num = self._fit_config.get_index(parameter)
        self._sigmas[par_num] = sigma

    def set_sigmas(self, sigmas):
        """ Set the sigmas array.

        Args:
          sigmas (:class:`numpy.ndarray`): Array of sigma values to set.

        Raises:
          TypeError: If sigmas is not a numpy array.
          ValueError: If the sigmas array does not have the required
            shape.
        """
        if not isinstance(sigmas, numpy.ndarray):
            raise TypeError("sigmas must be a numpy array")
        if sigmas.shape != self._sigmas.shape:
            raise ValueError("sigmas array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(sigmas.shape), str(self._sigmas.shape)))
        self._sigmas = sigmas

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

    def set_stat(self, stat, idx):
        """ Sets the test statistic values in array at idx

        Args:
          stat (:class:`numpy.ndarray`): Values of the test statistic.
          idx (int): Index in the array.

        Raises:
          TypeError: If stat is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(stat, numpy.ndarray):
            raise TypeError("stat must be a numpy array")
        if stat.shape != self._stats[idx].shape:
            raise ValueError("stat array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stat.shape), str(self._stats[idx].shape)))
        self._stats[idx] = stat


class ReducedSummary(Summary):
    """ This class provides similar functionality to the :class:`Summary`
    class, but is slightly reduced by the fact that test-statistic
    values for the spectral dimensions (per-bin) are not stored.

    .. warning:: This class assumes that any test statistic values
      set in the 1-D stats array, already have the correct penalty term
      contributions added.

    Args:
      name (string): Name of the summary object.
      num_scales (int): Number of signal scales used in the limit setting.
      fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): Fit
        config used during limit setting.

    Attributes:
      _name (string): Name of the summary object.
      _num_scales (int): Number of signal scales used in the limit setting.
      _spectra_config (:class:`echidna.core.spectra.SpectraConfig`): Config
        for the signal spectrum.
      _fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): Fit
        config used during limit setting.
      _best_fits (:class:`numpy.ndarray`): The best fit values of the fit
        parameter for each corresponding signal scale.
      _penalty_terms (:class:`numpy.ndarray`): The penalty term values of the
        fit parameter for each corresponding signal scale.
      _scales (:class:`numpy.ndarray`): The signal scales used in the limit
        setting.
      _stats (:class:`numpy.ndarray`): The total test statistic values for
        each corresponding signal scale.
      _priors (float): The prior values of each fit parameter
      _sigma (float): The systematic uncertainty value of the fit parameter
    """
    def __init__(self, name, num_scales, fit_config):
        """ Initialises the summary data container
        """
        self._name = name
        self._num_scales = num_scales
        self._fit_config = fit_config
        pars_shape = (num_scales, len(self._fit_config.get_pars()))
        self._best_fits = numpy.zeros(shape=pars_shape, dtype=float)
        self._penalty_terms = numpy.zeros(shape=pars_shape, dtype=float)
        self._scales = numpy.zeros(shape=num_scales, dtype=float)
        stats_shape = tuple([num_scales])
        self._stats = numpy.zeros(shape=stats_shape, dtype=float)
        self._priors = numpy.zeros(shape=len(self._fit_config.get_pars()))
        self._sigmas = numpy.zeros(shape=len(self._fit_config.get_pars()))

    def get_stats(self):
        """ Gets the total test statistics array.

        .. warning:: Assumes penalty term contributions are already
          included.

        Returns:
          :class:`numpy.array`: The total test statistics at the corresponding
            signal scales.
        """
        return self._stats

    def get_stat(self, idx):
        """ Gets the total test statistic from array at idx

        .. warning:: Assumes penalty term contributions are already
          included.

        Args:
          idx (int): Index in the array.

        Returns:
          float: The total stat at idx.
        """
        return self._stats[idx]

    def nd_project_stats(self, *parameters):
        """ *** NOT VALID IN THIS CLASS ***
        Projects the raw test statistic values (at each signal
        scale) onto the axes specified by the spectral parameters.

        This allows you to see the values of the test statistic per-bin
        in a given dimension.

        .. warning:: No penalty term contributions are included here.

        Args:
          parameters (string): Names of a valid spectal parameters onto
            which to project the test statistic values.

        Raises:
          AttributeError: If used from this class
        """
        raise AttributeError("ReducedSummary class has no attribute "
                             "nd_project_stats")

    def nd_project_stat(self, idx, *parameters):
        """ *** NOT VALID IN THIS CLASS ***
        Projects the raw test statistic values at the given index
        onto the axes specified by the spectral parameters.

        This allows you to see the values of the test statistic per-bin
        in a given dimension.

        .. warning:: No penalty term contributions are included here.

        Args:
          idx (int): The index of the array.
          parameters (string): Names of a valid spectal parameters onto
            which to project the test statistic values.

        Raises:
          AttributeError: If used from this class
        """
        raise AttributeError("ReducedSummary class has no attribute "
                             "nd_project_stats")

    def set_stat(self, stat, idx):
        """ Sets the test statistic values in array at idx

        Args:
          stat (float): Value of the test statistic.
          idx (int): Index in the array.

        Raises:
          TypeError: If stat is not a float.
        """
        if not isinstance(stat, float):
            raise TypeError("stat must be a numpy array")
        self._stats[idx] = stat
