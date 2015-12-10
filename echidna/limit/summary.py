import numpy


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

    Attributes:
      _name (string): Name of the summary object.
      _num_scales (int): Number of signal scales used in the limit setting.
      _best_fits (:class:`numpy.ndarray`): The best fit values of the fit
        parameter for each corresponding signal scale.
      _penalty_terms (:class:`numpy.ndarray`): The penalty term values of the
        fit parameter for each corresponding signal scale.
      _scales (:class:`numpy.ndarray`): The signal scales used in the limit
        setting.
      _stats (:class:`numpy.ndarray`): The total test statistic values for
        each corresponding signal scale.
      _prior (float): The prior value of the fit parameter
      _sigma (float): The systematic uncertainty value of the fit parameter
    """
    def __init__(self, name, num_scales):
        """ Initialises the summary data container
        """
        self._name = name
        self._num_scales = num_scales
        self._best_fits = numpy.zeros(shape=num_scales, dtype=float)
        self._penalty_terms = numpy.zeros(shape=num_scales, dtype=float)
        self._scales = numpy.zeros(shape=num_scales, dtype=float)
        self._stats = numpy.zeros(shape=num_scales, dtype=float)
        self._prior = None
        self._sigma = None

    def get_best_fits(self):
        """ Gets the best_fits array

        Returns:
          :class:`numpy.ndarray`: The best fits at each signal scaling for the
            fit parameter.
        """
        return self._best_fits

    def get_best_fit(self, idx):
        """ Gets the best_fit from array at idx

        Args:
          idx (int): Index in the array.

        Returns:
          float: The best fit value of the fit parameter at idx in the array.
        """
        return self._best_fits[idx]

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

    def get_penalty_terms(self):
        """ Gets the penalty_terms array

        Returns:
          :class:`numpy.array`: The values of the penalty terms of the fit
            parameter at each signal scale.
        """
        return self._penalty_terms

    def get_penalty_term(self, idx):
        """ Gets the penalty_term from array at idx

        Args:
          idx (int): Index in the array.

        Returns:
          float: The value of the penalty term of the fit parameter at idx in
            the array.
        """
        return self._penalty_terms[idx]

    def get_prior(self):
        """ Get the prior value.

        Returns:
          float: The prior
        """
        return self._prior

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

    def get_sigma(self):
        """ Get the sigma value.

        Returns:
          float: The value of the systematic uncertainty of the fit parameter.
        """
        return self._sigma

    def get_stats(self):
        """ Gets the total test statistics array

        Returns:
          :class:`numpy.array`: The total test statistics at the corresponding
            signal scales.
        """
        return self._stats

    def get_stat(self, idx):
        """ Gets the total test statistic from array at idx

        Args:
          idx (int): Index in the array.

        Returns:
          float: The total stat at idx.
        """
        return self._stats[idx]

    def set_best_fits(self, best_fits):
        """ Sets the array containing best fit values of the fit parameter.

        Args:
          best_fits (:class:`numpy.ndarray`): The best fits array.

        Raises:
          TypeError: If best_fits is not an :class:`numpy.ndarray`
          ValueError: If the length of the best_fits array is not equal to the
          number of signal scales.
        """
        if isinstance(best_fits, numpy.ndarray):
            if len(best_fits) == self._num_scales:
                self._best_fits = best_fits
            else:
                raise ValueError("Length of best_fits (%s) array is not equal"
                                 "to the number of scales (%s)"
                                 % (len(best_fits), self._num_scales))
        else:
            raise TypeError("best_fits must be a numpy array")

    def set_best_fit(self, best_fit, idx):
        """ Sets the best fit value of the fit parameter in array at idx

        Args:
          best_fit (float): Best fit value of the fit parameter.
          idx (int): Index in the array.

        Raises:
          TypeError: If best_fit is not a float.
        """
        if isinstance(best_fit, float):
            self._best_fits[idx] = best_fit
        else:
            raise TypeError("best_fit must be a float")

    def set_name(self, name):
        """ Sets the name of the summary object

        Args:
          name (string): Name of the summary object

        Raises:
          TypeError: If name is not a string.
        """
        if isinstance(name, str):
            self._name = name
        else:
            raise TypeError("Name must be a string")

    def set_penalty_terms(self, penalty_terms):
        """ Sets the array containing the penalty term values of the fit
          parameter.

        Args:
          penalty_terms (:class:`numpy.ndarray`): The penalty terms array.

        Raises:
          TypeError: If penalty_terms is not a :class:`numpy.ndarray`.
          ValueError: If length of penalty_terms array is not equal to the
            number of signal scales.
        """
        if isinstance(penalty_terms, numpy.ndarray):
            if len(penalty_terms) == self._num_scales:
                self._penalty_terms = penalty_terms
            else:
                raise ValueError("Length of penalty_terms (%s) array is not "
                                 "equal to the number of scales (%s)"
                                 % (len(penalty_terms), self._num_scales))
        else:
            raise TypeError("penalty_terms must be a numpy array")

    def set_penalty_term(self, penalty_term, idx):
        """ Sets the penalty term value of the fit parameter in array at idx

        Args:
          penalty_term (float): The value of the penalty term.
          idx (int): Index in the array.

        Raises:
          TypeError: If penalty_term is not a float.
        """
        if isinstance(penalty_term, float):
            self._penalty_terms[idx] = penalty_term
        else:
            raise TypeError("penalty_term must be a float")

    def set_prior(self, prior):
        """ Set the prior value of fit parameter to store.

        Raises:
          TypeError: If prior is not a float.
        """
        if isinstance(prior, float):
            self._prior = prior
        else:
            raise TypeError("prior must be a float")

    def set_scales(self, scales):
        """ Sets the signal scales array

        Args:
          scales (:class:`numpy.ndarray`): The signal scales array.

        Raises:
          TypeError: If scales is not a :class:`numpy.ndarray`.
          ValueError: If length of scales array is not equal to the
            number of signal scales.
        """
        if isinstance(scales, numpy.ndarray):
            if len(scales) == self._num_scales:
                self._scales = scales
            else:
                raise ValueError("Length of scales (%s) array is not equal to"
                                 " the (initialised) number of scales (%s)"
                                 % (len(scales), self._num_scales))
        else:
            raise TypeError("scales must be a numpy array")

    def set_scale(self, scale, idx):
        """ Sets the signal scale in array at idx

        Args:
          scale (float): The signal scale.
          idx (int): Index in the array.

        Raises:
          TypeError: If scale is not a float.
        """
        if isinstance(scale, float):
            self._scales[idx] = scale
        else:
            raise TypeError("scale must be a float")

    def set_sigma(self, sigma):
        """ Set the systematic uncertainty value of the fit parameter to store.

        Args:
          sigma (float): The systematic uncertainty value.

        Raises:
          TypeError: If sigma is not a float.
        """
        if isinstance(sigma, float):
            self._sigma = sigma
        else:
            raise TypeError("sigma must be a float")

    def set_stats(self, stats):
        """ Sets the total test statistics array.

        Args:
          stats (:class:`numpy.ndarray`): The total test statistics array.

        Raises:
          TypeError: If stats is not a :class:`numpy.ndarray`.
          ValueError: If length of stats array is not equal to the number of
            signal scales.
        """
        if isinstance(stats, numpy.ndarray):
            if len(stats) == self._num_scales:
                self._stats = stats
            else:
                raise ValueError("Length of stats (%s) array is not equal to"
                                 " the number of scales (%s)"
                                 % (len(stats), self._num_scales))
        else:
            raise TypeError("stats must be a numpy array")
        return self._stats

    def set_stat(self, stat, idx):
        """ Sets the total test statistic in array at idx

        Args:
          stat (float): Value of the total test statistic.
          idx (int): Index in the array.

        Raises:
          TypeError: If stat is not a float.
        """
        if isinstance(stat, float):
            self._stats[idx] = stat
        else:
            raise TypeError("stat must be a float")
