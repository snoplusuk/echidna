""" Echidna's parameter module.

Contains :class:`Parameter` and all classes that inherit from it.
"""
import numpy

import abc
import logging
import warnings

from echidna.core import scale, shift, smear


class Parameter(object):
    """ The base class for creating parameter classes.

    Args:
      type_name (string): The type of the parameter.
      name (str): The name of this parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values

    Attributes:
      _type (string): The type of the parameter.
      _name (str): The name of this parameter
      _low (float): The lower limit to float the parameter from
      _high (float): The higher limit to float the parameter from
      _bins (int): The number of steps between low and high values
      _min (float): Minimum bound for values - (-inf) by default
      _max (float): Maximum bound for values - (inf) by default

    """

    def __init__(self, type_name, name, low, high, bins):
        """ Initialise config class
        """
        self._type = type_name
        self._name = name
        self._low = float(low)
        self._high = float(high)
        self._bins = int(bins)
        self._min = -numpy.inf
        self._max = numpy.inf

    def get_bins(self):
        """ Get the number of bins.

        Returns:
          int: Number of bins for this parameter.
        """
        return self._bins

    def get_high(self):
        """ Get the high value of the parameter

        Returns:
          float: The high value of the parameter.
        """
        return self._high

    def get_low(self):
        """ Get the low value of the parameter.

        Returns:
          float: The low value the parameter.
        """
        return self._low

    def get_name(self):
        """ Get the name of the parameter.

        Returns:
          float: The name of the parameter.
        """
        return self._name

    def get_type(self):
        """ Get the type of the parameter.

        Returns:
          float: The type of the parameter.
        """
        return self._type

    def get_width(self):
        """Get the width of the binning for the parameter

        Returns:
          float: Bin width.
        """
        return (self._high - self._low) / float(self._bins)

    def to_dict(self, basic=False):
        """ Represent the properties of the parameter in a dictionary.

        .. note:: The attributes :attr:`_name`, :attr:`_type` are never
          included in the dictionary. This is because it is expected
          that the dictionary returned here will usually be used as
          part of a larger dictionary where type and/or parameter_name
          are keys.

        Returns:
          dict: Representation of the parameter in the form of a
            dictionary.
        """
        parameter_dict = {}
        parameter_dict["low"] = self._low
        parameter_dict["high"] = self._high
        parameter_dict["bins"] = self._bins
        return parameter_dict


class FitParameter(Parameter):
    """Simple data container that holds information for a fit parameter
    (i.e. a systematic to float).

    .. warning:: The sigma value can be explicitly set as None. This
      is so that you disable a penalty term for a floating parameter.
      If a parameter is being floated, but sigma is None, then no
      penalty term will be added for the parameter.

    .. note:: The :class:`FitParameter` class offers three different
      scales for constructing the array of values for the parameter.

    These are:

      * **linear**: A standard linear scale is the default option. This
        creates an array of equally spaced values, starting at
        :obj:`low` and ending at :obj:`high` (*includive*). The array
        will contain :obj:`bins` values.
      * **logscale**: This creates an array of values that are equally
        spaced in log-space, but increase exponentially in linear-space,
        starting at :obj:`low` and ending at :obj:`high` (*includive*).
        The array will contain :obj:`bins` values.
      * **logscale_deviation**: This creates an array of values -
        centred around the prior - whose absolute deviations from the
        prior are equally spaced in log-space, but increase
        exponentially in linear-space. The values start at :obj:`low`
        and end at :obj:`high` (*includive*). The array will contain
        :obj:`bins` values.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string, optional): The spectral dimension to which the
        fit parameter applies.
      values (:class:`numpy.array`, optional): Array of parameter
        values to test in fit.
      best_fit (float, optional): Best-fit value calculated by fit.
      penalty_term (float, optional): Penalty term value at best fit.
      logscale (bool, optional): Flag to create an logscale array of
        values, rather than a linear array.
      base (float, optional): Base to use when creating an logscale
        array. Default is base-e.
      logscale_deviation (bool, optional): Flag to create a logscale deviation
        array of values rather than a linear or logscale array.

    Attributes:
      _prior (float): The prior of the parameter
      _sigma (float): The sigma of the parameter
      _dimension (string): The spectral dimension to which the fit
        parameter applies.
      _values (:class:`numpy.array`): Array of parameter values to
        test in fit.
      _best_fit (float): Best-fit value calculated by fit.
      _penalty_term (float): Penalty term value at best fit.
      _logscale (bool): Flag to create an logscale array of values,
        rather than a linear array.
      _base (float): Base to use when creating an logscale array.
        Default is base-e
      _logscale_deviation (bool): Flag to create a logscale deviation
        array of values rather than a linear or logscale array.
      _bin_boundaries (:class:`numpy.array`): Array of bin boundaries
        corresponding to :attr:`_values`.

    """

    def __init__(self, name, prior, sigma, low, high, bins, dimension=None,
                 values=None, current_value=None, penalty_term=None,
                 best_fit=None, logscale=None, base=numpy.e,
                 logscale_deviation=None):
        """Initialise FitParameter class
        """
        super(FitParameter, self).__init__("fit", name, low, high, bins)
        self._logger = logging.getLogger("FitParameter")
        self._prior = float(prior)
        if sigma is None:
            self._logger.warning(
                "Setting sigma explicitly as None for %s - "
                "No penalty term will be added for this parameter!" % name)
        self._sigma = sigma
        self._dimension = dimension
        self._values = values
        self._current_value = current_value
        self._best_fit = best_fit
        self._penalty_term = penalty_term
        self._logscale = None
        self._base = None
        self._logscale_deviation = None
        self._bin_boundaries = None
        if logscale:
            self._logger.info("Setting logscale %s for parameter %s" %
                              (logscale, name))
            logging.getLogger("extra").info(" --> with base: %.4g" % base)
            if logscale_deviation is not None:
                self._logger.warning("Recieved logscale_deviation flag that "
                                     "will not have any effect")
            self._logscale = logscale
            self._base = base
        elif logscale_deviation:
            self._logger.info("Setting logscale_deviation %s for parameter %s"
                              % (logscale_deviation, name))
            self._logscale_deviation = logscale_deviation

    @abc.abstractmethod
    def apply_to(self, spectrum):
        """ Applies current value of fit parameter to spectrum.

        Args:
          spectrum (:class:`Spectra`): Spectrum to which current value
            of parameter should be applied.

        Returns:
          (:class:`Spectra`): Modified spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        pass

    def check_values(self):
        """ Check that no values/bin_boundaries fall outside the min
        and max bounds for the paramaeter type. Check that the prior is
        in the values.

        Warnings:
          UserWarning: If any values fall outside the boundaries for
            the parameter type - defined by :attr:`_min` and
            :attr:`_max`.
          UserWarning: If :attr:`_prior` is not present in the
            :attr:`_values` array.
        """
        values = self.get_values()

        # Check for bad values - fall outside of allowed range for parameter
        bad_indices = numpy.where((values < self._min) |
                                  (values > self._max))[0]
        if len(bad_indices) != 0:
            # Bad values present
            warnings.warn(
                "%d values fall outside bounds (%.4g, %.4g), for parameter %s"
                % (len(bad_indices), self._min, self._max, self._name))
            bad_values = numpy.take(values, bad_indices)
            self._logger.debug("The bad values are:")
            logging.getLogger("extra").debug("\n\n%s\n" % str(bad_values))

        # Check prior contained in values
        if not numpy.any(numpy.around(values / self._prior, 12) ==
                         numpy.around(1., 12)):
            warnings.warn("Prior not in values array "
                          "for parameter %s" % self._name)
            logging.getLogger("extra").warning(
                "\n\nUsers are strongly advised to include the value of the "
                "in all FitParameter value arrays as failure to do this "
                "could result in unusual Fit results.\n\n This can be "
                "achieved with an odd number of bins and low and high values "
                "symmetric about prior. Some scale types - logscale_deviation "
                "build the array of values around the prior, so it is "
                "included by definition.\n")
            log_text = "Values: %s\n" % str(values)
            log_text += "Prior: %.4g\n" % self._prior
            logging.getLogger("extra").debug("\n\n%s" % log_text)

    def get_best_fit(self):
        """
        Returns:
          float: Best fit value of parameter - stored in
            :attr:`_best_fit`.

        Raises:
          ValueError: If the value of :attr:`_best_fit` has not yet
            been set.
        """
        if self._best_fit is None:
            raise ValueError("Best fit value for parameter" +
                             self._name + " has not been set")
        return self._best_fit

    def get_bin_boundaries(self):
        """ ***PENDING DEPRECATION***

        Returns an array of bin boundaries, based on the :attr:`_low`,
        :attr:`_high` and :attr:`_bins` parameters, and any flags
        (:attr:`_logscale` or :attr:`_logscale_deviation`) that have
        been applied.

        Returns:
          (:class:`numpy.array`): Array of bin_baoundaries for the
            parameter values stored in :attr:`_values`.

        Warnings:
          PendingDeprecationWarning: This method will be deprecated
            soon. Bin boundaries shouldn't be required here as we are
            referring to points on a grid, not bins.
        """
        warnings.warn(PendingDeprecationWarning(
            "Bin boundaries shouldn't be required here "
            "as we are referring to points on a grid, not bins"))
        if self._bin_boundaries is None:  # Generate array of bin boundaries
            if self._logscale:
                if self._low <= 0.:  # set low = -log(high)
                    low = -numpy.log(self._high)
                    logging.warning("Correcting fit parameter value <= 0.0")
                    logging.debug(" --> changed to %.4g (previously %.4g)" %
                                  (numpy.exp(low), self._low))
                else:
                    low = numpy.log(self._low)
                high = numpy.log(self._high)
                width = (numpy.log(high) - numpy.log(low)) / int(self._bins)
                self._bin_boundaries = numpy.logspace(
                    low - 0.5*width, high + 0.5*width,
                    num=self._bins+1, base=numpy.e)
            elif self._logscale_deviation:
                delta = self._high - self._prior
                width = numpy.log(delta + 1.) / int(self._bins / 2)
                deltas = numpy.linspace(
                    0.5 * width, numpy.log(delta + 1.) + 0.5*width,
                    num=int((self._bins + 1) / 2))
                pos = self._prior + numpy.exp(deltas) - 1.
                neg = self._prior - numpy.exp(deltas[::-1]) + 1.
                self._bin_boundaries = numpy.append(neg, pos)
            else:
                width = self.get_width()
                self._bin_boundaries = numpy.linspace(self._low + 0.5*width,
                                                      self._high + 0.5*width,
                                                      self._bins + 1)
        return self._bin_boundaries

    def get_current_value(self):
        """
        Returns:
          float: Current value of fit parameter - stored in
            :attr:`_current_value`
        """
        if self._current_value is None:
            raise ValueError("Current value not yet set " +
                             "for parameter " + self._name)
        return self._current_value

    def get_dimension(self):
        """
        Returns:
          string: Dimension to which fit parameter is applied.
        """
        return self._dimension

    def get_penalty_term(self):
        """ Gets the value of the penalty term at the best fit.

        Returns:
          float: Penalty term value of parameter at best fit - stored in
            :attr:`_penalty_term`.

        Raises:
          ValueError: If the value of :attr:`_penalty_term` has not yet
            been set.
        """
        if self._penalty_term is None:
            raise ValueError("Penalty term value for parameter" +
                             self._name + " has not been set")
        return self._penalty_term

    def get_prior(self):
        """
        Returns:
          float: Prior value of fit parameter - stored in
            :attr:`_prior`
        """
        return self._prior

    def get_sigma(self):
        """
        Returns:
          float: Sigma of fit parameter - stored in :attr:`_sigma`
        """
        return self._sigma

    def get_values(self):
        """ Returns an array of values, based on the :attr:`_low`,
        :attr:`_high` and :attr:`_bins` parameters, and any flags
        (:attr:`_logscale` or :attr:`_logscale_deviation`) that have
        been applied.

        .. warning:: Calling this method with the
          :attr:`logscale_deviation` flag enabled, may alter the value
          of :attr:`_low`, as this scale must be symmetric about the
          prior.

        Returns:
          (:class:`numpy.array`): Array of parameter values to test in
            fit. Stored in :attr:`_values`.
        """
        if self._values is None:  # Generate array of values
            if self._logscale:
                # Create an array that is equally spaced in log-space
                self._logger.info("Creating logscale array of values "
                                  "for parameter %s" % self._name)
                if self._low <= 0.:  # set low = -log(high)
                    low = -numpy.log(self._high)
                    logging.warning("Correcting fit parameter value <= 0.0")
                    logging.debug(" --> changed to %.4g (previously %.4g)" %
                                  (numpy.exp(low), self._low))
                else:
                    low = numpy.log(self._low)
                high = numpy.log(self._high)
                self._values = numpy.logspace(low, high, num=self._bins,
                                              base=numpy.e)
            elif self._logscale_deviation:
                # Create an array where absolute deviations from the prior
                # increase linearly in logspace. The array is therefore
                # approximately symmetrical about the prior, but the positive
                # and negative deviations are treated separatedly, so can have
                # an extended range on either side.
                self._logger.info("Creating logscale_deviation array of "
                                  "values for parameter %s" % self._name)

                # Calculate maximum deviation above and below prior
                delta_low = numpy.absolute(self._low - self._prior)
                delta_high = numpy.absolute(self._high - self._prior)

                # Calculate bins above and below, distributing evenly
                bins_low = numpy.rint(
                    (delta_low) / (delta_low + delta_high) * (self._bins + 1))
                bins_high = numpy.rint(
                    (delta_high) / (delta_low + delta_high) * (self._bins + 1))

                # Calculate arrays of deviation, linear in logspace
                deltas_low = numpy.linspace(
                    0., numpy.log(delta_low + 1.), bins_low)
                deltas_high = numpy.linspace(
                    0., numpy.log(delta_high + 1.), bins_high)

                # Create positive and negative arrays of values
                # [::-1] reverses array
                # Prior is included in low
                low = self._prior - numpy.exp(deltas_low[::-1]) + 1.
                # Prior not included in high
                high = self._prior + numpy.exp(deltas_high[1:]) - 1.
                self._values = numpy.append(low, high)
            else:  # Create a normal linear array
                self._logger.info("Creating linear array of values "
                                  "for parameter %s" % self._name)
                self._values = numpy.linspace(
                    self._low, self._high, self._bins)

        return self._values

    def get_value_at(self, index):
        """ Access the parameter value at a given index in the array.

        Args:
          index (int): Index of parameter value requested.

        Returns:
          float: Parameter value at the given index.
        """
        return self.get_values()[index]

    def get_value_index(self, value):
        """ Get the index corresponding to a given parameter value.

        Args:
          value (float): Parameter value for which to get corresponding
            index.

        Returns:
          int: Index of corresponding to the given parameter value.

        .. warning:: If there are multiple occurences of ``value`` in
          the array of parameter values, only the index of the first
          occurence will be returned.
        """
        indices = numpy.where(self.get_values() == value)[0]
        if len(indices) == 0:
            raise ValueError("No value %.2g found in parameter values " +
                             "for parameter %s." % (value, self._name))
        return int(indices[0])

    def set_best_fit(self, best_fit):
        """ Set value for :attr:`_best_fit`.

        Args:
          best_fit (float): Best fit value for parameter
        """
        self._best_fit = best_fit

    def set_current_value(self, value):
        """ Set value for :attr:`_current_value`.

        Args:
          value (float): Current value of fit parameter
        """
        self._current_value = value

    def set_par(self, **kwargs):
        """Set a fitting parameter's values after initialisation.

        Args:
          kwargs (dict): keyword arguments

        .. note::

          Keyword arguments include:

            * prior (float): Value to set the prior to of the parameter
            * sigma (float): Value to set the sigma to of the parameter
            * low (float): Value to set the lower limit to of the parameter
            * high (float): Value to set the higher limit to of the parameter
            * bins (float): Value to set the size of the bins between low and
              high of the parameter
            * logscale (bool): Flag to create an logscale array of
              values, rather than a linear array.
            * base (float): Base to use when creating an logscale array.

        Raises:
          TypeError: Unknown variable type passed as a kwarg.
        """
        for kw in kwargs:
            self._logger.warning("Updating value for %s (%.4g --> %.4g)" %
                                 (kw, self.__dict__["_"+kw], kwargs[kw]))
            if kw == "prior":
                self._prior = float(kwargs[kw])
            elif kw == "sigma":
                if kwargs[kw] is None:
                    self._logger.warning("Setting sigma explicitly as None - "
                                         "No penalty term will be applied")
                self._sigma = kwargs[kw]
            elif kw == "low":
                self._low = float(kwargs[kw])
            elif kw == "high":
                self._high = float(kwargs[kw])
            elif kw == "bins":
                self._bins = float(kwargs[kw])
            elif kw == "logscale":
                self._logscale = bool(kwargs[kw])
            elif kw == "base":
                self._base = float(kwargs[kw])
            elif kw == "logscale_deviation":
                self._logscale_deviation = bool(kwargs[kw])
            elif kw == "dimension":
                self._dimension = str(kwargs[kw])
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)
        self._logger.warning("Setting _values and _bin_boundaries to None")
        self._values = None
        self._bin_boundaries = None

    def set_penalty_term(self, penalty_term):
        """ Set value for :attr:`_penalty_term`.

        Args:
          penalty_term (float): Value for penalty term of parameter at
            best fit.
        """
        self._penalty_term = penalty_term

    def to_dict(self, basic=False):
        """ Represent the properties of the parameter in a dictionary.

        Args:
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.

        .. note:: The attributes :attr:`_name`, :attr:`_dimension`,
          :attr:`_values` and :attr:`_logger` are never included in
          the dictionary. For the first two this is because it is
          expected that the dictionary returned here will usually be
          used as part of a larger dictionary where dimension and
          parameter_name are keys. The :attr:`values` attribute is not
          included because this is a lrge numpy array. The logger is
          not included as this is for internal use only.

        Returns:
          dict: Representation of the parameter in the form of a
            dictionary.
        """
        parameter_dict = {}
        # Add basic attributes
        parameter_dict["prior"] = self._prior
        parameter_dict["sigma"] = self._sigma
        parameter_dict["low"] = self._low
        parameter_dict["high"] = self._high
        parameter_dict["bins"] = self._bins
        parameter_dict["logscale"] = self._logscale
        parameter_dict["base"] = self._base
        parameter_dict["logscale_deviation"] = self._logscale_deviation
        if basic:
            return parameter_dict
        # Add non-basic attributes
        parameter_dict["current_value"] = self._current_value
        parameter_dict["best_fit"] = self._best_fit
        parameter_dict["penalty_term"] = self._best_fit
        return parameter_dict


class RateParameter(FitParameter):
    """ Data container that holds information for a rate parameter that
    is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      logscale (bool, optional): Flag to create an logscale array of
        values, rather than a linear array.
      base (float, optional): Base to use when creating an logscale array.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`

    Attributes:
      _logscale (bool): Flag to create an logscale array of values,
        rather than a linear array.
      _base (float): Base to use when creating an logscale array.
    """
    def __init__(self, name, prior, sigma, low, high,
                 bins, logscale=None, base=numpy.e,
                 logscale_deviation=None, **kwargs):
        super(RateParameter, self).__init__(
            name, prior, sigma, low, high, bins, logscale=logscale,
            base=base, logscale_deviation=logscale_deviation, **kwargs)
        self._min = 0.  # For rates

    def apply_to(self, spectrum):
        """ Scales spectrum to current value of rate parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum which should be scaled
            to current rate value.

        Returns:
          (:class:`Spectra`): Scaled spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        spectrum.scale(self._current_value)
        return spectrum


class ResolutionParameter(FitParameter):
    """ Data container that holds information for a resulution parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the
        resolution parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
    """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ResolutionParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Smears spectrum to current value of resolution.

        Args:
          spectrum (:class:`Spectra`): Spectrum which should be smeared.

        Returns:
          (:class:`Spectra`): Smeared spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        cur_value = self._current_value
        if cur_value > 10.:
            smearer = smear.EnergySmearLY()
        else:
            smearer = smear.EnergySmearRes()
        smearer.set_resolution(self._current_value)
        spectrum = smearer.weighted_smear(spectrum, self._dimension)
        return spectrum

    def get_pre_convolved(self, directory, filename, added_dim=False):
        """ Constructs the filename and directory from which a pre_convolved
          spectrum can be loaded from.

        .. note:: Before any calls to this function, the directory
          and filename is should be of the form::

              ../dimension/syst/file_XXyy.hdf5

          where dimension is the dimension you are applying systematics to e.g.
          `energy_mc` and syst is the type of systematic e.g. `smear`.
          XX and yy represent the syst value e.g. 200 and syst denoted type
          e.g. ly for 200 NHits/MeV light yield.
          For multiple dimensions and systematics then it is of the form::

              ../dimension/syst1/syst2/dimension/syst3/file_AAbb_CCdd_EEff.hdf5

          where the order of directories and filename is the order in which
          the systematics have been applied.

        Args:
          directory (string): Current or base directory containing
            pre-convolved :class:`Spectra` object
          filename (string): Current or base name of :class:`Spectra`
            object
          added_dim (bool, optional): If a dimension has just been added to the
            directory then this flag is True.

        Returns:
          string: Directory containing pre-convolved :class:`Spectra`,
            appended with name of this :class:`FitParameter`
          string: Name of pre-convolved :class:`Spectra`, appended with
            current value of this :class:`FitParameter`

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of fit parameter %s "
                             "has not been set" % self._name)

        if directory[-1] != '/':
            directory += '/'
        if not added_dim:
            dir_list = directory.split('/')
            # Find last occurance of dimension and search from here
            idx = dir_list[::-1].index(self._dimension)
            if 'smear' not in dir_list[-idx:]:
                directory += 'smear/'
        else:
            directory += 'smear/'
        if self._current_value > 10.:
            ext = 'ly'
        else:
            ext = 'rs'
        value_string = str(self._current_value)
        # Strip trailling zero in filename
        value_string = value_string.rstrip('0').rstrip('.')
        filename_list = filename.split('_')
        temp_fname = ''
        subbed = False
        for x in filename_list:
            if x[-2:] == ext:
                x = value_string + ext
                subbed = True
            elif x[-7:] == ext + ".hdf5":
                x = value_string + ext + ".hdf5"
                subbed = True
            if x[-5:] == ".hdf5":
                temp_fname += x
            else:
                temp_fname += x + "_"
        if not subbed:
            temp_fname = temp_fname[:-5] + "_" + value_string + ext + ".hdf5"
        return directory, temp_fname


class ScaleParameter(FitParameter):
    """ Data container that holds information for a scale parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the scale
        parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
    """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ScaleParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Convolves spectrum with current value of scale parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum to be convolved.

        Returns:
          (:class:`Spectra`): Convolved spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of scale parameter %s "
                             "has not been set" % self._name)
        scaler = scale.Scale()
        scaler.set_scale_factor(self._current_value)
        return scaler.scale(spectrum, self._dimension)

    def get_pre_convolved(self, directory, filename, added_dim=False):
        """ Constructs the filename and directory from which a pre_convolved
          spectrum can be loaded from.

        .. note:: Before any calls to this function, the directory
          and filename is should be of the form::

              ../dimension/syst/file_XXyy.hdf5

          where dimension is the dimension you are applying systematics to e.g.
          `energy_mc` and syst is the type of systematic e.g. `smear`.
          XX and yy represent the syst value e.g. 200 and syst denoted type
          e.g. ly for 200 NHits/MeV light yield.
          For multiple dimensions and systematics then it is of the form::

              ../dimension/syst1/syst2/dimension/syst3/file_AAbb_CCdd_EEff.hdf5

          where the order of directories and filename is the order in which
          the systematics have been applied.

        Args:
          directory (string): Current or base directory containing
            pre-convolved :class:`Spectra` object
          filename (string): Current or base name of :class:`Spectra`
            object
          added_dim (bool, optional): If a dimension has just been added to the
            directory then this flag is True.

        Returns:
          string: Directory containing pre-convolved :class:`Spectra`,
            appended with name of this :class:`FitParameter`
          string: Name of pre-convolved :class:`Spectra`, appended with
            current value of this :class:`FitParameter`

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of fit parameter %s "
                             "has not been set" % self._name)

        if directory[-1] != '/':
            directory += '/'
        if not added_dim:
            dir_list = directory.split('/')
            # Find last occurance of dimension and search from here
            idx = dir_list[::-1].index(self._dimension)
            if 'scale' not in dir_list[-idx:]:
                directory += 'scale/'
        else:
            directory += 'scale/'
        value_string = str(self._current_value)
        # Strip trailling zero in filename
        value_string = value_string.rstrip('0').rstrip('.')
        filename_list = filename.split('_')
        temp_fname = ''
        subbed = False
        for x in filename_list:
            if x[-2:] == "sc":
                x = value_string + "sc"
                subbed = True
            elif x[-7:] == "sc.hdf5":
                x = value_string + "sc.hdf5"
                subbed = True
            if x[-5:] == ".hdf5":
                temp_fname += x
            else:
                temp_fname += x + "_"
        if not subbed:
            temp_fname = temp_fname[:-5] + "_" + value_string + "sc.hdf5"
        return directory, temp_fname


class ShiftParameter(FitParameter):
    """ Data container that holds information for a shift parameter
    that is included in the fit.

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values
      dimension (string): The spectral dimension to which the shift
        parameter applies.
      kwargs (dict): Other keyword arguments to pass to
        :class:`FitParameter`
   """

    def __init__(self, name, prior, sigma, low,
                 high, bins, dimension, **kwargs):
        super(ShiftParameter, self).__init__(
            name, prior, sigma, low, high, bins, dimension, **kwargs)

    def apply_to(self, spectrum):
        """ Convolves spectrum with current value of shift parameter.

        Args:
          spectrum (:class:`Spectra`): Spectrum to be convolved.

        Returns:
          (:class:`Spectra`): Convolved spectrum.

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of shift parameter %s "
                             "has not been set" % self._name)
        shifter = shift.Shift()
        shifter.set_shift(self._current_value)
        return shifter.shift(spectrum, self._dimension)

    def get_pre_convolved(self, directory, filename, added_dim=False):
        """ Constructs the filename and directory from which a pre_convolved
          spectrum can be loaded from.

        .. note:: Before any calls to this function, the directory
          and filename is should be of the form::

              ../dimension/syst/file_XXyy.hdf5

          where dimension is the dimension you are applying systematics to e.g.
          `energy_mc` and syst is the type of systematic e.g. `smear`.
          XX and yy represent the syst value e.g. 200 and syst denoted type
          e.g. ly for 200 NHits/MeV light yield.
          For multiple dimensions and systematics then it is of the form::

              ../dimension/syst1/syst2/dimension/syst3/file_AAbb_CCdd_EEff.hdf5

          where the order of directories and filename is the order in which
          the systematics have been applied.

        Args:
          directory (string): Current or base directory containing
            pre-convolved :class:`Spectra` object
          filename (string): Current or base name of :class:`Spectra`
            object
          added_dim (bool, optional): If a dimension has just been added to the
            directory then this flag is True.

        Returns:
          string: Directory containing pre-convolved :class:`Spectra`,
            appended with name of this :class:`FitParameter`
          string: Name of pre-convolved :class:`Spectra`, appended with
            current value of this :class:`FitParameter`

        Raises:
          ValueError: If :attr:`_current_value` is not set.
        """
        if self._current_value is None:
            raise ValueError("Current value of fit parameter %s "
                             "has not been set" % self._name)

        if directory[-1] != '/':
            directory += '/'
        if not added_dim:
            dir_list = directory.split('/')
            # Find last occurance of dimension and search from here
            idx = dir_list[::-1].index(self._dimension)
            if 'shift' not in dir_list[-idx:]:
                directory += 'shift/'
        else:
            directory += 'shift/'
        value_string = str(self._current_value)
        # Strip trailling zero in filename
        value_string = value_string.rstrip('0').rstrip('.')
        filename_list = filename.split('_')
        temp_fname = ''
        subbed = False
        for x in filename_list:
            if x[-2:] == "sh":
                x = value_string + "sh"
                subbed = True
            elif x[-7:] == "sh.hdf5":
                x = value_string + "sh.hdf5"
                subbed = True
            if x[-5:] == ".hdf5":
                temp_fname += x
            else:
                temp_fname += x + "_"
        if not subbed:
            temp_fname = temp_fname[:-5] + "_" + value_string + "sh.hdf5"
        return directory, temp_fname


class SpectraParameter(Parameter):
    """Simple data container that holds information for a Spectra parameter
    (i.e. axis of the spectrum).

    Args:
      name (str): The name of this parameter
      low (float): The lower limit of this parameter
      high (float): The upper limit of this parameter
      bins (int): The number of bins for this parameter
    """

    def __init__(self, name, low, high, bins):
        """Initialise SpectraParameter class
        """
        super(SpectraParameter, self).__init__("spectra", name, low, high,
                                               bins)

    def get_bin(self, x):
        """ Gets the bin index which contains value x.

        Args:
          x (float): Value you wish to find the bin index for.

        Raises:
          ValueError: If x is less than parameter lower bounds
          ValueError: If x is more than parameter upper bounds

        Returns:
          int: Bin index
        """
        if x < self._low:
            raise ValueError("%s is below parameter lower bound %s"
                             % (x, self._low))
        if x > self._high:
            raise ValueError("%s is above parameter upper bound %s"
                             % (x, self._high))
        return int((x - self._low) / self.get_width())

    def get_bin_boundaries(self):
        """ Returns the bin boundaries for the parameter

        Returns:
          :class:`numpy.ndarray`: Bin boundaries for the parameter.
        """
        return numpy.linspace(self._low, self._high, self._bins+1)

    def get_bin_centre(self, bin):
        """ Calculates the central value of a given bin

        Args:
          bin (int): Bin number.

        Raises:
          TypeError: If bin is not int
          ValueError: If bin is less than zero
          ValueError: If bin is greater than the number of bins - 1

        Returns:
          float: value of bin centre
        """
        if type(bin) != int and type(bin) != numpy.int64:
            raise TypeError("Must pass an integer value")
        if bin < 0:
            raise ValueError("Bin number (%s) must be zero or positive" % bin)
        if bin > self._bins - 1:
            raise ValueError("Bin number (%s) is out of range. Max = %s"
                             % (bin, self._bins))
        return self._low + (bin + 0.5)*self.get_width()

    def get_bin_centres(self):
        """ Returns the bin centres of the parameter

        Returns:
          :class:`numpy.ndarray`: Bin centres of parameter.
        """
        return numpy.arange(self._low+self.get_width()*0.5,
                            self._high,
                            self.get_width())

    def get_unit(self):
        """Get the default unit for a given parameter

        Raises:
          Exception: Unknown parameter.

        Returns:
          string: Unit of the parameter
        """
        if self._name.split('_')[0] == "energy":
            return "MeV"
        if self._name.split('_')[0] == "radial":
            return "mm"

    def round(self, x):
        """ Round the value to nearest bin edge

        Args:
          x (float): Value to round.

        Returns:
          float: The value of the closest bin edge to x
        """
        return round(x/self.get_width())*self.get_width()

    def set_par(self, **kwargs):
        """Set a limit / binning parameter after initialisation.

        Args:
          kwargs (dict): keyword arguments

        .. note::

          Keyword arguments include:

            * low (float): Value to set the lower limit to of the parameter
            * high (float): Value to set the higher limit to of the parameter
            * bins (int): Value to set the number of bins of the parameter

        Raises:
          TypeError: Unknown variable type passed as a kwarg.
        """
        for kw in kwargs:
            if kw == "low":
                self._low = float(kwargs[kw])
            elif kw == "high":
                self._high = float(kwargs[kw])
            elif kw == "bins":
                self._bins = int(kwargs[kw])
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)
