import numpy
from scipy import interpolate

import collections
import yaml
import copy
import abc
import logging


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
    """

    def __init__(self, type_name, name, low, high, bins):
        """ Initialise config class
        """
        self._type = type_name
        self._name = name
        self._low = float(low)
        self._high = float(high)
        self._bins = int(bins)

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


class FitParameter(Parameter):
    """Simple data container that holds information for a fit parameter
    (i.e. a systematic to float).

    Args:
      name (str): The name of this parameter
      prior (float): The prior of the parameter
      sigma (float): The sigma of the parameter
      low (float): The lower limit to float the parameter from
      high (float): The higher limit to float the parameter from
      bins (int): The number of steps between low and high values

    Attributes:
      _prior (float): The prior of the parameter
      _sigma (float): The sigma of the parameter
      _values (:class:`numpy.array`): Array of parameter values to
        test in fit.
      _best_fit (float): Best-fit value calculated by fit.
      _penalty_term (float): Penalty term value at best fit.
      _spectra_specific (bool): Flag to show parameter applies to only
        a specific :class:`Spectra` instance.
    """

    def __init__(self, name, prior, sigma, low, high, bins):
        """Initialise FitParameter class
        """
        super(FitParameter, self).__init__("fit", name, low, high, bins)
        self._prior = float(prior)
        self._sigma = float(sigma)
        self._values = None  # Initially
        self._current_value = None  # Initially
        self._best_fit = None  # Initially
        self._penalty_term = None  # Initially
        self._spectra_specific = False
        self._logger = logging.getLogger("fit_parameter")
        self._logscale = None

    def check_values(self):
        """ For symmetric arrays, check that the prior is in the values.

        Raises:
          ValueError: If prior is not in the values array.
        """
        values = self.get_values()
        if not self._logscale:
            indices = numpy.where(values == self._prior)[0]
            if len(indices) == 0:
                log_text = ""
                log_text += "Values: %s\n" % str(values)
                log_text += "Prior: %.4g\n" % self._prior
                self._logger.debug("\n%s" % log_text)
                raise ValueError("Prior not in values array. "
                                 "This can be achieved with an odd number "
                                 "of bins and symmetric low and high values "
                                 "about the prior.")

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
            if kw == "prior":
                self._prior = float(kwargs[kw])
            elif kw == "sigma":
                self._sigma = float(kwargs[kw])
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
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)
        self._values = None

    def set_current_value(self, value):
        """ Set value for :attr:`_current_value`.

        Args:
          value (float): Current value of fit parameter
        """
        if self._current_value is None:
            self._logger.debug("Changing current value of %s, "
                               "from None to %.4g" % (str(self), value))
        else:
            self._logger.debug("Changing current value of %s, "
                               "from %.4g to %.4g" % (str(self),
                                                      self._current_value,
                                                      value))
        self._current_value = value

    def set_best_fit(self, best_fit):
        """ Set value for :attr:`_best_fit`.

        Args:
          best_fit (float): Best fit value for parameter
        """
        self._best_fit = best_fit

    def set_penalty_term(self, penalty_term):
        """ Set value for :attr:`_penalty_term`.

        Args:
          penalty_term (float): Value for penalty term of parameter at
            best fit.
        """
        self._penalty_term = penalty_term

    def get_values(self):
        """
        Returns:
          (:class:`numpy.array`): Array of parameter values to test in
            fit. Stored in :attr:`_values`.
        """
        if self._values is None:  # Generate array of values
            if self._logscale:  # Create a linear array in log-space
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
            else:  # Create a normal linear array
                self._values = numpy.linspace(self._low,
                                              self._high, self._bins)
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

    def get_prior(self):
        """
        Returns:
          float: Prior value of fit parameter - stored in
            :attr:`_prior`
        """
        if self._prior is None:
            raise ValueError("Prior value not yet set " +
                             "for parameter " + self._name)
        return self._prior

    def get_sigma(self):
        """
        Returns:
          float: Sigma of fit parameter - stored in :attr:`_sigma`
        """
        if self._sigma is None:
            raise ValueError("Sigma not yet set for parameter " + self._name)
        return self._sigma

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

    def get_pre_convolved(self, directory, filename):
        """ Appends the name and current value of a the :class:`FitParameter`

        .. note:: Before any calls to this function, the base directory
          should be of the form::

              ../hyphen-separated-dimensions/spectrum_name/

          and a base filename of the form ``spectrum_name``.

        .. note:: Each call to this method, then appends the name of
          the :class:`FitParamter` to the ``directory`` and its current
          value to the ``filename``. So for three :class:`FitParameters``,
          after three calls to this method, the directory should be e.g.::

              ../energy_mc-radial3_mc/Te130_0n2b/syst1/syst2/syst3/

          and the filename might be::

              Te130_0n2b_250.0_0.012_1.07

        .. note:: To construct the full path to pass to
          :funct:`echidna.output.store.load`, the ``directory`` and
          ``filename`` returned by the last call to this method,
          should be added together, and appended with ``".hdf5"``.

              path = director + filename + ".hdf5"

        Args:
          directory (string): Current or base directory containing
            pre-convolved :class:`Spectra` object
          name (string): Current or base name of :class:`Spectra`
            object

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
        directory += "_%s/" % self._name
        value_string = "%f" % self._current_value
        # Strip leading/trailling zeros in filename
        filename += ("_%s" % value_string.strip("0"))
        return directory, filename

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
        if self._current_value is None:
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        pass


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

    Attributes:
      _logscale (bool): Flag to create an logscale array of values,
        rather than a linear array.
      _base (float): Base to use when creating an logscale array.
    """
    def __init__(self, name, prior, sigma, low, high,
                 bins, logscale=False, base=numpy.e):
        super(RateParameter, self).__init__(name, prior, sigma,
                                            low, high, bins)
        self._logscale = logscale
        self._base = base

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
            self._logger.debug("Applying current value (None) to %s" %
                               str(self))
            raise ValueError("Current value of rate parameter %s "
                             "has not been set" % self._name)
        else:
            self._logger.debug("Applying current value (%.4g) to %s" %
                               (self._current_value, str(self)))
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
    """

    def __init__(self, name, prior, sigma, low, high, bins):
        super(ResolutionParameter, self).__init__(name, prior, sigma,
                                                  low, high, bins)

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
        NotImplementedError("ResolutionParameter.apply_to not yet implemented")


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
    """

    def __init__(self, name, prior, sigma, low, high, bins):
        super(ScaleParameter, self).__init__(name, prior, sigma,
                                             low, high, bins)

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
        NotImplementedError("ScaleParameter.apply_to not yet implemented")


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
    """

    def __init__(self, name, prior, sigma, low, high, bins):
        super(ShiftParameter, self).__init__(name, prior, sigma,
                                             low, high, bins)

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
        NotImplementedError("ShiftParameter.apply_to not yet implemented")


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
                            self._high+self.get_width()*0.5,
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


class Config(object):
    """ The base class for creating config classes.

    Args:
      name (string): The name of the config type.

    Attributes:
      _name (string): The name of the config type.
      _parameters (:class:`collections.OrderedDict`): Dictionary of parameters.
    """

    def __init__(self, name, parameters):
        """ Initialise config class
        """
        self._name = name
        self._parameters = parameters

    def add_par(self, par):
        """ Add parameter to the config.

        Args:
          par (:class:`echidna.core.spectra.Parameter`): The parameter you want
            to add.
        """
        self._parameters[par._name] = par

    def get_index(self, parameter):
        """Return the index of a parameter within the existing set

        Args:
          parameter (string): Name of the parameter.

        Raises:
          IndexError: parameter is not in the config.

        Returns:
          int: Index of the parameter
        """
        for i, p in enumerate(self.get_pars()):
            if p == parameter:
                return i
        raise IndexError("Unknown parameter %s" % parameter)

    def get_name(self):
        """
        Returns:
          string: Name of :class:`Config` class instance - stored in
            :attr:`_name`.
        """
        return self._name

    def get_par(self, name):
        """Get a named FitParameter.

        Args:
          name (string): Name of the parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Named parameter.
        """
        return self._parameters[name]

    def get_par_by_index(self, index):
        """ Get parameter corresponding to given index

        Args:
          index (int): Index of parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Corresponding
            parameter.
        """
        name = self.get_pars()[index]
        return self.get_par(name)

    def get_pars(self):
        """Get list of all parameter names in the config.

        Returns:
          list: List of parameter names
        """
        return self._parameters.keys()

    def get_shape(self):
        """ Get the shape of the parameter space.

        Returns:
          tuple: A tuple constructed of the number of bins for each
            parameter in the config - this can be thought of as the
            full shape of the parameter space, whether it is the shape
            of the parameter space for the fit, or the shape of the
            spectral dimensions.
        """
        return tuple([self.get_par(par).get_bins() for par in self.get_pars()])


class GlobalFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
    """

    def __init__(self, parameters):
        """Initialise GlobalFitConfig class
        """
        super(GlobalFitConfig, self).__init__("global_fit", parameters)

    def add_config(self, config):
        """ Add pars from a :class:`echidna.core.spectra.Config` to this
          :class:`echidna.core.spectra.GlobalFitConfig`

        Args:
          config (:class:`echidna.core.spectra.Config`): Config to be added.
        """
        if config._name == "spectra_fit":
            spectra_name = config._spectra_name
            for par_name in config.get_pars():
                name = spectra_name + "_" + par_name
                par = config.get_par(par_name)
                par._name = name
                self.add_par(par, "spectra")
        elif config._name == "global_fit":
            for par_name in config.get_pars():
                self.add_par(config.get_par(par_name), "global")
        else:
            raise ValueError("%s is not a valid config type" % config._name)

    def add_par(self, par, par_type):
        """ Add parameter to the global fit config.

        Args:
          par (:class:`echidna.core.spectra.FitParameter`): Parameter you want
            to add.
          par_type (string): The type of parameter (global or spectra).
        """
        if par_type != 'global' and par_type != 'spectra':
            raise IndexError("%s is an invalid par_type. Must be 'global' or "
                             "'spectra'." % par_type)
        self._parameters[par._name] = {'par': par, 'type': par_type}

    def get_par(self, name):
        """ Get requested parameter:

        Args:
          name (string): Name of the parameter

        Returns:
          :class:`echidna.core.spectra.FitParameter`: The requested parameter.
        """
        return self._parameters[name]['par']

    def get_global_pars(self):
        """ Gets the parameters which are applied to all spectra
          simultaneously.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'global':
                pars.append(self._parameters[name]['par'])
        return pars

    def get_spectra_pars(self):
        """ Gets the parameters that are applied to individual spectra.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'spectra':
                pars.append(self._parameters[name]['par'])
        return pars

    @classmethod
    def load_from_file(cls, filename):
        """Initialise GlobalFitConfig class from a config file (classmethod).

        Args:
          filename (str): path to config file

        Returns:
          (:class:`echidna.core.spectra.GlobalFitConfig`): A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        parameters = collections.OrderedDict()
        for dim in config['parameters']:
            for syst in config['parameters'][dim]:
                name = dim + "_" + syst
                prior = config['parameters'][dim][syst]['prior'],
                sigma = config['parameters'][dim][syst]['sigma'],
                low = config['parameters'][dim][syst]['low'],
                high = config['parameters'][dim][syst]['high'],
                bins = config['parameters'][dim][syst]['bins'],
                if syst == 'resolution' or syst == 'resolution_ly':
                    parameters[name] = {'par': ResolutionParameter(name, prior,
                                                                   sigma, low,
                                                                   high, bins),
                                        'type': 'global'}
                if syst == 'shift':
                    parameters[name] = {'par': ShiftParameter(name, prior,
                                                              sigma, low,
                                                              high, bins),
                                        'type': 'global'}
                if syst == 'scale':
                    parameters[name] = {'par': ScaleParameter(name, prior,
                                                              sigma, low,
                                                              high, bins),
                                        'type': 'global'}
                else:
                    raise IndexError("%s is not a global fit systematic."
                                     % syst)
        return cls(parameters)


class SpectraFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
      spectra_name (string): Name of the spectra associated with the
         :class:`echidna.core.spectra.SpectraFitConfig`

    Attributes:
      _spectra_name (string): Name of the spectra associated with the
        :class:`echidna.core.spectra.SpectraFitConfig`
    """

    def __init__(self, parameters, spectra_name):
        """Initialise SpectraFitConfig class
        """
        super(SpectraFitConfig, self).__init__("spectra_fit", parameters)
        self._spectra_name = spectra_name

    @classmethod
    def load_from_file(cls, filename, spectra_name):
        """Initialise SpectraFitConfig class from a config file (classmethod).

        Args:
          filename (str): path to config file
          spectra_name (string): Name of the spectra associated with the
            :class:`echidna.core.spectra.SpectraFitConfig`


        Returns:
          (:class:`echidna.core.spectra.SpectraFitConfig`): A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        parameters = collections.OrderedDict()
        for syst in config['parameters']:
            if syst == 'rate':
                rate_kwargs = config['parameters'][syst]
                parameters[syst] = RateParameter(syst, **rate_kwargs)
            else:
                raise IndexError("Unknown systematic in config %s" % syst)
        return cls(parameters, spectra_name)


class SpectraConfig(Config):
    """Configuration container for Spectra objects.  Able to load
    directly with a set list of SpectraParameters or from yaml
    configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        SpectraParameter objects
    """

    def __init__(self, parameters):
        """Initialise SpectraConfig class
        """
        super(SpectraConfig, self).__init__("spectra", parameters)

    @classmethod
    def load_from_file(cls, filename):
        """Initialise SpectraConfig class from a config file (classmethod).

        Args:
          filename (str) path to config file

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        parameters = collections.OrderedDict()
        for v in config['parameters']:
            parameters[v] = SpectraParameter(v, config['parameters'][v]['low'],
                                             config['parameters'][v]['high'],
                                             config['parameters'][v]['bins'])
        return cls(parameters)

    def get_dims(self):
        """Get list of dimension names.
        The _mc, _reco and _truth suffixes are removed.

        Returns:
          list: List of the dimensions names of the config.
        """
        dims = []
        for par in sorted(self._parameters.keys()):
            par = par.split('_')[:-1]
            dim = ""
            for entry in par:
                dim += entry+"_"
            dims.append(dim[:-1])
        return dims

    def get_dim(self, par):
        """Get the dimension of par.
        The _mc, _reco and _truth suffixes are removed.

        Args:
          par (string): Name of the parameter

        Returns:
          The dimension of par
        """
        dim = ""
        for entry in par.split('_')[:-1]:
            dim += entry+"_"
        return dim[:-1]

    def get_dim_type(self, dim):
        """Returns the type of the dimension i.e. mc, reco or truth.

        Args:
          dim (string): The name of the dimension

        Raises:
          IndexError: dim is not in the spectra.

        Returns:
          string: The type of the dimension (mc, reco or truth)
        """
        for par in sorted(self._parameters.keys()):
            par_split = par.split('_')[:-1]
            cur_dim = ""
            for entry in par_split:
                cur_dim += entry+"_"
            if cur_dim[:-1] == dim:
                return str(par.split('_')[-1])
        raise IndexError("No %s dimension in spectra" % dim)


class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.

    The spectra is stored as histogram binned in energy, x, radius, y, and
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d
    (energy).

    Args:
      name (str): The name of this spectra
      num_decays (float): The number of decays this spectra is created to
        represent.
      spectra_config (:class:`SpectraConfig`): The configuration object

    Attributes:
      _data (:class:`numpy.ndarray`): The histogram of data
      _name (str): The name of this spectra
      _config (:class:`SpectraConfig`): The configuration object
      _num_decays (float): The number of decays this spectra currently
        represents.
      _raw_events (int): The number of raw events used to generate the
        spectra. Increments by one with each fill independent of
        weight.
      _bipo (int): Flag to indicate whether the bipo cut was applied to the
        spectra. 0 is No Cut. 1 is Cut.
        Int type as HDF5 does not support bool.
      _style (string): Pyplot-style plotting style e.g. "b-" or
        {"color": "blue"}.
      _rois (dict): Dictionary containing the details of any ROI, along
        any axis, which has been defined.
    """
    def __init__(self, name, num_decays, spectra_config, fit_config=None):
        """ Initialise the spectra data container.
        """
        self._config = spectra_config
        self._raw_events = 0
        bins = []
        for v in self._config.get_pars():
            bins.append(self._config.get_par(v)._bins)
        self._data = numpy.zeros(shape=tuple(bins),
                                 dtype=float)
        self._fit_config = fit_config
        # Flag for indicating bipo cut. HDF5 does not support bool so
        # 0 = no cut and 1 = cut
        self._bipo = 0
        self._style = {"color": "blue"}  # default style for plotting
        self._rois = {}
        self._name = name
        self._num_decays = float(num_decays)

    def get_config(self):
        """ Get the config of the spectra.

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: The config of
            the spectra.
        """
        return self._config

    def get_data(self):
        """
        Returns:
          (:class:`numpy.ndarray`): The spectral data.
        """
        return self._data

    def get_fit_config(self):
        """ Get the config of the spectra.

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: The config of
            the spectra.
        """
        return self._fit_config

    def get_name(self):
        """
        Returns:
          string: The name of the spectra.
        """
        return self._name

    def set_fit_config(self, config):
        """ Get the config of the spectra.

        Args:
          config (:class:`echidna.core.spectra.SpectraFitConfig`): The fit
            config to assign to the spectra.
        """
        if isinstance(config, SpectraFitConfig):
            self._fit_config = config
        else:
            raise TypeError("Invalid config type: %s" % type(config))

    def fill(self, weight=1.0, **kwargs):
        """ Fill the bin with weight.  Note that values for all named
        parameters in the spectra's config (e.g. energy, radial) must be
        passed.

        Args:
          weight (float, optional): Defaults to 1.0, weight to fill the bin
            with.
          kwargs (float): Named values (e.g. for energy_mc, radial_mc)

        Raises:
          Exception: Parameter in kwargs is not in config.
          Exception: Parameter in config is not in kwargs.
          ValueError: If the energy, radius or time is beyond the bin limits.
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        for par in kwargs:
            if par not in self._config.get_pars():
                raise Exception('Unknown parameter %s' % par)
        for par in self._config.get_pars():
            if par not in kwargs:
                raise Exception('Missing parameter %s' % par)
        for v in self._config.get_pars():
            if not self._config.get_par(v)._low <= kwargs[v] < \
                    self._config.get_par(v)._high:
                raise ValueError("%s out of range: %s" % (v, kwargs[v]))
        bins = []
        for v in self._config.get_pars():
            bins.append(int((kwargs[v] - self._config.get_par(v)._low) /
                            (self._config.get_par(v)._high -
                             self._config.get_par(v)._low) *
                            self._config.get_par(v)._bins))
        self._data[tuple(bins)] += weight

    def shrink_to_roi(self, lower_limit, upper_limit, dimension):
        """ Shrink spectrum to a defined Region of Interest (ROI)

        Shrinks spectrum to given ROI and saves ROI parameters.

        Args:
          lower_limit (float): Lower bound of ROI, along given axis.
          upper_limit (float): Upper bound of ROI, along given axis.
          dimension (str): Name of the dimension to shrink.
        """
        integral_full = self.sum()  # Save integral of full spectrum

        # Shrink to ROI
        kw_low = dimension+"_low"
        kw_high = dimension+"_high"
        kw_args = {kw_low: lower_limit,
                   kw_high: upper_limit}
        self.shrink(**kw_args)

        # Calculate efficiency
        integral_roi = self.sum()  # Integral of spectrum over ROI
        efficiency = float(integral_roi) / float(integral_full)
        par = self.get_config().get_par(dimension)
        self._rois[dimension] = {"low": par._low,
                                 "high": par._high,
                                 "efficiency": efficiency}

    def get_roi(self, dimension):
        """ Access information about a predefined ROI for a given dimension

        Returns:
          dict: Dictionary containing parameters defining the ROI, on
            the given dimension.
        """
        return self._rois[dimension]

    def set_style(self, style):
        """ Sets plotting style.

        Styles should be valid pyplot style strings e.g. "b-", for a
        blue line, or dictionaries of strings e.g. {"color": "red"}.

        Args:
          style (string): Pyplot-style plotting style.
        """
        self._style = style

    def get_style(self):
        """
        Returns:
          string/dict: :attr:`_style` - pyplot-style plotting style.
        """
        return self._style

    def project(self, dimension):
        """ Project the histogram along an axis for a given dimension.
        Note that the dimension must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension (str): parameter to project onto

        Returns:
          :class:`numpy.ndarray`: The projection of the histogram onto the
            given axis
        """
        axis = self._config.get_index(dimension)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars()) - 1):
            if axis < i_axis+1:
                projection = projection.sum(1)
            else:
                projection = projection.sum(0)
        return projection

    def nd_project(self, dimensions):
        """ Project the histogram along an arbitary number of axes.

        Args:
          dimensions (str): List of axes to project onto

        Returns:
          :class:`numpy.ndarray`: The nd projection of the histogram.
        """
        axes = []
        for dim in dimensions:
            axes.append(self._config.get_index(dim))
        if len(axes) == len(self._config.get_pars()):
            return copy.copy(self._data)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars())):
            if i_axis not in axes:
                projection = projection.sum(i_axis)
        return projection

    def surface(self, dimension1, dimension2):
        """ Project the histogram along two axes for the given dimensions.
        Note that the dimensions must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension1 (str): first parameter to project onto
          dimension1 (str): second parameter to project onto

        Raises:
          IndexError: Axis of dimension1 is out of range
          IndexError: Axis of dimension2 is out of range

        Returns:
          :class:`numpy.ndarray`: The 2d surface of the histogram.
        """
        axis1 = self._config.get_index(dimension1)
        axis2 = self._config.get_index(dimension2)
        if axis1 < 0 or axis1 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis1)
        if axis2 < 0 or axis2 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis2)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars())):
            if i_axis != axis1 and i_axis != axis2:
                projection = projection.sum(i_axis)
        return projection

    def sum(self):
        """ Calculate and return the sum of the `_data` values.

        Returns:
          float: The sum of the values in the `_data` histogram.
        """
        return self._data.sum()

    def scale(self, num_decays):
        """ Scale THIS spectra to represent *num_decays* worth of decays over
        the entire unshrunken spectra.

        This rescales each bin by the ratio of *num_decays* to
        *self._num_decays*, i.e. it changes the spectra from representing
        *self._num_decays* to *num_decays*. *self._num_decays* is updated
        to equal *num_decays* after.

        Args:
          num_decays (float): Number of decays this spectra should represent.
        """
        self._data = numpy.multiply(self._data, num_decays / self._num_decays)
        # Make sure self._num_decays stays as a float
        self._num_decays = float(num_decays)

    def shrink(self, **kwargs):
        """ Shrink the data such that it only contains values between low and
        high for a given dimension by slicing. This updates the internal bin
        information as well as the data.

        Args:
          kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.

        .. note:

          The logic in this method is the same for each dimension, first
          check the new values are within the existing ones
          (can only compress). Then calculate the low bin number and high bin
          number (relative to the existing binning low).
          Finally update all the bookeeping and slice.

        Raises:
          IndexError: Parameter which is being shrank does not exist in the
            config file.
          ValueError: [parameter]_low value is lower than the parameters lower
            bound.
          ValueError: [parameter]_high value is lower than the parameters
            higher bound.
          IndexError: Suffix to [parameter] is not _high or _low.
        """
        # First check dimensions and bounds in kwargs are valid
        for arg in kwargs:
            pars = self.get_config().get_pars()
            high_low = arg.split("_")[-1]
            par = arg[:-1*(len(high_low)+1)]
            if par not in self._config.get_pars():
                raise IndexError("%s is not a parameter in the config" % par)
            if high_low == "low":
                if numpy.allclose(kwargs[arg], self._config.get_par(par)._low):
                    continue  # To avoid floating point errors
                if kwargs[arg] < self._config.get_par(par)._low:
                    raise ValueError("%s is below existing bound for %s (%s)"
                                     % (kwargs[arg], par,
                                        self._config.get_par(par)._low))
            elif high_low == "high":
                if numpy.allclose(kwargs[arg],
                                  self._config.get_par(par)._high):
                    continue  # To avoid floating point errors
                if kwargs[arg] > self._config.get_par(par)._high:
                    raise ValueError("%s is above existing bound for %s (%s)"
                                     % (kwargs[arg], par,
                                        self._config.get_par(par)._high))
            else:
                raise IndexError("%s index invalid. Index must be of the form"
                                 "[dimension name]_high or"
                                 "[dimension name]_low" % arg)
        slices_low = []
        slices_high = []
        for par_name in self._config.get_pars():
            par = self._config.get_par(par_name)
            kw_low = "%s_low" % par_name
            kw_high = "%s_high" % par_name
            if "%s_low" % par_name not in kwargs:
                kwargs[kw_low] = par._low
            if "%s_high" % par_name not in kwargs:
                kwargs[kw_high] = par._high
            # Round down the low bin
            low_bin = int((kwargs[kw_low] - par._low) / par.get_width())
            # Round up the high bin
            high_bin = numpy.ceil((kwargs[kw_high] - par._low) /
                                  par.get_width())
            # new_low is the new lower first bin edge
            new_low = par.round(par._low + low_bin * par.get_width())
            # new_high is the new upper last bin edge
            new_high = par.round(par._low + high_bin * par.get_width())
            # Correct floating point errors: If the difference between
            # input high/low and calculated new high/low is approximately
            # equal (<1%) to a bin width then assume user requested the bin
            # above/below to be cut.
            if numpy.fabs(new_high - kwargs[kw_high]) > (0.99 *
                                                         par.get_width()):
                # print ("WARNING: Correcting possible floating point error in"
                #       "spectra.Spectra.shrink\n%s was the input. %s is the "
                #        "calculated value for %s" % (kwargs[kw_low],
                #                                     new_low, kw_low))
                if (new_high - kwargs[kw_high]) > 0.0:
                    high_bin -= 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                else:
                    high_bin += 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                # print "Corrected %s to %s" % (kw_low, new_low)
            if numpy.fabs(new_low - kwargs[kw_low]) > (0.99 * par.get_width()):
                # print ("WARNING: Correcting possible floating point error in"
                #       "spectra.Spectra.shrink\n%s was the input. %s is the "
                #       "calculated value for %s" % (kwargs[kw_low],
                #                                    new_low, kw_low))
                if (new_low - kwargs[kw_low]) > 0.0:
                    low_bin -= 1
                    new_low = par._low + low_bin * par.get_width()
                else:
                    low_bin += 1
                    new_low = par._low + low_bin * par.get_width()
                print "Corrected %s to %s" % (kw_low, new_low)
            slices_high.append(high_bin)
            slices_low.append(low_bin)
            new_bins = high_bin - low_bin
            par.set_par(low=new_low, high=new_high, bins=new_bins)
        # First set up the command then evaluate. Hacky but hey ho.
        cmd = "self._data["
        for i in range(len(slices_low)):
            low = str(slices_low[i])
            high = str(slices_high[i])
            cmd += low+":"+high+","
        cmd = cmd[:-1]+"]"
        self._data = eval(cmd)

    def cut(self, **kwargs):
        """ Similar to :meth:`shrink`, but updates scaling information.

        If a spectrum is cut using :meth:`shrink`, subsequent calls to
        :meth:`scale` the spectrum must still scale the *full* spectrum
        i.e. before any cuts. The user supplies the number of decays
        the full spectrum should now represent.

        However, sometimes it is more useful to be able specify the
        number of events the revised spectrum should represent. This
        method updates the scaling information, so that it becomes the
        new *full* spectrum.

        Args:
          kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.
        """
        initial_count = self.sum()  # Store initial count
        self.shrink(**kwargs)
        new_count = self.sum()
        reduction_factor = float(new_count) / float(initial_count)
        # This reduction factor tells us how much the number of detected events
        # has been reduced by shrinking the spectrum. We want the number of
        # decays that the spectrum should now represent to be reduced by the
        # same factor
        self._num_decays *= reduction_factor

    def add(self, spectrum):
        """ Adds a spectrum to current spectra object.

        Args:
          spectrum (:class:`Spectra`): Spectrum to add.

        Raises:
          ValueError: spectrum has different dimenstions to the current
            spectra.
          IndexError: spectrum does not contain a dimension(s) that is in the
            current spectra config.
          IndexError: The current spectra does not contain a dimension(s) that
            is in the spectrum config.
          ValueError: The upper bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The lower bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The number of bins of a parameter in the current spectra
            and spectra are not equal.
        """
        if self._data.shape != spectrum._data.shape:
            raise ValueError("The spectra have different dimensions.\n"
                             "Dimension of self: %s. Dimension of spectrum %s"
                             % (self._data.shape, spectrum._data.shape))
        for v in self._config.get_dims():
            if v not in spectrum.get_config().get_dims():
                raise IndexError("%s not present in new spectrum" % v)
        for v in spectrum.get_config().get_dims():
            if v not in self._config.get_dims():
                raise IndexError("%s not present in this spectrum" % v)
        # Dictionary containing dimensions which have different types in the
        # two spectra. The type of the dimension of spectrum is the value
        types = {}
        for v in spectrum.get_config().get_pars():
            if v not in self._config.get_pars():
                dim = spectrum.get_config().get_dim(v)
                dim_type = spectrum._config.get_dim_type(dim)
                types[dim] = dim_type
        for v in self._config.get_pars():
            dim = self._config.get_dim(v)
            if dim in types:
                v_spec = dim+'_'+types[dim]
            else:
                v_spec = v
            if not numpy.allclose(self.get_config().get_par(v)._high,
                                  spectrum.get_config().get_par(v_spec)._high):
                raise ValueError("Upper %s bounds in spectra are not equal."
                                 "\n%s upper bound: %s\n%s upper bound: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._high,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._high))
            if not numpy.allclose(self.get_config().get_par(v)._low,
                                  spectrum.get_config().get_par(v_spec)._low):
                raise ValueError("Lower %s bounds in spectra are not equal."
                                 "\n%s lower bound: %s\n%s lower bound: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._low,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._low))
            if self.get_config().get_par(v)._bins != \
                    spectrum.get_config().get_par(v_spec)._bins:
                raise ValueError("Number of %s bins in spectra are not equal."
                                 "\n%s bins: %s\n%s lower bins: %s"
                                 % (v, self._name,
                                    self.get_config().get_par(v)._bins,
                                    spectrum._name,
                                    spectrum.get_config().get_par(v_spec)
                                    ._bins))
        self._data += spectrum._data
        self._raw_events += spectrum._raw_events
        self._num_decays += spectrum._num_decays

    def rebin(self, new_bins):
        """ Rebin spectra data into a smaller spectra of the same rank whose
        dimensions are factors of the original dimensions.

        Args:
          new_bins (tuple): new binning, this must match both the
            number and ordering of dimensions in the spectra config.
            For example if the old data shape is made of bins (1000, 10)
            and you would like to increase the bin width of both by 2 then
            you must pass the tuple (500, 5)

        Raises:
          ValueError: Shape mismatch. Number of dimenesions are different.
          ValueError: Old bins/ New bins must be integer
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        if len(new_bins) != len(self._config.get_pars()):
            raise ValueError('Incorrect number of dimensions; need %s'
                             % len(self._config.get_pars()))
        # Now do the rebinning
        for i, v in enumerate(self._config.get_pars()):
            if self._config.get_par(v)._bins % new_bins[i] != 0:
                raise ValueError("Old bins/New bins must be integer old: %s"
                                 " new: %s for parameter %s"
                                 % (self._config.get_par(v)._bins,
                                    new_bins[i], v))
            self._config.get_par(v)._bins = new_bins[i]

        compression_pairs = [(d, c//d) for d, c in zip(new_bins,
                                                       self._data.shape)]
        flattened = [l for p in compression_pairs for l in p]
        self._data = self._data.reshape(flattened)
        for i in range(len(new_bins)):
            self._data = self._data.sum(-1*(i+1))

    def interpolate1d(self, dimension, kind='cubic'):
        """ Interpolates a given dimension of a spectra.

        Args:
          dimension (string): Dimension you want to interpolate.
          kind (string): Method of interpolation.
            See :class:`scipy.interpolate.interp1d` for available methods.

        Returns:
          :class:`scipy.interpolate.interp1d`: Interpolation function.
        """
        x = self._config.get_par(dimension).get_bin_centres()
        y = self.project(dimension)
        return interpolate.interp1d(x, y, kind=kind, bounds_error=False)
