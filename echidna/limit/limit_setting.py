import numpy

from echidna.errors.custom_errors import CompatibilityError


class LimitSetting(object):
    """ Class to handle main limit setting.

    Args:
      signal (:class:`echidna.core.spectra.Spectra`): signal spectrum
      backgrounds (list): one :class:`echidna.core.spectra.Spectra` for
        each background
      data (:class:`numpy.array`, optional): 1D data energy spectrum
      **kwargs (dict): keyword arguments

    .. note::

      Keyword arguments include:

        * roi (*tuple*): (energy_lower, energy_upper)
        * pre_shrink (*bool*): If set to True, :meth:`shrink` method is
          called on all spectra before limit setting, shrinking to
          ROI. Only applies if ROI has been set via ``roi`` keyword.

    Attributes:
      _signal (:class:`echidna.core.spectra.Spectra`): signal spectrum
      _signal_config (:class:`echidna.limit.limit_config.LimitConfig`):
        signal configuration
      _backgrounds (list): one :class:`echidna.core.spectra.Spectra` for
        each background
      _background_configs (dict): one
        :class:`echidna.limit.limit_config.LimitConfig` for each
        background, with :attr:`Spectra._name` as the corresponding key
      _calculator (:class:`echidna.limit.chi_squared.ChiSquared`): chi
        squared calculator to use for limit setting
      _observed (:class:`numpy.array`): energy spectrum of observed
        events (data)
      _roi (tuple): (lower energy, upper energy)

    Raises:
      CompatibilityError: If any background spectrum is incompatible
        with the signal spectrum
    """
    def __init__(self, signal, backgrounds, data=None, **kwargs):
        self._signal = signal
        self._signal_config = None
        self._backgrounds = backgrounds
        self._background_configs = {}
        self._data = data
        self._calculator = None
        self._observed = None
        if kwargs.get("roi") is not None:
            self._roi = kwargs.get("roi")
            if kwargs.get("pre_shrink"):
                energy_low, energy_high = self._roi
                self._signal.shrink(energy_low, energy_high)
                for background in self._backgrounds:
                    background.shrink(energy_low, energy_high)
        else:
            self._roi = None

        # Check spectra are compatible
        for background in backgrounds:
            if (background._energy_width != signal._energy_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different energy bin widths")
            if (background._radial_width != signal._radial_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different radial bin widths")
            if (background._time_width != signal._time_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different time bin width")

    def configure_signal(self, signal_config):
        """ Supply a configuration object associated with the signal.

        Args:
          signal_config (:class:`echidna.limit.limit_config.LimitConfig`): signal
            configuration
        """
        self._signal_config = signal_config

    def configure_background(self, name, background_config):
        """ Supply configuration object associated with the background.

        Args:
          background_config (:class:`echidna.limit.limit_config.LimitConfig`): background
            configuration
        """
        self._background_configs[name] = background_config

    def set_calculator(self, calculator):
        """ Sets the chi squared calculator to use for limit setting

        Args:
          calculator (:class:`echidna.limit.chi_squared.ChiSquared`): chi
            squared calculator to use for limit setting
        """
        self._calculator = calculator

    def get_limit(self, limit_chi_squared=2.71):
        """ Get signal counts at limit.

        Args:
          limit_chi_squared (float, optional): chi squared required for
            limit.

        .. note::

          Default value for :obj:`limit_chi_squared` is 2.71, the chi
          squared value corresponding to a 90% confidence limit.

        Returns:
          float: Signal counts at required limit

        Raises:
          IndexError: If no limit can be calculated. Relies on finding
            the first bin with a chi squared value above
            :obj:`limit_chi_squared`. If no bin contains a chi squared
            value greater than :obj:`limit_chi_squared`, then there is
            no bin to be found, raising IndexError.
        """
        if self._signal_config is None:
            raise TypeError("signal configuration not set")
        if (len(self._background_configs) != len(self._backgrounds)):
            raise KeyError("missing configuration for one or more backgrounds")
        if self._calculator is None:
            raise TypeError("chi squared calculator not set")
        if self._data is None:
            observed = numpy.zeros(shape=[self._signal._energy_bins],
                                   dtype=float)
            for background in self._backgrounds:
                config = self._background_configs.get(background._name)
                background.scale(config._prior_count)
                observed += background.project(0)
            self._observed = observed
        else:  # _data is not None
            self._observed = self._data
        self._signal_config.reset_chi_squareds()
        for signal_count in self._signal_config.get_count():
            self._signal.scale(signal_count)
            self._signal_config.add_chi_squared(
                self._get_chi_squared(self._backgrounds,
                                      len(self._backgrounds)))
        try:
            return self._signal_config.get_first_bin_above(limit_chi_squared)
        except IndexError as detail:
            raise IndexError("unable to calculate confidence limit - " +
                             str(detail))

    def _get_chi_squared(self, backgrounds, total_backgrounds, current=-1):
        """ Internal method to minimise chi squared for each background.

        This method is called recursively to minimise the chi squared
        contribution for each background, and return the minimum to the
        previous background, which then minimises over all chi squareds
        it recieves.

        Args:
          backgrounds (list): one :class:`echidna.core.spectra.Spectra`
            for each background
          total_backgrounds (int): total number of backgrounds to
            include in fit
          current (int, optional): counter to keep track of the current
            background being floated. Default is -1, so that it is
            immediately set to zero after function call.

        Returns:
          float: Minimum chi squared obtained by floating backgrounds

        Raises:
          ValueError: If chi squared calculation raises ValueError (due
            to bins containing zero events) even after an attempt to
            resolve the error by shrinking each spectra to the supplied
            Region Of Interest (ROI).
        """
        current += 1
        background = backgrounds[current]
        name = background._name
        config = self._background_configs.get(name)
        config.reset_chi_squareds()

        # Loop over count values
        for count in config.get_count():  # Generator
            background.scale(count)
            if (current < total_backgrounds-1):
                config.add_chi_squared(
                    self._get_chi_squared(self._backgrounds,
                                          len(self._backgrounds),
                                          current))  # function recursion
            else:
                total_background = numpy.zeros(
                    shape=[self._signal._energy_bins],
                    dtype=float)
                for background in self._backgrounds:
                    total_background += background.project(0)
                expected = total_background + self._signal.project(0)
                if config._sigma is not None:
                    penalty_term = {
                        name: {
                            "parameter_value": count - config._prior_count,
                            "sigma": config._sigma
                            }
                        }
                else:
                    penalty_term = {}
                try:
                    config.add_chi_squared(
                        self._calculator.get_chi_squared(
                            self._observed,
                            expected,
                            penalty_term=penalty_term))
                except ValueError as detail:  # Either histogram has bins with
                                              # zero events
                    if self._roi is not None:
                        print "WARNING:", detail
                        print " --> shrinking spectra"
                        energy_low, energy_high = self._roi

                        # Shrink signal to ROI
                        self._signal.shrink(energy_low, energy_high)

                        # Create new total background with shrunk spectra
                        total_background = numpy.zeros(
                            shape=[self._signal._energy_bins],
                            dtype=float)
                        for background in self._backgrounds:
                            background.shrink(energy_low, energy_high)
                            total_background += background.project(0)

                        # Set make new expected and _observed (if required)
                        expected = total_background + self._signal.project(0)
                        if self._data is None:
                            self._observed = total_background
                        config.add_chi_squared(
                            self._calculator.get_chi_squared(
                                self._observed,
                                expected,
                                penalty_term=penalty_term))
                    else:
                        raise
        current -= 1
        return config.get_minimum()
