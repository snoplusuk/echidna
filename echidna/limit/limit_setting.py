import numpy


class LimitSetting(object):
    """ Class to handle the setting of the limit.
    """
    def __init__(self, signal, backgrounds, data=None, **kwargs):
        """
        Args:
          signal (:class: `spectra.Spectra`): signal spectrum
          backgrounds (list): one :class: `spectra.Spectra` for each
            background
          data (:class: `numpy.array`): 1D data energy spectrum
          **kwargs (dict): keyword arguments

        .. note::

        Keyword arguments include:

          * roi (*tuple*): (energy_lower, energy_upper)

        Attributes:
          _signal (:class: `spectra.Spectra`): signal spectrum
          _backgrounds (list): one :class: `spectra.Spectra` for each
            background
          _signal_config (:class: `limit_config.LimitConfig`): signal
            configuration
          _background_config (:class: `limit_config.LimitConfig`): 
            background configuration
          _calculator (:class: `chi_squared.ChiSquared`): chi squared
            calculator to use for limit setting
          _observed (:class: `numpy.array`): energy spectrum of observed
            events (data)
        """
        self._signal = signal
        self._signal_config = None
        self._backgrounds = backgrounds
        self._background_configs = {}
        self._data = data
        self._calculator = None
        self._observed = None
        if kwargs.get("roi") is not None:
            self._roi = kwargs.get("roi")
        else:
            self._roi = None

        # Check spectra are compatible
        for background in backgrounds:
            assert (background._energy_width == signal._energy_width), \
                "cannot compare histograms, different energy bin width"
            assert (background._radial_width == signal._radial_width), \
                "cannot compare histograms, different radial bin width"
            assert (background._time_width == signal._time_width), \
                "cannot compare histograms, different time bin width"

    def configure_signal(self, signal_config):
        """ Supply a configuration object associated with the signal

        Args:
          signal_config (:class: `limit_config.LimitConfig`): signal
            configuration
        """
        self._signal_config = signal_config

    def configure_background(self, name, background_config):
        """ Supply a configuration object associated with the background

        Args:
          background_config (:class: `limit_config.LimitConfig`): 
            background configuration
        """
        self._background_configs[name] = background_config

    def set_calculator(self, calculator):
        """ Sets the chi squared calculator to use for limit setting

        Args:
          calculator (:class: `chi_squared.ChiSquared`): chi squared
            calculator to use for limit setting
        """
        self._calculator = calculator

    def get_limit(self, limit_chi_squared=2.71):
        """ Get signal counts at limit

        Args:
          limit_chi_squared (float): chi squared required for limit
        """
        assert (self._signal_config is not None), \
            "signal limit configuration not set"
        assert (len(self._background_configs) == len(self._backgrounds)), \
            "missing limit configuration for one or more backgrounds"
        assert (self._calculator is not None), \
            "chi squared calculator not set"
        if self._data is None:
            observed = numpy.zeros(
                shape=[self._signal._energy_bins],
                dtype=float)
            for background in self._backgrounds:
                config = self._background_configs.get(background._name)
                background.scale(config._prior_count)
                observed += background.project(0)
            self._observed = observed
        else:  # _data is not None
            self,_observed = self._data
        self._signal_config.reset_chi_squareds()
        for signal_count in self._signal_config.get_count():
            print "LimitSetting.get_limit: signal_count =", signal_count
            self._signal.scale(signal_count)
            self._signal_config.add_chi_squared(
                self._get_chi_squared(
                    self._backgrounds, len(self._backgrounds)
                    )
                )
        try:
            return self._signal_config.get_first_bin_above(limit_chi_squared)
        except IndexError as detail:
            print ("LimitSetting.get_limit: unable to calculate confidence "
                   "limit - " + str(detail))

    def _get_chi_squared(self, backgrounds, total_backgrounds, current=-1):
        """
        Args:
          backgrounds (list): one :class: `spectra.Spectra` for each
            background
          total_backgrounds (int): total number of backgrounds to 
            include in fit
          current (int, optional): counter to keep track of the current 
            background being floated. Default is -1, so that it is 
            immediately set to zero after function call.

        Returns:
          *float*. Minimum chi squared obtained by floating backgrounds
        """
        current += 1
        background = backgrounds[current]
        name = background._name
        config = self._background_configs.get(name)
        config.reset_chi_squareds()

        # Loop over count values
        for count in config.get_count():
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
                            self._observed, expected, penalty_term=penalty_term
                            )
                        )
                except ValueError as detail:  # Either histogram has bins with
                                              # zero events
                    print "WARNING:", detail
                    if self._roi is not None:
                        print " --> shrinking spectra"
                        energy_low, energy_high = self._roi
                        print energy_low, energy_high
                        
                        # Shrink signal
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
                                self._observed, expected, penalty_term=penalty_term
                                )
                            )
                    else:
                        raise
        current -= 1
        return config.get_minimum()
