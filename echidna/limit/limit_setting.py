import numpy

from echidna.errors.custom_errors import CompatibilityError
import echidna.utilities as utilities


class SystAnalyser(object):
    """ Class to analyse the effect of systematics

    Records all chi squared data for a systematic and data about how the
    systematic affects the limit setting code.

    Args:
      name (float): Name of systematic. Ideally should be the same name
        as corresponding :class:`limit_config` key and penalty term.
      signal_counts (:class:`numpy.ndarray`): Array of signal counts
      syst_values (:class:`numpy.ndarray`): Array of values for systematic

    Attributes:
      _name (float): Name of systematic. Ideally should be the same name
        as corresponding :class:`limit_config` key and penalty term.
      _chi_squareds (:class:`numpy.ndarray`): Axes 0: signal counts,
        1: systematic values (e.g. background counts), 2: layer.
        Records the chi squared values for every iteration over the
        list of systematic values (layer).
      _preferred_values (:class:`numpy.ndarray`): Axes 0: signal counts,
        1: layer. Stores the prefered value, out of all systematic
        values in the loop, for each iteration (layer) over the values.
      _minima (:class:`numpy.ndarray`): Stores x,y coordinates for the
        position of each minima --> col 0: singnal count, col 1: value
        of systemaitc.
      _signal_counts (:class:`numpy.ndarray`): Array of signal counts.
      _syst_values (:class:`numpy.ndarray`): Array of values for 
        systematic
      _actual_counts (:class:`numpy.ndarray`): Array of actual signal
        counts. A copy of :obj:`signal_config._chi_squareds[2]` after
        limit setting is complete.
      _layer (int): current layer/iteration over systematic values.
      """
    def __init__(self, name, signal_counts, syst_values):
        self._name = name  # ideally same name as config and penalty term
        self._chi_squareds = numpy.zeros(shape=(len(signal_counts), 1,
                                                len(syst_values)), dtype=float)
        self._preferred_values = numpy.zeros(shape=(len(signal_counts), 1),
                                             dtype=float)
        self._penalty_values = numpy.zeros(shape=(2, 1), dtype=float)
        self._minima = numpy.zeros(shape=(2, 1), dtype=float)
        self._signal_counts = signal_counts
        self._syst_values = syst_values
        self._actual_counts = numpy.zeros(shape=(len(signal_counts)),
                                          dtype=float)
        self._layer = 1

    def add_chi_squareds(self, signal_bin, chi_squareds):
        """ Adds to :attr:`_chi_squareds`

        Args:
          signal_bin (int): Bin corresponding to current value in signal
            counts.
          chi_squareds (:class:`numpy.ndarray`): 1D array of chi
            squareds to add

        Raises:
          CompatibilityError: If supplied chi squareds array does not have
            correct shape.
        """
        required_shape = self._chi_squareds[0][0].shape
        if chi_squareds.shape != required_shape:
            raise CompatibilityError("chi_squareds array does not have "
                                     "required shape - " + str(required_shape))
        if self._chi_squareds.shape[1] < self._layer:  # layer doesn't exist
            # Create new layer
            new_layer = numpy.zeros(shape=self._chi_squareds[:, 0:1, :].shape,
                                    dtype=float)
            self._chi_squareds = numpy.append(self._chi_squareds, new_layer,
                                              axis=1)
        self._chi_squareds[signal_bin][self._layer-1] = chi_squareds

    def add_preferred_value(self, signal_bin, preferred_value):
        """ *** NOT IMPLEMENTED YET ***
        """
        if self._preferred_values.shape[1] < self._layer:  # doesn't exist
            new_layer = numpy.zeros(shape=self._preferred_values[:,0:1].shape,
                                    dtype=float)
            self._preferred_values = numpy.append(self._preferred_values,
                                                  new_layer, axis=1)
        self._preferred_values[signal_bin][self._layer-1] = preferred_value

    def add_penalty_value(self, syst_value, penalty_value):
        """ *** NOT IMPLEMENTED YET ***
        """
        entry_to_append = numpy.zeros((2, 1), dtype=float)
        entry_to_append[0][0] = syst_value
        entry_to_append[1][0] = penalty_value
        self._penalty_values = numpy.append(self._penalty_values,
                                            entry_to_append, axis=1)

    def add_minima(self, signal_count, syst_value):
        """ *** NOT IMPLEMENTED YET ***
        """
        entry_to_append = numpy.zeros((2, 1), dtype=float)
        entry_to_append[0][0] = signal_count
        entry_to_append[1][0] = syst_value
        self._minima = numpy.append(self._minima, entry_to_append, axis=1)


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
        * verbose (*bool*): If set to True, progress and timing
          information is printed to the terminal during limit setting.

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
      _verbose (bool): print progress and timing information to terminal
        during limit setting.

    Raises:
      CompatibilityError: If any background spectrum is incompatible
        with the signal spectrum
    """
    def __init__(self, signal, backgrounds, data=None, **kwargs):
        self._signal = signal
        self._signal_config = None
        self._backgrounds = backgrounds
        self._background_configs = {}
        self._syst_analysers = {}
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
        if kwargs.get("verbose"):
            self._verbose = True
        else:
            self._verbose = False

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

    def configure_background(self, name, background_config, **kwargs):
        """ Supply configuration object associated with the background.

        Args:
          background_config (:class:`echidna.limit.limit_config.LimitConfig`): background
            configuration

        .. note::

          Keyword arguments include:

            * plot_systematic (*bool*): if true produces signal vs systematic
              plots

        Raises:
          TypeError: If config has not been set for signal.
        """
        if self._signal_config is None:
            raise TypeError("signal configuration not set")
        self._background_configs[name] = background_config
        if kwargs.get("plot_systematic"):
            self._syst_analysers[name] = SystAnalyser(
                name+"_counts", self._signal_config._counts,
                background_config._counts)

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
          TypeError: If config has not been set for signal.
          KeyError: If config has not been set for one or more
            backgrounds.
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
                print background._name, background.sum(), config._prior_count
                background.scale(config._prior_count)
                observed += background.project(0)
            self._observed = observed
            print numpy.sum(self._observed)
            raw_input("RETURN to continue")
        else:  # _data is not None
            self._observed = self._data
        self._signal_config.reset_chi_squareds()
        for signal_count in self._signal_config.get_count():
            for syst_analyser in self._syst_analysers.values():
                syst_analyser._layer = 1  # reset layers 
            with utilities.Timer() as t:  # set timer
                self._signal.scale(signal_count)
                print self._signal._name, self._signal.sum()
                self._signal_config.add_chi_squared(
                    self._get_chi_squared(self._backgrounds,
                                          len(self._backgrounds)),
                    signal_count, self._signal.sum())
            if self._verbose:
                print ("Calculations for %.4f signal counts took %.03f "
                       "seconds." % (signal_count, t._interval))
        for syst_analyser in self._syst_analysers.values():
            syst_analyser._actual_counts = self._signal_config._chi_squareds[2]
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

        try:
            syst_analyser = self._syst_analysers[name]
        except KeyError:  # No syst_analyser set for this background
            syst_analyser = None
        sig_config = self._signal_config

        # Loop over count values
        for count in config.get_count():  # Generator
            background.scale(count)
            print background._name, background.sum()
            if (current < total_backgrounds-1):
                # Add penalty terms manually
                if config._sigma is not None:
                    penalty_term = {
                            "parameter_value": count - config._prior_count,
                            "sigma": config._sigma
                            }
                    self._calculator.set_penalty_term(name, penalty_term)
                config.add_chi_squared(
                    self._get_chi_squared(self._backgrounds,
                                          len(self._backgrounds),
                                          current),
                    count, background.sum())  # function recursion
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
                    print "observed:", numpy.sum(self._observed)
                    print "expected:", numpy.sum(expected)
                    config.add_chi_squared(
                        self._calculator.get_chi_squared(
                            self._observed, expected,
                            penalty_terms=penalty_term),
                        count, background.sum())
                    if syst_analyser is not None:
                        penalty_value = self._calculator._current_values[name]
                        syst_analyser.add_penalty_value(count, penalty_value)
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
                                self._observed, expected,
                                penalty_terms=penalty_term),
                            count, background.sum())
                        if syst_analyser is not None:
                            penalty_value = self._calculator._current_values[name]
                            syst_analyser.add_penalty_value(count, penalty_value)
                    else:
                        raise
        minimum, minimum_bin = config.get_minimum(minimum_bin=True)
        sig_bin = numpy.where(sig_config._counts == sig_config._current_count)[0][0]
        preferred_value = config._chi_squareds[1][minimum_bin][0]
        if syst_analyser is not None:
            syst_analyser.add_chi_squareds(sig_bin, config._chi_squareds[0])
            syst_analyser.add_preferred_value(sig_bin, preferred_value)
            syst_analyser.add_minima(self._signal.sum(), preferred_value)
            syst_analyser._layer += 1
        current -= 1
        return minimum
