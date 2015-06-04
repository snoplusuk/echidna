import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import echidna
from echidna.errors.custom_errors import CompatibilityError
import echidna.utilities as utilities
import echidna.output.plot as plot

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
      _name (string): Name of systematic. Ideally should be the same name
        as corresponding :class:`limit_config` key and penalty term.
      _chi_squareds (:class:`numpy.ndarray`): Axes 0: signal counts,
        1: systematic values (e.g. background counts), 2: layer.
        Records the chi squared values for every iteration over the
        list of systematic values (layer).
      _preferred_values (:class:`numpy.ndarray`): Axes 0: signal counts,
        1: layer. Stores the prefered value, out of all systematic
        values in the loop, for each iteration (layer) over the values.
      _penalty_values (:class:`numpy.ndarray`): col-0 = value of
        systematic, col-1 = value of penalty term.
      _minima (:class:`numpy.ndarray`): Stores x,y coordinates for the
        position of each minima --> col-0 = signal count, col-1: value
        of systematic.
      _signal_counts (:class:`numpy.ndarray`): Array of signal counts.
      _syst_values (:class:`numpy.ndarray`): Array of values for
        systematic.
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
        self._penalty_values = numpy.zeros(shape=(2, 0), dtype=float)
        self._minima = numpy.zeros(shape=(2, 0), dtype=float)
        self._signal_counts = signal_counts
        self._syst_values = syst_values
        self._actual_counts = numpy.zeros(shape=(len(signal_counts)),
                                          dtype=float)
        self._layer = 1

    def get_name(self):
        """
        Returns:
          string: :attr:`_name`
        """
        return self._name

    def add_chi_squareds(self, signal_bin, chi_squareds):
        """ Adds to :attr:`_chi_squareds`.

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

    def get_chi_squareds(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_chi_squareds` Axes 0: signal
            counts, 1: systematic values (e.g. background counts), 2:
            layer. Records the chi squared values for every iteration
            over the list of systematic values (layer).
        """
        return self._chi_squareds

    def add_preferred_value(self, signal_bin, preferred_value):
        """ Adds to :attr:`_preferred_values`

        Args:
          signal_bin (int): Bin corresponding to current value in signal
            counts.
          preferred_values (:class:`numpy.ndarray`): 1D array of
            preferred values to add.
        """
        if self._preferred_values.shape[1] < self._layer:  # doesn't exist
            new_layer = numpy.zeros(shape=self._preferred_values[:, 0: 1].shape,
                                    dtype=float)
            self._preferred_values = numpy.append(self._preferred_values,
                                                  new_layer, axis=1)
        self._preferred_values[signal_bin][self._layer-1] = preferred_value

    def get_preferred_values(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_preferred_values`. Axes 0:
          signal counts, 1: layer. Stores the prefered value, out of
          all systematic values in the loop, for each iteration (layer)
          over the values.
        """
        return self._preferred_values

    def add_penalty_value(self, syst_value, penalty_value):
        """ Adds to :attr:`_penalty_values`

        Args:
          syst_value (float): value of systematic.
          penalty_value (float): value of penalty term, corresponding
            to given systematic value.
        """
        entry_to_append = numpy.zeros((2, 1), dtype=float)
        entry_to_append[0][0] = syst_value
        entry_to_append[1][0] = penalty_value
        self._penalty_values = numpy.append(self._penalty_values,
                                            entry_to_append, axis=1)

    def get_penalty_values(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_penalty_values`. col-0 =
            value of systematic, col-1 = value of penalty term.
        """
        return self._penalty_values

    def add_minima(self, signal_count, syst_value):
        """ Adds to :attr:`_minima`

        Args:
          signal_count (float): current signal count
          syst_value (float): current value of systematic at given
            signal count.
        """
        entry_to_append = numpy.zeros((2, 1), dtype=float)
        entry_to_append[0][0] = signal_count
        entry_to_append[1][0] = syst_value
        self._minima = numpy.append(self._minima, entry_to_append, axis=1)

    def get_minima(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_minima`. col-0 = signal
            count, col-1 = value of systematic.
        """
        return self._minima

    def get_signal_counts(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_signal_counts`.
        """
        return self._signal_counts

    def get_syst_values(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_syst_values`.
        """
        return self._syst_values

    def get_actual_counts(self):
        """
        Returns:
          :class:`numpy.ndarray`: :attr:`_actual_counts`.
        """
        return self._actual_counts


class LimitSetting(object):
    """ Class to handle main limit setting.

    Args:
      signal (:class:`echidna.core.spectra.Spectra`): signal spectrum
      fixed_background (:class:`echidna.core.spectra.Spectra`): A spectrum
        containing all fixed backgrounds.
      floating_backgrounds (list): one :class:`echidna.core.spectra.Spectra`
        for each background
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
      _fixed_background (:class:`echidna.core.spectra.Spectra`): A spectrum
        containing all fixed backgrounds.
      _floating_backgrounds (list): one :class:`echidna.core.spectra.Spectra`
        for each background
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
      CompatibilityError: If fixed_background is not pre-shrunk and
        pre-shrinking is activated for floating backgrounds.
      CompatibilityError: If any background spectrum is incompatible
        with the signal spectrum
    """
    def __init__(self, signal, fixed_background=None,
                 floating_backgrounds=None, data=None, **kwargs):
        self._signal = signal
        self._signal_config = None
        self._fixed_background = fixed_background
        self._floating_backgrounds = floating_backgrounds
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
                if self._fixed_background:
                    if self._fixed_background._energy_low != energy_low:
                        raise CompatibilityError("Fixed background must be "
                                                 "pre-shrunk before"
                                                 "LimitSetting.")
                    if self._fixed_background._energy_high != energy_high:
                        raise CompatibilityError("Fixed background must be "
                                                 "pre-shrunk before"
                                                 "LimitSetting.")
                if self._floating_backgrounds:
                    for background in self._floating_backgrounds:
                        background.shrink(energy_low, energy_high)
        else:
            self._roi = None
        if kwargs.get("verbose"):
            self._verbose = True
        else:
            self._verbose = False
        # Check backgrounds have been added to class
        if not fixed_background and not floating_backgrounds:
            raise ValueError("Must provide fixed or floating backgrounds")
        # Check spectra are compatible
        if fixed_background:
            if (fixed_background._energy_width != signal._energy_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different energy bin widths")
            if (fixed_background._radial_width != signal._radial_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different radial bin widths")
            if (fixed_background._time_width != signal._time_width):
                raise CompatibilityError("cannot compare histograms with "
                                         "different time bin width")
        if self._floating_backgrounds:
            for background in floating_backgrounds:
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
          signal_config (:class:`echidna.limit.limit_config.LimitConfig`):
            signal configuration
        """
        self._signal_config = signal_config

    def configure_background(self, name, background_config, **kwargs):
        """ Supply configuration object associated with the background.

        Args:
          background_config (:class:`echidna.limit.limit_config.LimitConfig`):
          background configuration

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

    def get_limit(self, limit_chi_squared=2.71, **kwargs):
        """ Get signal counts at limit.

        Args:
          limit_chi_squared (float, optional): chi squared required for
            limit.

        .. note::

          Keyword arguments include:

            * debug (*int*): specify the level of debug information to
              output.

        .. note::

          Default value for :obj:`limit_chi_squared` is 2.71, the chi
          squared value corresponding to a 90% confidence limit.

        Returns:
          float: Signal counts in ROI at required limit

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
        if self._floating_backgrounds is None:
            return self.get_limit_no_float(limit_chi_squared)
        if self._signal_config is None:
            raise TypeError("signal configuration not set")
        if (len(self._background_configs) != len(self._floating_backgrounds)):
            raise KeyError("missing configuration for one or more backgrounds")
        if self._calculator is None:
            raise TypeError("chi squared calculator not set")
        if self._data is None:
            observed = numpy.zeros(shape=[self._signal._energy_bins],
                                   dtype=float)
            if self._fixed_background:
                observed += self._fixed_background.project(0)
            for background in self._floating_backgrounds:
                config = self._background_configs.get(background._name)
                background.scale(config._prior_count)
                observed += background.project(0)
            self._observed = observed
        else:  # _data is not None
            self._observed = self._data

        # Set-up debug output
        if kwargs.get("debug") == 1:
            path = echidna.__echidna_base__ + "/debug/"
            filename = self._signal._name + "_debug.pdf"
            debug_pdf = PdfPages(path + filename)
            print "DEBUG: Writing debug file to " + path + filename
        self._signal_config.reset_chi_squareds()
        fig_num = 0
        for signal_count in self._signal_config.get_count():
            for syst_analyser in self._syst_analysers.values():
                syst_analyser._layer = 1  # reset layers
            with utilities.Timer() as t:  # set timer
                self._signal.scale(signal_count)
                self._signal_config.add_chi_squared(
                    self._get_chi_squared(self._floating_backgrounds,
                                          len(self._floating_backgrounds)),
                    signal_count, self._signal.sum())
            if self._verbose:
                print ("Calculations for %.4f signal counts took %.03f "
                       "seconds." % (signal_count, t._interval))
        try:
            return self._signal_config.get_first_bin_above(limit_chi_squared)
        except IndexError as detail:
            raise IndexError("unable to calculate confidence limit - " +
                             str(detail))

    def get_limit_no_float(self, limit_chi_squared=2.71):
        """ Get signal counts at limit.

        Args:
          limit_chi_squared (float, optional): chi squared required for
            limit.

        .. note::

          Default value for :obj:`limit_chi_squared` is 2.71, the chi
          squared value corresponding to a 90% confidence limit.

        Returns:
          float: Signal counts in ROI at required limit

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
        if self._calculator is None:
            raise TypeError("chi squared calculator not set")
        if self._data is None:
            self._observed = self._fixed_background.project(0)
        else:
            self._observed = self._data
        self._signal_config.reset_chi_squareds()
        for signal_count in self._signal_config.get_count():
            with utilities.Timer() as t:  # set timer
                self._signal.scale(signal_count)
                expected = self._observed + self._signal.project(0)
                print 'observed', self._observed.sum(), 'expected', expected.sum()
                self._signal_config.add_chi_squared(
                    self._calculator.get_chi_squared(self._observed, expected),
                    signal_count, self._signal.sum())
            if self._verbose:
                print ("Calculations for %.4f signal counts took %.03f "
                       "seconds." % (signal_count, t._interval))

            if kwargs.get("debug") == 1:
                spectra = {}
                for background in self._backgrounds:
                    spectra[background._name] = {"spectra": background,
                                                 "style": background._style,
                                                 "type": "background",
                                                 "label": background._name}
                spectra[self._signal._name] = {"spectra": self._signal,
                                               "style": self._signal._style,
                                               "type": "signal",
                                               "label": self._signal._name}
                current_chi_squared = numpy.sum(self._calculator.get_chi_squared_per_bin())
                plot_text = ["$\chi^2$ = %.2g" % current_chi_squared]
                plot_text.append("Livetime = %.1f years" % self._signal._time_high)
                title = self._signal._name + " @ %.2g counts" % self._signal._data.sum()
                spectral_plot = plot.spectral_plot(spectra, 0, fig_num,
                                                   per_bin=self._calculator,
                                                   title=title,
                                                   text=plot_text,
                                                   log_y=True)
                debug_pdf.savefig(spectral_plot)
                spectral_plot.clear()
                fig_num += 1
        if kwargs.get("debug") == 1:
            debug_pdf.close()
        for syst_analyser in self._syst_analysers.values():
            print syst_analyser._syst_values
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
            if (current < total_backgrounds-1):
                # Add penalty terms manually
                if config._sigma is not None:
                    penalty_term = {
                            "parameter_value": count - config._prior_count,
                            "sigma": config._sigma
                            }
                    self._calculator.set_penalty_term(name, penalty_term)
                config.add_chi_squared(
                    self._get_chi_squared(self._floating_backgrounds,
                                          len(self._floating_backgrounds),
                                          current),
                    count, background.sum())  # function recursion
            else:
                if self._fixed_background:
                    total_background = self._fixed_background.project(0)
                else:
                    total_background = numpy.zeros(
                        shape=[self._signal._energy_bins],
                        dtype=float)
                for background in self._floating_backgrounds:
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
                            self._observed, expected,
                            penalty_terms=penalty_term),
                        count, background.sum())
                except ValueError as detail:
                    # Either histogram has bins with zero events
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
                        for background in self._floating_backgrounds:
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
                    else:
                        raise
            if syst_analyser is not None:
                penalty_value = self._calculator._current_values[name]
                syst_analyser.add_penalty_value(count, penalty_value)
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


def make_fixed_background(spectra, **kwargs):
    ''' Makes a spectrum for fixed backgrounds. If pre-shrinking spectra to the
      ROI in the LimitSetting class you *must* also pre-shrink here.

    Args:
      spectra (dictionary): Dictionary containing spectra as keys and priors as
        values.

    Keyword arguments include:

      * roi (*tuple*): (energy_lower, energy_upper)
      * pre_shrink (*bool*): If set to True, :meth:`shrink` method is
        called on all spectra before limit setting, shrinking to
        ROI. Only applies if ROI has been set via ``roi`` keyword.

    Returns: Spectrum containing all fixed backgrounds.
    '''
    first = True
    for spectrum, scaling in spectra.iteritems():
        if first:
            first = False
            if kwargs.get("roi") is not None:
                roi = kwargs.get("roi")
                if kwargs.get("pre_shrink"):
                    energy_low, energy_high = roi
                    spectrum.shrink(energy_low, energy_high)
            spectrum.scale(scaling)
            total_spectrum = spectrum
        else:
            if kwargs.get("roi") is not None:
                roi = kwargs.get("roi")
                if kwargs.get("pre_shrink"):
                    energy_low, energy_high = roi
                    spectrum.shrink(energy_low, energy_high)
            spectrum.scale(scaling)
            total_spectrum.add(spectrum)
    return total_spectrum
