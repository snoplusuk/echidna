import numpy

from echidna.errors.custom_errors import LimitError, CompatibilityError


class Limit(object):
    """ Class to handle main limit setting.

    Args:
      signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      shrink (bool, optional): If set to True, :meth:`shrink` method is
        called on the signal spectrum before limit setting, shrinking to
        ROI.

    Attributes:
      _signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      _fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
    """
    def __init__(self, signal, fitter, shrink=True):
        self._fitter = fitter
        self._fitter.check_fit_config(signal)
        self._fitter.set_signal(signal, shrink=shrink)
        self._signal = signal
        self._stats = numpy.zeros(
            self._signal.get_fit_config().get_par("rate")._bins,
            dtype=numpy.float64)

    def get_array_limit(self, array, limit=2.71):
        """ Get the limit from an array containing statisics

        Args:
          array (:class:`numpy.array`): The array you want to set a limit for.
          limit (float, optional): The value of the test statisic which
            corresponds to the limit you want to set. The default is 2.71
            which corresponds to 90% CL when using a chi-squared test
            statistic.

        Raises:
          CompatibilityError: If the length of the array is not equal to the
            number of signal scalings.
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        counts = self._signal.get_fit_config().get_par("rate").get_values()
        if len(counts) != len(array):
            raise CompatibilityError("Array length and number of signal "
                                     "scalings is different.")
        i = 0
        if not isinstance(array[0], float):  # is array
            array = self.sum_entries(array)
        for entry in array:
            if entry > limit:
                return counts[i]
            i += 1
        raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                         % (array[-1], limit))

    def get_limit(self, limit=2.71, stat_zero=None):
        """ Get the limit using the signal spectrum.

        Args:
          limit (float, optional): The value of the test statisic which
            corresponds to the limit you want to set. The default is 2.71
            which corresponds to 90% CL when using a chi-squared test
            statistic.
          stat_zero (float, optional): Enables calculation of e.g. delta
            chi-squared. Include value of test statistic for zero signal
            contribution, so this can be subtracted from the value of
            the test statistic, with signal.

        Raises:
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        par = self._signal.get_fit_config().get_par("rate")
        for i, scale in enumerate(par.get_values()):  # Loop signal scales
            if not numpy.isclose(scale, 0.):
                self._signal.scale(scale)
                self._fitter.set_signal(self._signal, shrink=False)
            else:
                self._fitter.remove_signal()
            stat = self._fitter.fit()
            if isinstance(stat, numpy.ndarray):  # Is per-bin array
                stat = numpy.sum(stat)
            self._stats[i] = stat
        if stat_zero:  # If supplied specific stat_zero use this
            min_stat = stat_zero
        else:
            # Find array minimum and fit for no signal - use whichever is
            # smallest
            min_stat = self._stats.min()
            # Check zero signal stat in case its not in self._stats
            self._fitter.remove_signal()
            stat = self._fitter.fit()
            if isinstance(stat, numpy.ndarray):  # Is per-bin array
                stat = numpy.sum(stat)
            if stat < min_stat:
                min_stat = stat

        # Also want to know index of minimum
        self._stats -= min_stat
        min_bin = numpy.argmin(self._stats)
        try:
            # Slice from min_bin upwards
            i_limit = numpy.where(self._stats[min_bin:] > limit)[0][0]
            return par.get_values()[min_bin+i_limit]
        except IndexError:
            raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                             % (self._stats.max(), limit))

    def get_statistics(self):
        """ Get the test statistics for all signal scalings.

        Returns:
          :class:`numpy.array`: Of test statisics for all signal scalings.
        """
        signal_config = self._signal.get_fit_config()
        stats = []
        for scale in signal_config.get_par("rate").get_values():
            if not numpy.isclose(scale, 0.):
                self._signal.scale(scale)
                self._fitter.set_signal(self._signal, shrink=False)
            else:
                self._fitter.remove_signal()
            stats.append(self._fitter.fit())
        return numpy.array(stats)

    def sum_entries(self, array):
        """ Sums entries of an array which contains arrays as entries.

        Args:
          array (:class:`numpy.array`): The array you want to sum the
            elements of.

        Returns:
          :class:`numpy.array`: The input array with its entries summed.
        """
        new_array = []
        for entry in array:
            new_array.append(entry.sum())
        return numpy.array(new_array)
