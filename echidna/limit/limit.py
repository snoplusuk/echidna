import numpy

from echidna.errors.custom_errors import LimitError, CompatibilityError
from echidna.limit import summary
from echidna.output import store

import logging
import time


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
      per_bin (bool, optional): If set to True, the values of the test
        statistic over spectral dimensions (per bin) will be stored.

    Attributes:
      _signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      _fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      _stats (:class:`numpy.ndarray`): Data container for test
        statistic values.
      _per_bin (bool): If set to True, the values of the test statistic
        over spectral dimensions (per bin) will be stored.
    """
    def __init__(self, signal, fitter, shrink=True, per_bin=False):
        if ((per_bin and not fitter._per_bin) or
                (not per_bin and fitter._per_bin)):
            raise ValueError("Mismatch in per_bin flags. To use per_bin "
                             "effectively, both Fitter and Limit instances "
                             "should have per_bin enabled.\n fitter: %s\n "
                             "limit: %s" (fitter._per_bin, per_bin))
        self._per_bin = per_bin
        self._logger = logging.getLogger(name="Limit")
        self._fitter = fitter
        self._fitter.check_fit_config(signal)
        self._fitter.set_signal(signal, shrink=shrink)
        self._signal = signal

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

    def get_limit(self, limit=2.71, stat_zero=None, store_summary=True):
        """ Get the limit using the signal spectrum.

        Args:
          limit (float, optional): The value of the test statisic which
            corresponds to the limit you want to set. The default is 2.71
            which corresponds to 90% CL when using a chi-squared test
            statistic.
          stat_zero (float or :class:`numpy.ndarray`, optional): Enables
            calculation of e.g. delta chi-squared. Include values of
            test statistic for zero signal contribution, so these can be
            subtracted from the values of the test statistic, with signal.
          store_summary (bool, optional):  If True (default) then a hdf5 file
            is produced containing best fit values for systematics, total
            delta chi-squared and penalty chi_squared of each systematic as a
            function of signal scaling. The prior and sigma values used are
            also stored. A log file is also produced for the values of best
            fits and penalty chi_squared of each systematic,
            total chi_squared, number of degrees of freedom and signal scaling
            at the requested limit.

        Raises:
          TypeError: If stat_zero is not a numpy array, when per_bin is
            enabled.
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        par = self._signal.get_fit_config().get_par("rate")

        # Create stats array
        shape = self._signal.get_fit_config().get_shape()
        stats = numpy.zeros(shape, dtype=numpy.float64)

        if stat_zero:  # If supplied specific stat_zero use this
            if self._per_bin:
                if not isinstance(stat_zero, numpy.ndarray):
                    raise TypeError("For per_bin enabled, "
                                    "stat_zero should be a numpy array")
                min_per_bin = stat_zero
                min_stat = numpy.sum(stat_zero)
            else:
                min_per_bin = None
                min_stat = stat_zero
        else:  # check zero signal stat in case its not in self._stats
            self._fitter.remove_signal()
            min_stat = self._fitter.fit()
            fit_results = self._fitter.get_fit_results()
            minimum_position = fit_results.get_minimum_position()
            # Get per_bin array getting stats at minimum position
            min_per_bin = fit_results.get_stat(minimum_position)

        # Create summary if required
        if store_summary and self._fitter.get_fit_config() is not None:
            scales = par.get_values()
            summary_name = self._fitter.get_fit_config().get_name()
            if self._per_bin:  # want full Summary class
                limit_summary = summary.Summary(
                    summary_name, len(scales),
                    spectra_config=self._signal.get_config(),
                    fit_config=self._fitter.get_fit_config())
            else:  # use ReducedSummary
                limit_summary = summary.ReducedSummary(
                    summary_name, len(scales),
                    spectra_config=self._signal.get_config(),
                    fit_config=self._fitter.get_fit_config())
            limit_summary.set_scales(scales)

            # Set prior and sigma values
            for par_name in self._fitter.get_fit_config().get_pars():
                cur_par = self._fitter.get_fit_config().get_par(par_name)
                limit_summary.set_prior(cur_par.get_prior(), par_name)
                limit_summary.set_sigma(cur_par.get_sigma(), par_name)

        # Loop through signal scalings
        for i, scale in enumerate(par.get_values()):
            self._logger.debug("signal scale: %.4g" % scale)
            if not numpy.isclose(scale, 0.):
                self._signal.scale(scale)
                self._fitter.set_signal(self._signal, shrink=False)
            else:
                self._fitter.remove_signal()
            stat = self._fitter.fit()  # best-fit test statistic for this scale
            stats[i] = stat

            if store_summary and self._fitter.get_fit_config() is not None:
                fit_results = self._fitter.get_fit_results()  # get results
                results_summary = fit_results.get_summary()
                for par_name, value in results_summary.iteritems():
                    limit_summary.set_best_fit(value.get("best_fit"),
                                               i, par_name)
                    limit_summary.set_penalty_term(value.get("penalty_term"),
                                                   i, par_name)
                if self._per_bin:
                    minimum_position = fit_results.get_minimum_position()
                    # Get per_bin array getting stats at minimum position
                    min_per_bin = fit_results.get_raw_stat(minimum_position)
                    limit_summary.set_stat(min_per_bin, i)
                else:  # just use single stat
                    limit_summary.set_stat(stat, i)

        # Find array minimum - use whichever is largest out of array min and
        # previously calculated min_stat
        if stats.min() > min_stat:
            min_stat = stats.min()
            if self._per_bin:
                # Now we want the corresponding per_bin values
                min_per_bin = limit_summary.get_raw_stat(stats.argmin())

        # Convert stats to delta - subtracting minimum
        stats -= min_stat
        limit_summary.set_stats(limit_summary.get_raw_stats() - min_per_bin)

        # Also want to know index of minimum
        min_bin = numpy.argmin(stats)

        try:
            # Slice from min_bin upwards
            log_text = ""
            i_limit = numpy.where(stats[min_bin:] > limit)[0][0]
            limit = par.get_values()[min_bin + i_limit]
            limit_summary.set_limit(limit)
            limit_summary.set_limit_idx(min_bin + i_limit)
            log_text += "\n===== Limit Summary =====\nLimit found at:\n"
            log_text += "Signal Decays: %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             limit_summary.get_best_fit(i_limit, parameter))
                log_text += ("Prior: %.4g\n" %
                             limit_summary.get_prior(parameter))
                log_text += ("Sigma: %.4g\n" %
                             limit_summary.get_sigma(parameter))
                log_text += ("Penalty term: %.4g\n" %
                             limit_summary.get_penalty_term(i_limit,
                                                            parameter))
            log_text += "----------------------------\n"
            log_text += "Test statistic: %.4f\n" % stats[i_limit]
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s\n" % log_text)

            if (store_summary and self._fitter.get_fit_config() is not None):
                timestamp = "%.f" % time.time()  # seconds since epoch
                fname = limit_summary.get_name() + "_" + timestamp + ".hdf5"
                store.dump_summary(fname, limit_summary)
                store.dump(fname, self._fitter.get_data(),
                           append=True, group_name="data")
                store.dump(fname, self._fitter.get_fixed_background(),
                           append=True, group_name="fixed")
                for background in self._fitter.get_floating_backgrounds():
                    store.dump(fname, background, append=True,
                               group_name=background.get_name())
                store.dump(fname, self._signal,
                           append=True, group_name="signal")
                self._logger.info("Saved summary of %s to file %s" %
                                  (limit_summary.get_name(), fname))

            return limit

        except IndexError as detail:
            # Slice from min_bin upwards
            log_text = ""
            i_limit = numpy.argmax(stats[min_bin:])
            limit = par.get_values()[min_bin + i_limit]
            log_text += "\n===== Limit Summary =====\nNo limit found:\n"
            log_text += "Signal Decays (at max stat): %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             limit_summary.get_best_fit(i_limit, parameter))
                log_text += ("Prior: %.4g\n" %
                             limit_summary.get_prior(parameter))
                log_text += ("Sigma: %.4g\n" %
                             limit_summary.get_sigma(parameter))
                log_text += ("Penalty term: %.4g\n" %
                             limit_summary.get_penalty_term(i_limit,
                                                            parameter))
            log_text += "----------------------------\n"
            log_text += "Test statistic: %.4f\n" % stats[i_limit]
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s" % log_text)

            if (store_summary and self._fitter.get_fit_config() is not None):
                timestamp = "%.f" % time.time()  # seconds since epoch
                fname = limit_summary.get_name() + "_" + timestamp + ".hdf5"
                store.dump_summary(fname, limit_summary)
                store.dump(fname, self._fitter.get_data(),
                           append=True, group_name="data")
                store.dump(fname, self._fitter.get_fixed_background(),
                           append=True, group_name="fixed")
                for background in self._fitter.get_floating_backgrounds():
                    store.dump(fname, background, append=True,
                               group_name=background.get_name())
                store.dump(fname, self._signal,
                           append=True, group_name="signal")
                self._logger.info("Saved summary of %s to file %s" %
                                  (limit_summary.get_name(), fname))

            self._logger.error("Recieived: IndexError: %s" % detail)
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
