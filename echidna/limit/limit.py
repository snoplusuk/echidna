import numpy
import copy
import time

from echidna.errors.custom_errors import LimitError, CompatibilityError
from echidna.limit import summary
from echidna.output import store


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
      _stats (:class:`numpy.ndarray`): Data container for
        test statistic values.
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

    def get_limit(self, limit=2.71, stat_zero=0., store_summary=True):
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
          store_summary (bool, optional):  If True (default) then a hdf5 file
            is produced containing best fit values for systematics, total
            delta chi-squared and penalty chi_squared of each systematic as a
            function of signal scaling. The prior and sigma values used are
            also stored. A log file is also produced for the values of best
            fits and penalty chi_squared of each systematic,
            total chi_squared, number of degrees of freedom and signal scaling
            at the requested limit.

        Raises:
          LimitError: If all values in the array are below limit.

        Returns:
          float: The signal scaling at the limit you are setting.
        """
        par = self._signal.get_fit_config().get_par("rate")
        # Check zero signal stat in case its not in self._stats
        self._fitter.remove_signal()
        min_stat = self._fitter.fit()
        if store_summary and \
                self._fitter.get_floating_backgrounds() is not None:
            summaries = {}
            scales = par.get_values()
            for par_name in self._fitter.get_fit_config().get_pars():
                summaries[par_name] = summary.Summary(par_name, len(scales))
                summaries[par_name].set_scales(scales)
                cur_par = self._fitter.get_fit_config().get_par(par_name)
                summaries[par_name].set_prior(cur_par.get_prior())
                summaries[par_name].set_sigma(cur_par.get_sigma())
        for i, scale in enumerate(par.get_values()):  # Loop signal scales
            print "signal scale:", scale
            if not numpy.isclose(scale, 0.):
                self._signal.scale(scale)
                self._fitter.set_signal(self._signal, shrink=False)
            else:
                self._fitter.remove_signal()
            stat = self._fitter.fit()
            if not isinstance(stat, float):  # Is per-bin array
                stat = stat.sum()
            self._stats[i] = stat
            if store_summary and \
                    self._fitter.get_floating_backgrounds() is not None:
                summary_results = self._fitter.get_minimiser().get_summary()
                for key in summary_results.keys():
                    summaries[key].set_best_fit(
                        summary_results[key]["best_fit"], i)
                    summaries[key].set_penalty_term(
                        summary_results[key]["penalty_term"], i)
        stat = self._stats.min()
        if stat < min_stat:
            min_stat = stat
        self._stats -= min_stat
        if store_summary and \
                self._fitter.get_floating_backgrounds() is not None:
            timestamp = "%.f" % time.time()  # seconds since epoch
            for name, cur_summary in summaries.iteritems():
                cur_summary.set_stats(self._stats)
                fname = name + "_" + timestamp + ".hdf5"
                store.dump_summary(fname, cur_summary)
                print "Saved summary of %s to file %s" % (name, fname)
        try:
            i_limit = numpy.where(self._stats > limit)[0][0]
            if store_summary and \
                    self._fitter.get_floating_backgrounds() is not None:
                log_name = "limit_" + timestamp + ".log"
                with open(log_name, "w") as fout:
                    fout.write("===== Limit Summary =====\nLimit found at:\n")
                    fout.write("Signal Decays: %s\n"
                               % par.get_values()[i_limit])
                    for name, cur_summary in summaries.iteritems():
                        fout.write("--- systematic: %s ---\n" % name)
                        fout.write("Best fit: %s\n"
                                   % cur_summary.get_best_fit(i_limit))
                        fout.write("Prior: %s\n"
                                   % cur_summary.get_prior())
                        fout.write("Sigma: %s\n"
                                   % cur_summary.get_sigma())
                        fout.write("Penalty term: %s\n"
                                   % cur_summary.get_penalty_term(i_limit))
                    fout.write("----------------------------\n")
                    fout.write("Test statistic: %s\n" % self._stats[i_limit])
                    fout.write("N.D.F.: 1\n")  # Only fit one dof currently
                print "Log saved to %s" % log_name
            return par.get_values()[i_limit]
        except:
            i_limit = numpy.where(self._stats == self._stats.max)[0][0]
            if store_summary and \
                    self._fitter.get_floating_backgrounds() is not None:
                log_name = "limit_" + timestamp + ".log"
                with open(log_name, "w") as fout:
                    fout.write("===== Limit Summary =====\nLimit not found.\n"
                               "Values at max test statistic:\n")
                    fout.write("Signal Decays: %s\n"
                               % par.get_values()[i_limit])
                    for name, cur_summary in summaries.iteritems():
                        fout.write("--- systematic: %s ---\n" % name)
                        fout.write("Best fit: %s\n"
                                   % cur_summary.get_best_fit(i_limit))
                        fout.write("Prior: %s\n"
                                   % cur_summary.get_prior())
                        fout.write("Sigma: %s\n"
                                   % cur_summary.get_sigma())
                        fout.write("Penalty term: %s\n"
                                   % cur_summary.get_penalty_term(i_limit))
                        fout.write("----------------------------\n")
                    fout.write("Max test statistic: %s\n"
                               % self._stats[i_limit])
                    fout.write("N.D.F.: 1\n")  # Only fit one dof currently
                    fout.write("Requested test statistic limit: %s" % limit)
            print "Log saved to %s" % log_name
            raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                             % (self._stats.max, limit))

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
