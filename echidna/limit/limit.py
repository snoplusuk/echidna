import numpy

from echidna.core.config import GlobalFitConfig
from echidna.fit.fit_results import FitResults
import echidna.output as output
from echidna.errors.custom_errors import LimitError, CompatibilityError
from echidna.limit import summary
from echidna.output import store

import logging
import collections
import yaml
import datetime
import json


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
      _per_bin (bool): If set to True, the values of the test statistic
        over spectral dimensions (per bin) will be stored.
      _min_per_bin (:class:`numpy.ndarray`): Per bin values for
        ``stat_zero``
      _fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      _signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      _fit_results (:class:`FitResults`): Fit results instance to report
        limit fit results
    """
    def __init__(self, signal, fitter, shrink=True, per_bin=False):
        if ((per_bin and not fitter._per_bin) or
                (not per_bin and fitter._per_bin)):
            raise ValueError("Mismatch in per_bin flags. To use per_bin "
                             "effectively, both Fitter and Limit instances "
                             "should have per_bin enabled.\n fitter: %s\n "
                             "limit: %s" (fitter._per_bin, per_bin))
        self._per_bin = per_bin
        self._min_per_bin = None
        self._logger = logging.getLogger(name="Limit")
        self._fitter = fitter
        self._fitter.check_fit_config(signal)
        self._fitter.set_signal(signal, shrink=shrink)
        self._signal = signal
        parameters = collections.OrderedDict()
        name = signal.get_name() + "_limit_fit_config"
        fit_config = GlobalFitConfig(name, parameters)
        fit_config.add_config(signal.get_fit_config())
        fit_config.add_config(fitter.get_fit_config())
        spectra_config = signal.get_config()
        name = signal.get_name() + "_limit_fit_results"
        self._fit_results = FitResults(fit_config, spectra_config, name)
        self._logger.info("Setting limit with the following parameters:")
        logging.getLogger("extra").info(
            yaml.dump(fit_config.dump(basic=True)))

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
        self._logger.debug("Creating stats array with shape %s" % str(shape))

        if stat_zero:  # If supplied specific stat_zero use this
            self._logger.warning("Overriding stat_zero with supplied value")
            logging.getLogger("extra").warning(" --> %s" % stat_zero)
            if self._per_bin:
                if not isinstance(stat_zero, numpy.ndarray):
                    raise TypeError("For per_bin enabled, "
                                    "stat_zero should be a numpy array")
                self._min_per_bin = stat_zero
                self._min_stat = numpy.sum(stat_zero)
            else:
                self._min_stat = stat_zero
        else:  # check zero signal stat in case its not in self._stats
            self._fitter.remove_signal()
            fit_stats = self._fitter.fit()
            if self._per_bin:
                if not isinstance(fit_stats, numpy.ndarray):
                    raise TypeError("For per_bin enabled, "
                                    "the fit output should be a numpy array")
                self._min_per_bin = fit_stats
                self._min_stat = numpy.sum(fit_stats)
            else:
                if not isinstance(fit_stats, float):
                    raise TypeError("per_bin disabled in limit and enabled "
                                    "in fit or test_statistic.")
                self._min_stat = fit_stats
            self._logger.info("Calculated stat_zero: %.4g" % self._min_stat)
            fit_results = self._fitter.get_fit_results()
            if fit_results:
                self._logger.info("Fit summary:")
                logging.getLogger("extra").info(
                    "\n%s\n" % json.dumps(fit_results.get_summary()))

        # Create summary
        scales = par.get_values()
        summary_name = (self._fitter.get_fit_config().get_name() + "_" +
                        self._signal.get_name())
        self._logger.info(summary_name)
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
        self._logger.debug("Testing signal scalings:\n\n")
        logging.getLogger("extra").debug(str(par.get_values()))
        for i, scale in enumerate(par.get_values()):
            self._logger.debug("signal scale: %.4g" % scale)
            if not numpy.isclose(scale, 0.):
                if self._fitter.get_signal() is None:
                    self._fitter.set_signal(self._signal, shrink=False)
                self._signal.scale(scale)
            else:  # want no signal contribution
                self._fitter.remove_signal()
                self._logger.warning(
                    "Removing signal in fit for scale %.4g" % scale)

            fit_stats = self._fitter.fit()
            stats[i] = numpy.sum(fit_stats)

            fit_results = self._fitter.get_fit_results()  # get results
            if fit_results:
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
                    limit_summary.set_stat(stats[i], i)

                # Update fit_results
                self._fit_results.set_stat(fit_results.get_raw_stats(), i)
                self._fit_results.set_penalty_term(
                    fit_results.get_penalty_terms(), i)

        # Convert stats to delta - subtracting minimum
        stats -= self._min_stat

        # Also want to know index of minimum
        min_bin = numpy.argmin(stats)

        # Now we want the corresponding per_bin values if required
        if self._per_bin:
            if self._min_per_bin is None:
                self._min_per_bin = limit_summary.get_raw_stat(min_bin)
            self._logger.debug("Values of _min_per_bin are:\n")
            logging.getLogger("extra").debug("\n%s\n" % str(self._min_per_bin))
            limit_summary.set_stats(limit_summary.get_raw_stats() -
                                    self._min_per_bin)
            self._fit_results.set_stats(self._fit_results.get_raw_stats() -
                                        self._min_per_bin)

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
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                path = output.__default_save_path__ + "/"
                fname = limit_summary.get_name() + "_" + timestamp + ".hdf5"
                store.dump_summary(path + fname, limit_summary)
                store.dump_fit_results(path + fname, self._fit_results)
                store.dump(path + fname, self._fitter.get_data(),
                           append=True, group_name="data")
                if self._fitter.get_fixed_background():
                    store.dump(path + fname,
                               self._fitter.get_fixed_background(),
                               append=True, group_name="fixed")
                if self._fitter.get_floating_backgrounds():
                    for background in self._fitter.get_floating_backgrounds():
                        store.dump(path + fname, background, append=True,
                                   group_name=background.get_name())
                store.dump(path + fname, self._signal,
                           append=True, group_name="signal")
                self._logger.info("Saved summary of %s to file %s" %
                                  (limit_summary.get_name(), path + fname))

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
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                path = output.__default_save_path__ + "/"
                fname = limit_summary.get_name() + "_" + timestamp + ".hdf5"
                store.dump_summary(path + fname, limit_summary)
                store.dump_fit_results(path + fname, self._fit_results)
                store.dump(path + fname, self._fitter.get_data(),
                           append=True, group_name="data")
                if self._fitter.get_fixed_background() is not None:
                    store.dump(path + fname,
                               self._fitter.get_fixed_background(),
                               append=True, group_name="fixed")
                if self._fitter.get_floating_backgrounds():
                    for background in self._fitter.get_floating_backgrounds():
                        store.dump(path + fname, background, append=True,
                                   group_name=background.get_name())
                store.dump(path + fname, self._signal,
                           append=True, group_name="signal")
                self._logger.info("Saved summary of %s to file %s" %
                                  (limit_summary.get_name(), fname))

            self._logger.error("Recieived: IndexError: %s" % detail)
            raise LimitError("Unable to find limit. Max stat: %s, Limit: %s"
                             % (stats.max(), limit))

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
