import numpy

from echidna.core.config import GlobalFitConfig
from echidna.fit.fit_results import LimitResults
import echidna.output as output
from echidna.errors.custom_errors import LimitError, CompatibilityError
from echidna.limit import summary
from echidna.output import store

import logging
import collections
import yaml
import datetime
import json
import copy


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
      _min_stat (float): Minimum test statistic value when limit setting.
      _logger (:class:`logging.Logger`): The output logger.
      _per_bin (bool): If set to True, the values of the test statistic
        over spectral dimensions (per bin) will be stored.
      _fitter (:class:`echidna.limit.fit.Fit`): The fitter used to set a
        a limit with.
      _signal (:class:`echidna.core.spectra.Spectra`): signal spectrum you wish
        to obtain a limit for.
      _limit_results (:class:`echidna.fit.fit_results.LimitResults`): Limit
        results instance to report limit fit results
    """
    def __init__(self, signal, fitter, shrink=True, per_bin=False,
                 store_all=False):
        if ((per_bin and not fitter._per_bin) or
                (not per_bin and fitter._per_bin)):
            raise ValueError("Mismatch in per_bin flags. To use per_bin "
                             "effectively, both Fitter and Limit instances "
                             "should have per_bin enabled.\n fitter: %s\n "
                             "limit: %s" (fitter._per_bin, per_bin))
        self._min_stat = 0.
        self._per_bin = per_bin
        self._logger = logging.getLogger(name="Limit")
        self._fitter = fitter
        self._fitter.check_fit_config(signal)
        self._fitter.set_signal(signal, shrink=shrink)
        self._signal = signal
        parameters = collections.OrderedDict()
        name = signal.get_name() + "_limit_fit_config"
        limit_config = signal.get_fit_config()
        fitter_config = fitter.get_fit_config()
        signal_config = signal.get_config()
        name = signal.get_name() + "_limit_results"
        self._limit_results = LimitResults(fitter_config, signal_config,
                                           limit_config, name)
        self._logger.info("Setting limit with the following parameters:")
        logging.getLogger("extra").info(
            yaml.dump(fitter_config.dump(basic=True)))

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

    def get_limit(self, limit=2.71, stat_zero=None, store_limit=True,
                  store_fits=False, store_spectra=False, limit_fname=None):
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
          store_limit (bool, optional):  If True (default) then a hdf5 file
            containing the :class:`echidna.fit.fit_results.LimitResults`
            object is saved.
          store_fits (bool, optional): If True then the
            :class:`echidna.fit.fit_results.FitResults` objects at each signal
            scale is stored in the
            :class:`echidna.fit.fit_results.LimitResults` object.
            Default is False.
          store_spectra (bool, optional): If True then the spectra used for
            fitting are saved to hdf5. Default is False.
          limit_fname (string): Filename to save the
            `:class:`echidna.fit.fit_results.LimitResults` to.

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
                self._min_stat = numpy.sum(stat_zero)
            else:
                self._min_stat = stat_zero
        else:  # check zero signal stat in case its not in self._stats
            self._fitter.remove_signal()
            fit_stats = self._fitter.fit()
            if type(fit_stats) is tuple:
                fit_stats = fit_stats[0]
            if self._per_bin:
                if not isinstance(fit_stats, numpy.ndarray):
                    raise TypeError("For per_bin enabled, "
                                    "the fit output should be a numpy array")
                self._min_stat = numpy.sum(fit_stats)
            else:
                if not isinstance(fit_stats, float):
                    raise TypeError("per_bin disabled in limit and enabled "
                                    "in fit or test_statistic.")
                self._min_stat = fit_stats
            self._logger.info("Calculated stat_zero: %.4g" % self._min_stat)
            fit_results = self._fitter.get_minimiser()
            if fit_results:
                self._logger.info("Fit summary:")
                logging.getLogger("extra").info(
                    "\n%s\n" % json.dumps(fit_results.get_summary()))

        # Create summary
        scales = par.get_values()

        # Loop through signal scalings
        self._logger.debug("Testing signal scalings:\n\n")
        logging.getLogger("extra").debug(str(par.get_values()))
        for i, scale in enumerate(par.get_values()):
            self._logger.debug("signal scale: %.4g" % scale)
            print "scale", scale
            if not numpy.isclose(scale, 0.):
                if self._fitter.get_signal() is None:
                    self._fitter.set_signal(self._signal, shrink=False)
                self._fitter._signal.scale(scale)
            else:  # want no signal contribution
                self._fitter.remove_signal()
                self._logger.warning(
                    "Removing signal in fit for scale %.4g" % scale)

            fit_stats = self._fitter.fit()
            if type(fit_stats) is tuple:
                fit_stats = fit_stats[0]
            stats[i] = numpy.sum(fit_stats)

            fit_results = copy.deepcopy(self._fitter.get_minimiser())
            if fit_results:
                results_summary = fit_results.get_summary()
                for par_name, value in results_summary.iteritems():
                    self._limit_results.set_best_fit(i, value.get("best_fit"),
                                                     par_name)
                    self._limit_results.set_penalty_term(
                        i, value.get("penalty_term"), par_name)
                if store_fits:
                    self._limit_results.set_fit_result(i, fit_results)

        # Convert stats to delta - subtracting minimum
        stats -= self._min_stat

        # Also want to know index of minimum
        min_bin = numpy.argmin(stats)

        if self._per_bin:
            stats = stats.sum(-1)

        self._limit_results._stats = stats
        stats = self._limit_results.get_full_stats()

        try:
            # Slice from min_bin upwards
            log_text = ""
            i_limit = numpy.where(stats[min_bin:] > limit)[0][0]
            limit = par.get_values()[min_bin + i_limit]
            log_text += "\n===== Limit Summary =====\nLimit found at:\n"
            log_text += "Signal Decays: %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                cur_par = self._fitter.get_fit_config().get_par(parameter)
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             self._limit_results.get_best_fit(i_limit,
                                                              parameter))
                log_text += ("Prior: %.4g\n" %
                             cur_par.get_prior())
                log_text += ("Sigma: %.4g\n" %
                             cur_par.get_sigma())
                log_text += ("Penalty term: %.4g\n" %
                             self._limit_results.get_penalty_term(i_limit,
                                                                  parameter))
            log_text += "----------------------------\n"
            log_text += "Test statistic: %.4f\n" % stats[i_limit]
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s\n" % log_text)

            if store_limit:
                if limit_fname:
                    if limit_fname[-5:] != '.hdf5':
                        limit_fname += '.hdf5'
                else:
                    timestamp = datetime.datetime.now().strftime(
                        "%Y-%m-%d_%H-%M-%S")
                    path = output.__default_save_path__ + "/"
                    fname = (self._limit_results.get_name() + "_" + timestamp +
                             ".hdf5")
                    limit_fname = path + fname
                store.dump_limit_results(limit_fname, self._limit_results)
                self._logger.info("Saved summary of %s to file %s" %
                                  (self._limit_results.get_name(),
                                   limit_fname))
            if store_spectra:
                path = output.__default_save_path__ + "/"
                fname = self._fitter.get_data()._name + "_data.hdf5"
                store.dump(path + fname, self._fitter.get_data(),
                           append=True, group_name="data")
                if self._fitter.get_fixed_background():
                    fname = (self._fitter.get_fixed_background()._name +
                             "_fixed.hdf5")
                    store.dump(path + fname,
                               self._fitter.get_fixed_background(),
                               append=True, group_name="fixed")
                if self._fitter.get_floating_backgrounds():
                    for background in self._fitter.get_floating_backgrounds():
                        fname = background._name + "_float.hdf5"
                        store.dump(path + fname, background, append=True,
                                   group_name=background.get_name())
                fname = self._signal._name + "_signal.hdf5"
                store.dump(path + fname, self._signal, append=True,
                           group_name="signal")

            return limit

        except IndexError as detail:
            # Slice from min_bin upwards
            log_text = ""
            i_limit = numpy.argmax(stats[min_bin:])
            limit = par.get_values()[min_bin + i_limit]
            log_text += "\n===== Limit Summary =====\nNo limit found:\n"
            log_text += "Signal Decays (at max stat): %.4g\n" % limit
            for parameter in self._fitter.get_fit_config().get_pars():
                cur_par = self._fitter.get_fit_config().get_par(parameter)
                log_text += "--- systematic: %s ---\n" % parameter
                log_text += ("Best fit: %4g\n" %
                             self._limit_results.get_best_fit(i_limit,
                                                              parameter))
                log_text += ("Prior: %.4g\n" % cur_par.get_prior())
                log_text += ("Sigma: %.4g\n" % cur_par.get_sigma())
                log_text += ("Penalty term: %.4g\n" %
                             self._limit_results.get_penalty_term(i_limit,
                                                                  parameter))
            log_text += "----------------------------\n"
            log_text += "Test statistic: %.4f\n" % stats[i_limit]
            log_text += "N.D.F.: 1\n"  # Only fit one dof currently
            logging.getLogger("extra").info("\n%s" % log_text)

            if store_limit:
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                path = output.__default_save_path__ + "/"
                fname = (self._limit_results.get_name() + "_" + timestamp +
                         ".hdf5")
                store.dump_limit_results(path + fname, self._limit_results)
                self._logger.info("Saved summary of %s to file %s" %
                                  (self._limit_results.get_name(),
                                   path + fname))
            if store_spectra:
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
                store.dump(path + fname, self._signal, append=True,
                           group_name="signal")
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
