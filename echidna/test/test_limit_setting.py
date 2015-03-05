import unittest

import numpy

import echidna.core.spectra as spectra
import echidna.limit.limit_setting as limit_setting
import echidna.limit.limit_config as limit_config
import echidna.limit.chi_squared as chi_squared
from echidna.errors.custom_errors import CompatibilityError


class TestLimitSetting(unittest.TestCase):

    def setUp(self):
        self._n_events = 1e5
        self._signal = spectra.Spectra("signal", self._n_events)
        self._backgrounds = [spectra.Spectra("bkg_1", self._n_events),
                             spectra.Spectra("bkg_2", self._n_events)]

    def test_init(self):
        """ Test initialisation of the class.

        Checks that the CompatibilityError is raised when necessary.
        """
        # Test energy compatibility
        width_orig = self._backgrounds[0]._energy_width
        self._backgrounds[0]._energy_width *= 0.5
        self.assertRaises(CompatibilityError,
                          limit_setting.LimitSetting,
                          self._signal,
                          self._backgrounds)
        self._backgrounds[0]._energy_width = width_orig

        # Test radial compatibility
        width_orig = self._backgrounds[0]._radial_width
        self._backgrounds[0]._radial_width *= 0.5
        self.assertRaises(CompatibilityError,
                          limit_setting.LimitSetting,
                          self._signal,
                          self._backgrounds)
        self._backgrounds[0]._radial_width = width_orig

        # Test time compatibility
        width_orig = self._backgrounds[0]._time_width
        self._backgrounds[0]._time_width *= 0.5
        self.assertRaises(CompatibilityError,
                          limit_setting.LimitSetting,
                          self._signal,
                          self._backgrounds)
        self._backgrounds[0]._time_width = width_orig

    def test_get_limit(self):
        """ Test main limit setting code
        """
        numpy.random.seed()
        mu = 5.0
        sigma = 1.0
        for energy, radius, time in zip(
                numpy.random.normal(mu, sigma, self._n_events),
                numpy.random.uniform(self._signal._radial_low,
                                     self._signal._radial_high,
                                     self._n_events),
                numpy.random.uniform(self._signal._time_low,
                                     self._signal._time_high,
                                     self._n_events)):
            self._signal.fill(energy, radius, time)
            for background in self._backgrounds:
                background.fill(energy, radius, time)

        # Configure signal
        signal_counts = numpy.arange(1.0, 55.0, 1.0, dtype=float)
        signal_prior = 18.0
        signal_config = limit_config.LimitConfig(signal_prior,
                                                 signal_counts)

        # Configure bkg_1
        # No penalty term to start with so just an array containing one value
        bkg_1_counts = numpy.arange(100.0, 101.0, 1.0, dtype=float)
        bkg_1_prior = 100.0
        bkg_1_config = limit_config.LimitConfig(bkg_1_prior, bkg_1_counts)

        # set chi squared calculator
        calculator = chi_squared.ChiSquared()

        # Test 1: test Exceptions
        limit_setter_1 = limit_setting.LimitSetting(self._signal,
                                                    self._backgrounds[:1])
        self.assertRaises(TypeError, limit_setter_1.get_limit)
        limit_setter_1.configure_signal(signal_config)
        self.assertRaises(KeyError, limit_setter_1.get_limit)
        limit_setter_1.configure_background("bkg_1", bkg_1_config)
        self.assertRaises(TypeError, limit_setter_1.get_limit)
        limit_setter_1.set_calculator(calculator)
        self.assertRaises(ValueError, limit_setter_1.get_limit)

        # Test 2: different limits with and without penalty term
        # Define ROI
        roi = (mu - sigma, mu + sigma)

        limit_setter_2 = limit_setting.LimitSetting(self._signal,
                                                    self._backgrounds[:1],
                                                    roi=roi, pre_shrink=True)
        limit_setter_2.configure_signal(signal_config)
        limit_setter_2.configure_background("bkg_1", bkg_1_config)
        limit_setter_2.set_calculator(calculator)
        limit_2 = limit_setter_2.get_limit()

        # Now try with a penalty term
        # set new config this time with more background counts to cycle through
        bkg_1_penalty_counts = numpy.arange(75.0, 125.0, 0.5, dtype=float)
        bkg_1_sigma = 25.0  # To use in penalty term
        bkg_1_penalty_config = limit_config.LimitConfig(bkg_1_prior,
                                                        bkg_1_penalty_counts,
                                                        bkg_1_sigma)

        limit_setter_2.configure_background("bkg_1", bkg_1_penalty_config)
        limit_2_penalty = limit_setter_2.get_limit()
        self.assertNotEqual(limit_2, limit_2_penalty)

        # Test 3: Try with a second background
        # Configure bkg_2
        # No penalty term so just an array containing one value
        bkg_2_counts = numpy.arange(10.0, 11.0, 1.0, dtype=float)
        bkg_2_prior = 10.0
        bkg_2_config = limit_config.LimitConfig(bkg_2_prior, bkg_2_counts)

        limit_setter_3 = limit_setting.LimitSetting(self._signal,
                                                    self._backgrounds,
                                                    roi=roi, pre_shrink=True)
        limit_setter_3.configure_signal(signal_config)
        limit_setter_3.configure_background("bkg_1", bkg_1_config)
        limit_setter_3.configure_background("bkg_2", bkg_2_config)
        limit_setter_3.set_calculator(calculator)
        limit_3 = limit_setter_3.get_limit()
        self.assertNotEqual(limit_2, limit_3)

        # Check chi squareds have been recorded for each background
        for config in limit_setter_3._background_configs.values():
            chi_squareds = config._chi_squareds
            self.assertTrue(chi_squareds.shape[1] > 0)
