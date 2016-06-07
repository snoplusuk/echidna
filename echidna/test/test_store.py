import numpy

from echidna.core.config import (SpectraConfig, SpectraFitConfig,
                                 GlobalFitConfig)
import echidna.core.spectra as spectra
import echidna.output.store as store
from echidna.fit.minimise import GridSearch

import unittest
import random


class TestStore(unittest.TestCase):

    def setUp(self):
        """ Set up before each test.
        """
        # Set up test decays
        self._test_decays = 1000

        # Set up spectra config
        self._spectra_config = SpectraConfig.load_from_file(
            "echidna/config/spectra_example2.yml")

        # Set up global fit config
        self._global_fit_config = GlobalFitConfig.load_from_file(
            "echidna/config/fit_example2.yml",
            sf_filename="echidna/config/spectra_fit_example.yml")

        # Set up spectra fit config
        self._spectra_fit_config = SpectraFitConfig.load_from_file(
            "echidna/config/spectra_fit_example.yml", "Test")

        self._test_spectra = spectra.Spectra(
            "Test", self._test_decays, self._spectra_config,
            fit_config=self._spectra_fit_config, background_name="Test")

    def test_serialisation(self):
        """ Test saving and then reloading a test spectra.
        """
        test_spectra = self._test_spectra

        # Save values
        spectra_config = test_spectra.get_config()
        spectra_pars = spectra_config.get_pars()
        energy_high = spectra_config.get_par("energy_mc").get_high()
        energy_bins = spectra_config.get_par("energy_mc").get_bins()
        energy_low = spectra_config.get_par("energy_mc").get_low()
        radial_high = spectra_config.get_par("radial_mc").get_high()
        radial_bins = spectra_config.get_par("radial_mc").get_bins()
        radial_low = spectra_config.get_par("radial_mc").get_low()
        energy_width = spectra_config.get_par("energy_mc").get_width()
        radial_width = spectra_config.get_par("radial_mc").get_width()

        spectra_fit_config = test_spectra.get_fit_config()
        spectra_fit_pars = spectra_fit_config.get_pars()
        rate_prior = spectra_fit_config.get_par("rate").get_prior()
        rate_sigma = spectra_fit_config.get_par("rate").get_sigma()
        rate_low = spectra_fit_config.get_par("rate").get_low()
        rate_high = spectra_fit_config.get_par("rate").get_high()
        rate_bins = spectra_fit_config.get_par("rate").get_bins()

        # Fill spectrum
        for x in range(0, self._test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)

        # Dump spectrum
        store.dump("test.hdf5", test_spectra)

        # Re-load spectrum
        loaded_spectra = store.load("test.hdf5")

        # Re-load saved values
        spectra_config = loaded_spectra.get_config()
        spectra_pars2 = spectra_config.get_pars()
        energy_high2 = spectra_config.get_par("energy_mc").get_high()
        energy_bins2 = spectra_config.get_par("energy_mc").get_bins()
        energy_low2 = spectra_config.get_par("energy_mc").get_low()
        radial_high2 = spectra_config.get_par("radial_mc").get_high()
        radial_bins2 = spectra_config.get_par("radial_mc").get_bins()
        radial_low2 = spectra_config.get_par("radial_mc").get_low()
        energy_width2 = spectra_config.get_par("energy_mc").get_width()
        radial_width2 = spectra_config.get_par("radial_mc").get_width()

        spectra_fit_config = loaded_spectra.get_fit_config()
        spectra_fit_pars2 = spectra_fit_config.get_pars()
        rate_prior2 = spectra_fit_config.get_par("rate").get_prior()
        rate_sigma2 = spectra_fit_config.get_par("rate").get_sigma()
        rate_low2 = spectra_fit_config.get_par("rate").get_low()
        rate_high2 = spectra_fit_config.get_par("rate").get_high()
        rate_bins2 = spectra_fit_config.get_par("rate").get_bins()

        # Run tests
        self.assertTrue(loaded_spectra.sum() == self._test_decays,
                        msg="Original decays: %.3f, loaded spectra sum %3f"
                        % (float(self._test_decays),
                           float(loaded_spectra.sum())))
        self.assertTrue(numpy.array_equal(self._test_spectra._data,
                                          loaded_spectra._data),
                        msg="Original _data does not match loaded _data")
        self.assertTrue(test_spectra._num_decays == loaded_spectra._num_decays,
                        msg="Original num decays: %.3f, Loaded: %.3f"
                        % (float(test_spectra._num_decays),
                           float(loaded_spectra._num_decays)))

        # Check order of parameters
        self.assertListEqual(spectra_pars, spectra_pars2)
        self.assertListEqual(spectra_fit_pars, spectra_fit_pars2)

        self.assertTrue(energy_low == energy_low2,
                        msg="Original energy low: %.4f, Loaded: %.4f"
                        % (energy_low, energy_low2))
        self.assertTrue(energy_high == energy_high2,
                        msg="Original energy high: %.4f, Loaded: %.4f"
                        % (energy_high, energy_high2))
        self.assertTrue(energy_bins == energy_bins2,
                        msg="Original energy bins: %.4f, Loaded: %.4f"
                        % (energy_bins, energy_bins2))
        self.assertTrue(energy_width == energy_width2,
                        msg="Original energy width: %.4f, Loaded: %.4f"
                        % (energy_width, energy_width2))
        self.assertTrue(radial_low == radial_low2,
                        msg="Original radial low: %.4f, Loaded: %.4f"
                        % (radial_low, radial_low2))
        self.assertTrue(radial_high == radial_high2,
                        msg="Original radial high: %.4f, Loaded: %.4f"
                        % (radial_high, radial_high2))
        self.assertTrue(radial_bins == radial_bins2,
                        msg="Original radial bins: %.4f, Loaded: %.4f"
                        % (radial_bins, radial_bins2))
        self.assertTrue(radial_width == radial_width2,
                        msg="Original radial width: %.4f, Loaded: %.4f"
                        % (radial_width, radial_width2))
        self.assertTrue(rate_prior == rate_prior2,
                        msg="Original rate prior: %.4f, Loaded: %.4f"
                        % (rate_prior, rate_prior2))
        self.assertTrue(rate_sigma == rate_sigma2,
                        msg="Original rate sigma: %.4f, Loaded: %.4f"
                        % (rate_sigma, rate_sigma2))
        self.assertTrue(rate_low == rate_low2,
                        msg="Original rate low: %.4f, Loaded: %.4f"
                        % (rate_low, rate_low2))
        self.assertTrue(rate_high == rate_high2,
                        msg="Original rate high: %.4f, Loaded: %.4f"
                        % (rate_high, rate_high2))
        self.assertTrue(rate_bins == rate_bins2,
                        msg="Original rate bins: %.4f, Loaded: %.4f"
                        % (rate_bins, rate_bins2))

    def test_serialisation_grid_search(self):
        """ Test saving and then reloading :class:`GridSearch`
        """
        fit_results = GridSearch(self._global_fit_config,
                                 self._spectra_config,
                                 name="test_fit_results",
                                 per_bin=True)

        # Fill with some random data
        shape = fit_results.get_fit_config().get_shape()
        penalty_terms = numpy.random.uniform(high=10, size=shape)
        fit_results.set_penalty_terms(penalty_terms)

        shape += fit_results._spectra_config.get_shape()
        stats = numpy.random.uniform(high=100, size=shape)
        fit_results.set_stats(stats)

        # Save values
        spectra_config = fit_results.get_spectra_config()
        spectra_pars = spectra_config.get_pars()
        energy_high = spectra_config.get_par("energy_mc").get_high()
        energy_bins = spectra_config.get_par("energy_mc").get_bins()
        energy_low = spectra_config.get_par("energy_mc").get_low()
        radial_high = spectra_config.get_par("radial_mc").get_high()
        radial_bins = spectra_config.get_par("radial_mc").get_bins()
        radial_low = spectra_config.get_par("radial_mc").get_low()
        energy_width = spectra_config.get_par("energy_mc").get_width()
        radial_width = spectra_config.get_par("radial_mc").get_width()

        fit_config = fit_results.get_fit_config()
        fit_pars = fit_config.get_pars()
        rate_prior = fit_config.get_par("rate").get_prior()
        rate_sigma = fit_config.get_par("rate").get_sigma()
        rate_low = fit_config.get_par("rate").get_low()
        rate_high = fit_config.get_par("rate").get_high()
        rate_bins = fit_config.get_par("rate").get_bins()

        # Dump fit results
        store.dump_fit_results("fit_results.hdf5", fit_results)

        # Re-load fit_results
        loaded = store.load_fit_results("fit_results.hdf5")

        # Re-load saved values
        spectra_config = loaded.get_spectra_config()
        spectra_pars2 = spectra_config.get_pars()
        energy_high2 = spectra_config.get_par("energy_mc").get_high()
        energy_bins2 = spectra_config.get_par("energy_mc").get_bins()
        energy_low2 = spectra_config.get_par("energy_mc").get_low()
        radial_high2 = spectra_config.get_par("radial_mc").get_high()
        radial_bins2 = spectra_config.get_par("radial_mc").get_bins()
        radial_low2 = spectra_config.get_par("radial_mc").get_low()
        energy_width2 = spectra_config.get_par("energy_mc").get_width()
        radial_width2 = spectra_config.get_par("radial_mc").get_width()

        fit_config = loaded.get_fit_config()
        fit_pars2 = fit_config.get_pars()
        rate_prior2 = fit_config.get_par("rate").get_prior()
        rate_sigma2 = fit_config.get_par("rate").get_sigma()
        rate_low2 = fit_config.get_par("rate").get_low()
        rate_high2 = fit_config.get_par("rate").get_high()
        rate_bins2 = fit_config.get_par("rate").get_bins()

        # Run tests
        self.assertTrue(numpy.array_equal(fit_results.get_raw_stats(),
                                          loaded.get_raw_stats()),
                        msg="Original _stats does not match loaded _stats")
        for par in fit_results._fit_config.get_pars():
            self.assertTrue(
                numpy.array_equal(fit_results.get_penalty_terms(par),
                                  loaded.get_penalty_terms(par)),
                msg="Original _penalty_terms "
                "does not match loaded _penlty_terms")

        # Check order of parameters
        self.assertListEqual(spectra_pars, spectra_pars2)
        self.assertListEqual(fit_pars, fit_pars2)

        self.assertTrue(energy_low == energy_low2,
                        msg="Original energy low: %.4f, Loaded: %.4f"
                        % (energy_low, energy_low2))
        self.assertTrue(energy_high == energy_high2,
                        msg="Original energy high: %.4f, Loaded: %.4f"
                        % (energy_high, energy_high2))
        self.assertTrue(energy_bins == energy_bins2,
                        msg="Original energy bins: %.4f, Loaded: %.4f"
                        % (energy_bins, energy_bins2))
        self.assertTrue(energy_width == energy_width2,
                        msg="Original energy width: %.4f, Loaded: %.4f"
                        % (energy_width, energy_width2))
        self.assertTrue(radial_low == radial_low2,
                        msg="Original radial low: %.4f, Loaded: %.4f"
                        % (radial_low, radial_low2))
        self.assertTrue(radial_high == radial_high2,
                        msg="Original radial high: %.4f, Loaded: %.4f"
                        % (radial_high, radial_high2))
        self.assertTrue(radial_bins == radial_bins2,
                        msg="Original radial bins: %.4f, Loaded: %.4f"
                        % (radial_bins, radial_bins2))
        self.assertTrue(radial_width == radial_width2,
                        msg="Original radial width: %.4f, Loaded: %.4f"
                        % (radial_width, radial_width2))
        self.assertTrue(rate_prior == rate_prior2,
                        msg="Original rate prior: %.4f, Loaded: %.4f"
                        % (rate_prior, rate_prior2))
        self.assertTrue(rate_sigma == rate_sigma2,
                        msg="Original rate sigma: %.4f, Loaded: %.4f"
                        % (rate_sigma, rate_sigma2))
        self.assertTrue(rate_low == rate_low2,
                        msg="Original rate low: %.4f, Loaded: %.4f"
                        % (rate_low, rate_low2))
        self.assertTrue(rate_high == rate_high2,
                        msg="Original rate high: %.4f, Loaded: %.4f"
                        % (rate_high, rate_high2))
        self.assertTrue(rate_bins == rate_bins2,
                        msg="Original rate bins: %.4f, Loaded: %.4f"
                        % (rate_bins, rate_bins2))

if __name__ == '__main__':
    unittest.main()
