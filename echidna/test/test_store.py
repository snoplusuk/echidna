import unittest
import echidna.core.spectra as spectra
import echidna.output.store as store
import random
import numpy


class TestStore(unittest.TestCase):

    def test_serialisation(self):
        """ Test saving and then reloading a test spectra.

        """
        test_decays = 10
        config_path = "echidna/config/spectra_example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        fit_config_path = "echidna/config/spectra_fit_example.yml"
        fit_config = spectra.SpectraFitConfig.load_from_file(fit_config_path,
                                                             "Test")

        test_spectra = spectra.Spectra("Test", test_decays, config,
                                       fit_config=fit_config)

        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        energy_bins = test_spectra.get_config().get_par("energy_mc")._bins
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        radial_bins = test_spectra.get_config().get_par("radial_mc")._bins
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        energy_width = test_spectra.get_config().get_par("energy_mc").\
            get_width()
        radial_width = test_spectra.get_config().get_par("radial_mc").\
            get_width()

        rate_prior = test_spectra.get_fit_config().get_par("rate").get_prior()
        rate_sigma = test_spectra.get_fit_config().get_par("rate").get_sigma()
        rate_low = test_spectra.get_fit_config().get_par("rate").get_low()
        rate_high = test_spectra.get_fit_config().get_par("rate").get_high()
        rate_bins = test_spectra.get_fit_config().get_par("rate").get_bins()

        for x in range(0, test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)

        store.dump("test.hdf5", test_spectra)
        loaded_spectra = store.load("test.hdf5")

        energy_high2 = loaded_spectra.get_config().get_par("energy_mc")._high
        energy_bins2 = loaded_spectra.get_config().get_par("energy_mc")._bins
        energy_low2 = loaded_spectra.get_config().get_par("energy_mc")._low
        radial_high2 = loaded_spectra.get_config().get_par("radial_mc")._high
        radial_bins2 = loaded_spectra.get_config().get_par("radial_mc")._bins
        radial_low2 = loaded_spectra.get_config().get_par("radial_mc")._low
        energy_width2 = loaded_spectra.get_config().get_par("energy_mc").\
            get_width()
        radial_width2 = loaded_spectra.get_config().get_par("radial_mc").\
            get_width()

        rate_prior2 = test_spectra.get_fit_config().get_par("rate").get_prior()
        rate_sigma2 = test_spectra.get_fit_config().get_par("rate").get_sigma()
        rate_low2 = test_spectra.get_fit_config().get_par("rate").get_low()
        rate_high2 = test_spectra.get_fit_config().get_par("rate").get_high()
        rate_bins2 = test_spectra.get_fit_config().get_par("rate").get_bins()

        self.assertTrue(loaded_spectra.sum() == test_decays,
                        msg="Input decays: %s, loaded spectra sum %s"
                        % (test_decays, loaded_spectra.sum()))
        self.assertTrue(numpy.array_equal(test_spectra._data,
                                          loaded_spectra._data))
        self.assertTrue(energy_low == energy_low2,
                        msg="Original energy low: %s, Loaded: %s"
                        % (energy_low, energy_low2))
        self.assertTrue(energy_high == energy_high2,
                        msg="Original energy high: %s, Loaded: %s"
                        % (energy_high, energy_high2))
        self.assertTrue(energy_bins == energy_bins2,
                        msg="Original energy bins: %s, Loaded: %s"
                        % (energy_bins, energy_bins2))
        self.assertTrue(energy_width == energy_width2,
                        msg="Original energy width: %s, Loaded: %s"
                        % (energy_width, energy_width2))
        self.assertTrue(radial_low == radial_low2,
                        msg="Original radial low: %s, Loaded: %s"
                        % (radial_low, radial_low2))
        self.assertTrue(radial_high == radial_high2,
                        msg="Original radial high: %s, Loaded: %s"
                        % (radial_high, radial_high2))
        self.assertTrue(radial_bins == radial_bins2,
                        msg="Original radial bins: %s, Loaded: %s"
                        % (radial_bins, radial_bins2))
        self.assertTrue(radial_width == radial_width2,
                        msg="Original radial width: %s, Loaded: %s"
                        % (radial_width, radial_width2))
        self.assertTrue(rate_prior == rate_prior2,
                        msg="Original rate prior: %s, Loaded: %s"
                        % (rate_prior, rate_prior2))
        self.assertTrue(rate_sigma == rate_sigma2,
                        msg="Original rate sigma: %s, Loaded: %s"
                        % (rate_sigma, rate_sigma2))
        self.assertTrue(rate_low == rate_low2,
                        msg="Original rate low: %s, Loaded: %s"
                        % (rate_low, rate_low2))
        self.assertTrue(rate_high == rate_high2,
                        msg="Original rate high: %s, Loaded: %s"
                        % (rate_high, rate_high2))
        self.assertTrue(rate_bins == rate_bins2,
                        msg="Original rate bins: %s, Loaded: %s"
                        % (rate_bins, rate_bins2))
        self.assertTrue(test_spectra._num_decays == loaded_spectra._num_decays,
                        msg="Original num decays: %s, Loaded: %s"
                        % (test_spectra._num_decays,
                           loaded_spectra._num_decays))
