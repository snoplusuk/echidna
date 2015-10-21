import unittest
import echidna.core.spectra as spectra
import random
import numpy


class TestSpectra(unittest.TestCase):

    def test_fill(self):
        """ Test the fill method.

        Basically tests bin positioning makes sense.
        """
        test_decays = 10
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        energy_bins = test_spectra.get_config().get_par("energy_mc")._bins
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        radial_bins = test_spectra.get_config().get_par("radial_mc")._bins
        for x in range(0, test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
            x_bin = energy / energy_high * energy_bins
            y_bin = radius / radial_high * radial_bins
            self.assertTrue(test_spectra._data[x_bin, y_bin] > 0)
        # Also test the sum method at the same time
        self.assertTrue(test_spectra.sum() == test_decays,
                        msg="Input decays %s, spectra sum %s"
                        % (test_decays, test_spectra.sum()))
        self.assertRaises(ValueError, test_spectra.fill,
                          energy_mc=energy_low - 1, radial_mc=0)
        self.assertRaises(ValueError, test_spectra.fill, energy_mc=0,
                          radial_mc=radial_low - 1)
        self.assertRaises(ValueError, test_spectra.fill,
                          energy_mc=energy_high + 1, radial_mc=0)
        self.assertRaises(ValueError, test_spectra.fill, energy_mc=0,
                          radial_mc=radial_high + 1)

    def test_project(self):
        """ Test the projection method of the spectra.

        This creates projected spectra alongside the actual spectra.
        """
        test_decays = 10
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy_bins = test_spectra.get_config().get_par("energy_mc")._bins
        radial_bins = test_spectra.get_config().get_par("radial_mc")._bins
        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        energy_projection = numpy.ndarray(shape=(energy_bins), dtype=float)
        energy_projection.fill(0)
        radial_projection = numpy.ndarray(shape=(radial_bins), dtype=float)
        radial_projection.fill(0)
        for x in range(0, test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
            x_bin = energy / energy_high * energy_bins
            y_bin = radius / radial_high * radial_bins
            energy_projection[x_bin] += 1.0
            radial_projection[y_bin] += 1.0
        self.assertTrue(numpy.array_equal(energy_projection,
                                          test_spectra.project("energy_mc")))
        self.assertTrue(numpy.array_equal(radial_projection,
                                          test_spectra.project("radial_mc")))

    def test_scale(self):
        """ Test the scale method of the spectra.

        This creates a spectra and then scales it.
        """
        test_decays = 10  # should be a float
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        for x in range(0, test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
        self.assertTrue(test_spectra.sum() == test_decays)
        count = 150  # int
        test_spectra.scale(count)
        count = 110  # int --> int(110) / int(150) = 0
        test_spectra.scale(count)
        # Check sum != 0.0
        self.assertNotEqual(test_spectra.sum(), 0.0)
        self.assertTrue(test_spectra.sum() == count,
                        msg="Spectra sum: %s, scaling %s"
                        % (test_spectra.sum(), count))

    def test_slicing(self):
        """ Test the slicing shirnks the spectra in the correct way.

        """
        test_decays = 10
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        self.assertRaises(ValueError,
                          test_spectra.shrink,
                          energy_mc_low=energy_low,
                          energy_mc_high=2 * energy_high,
                          radial_mc_low=radial_low,
                          radial_mc_high=radial_high)
        self.assertRaises(ValueError,
                          test_spectra.shrink,
                          energy_mc_low=energy_low,
                          energy_mc_high=energy_high,
                          radial_mc_low=radial_low,
                          radial_mc_high=2 * radial_high)
        test_spectra.shrink(energy_mc_low=energy_low,
                            energy_mc_high=energy_high / 2,
                            radial_mc_low=radial_low,
                            radial_mc_high=radial_high / 2)
        energy_bins = test_spectra.get_config().get_par("energy_mc")._bins
        radial_bins = test_spectra.get_config().get_par("radial_mc")._bins
        self.assertTrue(test_spectra._data.shape == (energy_bins, radial_bins),
                        msg="Spectra shape %s, energy bins %s, radial bins %s"
                        % (test_spectra._data.shape, energy_bins, radial_bins))

    def test_rebin(self):
        """ Tests that the spectra are being rebinned correctly.

        """
        test_decays = 10
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        old_energy_width = test_spectra.get_config().get_par("energy_mc")\
            .get_width()
        old_radial_width = test_spectra.get_config().get_par("radial_mc")\
            .get_width()
        energy_high = test_spectra.get_config().get_par("energy_mc")._high
        radial_high = test_spectra.get_config().get_par("radial_mc")._high
        energy_low = test_spectra.get_config().get_par("energy_mc")._low
        radial_low = test_spectra.get_config().get_par("radial_mc")._low
        for decay in range(test_decays):
            energy = random.uniform(energy_low, energy_high)
            radius = random.uniform(radial_low, radial_high)
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
        new_bins = (1, 2, 3, 4)
        self.assertRaises(ValueError, test_spectra.rebin, new_bins)
        new_bins = (99999, 99999)
        self.assertRaises(ValueError, test_spectra.rebin, new_bins)
        old_sum = test_spectra.sum()
        new_bins = ()
        for par in test_spectra.get_config().get_pars():
            bins = test_spectra.get_config().get_par(par)._bins / 2.
            new_bins += (bins,)
        test_spectra.rebin(new_bins)
        self.assertTrue(old_sum == test_spectra.sum(),
                        msg="Sum pre rebin %s, post bin %s"
                        % (old_sum, test_spectra.sum()))
        new_energy_width = test_spectra.get_config().get_par("energy_mc")\
            .get_width()
        new_radial_width = test_spectra.get_config().get_par("radial_mc")\
            .get_width()
        self.assertTrue(test_spectra._data.shape == new_bins,
                        msg="Spectra shape %s, expected shape %s"
                        % (test_spectra._data.shape, new_bins))
        self.assertTrue(new_energy_width == old_energy_width*2.,
                        msg="New width %s, Expected width %s"
                        % (new_energy_width, old_energy_width))
        self.assertTrue(new_radial_width == old_radial_width*2.,
                        msg="New width %s, Expected width %s"
                        % (new_radial_width, old_radial_width))
