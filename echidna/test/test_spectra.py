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
        test_spectra = spectra.Spectra("Test", test_decays)
        for x in range(0, test_decays):
            energy = random.uniform(0, test_spectra._energy_high)
            radius = random.uniform(0, test_spectra._radial_high)
            time = random.uniform(0, test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
            x_bin = energy / test_spectra._energy_high * test_spectra._energy_bins
            y_bin = radius / test_spectra._radial_high * test_spectra._radial_bins
            z_bin = time / test_spectra._time_high * test_spectra._time_bins
            self.assertTrue(test_spectra._data[x_bin, y_bin, z_bin] > 0)
        # Also test the sum method at the same time
        self.assertTrue(test_spectra.sum() == test_decays)
        self.assertRaises(ValueError, test_spectra.fill, -1, 0, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, -1, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, 0, -1)
        self.assertRaises(ValueError, test_spectra.fill,
                          test_spectra._energy_high + 1, 0, 0)
        self.assertRaises(ValueError, test_spectra.fill,
                          0, test_spectra._radial_high + 1, 0)
        self.assertRaises(ValueError, test_spectra.fill,
                          0, 0, test_spectra._time_high + 1)

    def test_project(self):
        """ Test the projection method of the spectra.

        This creates projected spectra alongside the actual spectra.
        """
        test_decays = 10
        test_spectra = spectra.Spectra("Test", test_decays)
        energy_projection = numpy.ndarray(shape=(test_spectra._energy_bins),
                                          dtype=float)
        energy_projection.fill(0)
        radial_projection = numpy.ndarray(shape=(test_spectra._radial_bins),
                                          dtype=float)
        radial_projection.fill(0)
        time_projection = numpy.ndarray(shape=(test_spectra._time_bins),
                                        dtype=float)
        time_projection.fill(0)
        for x in range(0, test_decays):
            energy = random.uniform(0, 10.0)
            radius = random.uniform(0, 6000.0)
            time = random.uniform(0, 10.0)
            test_spectra.fill(energy, radius, time)
            x_bin = energy / test_spectra._energy_high * test_spectra._energy_bins
            y_bin = radius / test_spectra._radial_high * test_spectra._radial_bins
            z_bin = time / test_spectra._time_high * test_spectra._time_bins
            energy_projection[x_bin] += 1.0
            radial_projection[y_bin] += 1.0
            time_projection[z_bin] += 1.0
        self.assertTrue(numpy.array_equal(energy_projection,
                                          test_spectra.project(0)))
        self.assertTrue(numpy.array_equal(radial_projection,
                                          test_spectra.project(1)))
        self.assertTrue(numpy.array_equal(time_projection,
                                          test_spectra.project(2)))

    def test_scale(self):
        """ Test the scale method of the spectra.

        This creates a spectra and then scales it.
        """
        test_decays = 10
        test_spectra = spectra.Spectra("Test", test_decays)
        for x in range(0, test_decays):
            energy = random.uniform(test_spectra._energy_low,
                                    test_spectra._energy_high)
            radius = random.uniform(test_spectra._radial_low,
                                    test_spectra._radial_high)
            time = random.uniform(test_spectra._time_low,
                                  test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
        self.assertTrue(test_spectra.sum() == test_decays)
        count = 150
        test_spectra.scale(count)
        self.assertTrue(test_spectra.sum() == count)

    def test_slicing(self):
        """ Test the slicing shirnks the spectra in the correct way.

        """
        test_decays = 10
        test_spectra = spectra.Spectra("Test", test_decays)
        self.assertRaises(ValueError,
                          test_spectra.shrink,
                          test_spectra._energy_low,
                          2 * test_spectra._energy_high,
                          test_spectra._radial_low,
                          test_spectra._radial_high,
                          test_spectra._time_low,
                          test_spectra._time_high)
        self.assertRaises(ValueError,
                          test_spectra.shrink,
                          test_spectra._energy_low,
                          test_spectra._energy_high,
                          test_spectra._radial_low,
                          2 * test_spectra._radial_high,
                          test_spectra._time_low,
                          test_spectra._time_high)
        self.assertRaises(ValueError,
                          test_spectra.shrink,
                          test_spectra._energy_low,
                          test_spectra._energy_high,
                          test_spectra._radial_low,
                          test_spectra._radial_high,
                          test_spectra._time_low,
                          2 * test_spectra._time_high)
        test_spectra.shrink(test_spectra._energy_low,
                            test_spectra._energy_high / 2,
                            test_spectra._radial_low,
                            test_spectra._radial_high / 2,
                            test_spectra._time_low,
                            test_spectra._time_high / 2)
        self.assertTrue(test_spectra._data.shape == (test_spectra._energy_bins,
                                                     test_spectra._radial_bins,
                                                     test_spectra._time_bins))

    def test_rebin(self):
        """ Tests that the spectra are being rebinned correctly.

        """
        test_decays = 10
        test_spectra = spectra.Spectra("Test", test_decays)
        test_spectra._energy_bins = 1000
        test_spectra._radial_bins = 1000
        test_spectra._time_bins = 10
        test_spectra.calc_widths()
        old_energy_width = test_spectra._energy_width
        old_radial_width = test_spectra._radial_width
        old_time_width = test_spectra._time_width
        for decay in range(test_decays):
            energy = random.uniform(test_spectra._energy_low,
                                    test_spectra._energy_high)
            radius = random.uniform(test_spectra._radial_low,
                                    test_spectra._radial_high)
            time = random.uniform(test_spectra._time_low,
                                  test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
        new_bins = (1, 2, 3, 4)
        self.assertRaises(ValueError, test_spectra.rebin, new_bins)
        new_bins = (99999, 99999, 99999)
        self.assertRaises(ValueError, test_spectra.rebin, new_bins)
        old_sum = test_spectra.sum()
        new_bins = (500, 250, 2)
        test_spectra.rebin(new_bins)
        self.assertTrue(old_sum == test_spectra.sum())
        self.assertTrue(test_spectra._data.shape == new_bins)
        self.assertTrue(test_spectra._energy_width == old_energy_width*2.)
        self.assertTrue(test_spectra._radial_width == old_radial_width*4.)
        self.assertTrue(test_spectra._time_width == old_time_width*5.)

    def test_copy(self):
        """ Tests that the spectra are being copied correctly.

        """
        test_decays = 10
        test_spectra = spectra.Spectra("Test", test_decays)
        # Modify spectra to non default values
        test_spectra._energy_bins = 250
        test_spectra._radial_bins = 500
        test_spectra._time_bins = 2
        test_spectra._energy_low = 1.
        test_spectra._energy_high = 11.
        test_spectra._radial_low = 100.
        test_spectra._radial_high = 10100.
        test_spectra._time_low = 1.
        test_spectra._time_high = 11.
        test_spectra._raw_events = 9.
        test_spectra.calc_widths()
        for decay in range(test_decays):
            energy = random.uniform(test_spectra._energy_low,
                                    test_spectra._energy_high)
            radius = random.uniform(test_spectra._radial_low,
                                    test_spectra._radial_high)
            time = random.uniform(test_spectra._time_low,
                                  test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
        new_spectra = test_spectra.copy()
        self.assertTrue(new_spectra._data.all() == test_spectra._data.all())
        self.assertTrue(new_spectra._name == test_spectra._name)
        self.assertTrue(new_spectra._energy_low == test_spectra._energy_low)
        self.assertTrue(new_spectra._energy_high == test_spectra._energy_high)
        self.assertTrue(new_spectra._energy_bins == test_spectra._energy_bins)
        self.assertTrue(new_spectra._energy_width == test_spectra._energy_width)
        self.assertTrue(new_spectra._radial_low == test_spectra._radial_low)
        self.assertTrue(new_spectra._radial_high == test_spectra._radial_high)
        self.assertTrue(new_spectra._radial_bins == test_spectra._radial_bins)
        self.assertTrue(new_spectra._radial_width == test_spectra._radial_width)
        self.assertTrue(new_spectra._time_low == test_spectra._time_low)
        self.assertTrue(new_spectra._time_high == test_spectra._time_high)
        self.assertTrue(new_spectra._time_bins == test_spectra._time_bins)
        self.assertTrue(new_spectra._time_width == test_spectra._time_width)
        self.assertTrue(new_spectra._num_decays == test_spectra._num_decays)
        self.assertTrue(new_spectra._raw_events == test_spectra._raw_events)
        new_spectra2 = test_spectra.copy(name="Copy")
        self.assertTrue(new_spectra2._name != test_spectra._name)
        self.assertTrue(new_spectra2._name == "Copy")
