import unittest
import echidna.core.spectra as spectra
import random
import numpy

class TestSpectra(unittest.TestCase):
    
    def test_fill(self):
        """ Test the fill method.

        Basically tests bin positioning makes sense.
        """
        test_spectra = spectra.Spectra("Test")
        test_points = 10
        for x in range(0, test_points):
            energy = random.uniform(0, test_spectra._energy_high)
            radius = random.uniform(0, test_spectra._radial_high)
            time = random.uniform(0, test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
            x_bin = energy / test_spectra._energy_high * test_spectra._energy_bins
            y_bin = radius / test_spectra._radial_high * test_spectra._radial_bins
            z_bin = time / test_spectra._time_high * test_spectra._time_bins
            self.assertTrue(test_spectra._data[x_bin, y_bin, z_bin] > 0)
        # Also test the sum method at the same time
        self.assertTrue(test_spectra.sum() == test_points)
        self.assertRaises(ValueError, test_spectra.fill, -1, 0, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, -1, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, 0, -1)
        self.assertRaises(ValueError, test_spectra.fill, test_spectra._energy_high + 1, 0, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, test_spectra._radial_high + 1, 0)
        self.assertRaises(ValueError, test_spectra.fill, 0, 0, test_spectra._time_high + 1)

    def test_project(self):
        """ Test the projection method of the spectra.

        This creates projected spectra alongside the actual spectra.
        """
        test_spectra = spectra.Spectra("Test")
        test_points = 10
        energy_projection = numpy.ndarray(shape=(test_spectra._energy_bins), dtype=float)
        energy_projection.fill(0)
        radial_projection = numpy.ndarray(shape=(test_spectra._radial_bins), dtype=float)
        radial_projection.fill(0)
        time_projection = numpy.ndarray(shape=(test_spectra._time_bins), dtype=float)
        time_projection.fill(0)
        for x in range(0, test_points):
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
        self.assertTrue(numpy.array_equal(energy_projection, test_spectra.project(0)))
        self.assertTrue(numpy.array_equal(radial_projection, test_spectra.project(1)))
        self.assertTrue(numpy.array_equal(time_projection, test_spectra.project(2)))

    def test_normalise(self):
        """ Test the normalisation method of the spectra.

        This creates a spectra and then normalises it.
        """
        test_spectra = spectra.Spectra("Test")
        test_points = 10
        for x in range(0, test_points):
            energy = random.uniform(test_spectra._energy_low, 
                                    test_spectra._energy_high)
            radius = random.uniform(test_spectra._radial_low,
                                    test_spectra._radial_high)
            time = random.uniform(test_spectra._time_low,
                                  test_spectra._time_high)
            test_spectra.fill(energy, radius, time)
        self.assertTrue(test_spectra.sum(), test_points)
        count = 150
        test_spectra.normalise(count)
        self.assertTrue(test_spectra.sum(), count)

    def test_slicing(self):
        """ Test the slicing shirnks the spectra in the correct way.
        
        """
        test_spectra = spectra.Spectra("Test")
        self.assertRaises(ValueError, 
                          test_spectra.shrink(test_spectra._energy_low,
                                              2 * test_spectra._energy_high,
                                              test_spectra._radial_low,
                                              test_spectra._radial_high,
                                              test_spectra._time_low,
                                              test_spectra._time_high))
        self.assertRaises(ValueError, 
                          test_spectra.shrink(test_spectra._energy_low,
                                              test_spectra._energy_high,
                                              test_spectra._radial_low,
                                              2 * test_spectra._radial_high,
                                              test_spectra._time_low,
                                              test_spectra._time_high))
        self.assertRaises(ValueError, 
                          test_spectra.shrink(test_spectra._energy_low,
                                              test_spectra._energy_high,
                                              test_spectra._radial_low,
                                              test_spectra._radial_high,
                                              test_spectra._time_low,
                                              2 * test_spectra._time_high))
        test_spectra.shrink(2 * test_spectra._energy_low,
                            test_spectra._energy_high,
                            2 *test_spectra._radial_low,
                            test_spectra._radial_high,
                            2 * test_spectra._time_low,
                            test_spectra._time_high)
        self.assertTrue(test_spectra._data.shape == (test_spectra._energy_bins / 2,
                                                     test_spectra._radial_bins / 2,
                                                     test_spectra._time_bins / 2))
