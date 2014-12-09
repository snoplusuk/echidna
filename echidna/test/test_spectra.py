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
            energy = random.randrange(0, spectra.Spectra._energy_high)
            radius = random.randrange(0, spectra.Spectra._radial_high)
            time = random.randrange(0, spectra.Spectra._time_high)
            test_spectra.fill(energy, radius, time)
            x_bin = energy / spectra.Spectra._energy_high * spectra.Spectra._energy_bins
            y_bin = radius / spectra.Spectra._radial_high * spectra.Spectra._radial_bins
            z_bin = time / spectra.Spectra._time_high * spectra.Spectra._time_bins
            self.assertTrue(test_spectra._data[x_bin, y_bin, z_bin] > 0)
        # Also test the sum method at the same time
        self.assertTrue(test_spectra.sum() == test_points)

    def test_project(self):
        """ Test the projection method of the spectra.

        This creates projected spectra alongside the actual spectra.
        """
        test_spectra = spectra.Spectra("Test")
        test_points = 10
        energy_projection = numpy.ndarray(shape=(spectra.Spectra._energy_bins), dtype=float)
        energy_projection.fill(0)
        radial_projection = numpy.ndarray(shape=(spectra.Spectra._radial_bins), dtype=float)
        radial_projection.fill(0)
        time_projection = numpy.ndarray(shape=(spectra.Spectra._time_bins), dtype=float)
        time_projection.fill(0)
        for x in range(0, test_points):
            energy = random.randrange(0, 10.0)
            radius = random.randrange(0, 6000.0)
            time = random.randrange(0, 10.0)
            test_spectra.fill(energy, radius, time)
            x_bin = energy / spectra.Spectra._energy_high * spectra.Spectra._energy_bins
            y_bin = radius / spectra.Spectra._radial_high * spectra.Spectra._radial_bins
            z_bin = time / spectra.Spectra._time_high * spectra.Spectra._time_bins
            energy_projection[x_bin] += 1.0
            radial_projection[y_bin] += 1.0
            time_projection[z_bin] += 1.0
        self.assertTrue(numpy.array_equal(energy_projection, test_spectra.project(0)))
        self.assertTrue(numpy.array_equal(radial_projection, test_spectra.project(1)))
        self.assertTrue(numpy.array_equal(time_projection, test_spectra.project(2)))
