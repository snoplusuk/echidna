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
        test_spectra = spectra.Spectra("Test", test_decays)
        for x in range(0, test_decays):
            energy = random.uniform(0, test_spectra._energy_high)
            radius = random.uniform(0, test_spectra._radial_high)
            time = random.uniform(0, test_spectra._time_high)
            test_spectra.fill(energy, radius, time)

        store.dump("test.hdf5", test_spectra)
        loaded_spectra = store.load("test.hdf5")
        self.assertTrue(loaded_spectra.sum() == test_decays)
        self.assertTrue(numpy.array_equal(test_spectra._data, loaded_spectra._data))
        self.assertTrue(test_spectra._energy_low == loaded_spectra._energy_low)
        self.assertTrue(test_spectra._energy_high == loaded_spectra._energy_high)
        self.assertTrue(test_spectra._energy_bins == loaded_spectra._energy_bins)
        self.assertTrue(test_spectra._energy_width == loaded_spectra._energy_width)
        self.assertTrue(test_spectra._radial_low == loaded_spectra._radial_low)
        self.assertTrue(test_spectra._radial_high == loaded_spectra._radial_high)
        self.assertTrue(test_spectra._radial_bins == loaded_spectra._radial_bins)
        self.assertTrue(test_spectra._radial_width == loaded_spectra._radial_width)
        self.assertTrue(test_spectra._time_low == loaded_spectra._time_low)
        self.assertTrue(test_spectra._time_high == loaded_spectra._time_high)
        self.assertTrue(test_spectra._time_bins == loaded_spectra._time_bins)
        self.assertTrue(test_spectra._time_width == loaded_spectra._time_width)
        self.assertTrue(test_spectra._num_decays == loaded_spectra._num_decays)
