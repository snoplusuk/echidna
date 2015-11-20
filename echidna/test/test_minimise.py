""" Tests minimise module
"""
import numpy

from echidna.limit.minimise import GridSearch
from echidna.core.spectra import GlobalFitConfig, FitParameter

from collections import OrderedDict
import unittest


def funct(*args):
    """ Function to test minimisation.

    Best fit values should be (4.0, -2.5, 0.5)
    """
    return (args[0] - 4.)**2 + (args[1] + 2.5)**2 + (args[2] - 0.5)**2


class TestGridSearch(unittest.TestCase):
    """ Tests for the class :class:`echidna.limit.minimise.GridSearch`.
    """
    def setUp(self):
        # Cannot load from file, because not using standard echidna parameters
        parameters = OrderedDict({})
        fit_config = GlobalFitConfig(parameters)
        x = FitParameter(name="x", prior=4.0, sigma=0.1,
                         low=3.0, high=5.0, bins=21)
        fit_config.add_par(x, "global")
        y = FitParameter(name="y", prior=-2.5, sigma=0.1,
                         low=-5.0, high=0.0, bins=51)
        fit_config.add_par(y, "global")
        z = FitParameter(name="z", prior=0.5, sigma=0.1,
                         low=0.0, high=1.0, bins=11)
        fit_config.add_par(z, "global")

        # Initialise default GridSearch
        self._default_grid_search = GridSearch(fit_config,
                                               "default_grid_search")
        # Initialise GridSearch using find_minimum
        self._grid_search = GridSearch(fit_config,
                                       "grid_search",
                                       use_numpy=True)

    def test_find_minimum(self):
        """ Test the :meth:`GridSearch.find_minimum` method.
        """
        # Create a 100 * 100 * 100 array of uniform random numbers, in the
        # range (0.01 < x < 1.0)
        shape = tuple(numpy.repeat(100, 3))
        array = numpy.random.uniform(low=0.01, high=1.0, size=shape)

        # Generate a random coordinate position
        coords = tuple(numpy.random.randint(low=0, high=100, size=3))
        # Generate random minimum < 0.01
        minimum = numpy.random.uniform(low=0, high=0.01)
        # Set minimum at generated coordinates
        array[coords[0], coords[1], coords[2]] = minimum

        # Find minimum of array
        fit_min, fit_coords = self._default_grid_search.find_minimum(array)
        self.assertIsInstance(fit_min, float)
        self.assertIsInstance(fit_coords, tuple)
        self.assertEqual(fit_min, minimum)
        self.assertEqual(fit_coords, coords)

        # Now try with two equal minima
        coords2 = tuple(numpy.random.randint(low=0, high=100, size=3))
        # Set second minimum at second generated coordinates
        array[coords2[0], coords2[1], coords2[2]] = minimum

        # Find minimum of array
        fit_min, fit_coords = self._default_grid_search.find_minimum(array)
        self.assertEqual(fit_min, minimum)
        if coords[0] < coords2[0]:  # coords should be returned as best fit
            self.assertEqual(fit_coords, coords)
        else:  # coords2 should be returned as best fit
            self.assertEqual(fit_coords, coords2)

    def test_minimise(self):
        """ Test the :meth:`GridSearch.minimise` method.
        """
        fit_x = 4.0
        fit_y = -2.5
        fit_z = 0.5

        # Test default grid search, with numpy
        minimum = self._default_grid_search.minimise(funct)
        self.assertIsInstance(minimum, float)
        results = self._default_grid_search.get_summary()
        self.assertAlmostEqual(results.get("x"), fit_x)
        self.assertAlmostEqual(results.get("y"), fit_y)
        self.assertAlmostEqual(results.get("z"), fit_z)

        # Try grid search using find_minimum
        self._grid_search.minimise(funct)
        results = self._grid_search.get_summary()
        self.assertAlmostEqual(results.get("x"), fit_x)
        self.assertAlmostEqual(results.get("y"), fit_y)
        self.assertAlmostEqual(results.get("z"), fit_z)
