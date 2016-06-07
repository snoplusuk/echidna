""" Tests minimise module.

Tests the :meth:`GridSearch.fin_minimum`, method - an alternative to
calling :meth:`numpy.argmin` and :meth:`numpy.nanmin`, which are used
by default.

The main part of this test is verifying the :class:`GridSearch` class,
particularly the `meth:`GridSearch.minimise` method.
"""
import numpy

import echidna.core.spectra as spectra
import echidna.fit.test_statistic as test_statistic
from echidna.core.config import (GlobalFitConfig, SpectraFitConfig,
                                 SpectraConfig)
from echidna.fit.minimise import GridSearch

import unittest
import copy


class TestGridSearch(unittest.TestCase):
    """ Tests for the class :class:`echidna.limit.minimise.GridSearch`.
    """
    def setUp(self):
        """ Set up attributes for tests.

        Attributes:
          _A (:class:`spectra.Spectra`): Test spectrum A - to use in test
          _B (:class:`spectra.Spectra`): Test spectrum B - to use in test
          _C (:class:`spectra.Spectra`): Test spectrum C - to use in test
          _test_statistic (:class:`test_statistic.BakerCousinsChi`): Test
            statistic to use in test
          _default_grid_search (:class:`GridSearch`): GridSearch using
            default method (numpy) for minimisation.
          _grid_search (:class:`GridSearch`): GridSearch using
            :meth:`GridSearch.find_minimum`, to find minimum.
        """
        # Create spectra
        num_decays = 1.e4
        spectra_config = SpectraConfig.load(
            {"parameters":
                {"x":
                    {"low": 2.0,
                     "high": 3.0,
                     "bins": 10}}},
            name="spectra_config")
        spectrum = spectra.Spectra("spectra", num_decays, spectra_config)

        # Fill spectrum with random Gaussian data
        for i_decay in range(int(num_decays)):
            x = numpy.random.normal(loc=2.5, scale=0.1)
            if numpy.random.uniform() > 0.1:
                if x > 2.0 and x < 3.0:
                    spectrum.fill(x=x)

        # Save three copies of spectrum
        self._A = copy.copy(spectrum)
        self._A._name = "A"
        self._B = copy.copy(spectrum)
        self._B._name = "B"
        self._C = copy.copy(spectrum)
        self._C._name = "C"

        # Make Global fit config
        fit_config = GlobalFitConfig.load({
            "global_fit_parameters": {
                "x": {}}})

        fit_config_A = SpectraFitConfig.load({
            "spectral_fit_parameters": {
                "rate": {
                    "prior": 5.0,
                    "sigma": None,
                    "low": 4.0,
                    "high": 6.0,
                    "bins": 11}}},
            spectra_name=self._A.get_name())
        fit_config_B = SpectraFitConfig.load({
            "spectral_fit_parameters": {
                "rate": {
                    "prior": 12.0,
                    "sigma": None,
                    "low": 11.0,
                    "high": 13.0,
                    "bins": 11}}},
            spectra_name=self._B.get_name())
        fit_config_C = SpectraFitConfig.load({
            "spectral_fit_parameters": {
                "rate": {
                    "prior": 13.0,
                    "sigma": None,
                    "low": 12.0,
                    "high": 14.0,
                    "bins": 11}}},
            spectra_name=self._C.get_name())

        fit_config.add_config(fit_config_A)
        fit_config.add_config(fit_config_B)
        fit_config.add_config(fit_config_C)

        # Initialise test statistic
        self._test_statistic = test_statistic.BakerCousinsChi(per_bin=True)

        # Initialise default GridSearch
        self._default_grid_search = GridSearch(fit_config,
                                               spectra_config,
                                               "default_grid_search",
                                               per_bin=True)
        # Initialise GridSearch using find_minimum
        self._grid_search = GridSearch(fit_config,
                                       spectra_config,
                                       "grid_search",
                                       use_numpy=False,
                                       per_bin=True)

    def _funct(self, *args):
        """ Callable to pass to minimiser.

        Fits A**2 + B**2 to C**2.

        Returns:
          :class:`numpy.ndarray`: Test statistic values
          float: Penalty term - in this case always zero.
        """
        a = args[0]
        b = args[1]
        c = args[2]

        # Scale spectra
        self._A.scale(a ** 2)
        self._B.scale(b ** 2)
        self._C.scale(c ** 2)

        observed = self._C.project("x")
        expected = self._A.project("x") + self._B.project("x")

        # Return test statistics and penalty term (always zero)
        return self._test_statistic.compute_statistic(observed, expected), 0.

    def test_find_minimum(self):
        """ Test the :meth:`GridSearch.find_minimum` method.

        Tests:
          * That :meth:`GridSearch.find_minimum` correctly locates the
            minimum.
          * That :meth:`GridSearch.find_minimum` exhibits the correct
            behaviour when there are two minima.
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

        Floats A, B and C (rates) and fits A^2 + B^2 to C^2, using a
        a :class:`test_statistic.BakerCousinsChi` test statistic.

        The fitted rates should form the pythagorean triple:
        5^2 + 12^2 = 13^2.
        """
        fit_A = 5.0
        fit_B = 12.0
        fit_C = 13.0

        # Test default grid search, with numpy
        minimum, penalty = self._default_grid_search.minimise(
            self._funct, self._test_statistic)
        self.assertIsInstance(minimum, numpy.ndarray)
        results = self._default_grid_search.get_summary()
        self.assertAlmostEqual(results.get("A_rate").get("best_fit"), fit_A)
        self.assertAlmostEqual(results.get("B_rate").get("best_fit"), fit_B)
        self.assertAlmostEqual(results.get("C_rate").get("best_fit"), fit_C)

        # Try grid search using find_minimum
        self._grid_search.minimise(self._funct, self._test_statistic)
        results = self._grid_search.get_summary()
        self.assertAlmostEqual(results.get("A_rate").get("best_fit"), fit_A)
        self.assertAlmostEqual(results.get("B_rate").get("best_fit"), fit_B)
        self.assertAlmostEqual(results.get("C_rate").get("best_fit"), fit_C)
