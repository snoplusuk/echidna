"""
"""
import numpy

from echidna.limit.test_statistic import (TestStatistic, BakerCousinsChi,
                                          BakerCousinsLL, Neyman, Pearson)

import unittest


class TestTestStatistic(unittest.TestCase):
    """ Tests for the :class:`echidna.limit.test_statistic.TestStatistic`
    class and it's derived classes.

    """
    def setUp(self):
        # Set up observed and expected arrays
        # Include a zero value in both to check epsilon behaviour
        self._observed = numpy.array([8.89, 4.03, 0.97, 0.,
                                      0.02, 1.04, 3.96, 9.08])
        self._expected = numpy.array([9., 4., 1., 0.1, 0., 1., 4., 9.])

    def test_abstract_base_class(self):
        """ Tests the base class
        """
        self.assertRaises(TypeError, lambda: TestStatistic("test_base_class"))

        class ABCTest(TestStatistic):
            """ Class to test abstract base class (ABC) exceptions
            """
            def __init__(self, name, per_bin=False):
                super(ABCTest, self).__init__(name, per_bin)
        # Test exception raised when method not overridden in ABC
        self.assertRaises(TypeError, lambda: ABCTest("abc_test"))

    def test_pearson(self):
        """ Test the Pearson chi-squared class
        """
        # Test some standard values
        self.assertAlmostEqual(Pearson._compute(100., 110.), (10. / 11.))
        self.assertAlmostEqual(Pearson._compute(100., 90.), (10. / 9.))
        self.assertAlmostEqual(Pearson._compute(100., 100.), 0.)

        # Test arrays
        test_statistic = Pearson()
        # Check correct exceptions raised - only need to do once
        # Following example from:
        # http://www.lengrand.fr/2011/12/pythonunittest-assertraises-raises-error/
        # to ensure Exceptions are caught
        # Observed wrong shape
        self.assertRaises(
            TypeError,
            lambda: test_statistic.compute_statistic(
                self._observed.reshape(2, 4), self._expected))
        # Expected wrong shape
        self.assertRaises(
            TypeError,
            lambda: test_statistic.compute_statistic(
                self._observed, self._expected.reshape(2, 4)))
        # Different lengths
        self.assertRaises(
            ValueError,
            lambda: test_statistic.compute_statistic(
                self._observed, numpy.append(self._expected, [16.])))

        test_statistic_pb = Pearson(per_bin=True)
        result = test_statistic.compute_statistic(self._observed,
                                                  self._expected)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual((result - 4.0e30)/result, 0.)
        result = test_statistic_pb.compute_statistic(self._observed,
                                                     self._expected)
        calculated = numpy.array([1.34444444e-3, 2.25e-4, 9.e-4, 0.1,
                                  4.e30, 1.6e-3, 4.e-4, 7.11111111e-4])
        self.assertIsInstance(result, numpy.ndarray)
        self.assertTrue(numpy.allclose(result, calculated))

    def test_neyman(self):
        """ Test the Neyman chi-squared class
        """
        # Test some standard values
        self.assertAlmostEqual(Neyman._compute(100., 110.), 1.)
        self.assertAlmostEqual(Neyman._compute(100., 90.), 1.)
        self.assertAlmostEqual(Neyman._compute(100., 100.), 0.)

        # Test arrays
        test_statistic = Neyman()
        test_statistic_pb = Neyman(per_bin=True)
        result = test_statistic.compute_statistic(self._observed,
                                                  self._expected)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual((result - 1.0e32)/result, 0.)
        result = test_statistic_pb.compute_statistic(self._observed,
                                                     self._expected)
        calculated = numpy.array([1.36107987e-3, 2.23325062e-4, 9.27835052e-4,
                                  1.e32, 0.02, 1.53846154e-3, 4.04040404e-4,
                                  7.04845815e-4])
        self.assertIsInstance(result, numpy.ndarray)
        self.assertTrue(numpy.allclose(result, calculated))

    def test_log_likelihood(self):
        """ Test the :class:`echidna.limit.test_statistic.BakerCousinLL`
        class that calculates the log likelihood ratio.
        """
        # Test some standard values
        self.assertAlmostEqual(BakerCousinsLL._compute(100., 110.),
                               0.468982019568)
        self.assertAlmostEqual(BakerCousinsLL._compute(100., 90.),
                               0.536051565783)
        self.assertAlmostEqual(BakerCousinsLL._compute(100., 100.), 0.)

        # Test arrays
        test_statistic = BakerCousinsLL()
        test_statistic_pb = BakerCousinsLL(per_bin=True)
        result = test_statistic.compute_statistic(self._observed,
                                                  self._expected)
        self.assertIsInstance(result, float)
        self.assertAlmostEquals((result - 1.57010389)/result, 0.)
        result = test_statistic_pb.compute_statistic(self._observed,
                                                     self._expected)
        calculated = numpy.array([6.74977765e-4, 1.12219800e-4, 4.54568740e-4,
                                  0.1, 1.46751740, 7.89541679e-4,
                                  2.00670020e-4, 3.54506715e-4])
        self.assertIsInstance(result, numpy.ndarray)
        self.assertTrue(numpy.allclose(result, calculated))

    def test_poisson_likelihood_chi(self):
        """ Test the :class:`echidna.limit.test_statistic.BakerCousinChi`
        class that calculates the poisson likelihood chi-squared.
        """
        self.assertAlmostEqual(BakerCousinsChi._compute(100., 110.),
                               0.937964039135)
        self.assertAlmostEqual(BakerCousinsChi._compute(100., 90.),
                               1.072103131565)
        self.assertAlmostEqual(BakerCousinsChi._compute(100., 100.), 0.)

        # Test arrays
        test_statistic = BakerCousinsChi()
        test_statistic_pb = BakerCousinsChi(per_bin=True)
        result = test_statistic.compute_statistic(self._observed,
                                                  self._expected)
        self.assertIsInstance(result, float)
        self.assertAlmostEquals((result - 3.14020778), 0.)
        result = test_statistic_pb.compute_statistic(self._observed,
                                                     self._expected)
        calculated = numpy.array([6.74977765e-4, 1.12219800e-4, 4.54568740e-4,
                                  0.1, 1.46751740, 7.89541679e-4,
                                  2.00670020e-4, 3.54506715e-4]) * 2.
        self.assertIsInstance(result, numpy.ndarray)
        self.assertTrue(numpy.allclose(result, calculated))
