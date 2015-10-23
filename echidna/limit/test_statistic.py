""" Module to hold classes for various test statistics that can be used
for fitting.
"""
import numpy
import abc


class TestStatistic(object):
    """ Base class for the calculation of a test statistic.

    The calculation of any test statistic is based on one spectrum
    containing observed events and one containing expected events. It
    is assumed that the observed events form the "data" spectrum and
    the expected events form the spectrum predicted by the model.

    Args:
      name (string): Name of test statistic.
      per_bin (bool): If True the statistic in each bin is returned as an
        :class:`numpy.array`. If False one value for the statistic is returned
        for the entire array.

    Attributes:
      _name (string): Name of test statistic.
      _per_bin (bool): If True the statistic in each bin is returned as an
        :class:`numpy.array`. If False one value for the statistic is returned
        for the entire array.
    """
    __metaclass__ = abc.ABCMeta  # Only required for python 2

    def __init__(self, name, per_bin):
        self._name = name
        self._per_bin = per_bin

    def get_name(self):
        """
        Returns:
          string: Name of test statistic, stored in :attr:`_name`
        """
        return self._name

    def compute_statistic(self, observed, expected):
        """ Compute the value of the test statistic.

        Args:
          observed (:class:`numpy.ndarray`): 1D array containing the
            observed data points.
          expected (:class:`numpy.ndarray`): 1D array containing the
            expected values predicted by the model.

        Returns:
          float: Computed value of test statistic if :attr:`_per_bin` is False.
          :class:`numpy.array`: Of computed values of test statistic in each
            bin if :attr:`per_bin` is True.
        """
        if len(observed.shape) != 1:
            raise TypeError("Incompatible shape %s for observed array, "
                            "expected 1-D array" % str(observed.shape))
        if len(expected.shape) != 1:
            raise TypeError("Incompatible shape %s for expected array, "
                            "expected 1-D array" % str(expected.shape))
        if len(observed) != len(expected):
            raise ValueError(
                "Number of bins mismatch, expecting %d bins, found %d"
                % (observed.shape[0], self.get_spectra_par()._bins))
        if not self._per_bin:
            return self._compute(observed, expected)
        else:
            return self._get_stats(observed, expected)

    # Method adapted from http://codereview.stackexchange.com/a/47115
    @abc.abstractmethod
    def _compute(self, observed, expected):
        """ Calculates the test statistic.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Returns:
          float: Calculated test statistic.
        """
        return None

    @abc.abstractmethod
    def _get_stats(self, observed, expected):
        """ Gets the test statistic for each bin.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Raises:
          ValueError: If arrays are different lengths.

        Returns:
          :class:`numpy.array`: Of the test statistic in each bin.
        """
        return None


class BakerCousinsChi(TestStatistic):
    """ Test statistic class for calculating the Baker-Cousins chi-squared test
    statistic.

    Args:
      per_bin (bool, optional): If True the statistic in each bin is returned
        as an :class:`numpy.array`. If False (default) one value for the
        statistic is returned for the entire array.
    """
    def __init__(self, per_bin=False):
        super(ChiSquared, self).__init__("baker_cousins", per_bin)

    def _compute(self, observed, expected):
        """ Calculates the chi-squared.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Returns:
          float: Calculated chi squared.
        """
        epsilon = 1e-34  # In the limit of zero
        total = 0
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = expected[i] - observed[i] + observed[i] *\
                    numpy.log(observed[i] / expected[i])
            total += bin_value
        return 2. * total

    def _get_stats(self, observed, expected):
        """ Gets chi squared for each bin.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Raises:
          ValueError: If arrays are different lengths.

        Returns:
          :class:`numpy.array`: Of the chi squared in each bin.
        """
        epsilon = 1e-34  # In the limit of zero
        stats = []
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = expected[i] - observed[i] + observed[i] *\
                    numpy.log(observed[i] / expected[i])
            stats.append(2.*bin_value)
        return numpy.array(stats)


class BakerCousinsLL(TestStatistic):
    """ Test statistic class for calculating the Baker-Cousins log likelihood
      ratio test statistic.

    Args:
      per_bin (bool, optional): If True the statistic in each bin is returned
        as an :class:`numpy.array`. If False (default) one value for the
        statistic is returned for the entire array.
    """
    def __init__(self, per_bin=False):
        super(ChiSquared, self).__init__("baker_cousins", per_bin)

    def _compute(self, observed, expected):
        """ Calculates the log likelihood.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Returns:
          float: Calculated Neyman's chi squared
        """
        epsilon = 1e-34  # In the limit of zero
        total = 0
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = expected[i] - observed[i] + observed[i] *\
                    numpy.log(observed[i] / expected[i])
            total += bin_value
        return total

    def _get_stats(self, observed, expected):
        """ Gets chi squared for each bin.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events

        Returns:
          :class:`numpy.array`: Of the chi squared in each bin.
        """
        epsilon = 1e-34  # In the limit of zero
        stats = []
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = expected[i] - observed[i] + observed[i] *\
                    numpy.log(observed[i] / expected[i])
            stats.append(bin_value)
        return numpy.array(stats)


class Neyman(TestStatistic):
    """ Test statistic class for calculating the Neyman chi-squared test
    statistic.

    Args:
      per_bin (bool, optional): If True the statistic in each bin is returned
        as an :class:`numpy.array`. If False (default) one value for the
        statistic is returned for the entire array.
    """
    def __init__(self, per_bin=False):
        super(ChiSquared, self).__init__("neyman", per_bin)

    def _compute(self, observed, expected, per_bin=False):
        """ Calculates chi squared.

        Args:
          observed (:class:`numpy.array`/float): Number of observed
            events
          expected (:class:`numpy.array`/float): Number of expected
            events

        Returns:
          float: Calculated Neyman's chi squared
        """
        # Chosen due to backgrounds with low rates in ROI
        epsilon = 1e-34  # In the limit of zero
        total = 0
        for i in range(len(observed)):
            if observed[i] < epsilon:
                expected[i] = epsilon
            if expected[i] < epsilon:
                bin_value = observed[i]
            else:
                bin_value = (expected[i] - observed[i])**2 / observed[i]
            total += bin_value
        return total

    def _get_stats(self, observed, expected):
        """ Gets chi squared for each bin.

        Args:
          observed (:class:`numpy.array`/float): Number of observed
            events
          expected (:class:`numpy.array`/float): Number of expected
            events

        Returns:
          :class:`numpy.array`: Of the chi squared in each bin.
        """
        # Chosen due to backgrounds with low rates in ROI
        epsilon = 1e-34  # In the limit of zero
        stats = []
        for i in range(len(observed)):
            if observed[i] < epsilon:
                expected[i] = epsilon
            if expected[i] < epsilon:
                bin_value = observed[i]
            else:
                bin_value = (expected[i] - observed[i])**2 / observed[i]
            stats.append(bin_value)
        return numpy.array(stats)


class Pearson(TestStatistic):
    """ Test statistic class for calculating the Pearson chi-squared test
    statistic.

    Args:
      per_bin (bool, optional): If True the statistic in each bin is returned
        as an :class:`numpy.array`. If False (default) one value for the
        statistic is returned for the entire array.
    """
    def __init__(self):
        super(ChiSquared, self).__init__("pearson")

    def _compute(self, observed, expected, per_bin=False):
        """ Calculates chi squared.

        Args:
          observed (:class:`numpy.array`/float): Number of observed
            events
          expected (:class:`numpy.array`/float): Number of expected
            events
          per_bin (bool, optional): If True updates :attr:`_per_bin`
            with value calculated for each bin.

        Raises:
          ValueError: If arrays are different lengths.

        Returns:
          float: Calculated Pearson's chi squared
        """
        # Chosen due to backgrounds with low rates in ROI
        epsilon = 1e-34  # Limit of zero
        total = 0
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = (observed[i] - expected[i])**2 / expected[i]
            if per_bin:
                self._per_bin[i] = bin_value
            total += bin_value
        return total

    def _get_stats(self, observed, expected):
        """ Gets chi squared for each bin.

        Args:
          observed (:class:`numpy.array`): Array of number of observed events
          expected (:class:`numpy.array`): Array of number of expected events

        Raises:
          ValueError: If arrays are different lengths.

        Returns:
          :class:`numpy.array`: Of the chi squared in each bin.
        """
        # Chosen due to backgrounds with low rates in ROI
        epsilon = 1e-34  # Limit of zero
        total = 0
        for i in range(len(observed)):
            if expected[i] < epsilon:
                expected[i] = epsilon
            if observed[i] < epsilon:
                bin_value = expected[i]
            else:
                bin_value = (observed[i] - expected[i])**2 / expected[i]
            if per_bin:
                self._per_bin[i] = bin_value
            total += bin_value
        return total
