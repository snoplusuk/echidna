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
      type_name (string): Type of test statistic.
      spectra_par (:class:`echidna.core.spectra.SpectraParameter): The
        parameter class for the projection dimension e.g.
        ``energy_mc``.

    Attributes:
      _type (string): Type of test statistic.
      _spectra_par (:class:`echidna.core.spectra.SpectraParameter): The
        parameter class for the projection dimension e.g.
        ``energy_mc``.
      _per_bin (:class:`numpy.ndarray`): 1D numpy array to hold value
        of test statistic calculated for each bin

    """
    __metaclass__ = abc.ABCMeta  # Only required for python 2

    def __init__(self, type_name, spectra_par):
        self._type = type_name
        self._spectra_par = spectra_par
        self._per_bin = numpy.zeros(shape=(spectra_par._bins))

    def get_type(self):
        """
        Returns:
          string: Type of test statistic, stored in :attr:`_type`
        """
        return self._type

    def get_spectra_par(self):
        """
        Returns:
          (:class:`echidna.core.spectra.SpectraParameter):The parameter
           class for the projection dimension, stored in
           :attr:`_spectra_par`.
        """
        return self._spectra_par

    def get_per_bin(self):
        """
        Returns:
          (:class:`numpy.ndarray`): 1D numpy array to hold value
            of test statistic calculated for each bin from the most
            recent call to :meth:`compute_statistic`.
        """
        return self._per_bin

    # Method adapted from http://codereview.stackexchange.com/a/47115
    @abc.abstractmethod
    def compute_statistic(self, observed, expected):
        """ Compute the value of the test statistic.

        .. warning:: This method must be overridden by the derived
          class for each test statistic. Failure to do this will raise
          ``TypeError``.

        Args:
          observed (:class:`numpy.ndarray`): 1D array containing the
            observed data points.
          expected (:class:`numpy.ndarray`): 1D array containing the
            expected values predicted by the model.

        Returns:
          float: Computed value of test statistic.
        """
        return None


class ChiSquared(TestStatistic):
    """ Test statistic class for calculating the chi-squared test
    satatistic.

    Args:
      spectra_par (:class:`echidna.core.spectra.SpectraParameter): The
        parameter class for the projection dimension e.g.
        ``energy_mc``.
      form (string, optional): Form of chi-squared to use in calculation
        of the test statistic. Default is "poisson_likelihood"
      single_bin (bool, optional): If ``True``, the test statistic is
        calculated for a single bin, i.e. just using the sum of
        ``observed`` and ``expected``.

    .. note:: All forms of chi-squared are as defined in::

        * REF: S. Baker & R. D. Cousins, Nucl. Inst. and Meth. in Phys.
          Res. 211, 437-442 (1984)

      Available forms of chi-squared include::

        * ``poisson_likelihood`` (default)
        * ``pearson``
        * ``neyman``

    Raises:
      TypeError: If class does not override abstract methods of base
        class.
    """
    def __init__(self, spectra_par,
                 form="poisson_likelihood", single_bin=False):
        super(ChiSquared, self).__init__("chi_squared", spectra_par)
        self._form = form
        self._single_bin

    def compute_statistic(self, observed, expected):
        """ Compute the value of the test statistic.

        Args:
          observed (:class:`numpy.ndarray`): 1D array containing the
            observed data points.
          expected (:class:`numpy.ndarray`): 1D array containing the
            expected values predicted by the model.

        Returns:
          float: Computed value of test statistic.
        """
        if len(observed.shape) != 1:
            raise TypeError("Incompatible shape %s for observed array, "
                            "expected 1-D array" % str(observed.shape))
        if len(expected.shape) != 1:
            raise TypeError("Incompatible shape %s for expected array, "
                            "expected 1-D array" % str(expected.shape))
        if observed.shape[0] != self.get_spectra_par()._bins:
            raise ValueError(
                "Number of bins mismatch, expecting %d bins, found %d"
                % (observed.shape[0], self.get_spectra_par()._bins))
        if expected.shape[0] != self.get_spectra_par()._bins:
            raise ValueError(
                "Number of bins mismatch, expecting %d bins, found %d"
                % (expected.shape[0], self.get_spectra_par()._bins))

        if self._single_bin:  # prepare single bin arrays
            observed = numpy.sum(observed)
            expected = numpy.sum(expected)
            # Both now converted to type numpy.float64 (float)

        if self._form is "poisson_likelihood":
            return 2. * self.log_likelihood(observed, expected, per_bin=True)

    @classmethod
    def log_likelihood(self, observed, expected, per_bin=False):
        """ Calculates the (Baker-Cousins) log likelihood.

        Args:
          observed (:class:`numpy.array`, *float*): Number of observed
            events
          expected (:class:`numpy.array`, *float*): Number of expected
            events
          per_bin (bool, optional): If True updates :attr:`_per_bin`
            with value calculated for each bin.

        Raises:
          ValueError: If arrays are different lengths.

        Returns:
          float: Calculated Neyman's chi squared
        """
        # Create chi-squared per bin array
        if len(observed) != len(expected):
            raise ValueError("Arrays are different lengths")
        # Chosen due to backgrounds with low rates in ROI
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
            if per_bin:
                self._per_bin[i] = bin_value
            total += bin_value
        return total

    @classmethod
    def pearson(self, observed, expected, per_bin=False):
        """ Calculates Pearson's chi squared.

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
        if isinstance(observed, float):
            observed = numpy.array([observed])
        if isinstance(expected, float):
            expected = numpy.array([expected])
        if len(observed) != len(expected):
            raise ValueError("Arrays are different lengths")
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

    @classmethod
    def neyman(self, observed, expected, per_bin=False):
        """ Calculates Neyman's chi squared.

        Args:
          observed (:class:`numpy.array`/float): Number of observed
            events
          expected (:class:`numpy.array`/float): Number of expected
            events
          per_bin (bool, optional): If True updates :attr:`_per_bin`
            with value calculated for each bin.

        Raises:
          ValueError: If arrays are different lengths

        Returns:
          float: Calculated Neyman's chi squared
        """
        if isinstance(observed, float):
            observed = numpy.array([observed])
        if isinstance(expected, float):
            expected = numpy.array([expected])
        if len(observed) != len(expected):
            raise ValueError("Arrays are different lengths")
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
            if per_bin:
                self._per_bin[i] = bin_value
            total += bin_value
        return total
