import numpy


class LimitConfig(object):
    """ Class to handle configuration parameters for each spectrum.

    In limit setting we want to be able to add multiple backgrounds and
    a signal spectrum. Each background will have a different range of
    counts to use in floating it and some will be constrained by a
    penalty term. This class keeps track of all this information.

    Attributes:
      _prior_counts (float): prior/expected counts
      _counts (list): list of count rates (*float*) to use for scaling
      _sigma (float): prior constraint on counts
      _chi_squareds (:class:`numpy.array`): array filled with chi
        squared corresponding to each count value

    Args:
      prior_counts (float): prior/expected counts
      counts (list): list of count (*float*) to use for scaling
      sigma (float, optional): prior constraint on counts
    """
    def __init__(self, prior_count, counts, sigma=None):
        self._prior_count = prior_count
        self._counts = counts
        self._sigma = sigma
        self._chi_squareds = numpy.zeros(shape=[0], dtype=float)

    def get_count(self):
        """
        Yields:
          float: next count in array

        Examples:
          When dealing with multiple backgrounds, just select the
          corresponding config for each background (e.g. ``bkg_config``)
          then you can correctly loop over the background scalings

          >>> for count in bkg_config.get_count():
          ...     spectrum.scale(count)
          ...     # spectrum scaled to each value of count
        """
        for count in self._counts:
            yield count

    def add_chi_squared(self, chi_squared):
        """ Add chi squared to config chi squared array.

        Args:
          chi_squared (float): chi squared value to add
        """
        self._chi_squareds = numpy.append(self._chi_squareds, chi_squared)

    def get_minimum(self):
        """ Get minimum value from chi_squared array.

        Returns:
          float: Minimum of _chi_squareds
        """
        return numpy.min(self._chi_squareds)

    def get_first_bin_above(self, limit):
        """ For signal, determine the count corresponding to limit.

        The limit count is the first count value that takes the chi
        squared over the threshold set by the desired confidence limit.

        Args:
          limit (float): chi squared value corresponding to desired
            confidence limit

        Returns:
          float: Count at limit
        """
        return self._counts[numpy.where(self._chi_squareds > limit)[0][0]]

    def reset_chi_squareds(self):
        """ Resets :attr:`_chi_squareds` to an empty
          :class:`numpy.array`
        """
        self._chi_squareds = numpy.zeros(shape=(0), dtype=float)
