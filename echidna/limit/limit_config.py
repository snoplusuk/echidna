import numpy


class LimitConfig(object):
    """ Class to hadle configuration parameters for each spectrum

    In limit setting we want to be able to add multiple backgrounds and a
    signal spectrum. Each background will have a different range of counts to
    uses in floating it and some will be constrained by a penalty term, which 
    also has a unique range of parameter values to loop over. This class keeps
    track of all this information.

    Attributes:
      _prior_counts (float): prior/expected counts
      _counts (list): list of count rates (*float*) to use for scaling
      _sigma (float): prior constraint on counts
      _chi_squareds (:class: `numpy.array`): array filled with chi 
        squared corresponding to each count value
 
    Args:
      prior_counts (float): prior/expected counts
      counts (list): list of count (*float*) to use for scaling
      sigma (float): prior constraint on counts
    """
    def __init__(self, prior_count, counts, sigma=None):
        self._prior_count = prior_count
        self._counts = counts
        self._sigma = sigma
        self._chi_squareds = numpy.zeros(shape=[0], dtype=float)

    def get_count(self):
        """ Generator
        
        This method is included so you can do something like:
        
        for count in bkg_config.get_count():
            spectrum.normalise(count)
            ...
        
        Yield:
          count (float)
        """
        for count in self._counts:
            yield count
    
    def add_chi_squared(self, chi_squared):
        """ Add chi squared to config chi squared array
        
        Args:
          chi_squared (float): chi squared value to add
        """
        self._chi_squareds = numpy.append(self._chi_squareds, chi_squared)

    def get_minimum(self):
        """ Get minimum value from chi_squared array

        Returns:
          *float*. Minimum of _chi_squareds
        """
        return numpy.min(self._chi_squareds)

    def get_first_bin_above(self, limit):
        """ For signal, determine the count corresponding to limit

        The limit count is the first count value that takes the chi
        squared over the threshold set by the desired confidence limit.

        Args:
          limit (float): chi squared value corresponding to desired
            confidence limit

        Returns:
          *float*. Count at limit
        """
        return self._counts[numpy.where(self._chi_squareds > limit)[0][0]]

    def reset_chi_squareds(self):
        """ Resets _chi_squareds to original state
        """
        self._chi_squareds = numpy.zeros(shape=(0), dtype=float)
