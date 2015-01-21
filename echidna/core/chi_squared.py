import numpy

import math

class ChiSquared(object):
    """ This class calculates the chi squared comparing "data" to "montecarlo".
    
    The calculation is based on a spectrum containing observed events and one
    containing expected events. The paradigm assumed is that the observed 
    events form the data spectrum and the expected events form the montecarlo
    spectrum. A couple of different methods for calculating chi squared are 
    included, as well as the option to add constraints via a penalty term.
    
    Args:
      form (str, optional): specify form of chi squared calculation to use
      **kwargs: Keyword arguments
      
    .. note::

      Forms of chi squared include
        
        * "pearson"
        * "neyman"
        * "poisson_likelihood" (*default*)
         
    .. note::

      Keyword arguments include

        * penalty_term (*dict*): specify value for

          * "parameter_value" (*optional*)
          * "sigma"
        
    Attributes:
      _form (str): form of chi squared calculation to use
    """
    def __init__(self, form="poisson_likelihood", **kwargs):
        self._form = form
        if (kwargs.get("penalty_term") != None):
            self._penalty_term = kwargs.get("penalty_term")
        else:
            self._penalty_term = None 

    def get_chi_squared(self, observed, expected, **kwargs):
        """ Calculate the chi squared comparing observed to expected.

        Args:
          observed (:class:`numpy.array`): energy spectrum of observed events
          expected (:class:`numpy.array`): energy spectrum of expected events
          **kwargs: keyword argumets
          
        .. note::
        
          Keyword arguments include

            * penalty_term (*dict*): specify values for

              * "parameter_value"
              * "sigma" (*optional*)

        .. warning::

          A penalty term sigma defined here will overwrite one defined in the
          constructor.

        Returns:
          float. Value of chi squared calculated
        """
        # Set up penalty term
        penalty_term_set = False
        if (kwargs.get("penalty_term") != None):
            parameter_value = kwargs.get("penalty_term").get("parameter_value")
            if (kwargs.get("penalty_term").get("sigma") != None):
                sigma = kwargs.get("penalty_term").get("sigma")
            else:
                sigma = self._penalty_term.get("sigma")
            penalty_term_set = True
        elif (self._penalty_term != None):
            parameter_value = self._penalty_term.get("parameter_value")
            sigma = self._penalty_term.get("sigma")
            penalty_term_set = True

        # Calculate chi squared
        if (self._form == "pearson"):
            chi_squared = pearson_chi_squared(observed, expected)
        elif (self._form == "neyman"):
            chi_squared = neyman_chi_squared(observed, expected)
        else: # (self._form == "poisson_likelihood")
            chi_squared = 2.0 * log_likelihood(observed, expected)

        # Add penalty term
        if penalty_term_set:
            chi_squared += math.pow(parameter_value/sigma, 2)
        return chi_squared

def check_bin_content(array):
    """ Checks bin content of a numpy array.

    Bin content must be > 0 to be used in chi squared calculations

    Args:
      array (:class:`numpy.array`, *float*): Array or value to check

    Returns:
      bool, string. Tuple containing result of check and an error message 
    """
    if (numpy.sum(array <= 0.0) != 0):
        result = False
        zero_bins, = numpy.where(array <= 0.0)
        message = "array contains " + str(len(zero_bins)) + " bins with content <= 0.0. " 
        message += "First instance at bin " + str(zero_bins[0]) + "."
    else:
        result = True
        message = "array - all bins > 0.0"
    return result, message

def pearson_chi_squared(observed, expected):
    """ Calculates Pearson's chi squared.

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed events
      expected (:class:`numpy.array`, *float*): Number of expected events
      
    Raises:
      ValueError: If either array contains a bin with content <= 0.0
      
    Returns:
      float. Calculated Pearson's chi squared
    """
    correct_bin_content, message = check_bin_content(observed)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: observed " + message)
    correct_bin_content, message = check_bin_content(expected)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: expected " + message)
    return numpy.sum((observed-expected)**2 / expected)

def neyman_chi_squared(observed, expected):
    """ Calculates Neyman's chi squared.

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed events.
      expected (:class:`numpy.array`, *float*): Number of expected events.

    Raises:
      ValueError: If either array contains a bin with content <= 0.0

    Returns:
      float. Calculated Neyman's chi squared
    """
    correct_bin_content, message = check_bin_content(observed)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: observed " + message)
    correct_bin_content, message = check_bin_content(expected)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: expected " + message)
    return numpy.sum((observed-expected)**2 / observed)

def log_likelihood(observed, expected):
    """ Calculates the log likelihood.

    .. note::
    
      For calculation of Poisson likelihood chi squared.

    Args:
      observed (:class:`numpy.array`, *float*): Number of observed events
      expected (:class:`numpy.array`, *float*): Number of expected events

    Raises:
      ValueError: If either array contains a bin with content <= 0.0

    Returns:
      float. Calculated Neyman's chi squared
    """
    correct_bin_content, message = check_bin_content(observed)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: observed " + message)
    correct_bin_content, message = check_bin_content(expected)
    if not correct_bin_content:
        raise ValueError("chi_squared.pearson_chi_squared: expected " + message)
    return numpy.sum(expected - observed + observed*numpy.log(observed/expected))
