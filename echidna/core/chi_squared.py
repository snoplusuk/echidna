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

        * penalty_terms (*dict*): specify (for each penalty term) values for:

          * "parameter_value" (*optional*)
          * "sigma" 
        
    Attributes:
      _form (str): form of chi squared calculation to use
      _penalty_terms (dict): information about each penalty term
      _penalty_terms_set (bool): True if one or more penalty terms have been set
    """
    def __init__(self, form="poisson_likelihood", **kwargs):
        self._form = form
        if (kwargs.get("penalty_terms") != None):
            self._penalty_terms = kwargs.get("penalty_terms")
            self._penalty_terms_set = True
        else:
            self._penalty_terms = None
            self._penalty_terms_set = False

    def get_chi_squared(self, observed, expected, **kwargs):
        """ Calculate the chi squared comparing observed to expected.

        Args:
          observed (:class:`numpy.array`): energy spectrum of observed events
          expected (:class:`numpy.array`): energy spectrum of expected events
          **kwargs: keyword argumets
          
        .. note::
        
          Keyword arguments include

            * penalty_terms (*dict*): specify (for each penalty term) values for:

              * "parameter_value"
              * "sigma" (*optional*)

        .. warning::

          A named penalty term defined here will overwrite one with the same
          name defined in the constructor.

        Returns:
          float. Value of chi squared calculated
        """
        # Set up penalty term
        if (kwargs.get("penalty_terms") != None):
            if self._penalty_terms_set:
                for name, penalty_term in kwargs.get("penalty_terms").iteritems():
                    if (self._penalty_terms.get(name) != None):
                        _penalty_term = self._penalty_terms.get(name)
                        # overwrite existing entries
                        if (penalty_term.get("parameter_value") != None):
                            _penalty_term["parameter_value"] = penalty_term.get("parameter_value")
                        if (penalty_term.get("sigma") != None):
                            _penalty_term["sigma"] = penalty_term.get("sigma")
                    else: # create new entry
                        self._penalty_terms[name] = penalty_term
            else: # no penalty term information currently set
                self._penalty_terms = kwargs.get("penalty_terms")
                self._penalty_terms_set = True

        # Calculate chi squared
        if (self._form == "pearson"):
            chi_squared = pearson_chi_squared(observed, expected)
        elif (self._form == "neyman"):
            chi_squared = neyman_chi_squared(observed, expected)
        else: # (self._form == "poisson_likelihood")
            chi_squared = 2.0 * log_likelihood(observed, expected)

        # Add penalty term(s)
        if self._penalty_terms_set:
            for name, penalty_term in self._penalty_terms.iteritems():
                chi_squared += numpy.power(penalty_term.get("parameter_value")/penalty_term.get("sigma"), 2.0)
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
