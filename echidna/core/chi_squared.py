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
        # Set up penalty term.
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

        # Calculate chi squared.
        chi_squared_sum = 0.0
        for n_observed, n_expected in zip(numpy.nditer(observed),
                                          numpy.nditer(expected)):
            if (self._form == "pearson"):
                chi_squared_sum += pearson_chi_squared(n_observed, n_expected)
            elif (self._form == "neyman"):
                chi_squared_sum += neyman_chi_squared(n_observed, n_expected)
            elif (self._form == "poisson_likelihood"):
                chi_squared_sum += log_likelihood(n_observed, n_expected)
        if (self._form == "poisson_likelihood"):
            chi_squared_sum *= 2.0

        # Add penalty term
        if penalty_term_set:
            chi_squared_sum += math.pow(parameter_value/sigma, 2)
        return chi_squared_sum
        
def pearson_chi_squared(n_observed, n_expected):
    """ Calculates Pearson's chi squared.

    Args:
      n_observed (float): Number of observed events
      n_expected (float): Number of expected events

    Returns:
      float. Calculated Pearson's chi squared
    """
    if (n_expected == 0.0):
        raise ZeroDivisionError("expected events --> float divison by zero")
    return math.pow(n_observed - n_expected, 2) / n_expected

def neyman_chi_squared(n_observed, n_expected):
    """ Calculates Neyman's chi squared.

    Args:
      n_observed (float): Number of observed events.
      n_expected (float): Number of expected events.

    Returns:
      float. Calculated Neyman's chi squared
    """
    if (n_observed == 0.0):
        raise ZeroDivisionError("observed events --> float divison by zero")
    return math.pow(n_observed - n_expected, 2) / n_observed

def log_likelihood(n_observed, n_expected):
    """ Calculates the log likelihood.

    .. note::
    
      For calculation of Poisson likelihood chi squared.

    Args:
      n_observed (float): Number of observed events
      n_expected (float): Number of expected events

    Returns:
      float. Calculated Neyman's chi squared
    """
    if (n_expected == 0.0):
        raise ZeroDivisionError("expected events --> float divison by zero")
    if (n_observed == 0.0):
        raise ValueError("observed events --> math domain error - ln(0)")
    return n_expected - n_observed + n_observed*math.log(n_observed/n_expected)
