""" Physics tests are like unittests but for testing physics results.

Example:
  To run all physics tests::

    $ python physics_tests.py

"""
import numpy


def test_function_float(function, expected, tolerance=0.001, **kwargs):
    """ Tests a function that returns a physics result (float)

    Args:
      function (function): An instance of the function/method. e.g.
        :obj:`caculator.get_n_atoms` - notice no parentheses.
      expected (float): Expected physics result of function.
      tolerance (float, optional): Tolerance in agreement of results.
        Evaluated as :obj:`tolerance*expected`, i.e. a tolerance of
        0.001 ensures agreement to within 0.1% of expected value.
      \**kwargs: Keyword arguments to pass to the function

    Returns:
      (bool, string): Tuple containing the result of the check True/
        False for Pass/Fail and a message.
    """
    # Evaluate function
    observed = function(**kwargs)

    result = numpy.allclose(observed, expected, atol=tolerance*expected)
    if result:
        message = function.__name__ + ": OK"
    else:
        message = function.__name__ + ": FAIL"
        message += ("\n -->" + str(observed) + " != " + str(expected) +
                    " to within " + str(tolerance*100) + "%")
    return result, message
