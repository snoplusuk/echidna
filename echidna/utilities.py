""" Utilities module to include functions that may be useful throughout
echidna.
"""
import time


class Timer:
    """ Useful timer class to show time elapsed performing a code block.

    Examples:
      >>> With Timer() as t:
      ...     # block of code to time
      >>> print ('Code executed in %.03f sec.' % t._interval)
    """
    def __enter__(self):
        """
        Attributes:
          _start (float): start time of code block

        Returns:
          :class:`Timer`: class instance
        """
        self._start = time.clock()
        return self

    def __exit__(self, *args):
        """
        Attributes:
          _end (float): end time of code block
          _interval (float): time elapsed during code block
        """
        self._end = time.clock()
        self._interval = self._end - self._start
