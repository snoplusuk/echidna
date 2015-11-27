""" Utilities module to include functions that may be useful throughout
echidna.
"""
import echidna

import time
import logging
import os
import socket


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


def start_logging(short_name=False):
    """ Function to initialise logging output

    Adapted from:
    https://docs.python.org/2/howto/logging-cookbook.html#logging-to-multiple-destinations
    """
    # set up logging to file
    if short_name:
        logging.basicConfig(
            level=logging.DEBUG,  # Include all logging levels in file
            # Format filename.py:XX [function_name()] LEVEL: message
            format=("%(filename)s:%(lineno)s [%(funcName)s()] "
                    "%(levelname)-8s: %(message)s"),
            filename="echidna.log",
            filemode='w')
    else:
        logging.basicConfig(
            level=logging.DEBUG,  # Include all logging levels in file
            # Format filename.py:XX [function_name()] LEVEL: message
            format=("%(filename)s:%(lineno)s [%(funcName)-20s()] "
                    "%(levelname)-8s: %(message)s"),
            filename="echidna.%s.%d.log" % (socket.gethostname(), os.getpid()),
            filemode='w')

    # define Handler which writes WARNING messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-8s %(name)20s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logging.info("echidna-v%s" % echidna.__version__)
