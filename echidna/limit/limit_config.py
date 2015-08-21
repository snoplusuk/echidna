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
      _counts_down (bool): True if :attr:`_counts` is defined such that
        it counts down from a high count to a low count. Otherwise
        set as False - countin up.
      _current_count (float): current count (scaling). The value most
        recently returned by :meth:`get_count()`.
      _sigma (float): prior constraint on counts
      _chi_squareds (:class:`numpy.array`): Array col-0 =
        :obj:`chi_squared`, col-1 = :obj:`get_counts`, col-2 =
        :meth:`spectra.sum()`.
      _minium_bin (float): Used to save position (bin number) of minimum
        of :attr:`_chi_squareds`.

    Args:
      prior_counts (float): prior/expected counts
      counts (list): list of count (*float*) to use for scaling
      sigma (float, optional): prior constraint on counts
    """
    def __init__(self, prior_count, counts, sigma=None):
        self._prior_count = prior_count
        self._counts = counts
        if (counts[-1] - counts[0]) < 0:  # Counts list counts down
            self._counts_down = True
        else:
            self._counts_down = False
        self._current_count = None
        self._sigma = sigma
        self._chi_squareds = numpy.zeros(shape=(3, 0), dtype=float)
        self._minimum_bin = None
        self._limit_count = None

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
            self._current_count = count
            yield count

    def get_current_count(self):
        """
        Returns:
          float: current count (scaling). The value most recently
            returned by :meth:`get_count()`.
        """
        return self._current_count

    def get_sigma(self):
        """
        Returns:
          float: :attr:`_sigma`
        """
        return self._sigma

    def add_chi_squared(self, chi_squared, scaling, events):
        """ Add chi squared to config chi squared array.

        Args:
          chi_squared (float): chi squared value to add
        """
        entry_to_append = numpy.zeros((3, 1), dtype=float)
        entry_to_append[0][0] = chi_squared
        entry_to_append[1][0] = scaling
        entry_to_append[2][0] = events
        # Append new entry, axis is 1 as we are appending an entry not a column
        self._chi_squareds = numpy.append(self._chi_squareds,
                                          entry_to_append, axis=1)

    def get_chi_squareds(self):
        """
        Returns:
          :class:`numpy.ndarray`: Array col-0 = :obj:`chi_squared`s,
            col-1 = :obj:`get_counts`, col-2 = :meth:`spectra.sum()`.
        """
        return self._chi_squareds

    def get_minimum(self, minimum_bin=False):
        """ Get minimum value from chi_squared array.

        Args:
          minimum_bin (bool, optional): If True, returns the position (bin
            number) in addition to minimum chi squared. Default is False.

        Returns:
          float: Minimum of :attr:`_chi_squareds`
          tuple: (Minimum of :attr:`_chi_squareds`, bin number of minimum)
        """
        minimum = numpy.min(self._chi_squareds[0])
        minimum_bin = numpy.where(self._chi_squareds[0] == minimum)
        # Set counts corresponding to minimum
        # --> Note scaling counts NOT actual events
        self._limit_count = self._counts[minimum_bin]
        if minimum_bin:
            return minimum, minimum_bin
        else:
            return minimum

    def get_counts_down(self):
        """
        Returns:
          bool: :attr:`_counts_down`.
        """
        return self._counts_down

    def get_first_bin_above(self, limit):
        """ For signal, determine the count corresponding to limit.

        The limit count is the first count value that takes the chi
        squared over the threshold set by the desired confidence limit.

        Args:
          limit (float): chi squared value corresponding to desired
            confidence limit

        Raises:
          AttributeError: If :attr:`_counts` is defined to count down.

        Returns:
          float: number of events (result of
            :meth:`echidna.core.spectra.Spectra.sum()`) at limit
        """
        if self._counts_down:  # Don't want to use this method
            raise AttributeError("Method `get_first_bin_above` will give "
                                 "incorrect result, use `get_last_bin_above` "
                                 "instead")
        first_bin_above = numpy.where(self._chi_squareds[0] > limit)[0][0]
        # Set counts corresponding to minimum
        # --> Note scaling counts NOT actual events
        self._limit_count = self._counts[first_bin_above]
        return self._chi_squareds[2, first_bin_above]

    def get_last_bin_above(self, limit):
        """ For signal, determine the count corresponding to limit.

        Method to use if signal counts have been defined in such a way
        that they count down to zero signal contribution. Produces
        exactly the same result as :meth:`get_first_bin_above` but
        where you have this set-up.

        Args:
          limit (float): chi squared value corresponding to desired
            confidence limit

        Raises:
          AttributeError: If :attr:`_counts` is defined to count up.

        Returns:
          float: number of events (result of
            :meth:`echidna.core.spectra.Spectra.sum()`) at limit
        """
        if not self._counts_down:  # Don't want to use this method
            raise AttributeError("Method `get_last_bin_above` will give "
                                 "incorrect result, use `get_first_bin_above` "
                                 "instead")
        last_bin_above = numpy.where(self._chi_squareds[0] > limit)[0][-1]
        # Set counts corresponding to minimum
        # --> Note scaling counts NOT actual events
        self._limit_count = self._counts[last_bin_above]
        return self._chi_squareds[2, last_bin_above]

    def get_limit_count(self):
        """ Returns the count for the full spectrum, at the limit.

        Scaling the :class:`Spectra` instance, corresponding to this
        :class:`LimitConfig` instance, to the value returned by this
        method provides a representation of the full spectrum at the
        limit.

        Returns:
          float: count (from :attr:`_counts`) at limit.

        Raises:
          AssertionError: if :attr:`_limit_counts` is None

        .. note::

          Requires either :meth:`get_minimum` or :meth:`get_first_bin_above`
          to be called, before it can be called.

        .. note::

          For backgrounds, where multiple backgrounds have been floated
          and :meth:`reset_chi_squareds` has been called, this will
          only give you the limit count for the last iteration. For the
          preferred value over the full float of systematics use the
          :class:`SystAnalyser` class.
        """
        if self._limit_count is None:
            raise AssertionError("Limit count not set, call get_minimum or"
                                 " get_first_bin_above first")
        else:
            return self._limit_count

    def reset_chi_squareds(self):
        """ Resets :attr:`_chi_squareds` to an empty
          :class:`numpy.ndarray`
        """
        self._chi_squareds = numpy.zeros(shape=(3, 0), dtype=float)
