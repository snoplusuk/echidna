import numpy


class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.

    The spectra is stored as histogram binned in energy, x, radius, y, and
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d
    (energy).

    Args:
      name (str): The name of this spectra
      num_decays (float): The number of decays this spectra is created to
        represent.

    Attributes:
      _data (:class:`numpy.ndarray`): The histogram of data
      _name (str): The name of this spectra
      _energy_low (float): Lowest bin edge in MeV
      _energy_high (float): Highest bin edge in MeV
      _energy_bins (int): Number of energy bins
      _energy_width (float): Width of a single bin in MeV
      _radial_low (float): Lowest bin edge in mm
      _radial_high (float): Highest bin edge in mm
      _radial_bins (int): Number of raidal bins
      _radial_width (float): Width of a single bin in mm
      _time_low (float): Lowest bin edge in years
      _time_high (float): Highest bin edge in years
      _time_bins (int): Number of time bins
      _time_width (float): Width of a single bin in yr
      _num_decays (float): The number of decays this spectra currently
        represents.
      _raw_events (int): The number of raw events used to generate the
        spectra. Increments by one with each fill independent of
        weight.
      _style (string): Pyplot-style plotting style e.g. "b-" or
        {"color": "blue"}.
      _rois (dict): Dictionary containing the details of any ROI, along
        any axis, which has been defined.
    """
    def __init__(self, name, num_decays):
        """ Initialise the spectra data container.
        """
        self._energy_low = 0.0  # MeV
        self._energy_high = 10.0  # MeV
        self._energy_bins = 1000
        self._radial_low = 0.0  # mm
        self._radial_high = 10000.0  # mm
        self._radial_bins = 1000
        self._time_low = 0.0  # years
        self._time_high = 10.0  # years
        self._time_bins = 10
        self.calc_widths()
        self._num_decays = num_decays
        self._raw_events = 0
        self._data = numpy.zeros(shape=(self._energy_bins,
                                        self._radial_bins,
                                        self._time_bins),
                                 dtype=float)
        self._style = {"color": "blue"}  # default style for plotting
        self._rois = {}
        self._name = name

    def fill(self, energy, radius, time, weight=1.0):
        """ Fill the bin for the `energy` `radius` and `time` with weight.

        Args:
          energy (float): Energy value to fill.
          raidus (float): Radial value to fill.
          time (float): Time value to fill.
          weight (float, optional): Defaults to 1.0, weight to fill the bin
            with.

        Raises:
          ValueError: If the energy, radius or time is beyond the bin limits.
        """
        if not self._energy_low <= energy < self._energy_high:
            raise ValueError("Energy out of range")
        if not self._radial_low <= radius < self._radial_high:
            raise ValueError("Radius out of range")
        if not self._time_low <= time < self._time_high:
            raise ValueError("Time out of range")
        energy_bin = (energy - self._energy_low) / (self._energy_high - self._energy_low) * self._energy_bins
        radial_bin = (radius - self._radial_low) / (self._radial_high - self._radial_low) * self._radial_bins
        time_bin = (time - self._time_low) / (self._time_high - self._time_low) * self._time_bins
        self._data[energy_bin, radial_bin, time_bin] += weight

    def shrink_to_roi(self, lower_limit, upper_limit, axis):
        """ Shrink spectrum to a defined Region of Interest (ROI)

        Shrinks spectrum to given ROI and saves ROI parameters.

        Args:
          lower_limit (float): Lower bound of ROI, along given axis.
          upper_limit (float): Upper bound of ROI, along given axis.
          axis (int): Axis along which to define ROI.
        """
        integral_full = self.sum()  # Save integral of full spectrum

        # Shrink to ROI
        if axis == 0:
            self.shrink(energy_low=lower_limit, energy_high=upper_limit)
        elif axis == 1:
            self.shrink(radial_low=lower_limit, radial_high=upper_limit)
        elif axis == 2:
            self.shrink(time_low=lower_limit, time_high=upper_limit)

        # Calculate efficiency
        integral_roi = self.sum()  # Integral of spectrum over ROI
        efficiency = float(integral_roi) / float(integral_full)
        self._rois[axis] = {"low": lower_limit, "high": upper_limit,
                            "efficiency": efficiency}

    def get_roi(self, axis):
        """ Access information about a predefined ROI on a given axis

        Returns:
          dict: Dictionary containing parameters defining the ROI, on
            the given axis.
        """
        return self._rois[axis]

    def set_style(self, style):
        """ Sets plotting style.

        Styles should be valid pyplot style strings e.g. "b-", for a
        blue line, or dictionaries of strings e.g. {"color": "red"}.

        Args:
          style (string): Pyplot-style plotting style.
        """
        self._style = style

    def get_style(self):
        """
        Returns:
          string/dict: :attr:`_style` - pyplot-style plotting style.
        """
        return self._style

    def project(self, axis):
        """ Project the histogram along an `axis`.

        Args:
          axis (int): To project onto

        Returns:
          The projection of the histogram onto the given axis
        """
        if axis == 0:
            return self._data.sum(1).sum(1)
        elif axis == 1:
            return self._data.sum(0).sum(1)
        elif axis == 2:
            return self._data.sum(0).sum(0)

    def surface(self, axis):
        """ Project the histogram along two axis, along the `axis`.

        Args:
          axis (int): To project away

        Returns:
          The 2d surface of the histogram.
        """
        return self._data.sum(axis)

    def sum(self):
        """ Calculate and return the sum of the `_data` values.

        Returns:
          The sum of the values in the `_data` histogram.
        """
        return self._data.sum()

    def scale(self, num_decays):
        """ Scale THIS spectra to represent *num_decays* worth of decays over
        the entire unshrunken spectra.

        This rescales each bin by the ratio of *num_decays* to
        *self._num_decays*, i.e. it changes the spectra from representing
        *self._num_decays* to *num_decays*. *self._num_decays* is updated
        to equal *num_decays* after.

        Args:
          num_decays (float): Number of decays this spectra should represent.
        """
        self._data = numpy.multiply(self._data, num_decays / self._num_decays)
        self._num_decays = num_decays

    def shrink(self, energy_low=None, energy_high=None, radial_low=None,
               radial_high=None, time_low=None, time_high=None):
        """ Shrink the data such that it only contains values between energy_low
        and energy_high (for example) by slicing. This updates the internal bin
        information as well as the data.

        Args:
          energy_low (float): Optional new low bound of the energy.
          energy_low (float): Optional new high bound of the energy.
          radial_low (float): Optional new low bound of the radius.
          radial_low (float): Optional new high bound of the radius.
          time_low (float): Optional new low bound of the time.
          time_low (float): Optional new high bound of the time.

        Notes:
          The logic in this method is the same for each dimension, first
        check the new values are within the existing ones (can only compress).
        Then calculate the low bin number and high bin number (relative to the
        existing binning low). Finally update all the bookeeping and slice.
        """
        if(energy_low is not None and energy_low < self._energy_low):
            raise ValueError("Energy low is below existing bound")
        if(energy_high is not None and energy_high > self._energy_high):
            raise ValueError("Energy high is above existing bound")
        if(radial_low is not None and radial_low < self._radial_low):
            raise ValueError("Radial low is below existing bound")
        if(radial_high is not None and radial_high > self._radial_high):
            raise ValueError("Radial high is above existing bound")
        if(time_low is not None and time_low < self._time_low):
            raise ValueError("Time low is below existing bound")
        if(time_high is not None and time_high > self._time_high):
            raise ValueError("Time high is above existing bound")

        energy_low_bin = 0
        energy_high_bin = self._energy_bins
        if(energy_low is not None and energy_high is not None):
            energy_low_bin = numpy.rint((energy_low - self._energy_low) / self._energy_width)
            energy_high_bin = numpy.rint((energy_high - self._energy_low) / self._energy_width)
            self._energy_low = energy_low
            self._energy_high = energy_high
            self._energy_bins = int(energy_high_bin - energy_low_bin)

        radial_low_bin = 0
        radial_high_bin = self._radial_bins
        if(radial_low is not None and radial_high is not None):
            radial_low_bin = numpy.rint((radial_low - self._radial_low) / self._radial_width)
            radial_high_bin = numpy.rint((radial_high - self._radial_low) / self._radial_width)
            self._radial_low = radial_low
            self._radial_high = radial_high
            self._radial_bins = int(radial_high_bin - radial_low_bin)

        time_low_bin = 0
        time_high_bin = self._time_bins
        if(time_low is not None and time_high is not None):
            time_low_bin = numpy.rint((time_low - self._time_low) / self._time_width)
            time_high_bin = numpy.rint((time_high - self._time_low) / self._time_width)
            self._time_low = time_low
            self._time_high = time_high
            self._time_bins = int(time_high_bin - time_low_bin)

        # Internal bookeeping complete, now slice the data
        self._data = self._data[energy_low_bin:energy_high_bin,
                                radial_low_bin:radial_high_bin,
                                time_low_bin:time_high_bin]

    def cut(self, energy_low=None, energy_high=None, radial_low=None,
            radial_high=None, time_low=None, time_high=None):
        """ Similar to :meth:`shrink`, but updates scaling information.

        If a spectrum is cut using :meth:`shrink`, subsequent calls to
        :meth:`scale` the spectrum must still scale the *full* spectrum
        i.e. before any cuts. The user supplies the number of decays
        the full spectrum should now represent.

        However, sometimes it is more useful to be able specify the
        number of events the revised spectrum should represent. This
        method updates the scaling information, so that it becomes the
        new *full* spectrum.

        Args:
          energy_low (float, optional): New low bound of the energy.
          energy_low (float, optional): New high bound of the energy.
          radial_low (float, optional): New low bound of the radius.
          radial_low (float, optional): New high bound of the radius.
          time_low (float, optional): New low bound of the time.
          time_low (float, optional): New high bound of the time.
        """
        initial_count = self.sum()  # Store initial count
        self.shrink(energy_low, energy_high, radial_low, radial_high,
                    time_low, time_high)
        new_count = self.sum()
        reduction_factor = float(new_count) / float(initial_count)
        # This reduction factor tells us how much the number of detected events
        # has been reduced by shrinking the spectrum. We want the number of
        # decays that the spectrum should now represent to be reduced by the
        # same factor
        self._num_decays *= reduction_factor

    def add(self, spectrum):
        """ Adds a spectrum to current spectra object.

        Args:
          spectrum (:class:`core.spectra`): Spectrum to add.
        """
        if not numpy.allclose(self._energy_low, spectrum._energy_low):
            raise ValueError("Lower energy bounds in spectra are not equal.")
        if not numpy.allclose(self._energy_high, spectrum._energy_high):
            raise ValueError("Upper energy bounds in spectra are not equal.")
        if not numpy.allclose(self._energy_bins, spectrum._energy_bins):
            raise ValueError("Number of energy bins in spectra are not equal.")
        if not numpy.allclose(self._energy_width, spectrum._energy_width):
            raise ValueError("Width of energy bins in spectra are not equal.")
        if not numpy.allclose(self._radial_low, spectrum._radial_low):
            raise ValueError("Lower radial bounds in spectra are not equal.")
        if not numpy.allclose(self._radial_high, spectrum._radial_high):
            raise ValueError("Upper radial bounds in spectra are not equal.")
        if not numpy.allclose(self._radial_bins, spectrum._radial_bins):
            raise ValueError("Number of radial bins in spectra are not equal.")
        if not numpy.allclose(self._radial_width, spectrum._radial_width):
            raise ValueError("Width of radial bins in spectra are not equal.")
        if not numpy.allclose(self._time_low, spectrum._time_low):
            raise ValueError("Lower time bounds in spectra are not equal.")
        if not numpy.allclose(self._time_high, spectrum._time_high):
            raise ValueError("Upper time bounds in spectra are not equal.")
        if not numpy.allclose(self._time_bins, spectrum._time_bins):
            raise ValueError("Number of time bins in spectra are not equal.")
        if not numpy.allclose(self._time_width, spectrum._time_width):
            raise ValueError("Width of time bins in spectra are not equal.")
        self._data += spectrum._data
        self._raw_events += spectrum._raw_events
        self._num_decays += spectrum._num_decays

    def rebin(self, new_bins):
        """ Rebin spectra data into a smaller spectra of the same rank whose
        dimensions are factors of the original dimensions.

        Args:
          new_bins (tuple): New bin sizes for spectra. Should be in order of
            energy, radial, time.

        Raises:
          ValueError: Shape mismatch. Number of dimenesions are different.
          ValueError: Old bins/ New bins must be integer
        """
        if self._data.ndim != len(new_bins):
            raise ValueError("Shape mismatch: %s->%s" % (self._data.shape,
                                                         new_bins))
        for i in range(len(new_bins)):
            if self._data.shape[i] % new_bins[i] != 0:
                raise ValueError("Old bins/New bins must be integer old: %s"
                                 " new: %s" % (self._data.shape, new_bins))
        compression_pairs = [(d, c//d) for d, c in zip(new_bins,
                                                       self._data.shape)]
        flattened = [l for p in compression_pairs for l in p]
        self._data = self._data.reshape(flattened)
        for i in range(len(new_bins)):
            self._data = self._data.sum(-1*(i+1))
        self._energy_bins = new_bins[0]
        self._radial_bins = new_bins[1]
        self._time_bins = new_bins[2]
        self.calc_widths()

    def calc_widths(self):
        """ Recalculates bin widths
        """
        self._energy_width = (self._energy_high - self._energy_low) / self._energy_bins
        self._radial_width = (self._radial_high - self._radial_low) / self._radial_bins
        self._time_width = (self._time_high - self._time_low) / self._time_bins

    def copy(self, name=None):
        """ Copies the current spectra and returns a new identical one.

        Args:
          name (string, optional): Name of the new copied spectrum.
            Default is the name of the current spectrum.

        Returns:
          :class:`echidna.core.spectra.Spectra` object which is identical to
            the current spectra apart from possibly its name.
        """
        if not name:
            name = self._name
        new_spectrum = Spectra(name, 0.)
        new_spectrum._energy_low = self._energy_low
        new_spectrum._energy_high = self._energy_high
        new_spectrum._energy_bins = self._energy_bins
        new_spectrum._radial_low = self._radial_low
        new_spectrum._radial_high = self._radial_high
        new_spectrum._radial_bins = self._radial_bins
        new_spectrum._time_low = self._time_low
        new_spectrum._time_high = self._time_high
        new_spectrum._time_bins = self._time_bins
        new_spectrum.calc_widths()
        new_spectrum._data = numpy.zeros(shape=numpy.shape(self._data),
                                         dtype=float)
        new_spectrum.add(self)
        new_spectrum._style = self._style
        new_spectrum._rois = self._rois
        return new_spectrum
