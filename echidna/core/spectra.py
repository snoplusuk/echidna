import numpy


class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.

    The spectra is stored as histogram binned in energy, x, radius, y, and
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d
    (energy).

    """
    def __init__(self, name, num_decays):
        """ Initialise the spectra data container.

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
        """
        self._energy_low = 0.0  # MeV
        self._energy_high = 10.0  # MeV
        self._energy_bins = 1000
        self._energy_width = (self._energy_high - self._energy_low) / self._energy_bins
        self._radial_low = 0.0  # mm
        self._radial_high = 6000.0  # mm
        self._radial_bins = 600
        self._radial_width = (self._radial_high - self._radial_low) / self._radial_bins
        self._time_low = 0.0  # years
        self._time_high = 10.0  # years
        self._time_bins = 10
        self._time_width = (self._time_high - self._time_low) / self._time_bins
        self._num_decays = num_decays
        self._data = numpy.zeros(shape=(self._energy_bins,
                                        self._radial_bins,
                                        self._time_bins),
                                 dtype=float)
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
            raise ValueError("Energy low is below exist bound")
        if(energy_high is not None and energy_high > self._energy_high):
            raise ValueError("Energy high is below exist bound")
        if(radial_low is not None and radial_low < self._radial_low):
            raise ValueError("Radial low is below exist bound")
        if(radial_high is not None and radial_high > self._radial_high):
            raise ValueError("Radial high is below exist bound")
        if(time_low is not None and time_low < self._time_low):
            raise ValueError("Time low is below exist bound")
        if(time_high is not None and time_high > self._time_high):
            raise ValueError("Time high is below exist bound")

        energy_low_bin = 0
        energy_high_bin = self._energy_bins
        if(energy_low is not None and energy_high is not None):
            energy_low_bin = (energy_low - self._energy_low) / self._energy_width
            energy_high_bin = (energy_high - self._energy_low) / self._energy_width
            self._energy_low = energy_low
            self._energy_high = energy_high
            self._energy_bins = int(energy_high_bin - energy_low_bin)

        radial_low_bin = 0
        radial_high_bin = self._radial_bins
        if(radial_low is not None and radial_high is not None):
            radial_low_bin = (radial_low - self._radial_low) / self._radial_width
            radial_high_bin = (radial_high - self._radial_low) / self._radial_width
            self._radial_low = radial_low
            self._radial_high = radial_high
            self._radial_bins = int(radial_high_bin - radial_low_bin)

        time_low_bin = 0
        time_high_bin = self._time_bins
        if(time_low is not None and time_high is not None):
            time_low_bin = (time_low - self._time_low) / self._time_width
            time_high_bin = (time_high - self._time_low) / self._time_width
            self._time_low = time_low
            self._time_high = time_high
            self._time_bins = int(time_high_bin - time_low_bin)

        # Internal bookeeping complete, now slice the data
        self._data = self._data[energy_low_bin:energy_high_bin,
                                radial_low_bin:radial_high_bin,
                                time_low_bin:time_high_bin]
