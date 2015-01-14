import numpy

class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.
    
    The spectra is stored as histogram binned in energy, x, radius, y, and 
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d 
    (energy).

    Attributes:
      _energy_low (float): Lowest bin edge in MeV
      _energy_high (float): Highest bin edge in MeV
      _energy_bins (int): Number of energy bins
      _radial_low (float): Lowest bin edge in mm
      _radial_high (float): Highest bin edge in mm
      _radial_bins (int): Number of raidal bins
      _time_low (float): Lowest bin edge in years
      _time_high (float): Highest bin edge in years
      _time_bins (int): Number of time bins
    """
    _energy_low = 0.0 # MeV
    _energy_high = 10.0 # MeV
    _energy_bins = 1000
    _radial_low = 0.0 # mm
    _radial_high = 6000.0 # mm
    _radial_bins = 600
    _time_low = 0.0 # years
    _time_high = 10.0 # years
    _time_bins = 10
    def __init__(self, name):
        """ Initialise the spectra data container.

        Args:
          name (str): The name of this spectra

        Attributes:
          _data (:class:`numpy.ndarray`): The histogram of data
          _name (str): The name of this spectra
        """
        self._data = numpy.zeros(shape=(Spectra._energy_bins, 
                                        Spectra._radial_bins, 
                                        Spectra._time_bins), 
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
        if not Spectra._energy_low <= energy < Spectra._energy_high:
            raise ValueError("Energy out of range")
        if not Spectra._radial_low <= radius < Spectra._radial_high:
            raise ValueError("Radius out of range")
        if not Spectra._time_low <= time < Spectra._time_high:
            raise ValueError("Time out of range")
        energy_bin = (energy - Spectra._energy_low) / (Spectra._energy_high - Spectra._energy_low) * Spectra._energy_bins
        radial_bin = (radius - Spectra._radial_low) / (Spectra._radial_high - Spectra._radial_low) * Spectra._radial_bins
        time_bin = (time - Spectra._time_low) / (Spectra._time_high - Spectra._time_low) * Spectra._time_bins
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

    def sum(self):
        """ Calculate and return the sum of the `_data` values.

        Returns:
          The sum of the values in the `_data` histogram.
        """
        return self._data.sum()

    def normalise(self, count):
        """ Normalise the total counts in the spectra to count, i.e. times each
        bin by count / self.sum().
        
        Args:
          count (float): Total number of events to normalise to.
        """
        numpy.multiply(self._data, count / self.sum())
