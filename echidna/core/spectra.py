import collections
import numpy
import yaml
import copy

class SpectraParameter(object):
    def __init__(self, name, low, high, bins):
        self._name = name
        self.high = high
        self.low = low
        self.bins = bins
    def setvar(self, variable, value):
        if variable=="low":
            self.low = value
        elif variable=="high":
            self.high = value
        elif variable=="bins":
            self.bins = value
        else:
            raise TypeError("Unknown parameter name")
    def get_width(self):
        return ((self.high - self.low) / self.bins)


class SpectraConfig(object):

    def __init__(self, parameters):
        self._parameters = parameters

    @classmethod
    def load_from_file(cls, filename):
        config = yaml.load(open(filename, 'r'))
        parameters = collections.OrderedDict()
        for v in config['parameters']:
            parameters[v] = SpectraParameter(v, config['parameters'][v]['low'],
                                             config['parameters'][v]['high'],
                                             config['parameters'][v]['bins'])
        return cls(parameters)

    def getpar(self, name):
        return self._parameters[name]

    def getpars(self):
        return self._parameters.keys()

    def get_index(self, parameter):
        for i, p in enumerate(self._parameters.keys()):
            if p == parameter:
                return i
        raise IndexError("Unknown parameter %s" % parameter)

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
    """
    def __init__(self, name, num_decays, spectra_config):
        """ Initialise the spectra data container.
        """
        self._config = spectra_config
        self._raw_events = 0
        bins = []
        for v in self._config.getpars():
            bins.append(self._config.getpar(v).bins)
        self._data = numpy.zeros(shape=tuple(bins),
                                 dtype=float)
        self._name = name
        self._num_decays = num_decays

    def get_config(self):
        return self._config

    def fill(self, weight=1.0, **kwargs):
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
        # Check all keys in kwargs are in the config variables and visa versa
        for var in kwargs:
            if var not in self._config.getpars():
                raise Exception('Unknown parameter %s' % var)
        for var in self._config.getpars():
            if var not in kwargs:
                raise Exception('Missing parameter %s' % var)
        for v in self._config.getpars():
            if not self._config.getpar(v).low <= kwargs[v] < self._config.getpar(v).high:
                raise ValueError("%s out of range" % v)
        bins = []
        for v in self._config.getpars():
            bins.append((kwargs[v] - self.config.getpar(v).low) / \
                        (self._config.getpar(v).high - self._config.getpar(v).low) * \
                        self._config.getpar(v).bins)
        # Cross fingers the ordering is the same!
        self._data[tuple(bins)] += weight
        
    def project(self, parameter):
        """ Project the histogram along an `axis`.

        Args:
          axis (int): To project onto

        Returns:
          The projection of the histogram onto the given axis
        """
        axis = self._config.get_index(parameter)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.getpars()) - 1):
            if axis < i_axis+1:
                projection = projection.sum(1)
            else:
                projection = projection.sum(0)
        return projection

    def surface(self, parameter1, parameter2):
        """ Project the histogram along two axis, along the `axis`.

        Args:
          axis (int): To project away

        Returns:
          The 2d surface of the histogram.
        """
        axis1 = self._config.get_index(parameter1)
        axis2 = self._config.get_index(parameter2)
        if axis1 < 0 or axis1 > len(self._config.getpars()):
            raise IndexError("Axis index %s out of range" % axis1)
        if axis2 < 0 or axis2 > len(self._config.getpars()):
            raise IndexError("Axis index %s out of range" % axis2)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.getpars())):
            if i_axis != axis1 and i_axis != axis2:
                projection = projection.sum(i_axis)
        return projection

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

    def shrink(self, **kwargs):
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
        params = set()
        for arg in kwargs:
            var, _, high_low = arg.rpartition("_")
            params.add(var)
            if high_low == "low":
                if kwargs[arg] < self._config.getpar(var).low:
                    raise ValueError("%s low is below existing bound")
            if high_low == "high":
                if kwargs[arg] > self._config.getpar(var).high:
                    raise ValueError("%s high is above existing bound")

        slices = []
        for var in params:
            if "%s_low" not in kwargs and "%s_high" not in kwargs:
                slices.append(0, self._config.getpar(var).bins)
            # FIXME: we need to ensure that the new limits are at bin edges
            if "%s_low" % var not in kwargs:
                kwargs["%s_low" % var] = self._config.getpar(var).low
            if "%s_high" % var not in kwargs:
                kwargs["%s_high" % var] = self._config.getpar(var).high
            low_bin = (kwargs["%s_low"] - self._config.getpar(var).low) / self._config.getpar(var).get_width()
            high_bin = (kwargs["%s_high"] - self._config.getpar(var).high) / self._config.getpar(var).get_width()
            bins = int(high_bin - low_bin)
            self._config.setpar(var, kwargs["%s_low" % var], kwargs["%s_high" % var], bins=bins)
            slices.append(low_bin, high_bin)

        # Internal bookeeping complete, now slice the data
        self._data = self._data[slices]

    def add(self, spectrum):
        """ Adds a spectrum to current spectra object.

        Args:
          spectrum (:class:`core.spectra`): Spectrum to add.
        """
        for v in self._config.getpars():
            if v not in spectrum.get_config().getpars():
                raise IndexError("%s not present in new spectrum" % v)
        for v in spectrum.get_config().getpars():
            if v not in self._config.getpars():
                raise IndexError("%s not present in this spectrum" % v)
        for v in self._config.getpars():
            if self._config.getpar(v).high != spectrum.get_config().getpar(v).high:
                raise ValueError("Upper %s bounds in spectra are not equal." % v)
            if self._config.getpar(v).low != spectrum.get_config().getpar(v).low:
                raise ValueError("Lower %s bounds in spectra are not equal." % v)
            if self._config.getpar(v).bins != spctrum.get_config().getpar(v).bins:
                raise ValueError("Number of %s bins in spectra are not equal." % v)
        self._data += spectrum._data
        self._raw_events += spectrum._raw_events
        self._num_decays += spectrum._num_decays
