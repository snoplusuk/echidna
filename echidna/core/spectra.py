import collections
import numpy
import yaml
import copy


class SpectraParameter(object):
    """Simple data container that holds information for a Spectra parameter
    (i.e. axis of the spectrum).

    Args:
      name (str): The name of this parameter
      low (float): The lower limit of this parameter
      high (float): The upper limit of this parameter
      bins (int): The number of bins for this parameter

    Attributes:
      _name (str): The name of this parameter
      _low (float): The lower limit of this parameter
      _high (float): The upper limit of this parameter
      _bins (int): The number of bins for this parameter
      _dimension (int): The position in the data array for this parameter.
    """

    def __init__(self, name, low, high, bins):
        """Initialise SpectraParameter class
        """
        self._name = name
        self._high = high
        self._low = low
        self._bins = bins

    def set_par(self, **kwargs):
        """Set a limit / binning parameter after initialisation.

        Args:
          \**kwargs (dict): keyword arguments

        .. note::

        Keyword arguments include:

        * low (float): Value to set the lower limit to of the parameter
        * high (float): Value to set the higher limit to of the parameter
        * bins (int): Value to set the number of bins of the parameter

        Raises:
          TypeError: Unknown variable type passed as a kwarg.
        """
        for kw in kwargs:
            if kw == "low":
                self._low = kwargs[kw]
            elif kw == "high":
                self._high = kwargs[kw]
            elif kw == "bins":
                self._bins = kwargs[kw]
            else:
                raise TypeError("Unhandled parameter name / type %s" % kw)

    def get_width(self):
        """Get the width of the binning for the parameter

        Returns:
          float: Bin width.
        """
        return (self._high - self._low) / float(self._bins)

    def get_unit(self):
        """Get the default unit for a given parameter

        Raises:
          Exception: Unknown parameter.

        Returns:
          string: Unit of the parameter
        """
        if self._name.split['_'][0] == "energy":
            return "MeV"
        if self._name.split['_'][0] == "radial":
            return "mm"
        if self._name.split['_'][0] == "time":
            return "years"
        else:
            raise Exception("%s is an unknown parameter"
                            % self._name.split('_')[0])

    def round(self, x):
        """ Round the value to nearest bin edge

        Args:
          x (float): Value to round.

        Returns:
          float: The value of the closest bin edge to x
        """
        return round(x/self.get_width())*self.get_width()


class SpectraConfig(object):
    """Configuration container for Spectra objects.  Able to load
    directly with a set list of SpectraParameters or from yaml
    configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        SpectraParameter objects

    Attributes:
      _parameters (:class:`collections.OrderedDict`): List of
        SpectraParameter objects
    """

    def __init__(self, parameters):
        """Initialise SpectraConfig class
        """
        self._parameters = parameters

    @classmethod
    def load_from_file(cls, filename):
        """Initialise SpectraConfig class from a config file (classmethod).

        Args:
          filename (str) path to config file

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        parameters = collections.OrderedDict()
        for v in config['parameters']:
            parameters[v] = SpectraParameter(v, config['parameters'][v]['low'],
                                             config['parameters'][v]['high'],
                                             config['parameters'][v]['bins'])
        return cls(parameters)

    def get_par(self, name):
        """Get a named SpectraParameter.

        Args:
          name (string): Name of the parameter.

        Returns:
          :class:`echidna.core.spectra.SpectraParameter`: Named parameter.
        """
        return self._parameters[name]

    def get_pars(self):
        """Get list of all parameter names in the config.

        Returns:
          list: List of parameter names
        """
        return sorted(self._parameters.keys())

    def get_index(self, parameter):
        """Return the index of a parameter within the existing set

        Args:
          parameter (string): Name of the parameter.

        Raises:
          IndexError: parameter is not in the config.

        Returns:
          int: Index of the parameter
        """
        for i, p in enumerate(self.get_pars()):
            if p == parameter:
                return i
        raise IndexError("Unknown parameter %s" % parameter)

    def get_dims(self):
        """Get list of dimension names.
        The _mc, _reco and _truth suffixes are removed.

        Returns:
          list: List of the dimensions names of the config.
        """
        dims = []
        for par in sorted(self._parameters.keys()):
            par = par.split('_')[:-1]
            dim = ""
            for entry in par:
                dim += entry+"_"
            dims.append(dim[:-1])
        return dims

    def get_dim(self, par):
        """Get the dimension of par.
        The _mc, _reco and _truth suffixes are removed.

        Args:
          par (string): Name of the parameter

        Returns:
          The dimension of par
        """
        dim = ""
        for entry in par.split('_')[:-1]:
            dim += entry+"_"
        return dim[:-1]

    def get_dim_type(self, dim):
        """Returns the type of the dimension i.e. mc, reco or truth.

        Args:
          dim (string): The name of the dimension

        Raises:
          IndexError: dim is not in the spectra.

        Returns:
          string: The type of the dimension (mc, reco or truth)
        """
        for par in sorted(self._parameters.keys()):
            par_split = par.split('_')[:-1]
            cur_dim = ""
            for entry in par_split:
                cur_dim += entry+"_"
            if cur_dim[:-1] == dim:
                return par.split('_')[-1]
        raise IndexError("No %s dimension in spectra" % dim)


class Spectra(object):
    """ This class contains a spectra as a function of energy, radius and time.

    The spectra is stored as histogram binned in energy, x, radius, y, and
    time, z. This histogram can be flattened to 2d (energy, radius) or 1d
    (energy).

    Args:
      name (str): The name of this spectra
      num_decays (float): The number of decays this spectra is created to
        represent.
      spectra_config (:class:`SpectraConfig`): The configuration object

    Attributes:
      _data (:class:`numpy.ndarray`): The histogram of data
      _name (str): The name of this spectra
      _config (:class:`SpectraConfig`): The configuration object
      _num_decays (float): The number of decays this spectra currently
        represents.
      _raw_events (int): The number of raw events used to generate the
        spectra. Increments by one with each fill independent of
        weight.
      _bipo (int): Flag to indicate whether the bipo cut was applied to the
        spectra. 0 is No Cut. 1 is Cut.
        Int type as HDF5 does not support bool.
      _style (string): Pyplot-style plotting style e.g. "b-" or
        {"color": "blue"}.
      _rois (dict): Dictionary containing the details of any ROI, along
        any axis, which has been defined.
    """
    def __init__(self, name, num_decays, spectra_config):
        """ Initialise the spectra data container.
        """
        self._config = spectra_config
        self._raw_events = 0
        bins = []
        for v in self._config.get_pars():
            bins.append(self._config.get_par(v)._bins)
        self._data = numpy.zeros(shape=tuple(bins),
                                 dtype=float)
        # Flag for indicating bipo cut. HDF5 does not support bool so
        # 0 = no cut and 1 = cut
        self._bipo = 0
        self._style = {"color": "blue"}  # default style for plotting
        self._rois = {}
        self._name = name
        self._num_decays = num_decays

    def get_config(self):
        """ Get the config of the spectra.

        Returns:
          :class:`echidna.core.spectra.SpectraConfig`: The config of
            the spectra.
        """
        return self._config

    def fill(self, weight=1.0, **kwargs):
        """ Fill the bin with weight.  Note that values for all named
        parameters in the spectra's config (e.g. energy, radial) must be
        passed.

        Args:
          weight (float, optional): Defaults to 1.0, weight to fill the bin
            with.
          \**kwargs (float): Named values (e.g. for energy_mc, radial_mc)

        Raises:
          Exception: Parameter in \**kwargs is not in config.
          Exception: Parameter in config is not in \**kwargs.
          ValueError: If the energy, radius or time is beyond the bin limits.
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        for par in kwargs:
            if par not in self._config.get_pars():
                raise Exception('Unknown parameter %s' % par)
        for par in self._config.get_pars():
            if par not in kwargs:
                raise Exception('Missing parameter %s' % par)
        for v in self._config.get_pars():
            if not self._config.get_par(v)._low <= kwargs[v] < \
                    self._config.get_par(v)._high:
                raise ValueError("%s out of range: %s" % (v, kwargs[v]))
        bins = []
        for v in self._config.get_pars():
            bins.append((kwargs[v] - self._config.get_par(v)._low) /
                        (self._config.get_par(v)._high -
                         self._config.get_par(v)._low) *
                        self._config.get_par(v)._bins)
        # Cross fingers the ordering is the same!
        self._data[tuple(bins)] += weight

    def shrink_to_roi(self, lower_limit, upper_limit, dimension):
        """ Shrink spectrum to a defined Region of Interest (ROI)

        Shrinks spectrum to given ROI and saves ROI parameters.

        Args:
          lower_limit (float): Lower bound of ROI, along given axis.
          upper_limit (float): Upper bound of ROI, along given axis.
          dimension (str): Name of the dimension to shrink.
        """
        integral_full = self.sum()  # Save integral of full spectrum

        # Shrink to ROI
        kw_low = dimension+"_low"
        kw_high = dimension+"_high"
        kw_args = {kw_low: lower_limit,
                   kw_high: upper_limit}
        self.shrink(**kw_args)

        # Calculate efficiency
        integral_roi = self.sum()  # Integral of spectrum over ROI
        efficiency = float(integral_roi) / float(integral_full)
        self._rois[dimension] = {"low": lower_limit, "high": upper_limit,
                                 "efficiency": efficiency}

    def get_roi(self, dimension):
        """ Access information about a predefined ROI for a given dimension

        Returns:
          dict: Dictionary containing parameters defining the ROI, on
            the given dimension.
        """
        return self._rois[dimension]

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

    def project(self, dimension):
        """ Project the histogram along an axis for a given dimension.
        Note that the dimension must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension (str): parameter to project onto

        Returns:
          :class:`numpy.ndarray`: The projection of the histogram onto the
            given axis
        """
        axis = self._config.get_index(dimension)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars()) - 1):
            if axis < i_axis+1:
                projection = projection.sum(1)
            else:
                projection = projection.sum(0)
        return projection

    def surface(self, dimension1, dimension2):
        """ Project the histogram along two axes for the given dimensions.
        Note that the dimensions must be one of the named parameters in
        the SpectraConfig.

        Args:
          dimension1 (str): first parameter to project onto
          dimension1 (str): second parameter to project onto

        Raises:
          IndexError: Axis of dimension1 is out of range
          IndexError: Axis of dimension2 is out of range

        Returns:
          :class:`numpy.ndarray`: The 2d surface of the histogram.
        """
        axis1 = self._config.get_index(dimension1)
        axis2 = self._config.get_index(dimension2)
        if axis1 < 0 or axis1 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis1)
        if axis2 < 0 or axis2 > len(self._config.get_pars()):
            raise IndexError("Axis index %s out of range" % axis2)
        projection = copy.copy(self._data)
        for i_axis in range(len(self._config.get_pars())):
            if i_axis != axis1 and i_axis != axis2:
                projection = projection.sum(i_axis)
        return projection

    def sum(self):
        """ Calculate and return the sum of the `_data` values.

        Returns:
          float: The sum of the values in the `_data` histogram.
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
        """ Shrink the data such that it only contains values between low and
        high for a given dimension by slicing. This updates the internal bin
        information as well as the data.

        Args:
          \**kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.

        .. note:

          The logic in this method is the same for each dimension, first
        check the new values are within the existing ones (can only compress).
        Then calculate the low bin number and high bin number (relative to the
        existing binning low). Finally update all the bookeeping and slice.

        Raises:
          IndexError: Parameter which is being shrank does not exist in the
            config file.
          ValueError: [parameter]_low value is lower than the parameters lower
            bound.
          ValueError: [parameter]_high value is lower than the parameters
            higher bound.
          IndexError: Suffix to [parameter] is not _high or _low.
        """
        # First check dimensions and bounds in kwargs are valid
        for arg in kwargs:
            high_low = arg.split("_")[-1]
            par = arg[:-1*(len(high_low)+1)]
            if par not in self._config.get_pars():
                raise IndexError("%s is not a parameter in the config" % par)
            if high_low == "low":
                if kwargs[arg] < self._config.get_par(par)._low:
                    raise ValueError("%s low is below existing bound"
                                     % kwargs[par])
            elif high_low == "high":
                if kwargs[arg] > self._config.get_par(par)._high:
                    raise ValueError("%s high is above existing bound")
            else:
                raise IndexError("%s index invalid. Index must be of the form"
                                 "[dimension name]_high or"
                                 "[dimension name]_low" % arg)
        slices_low = []
        slices_high = []
        for par_name in self._config.get_pars():
            par = self._config.get_par(par_name)
            kw_low = "%s_low" % par_name
            kw_high = "%s_high" % par_name
            if "%s_low" % par_name not in kwargs:
                kwargs[kw_low] = par._low
            if "%s_high" % par_name not in kwargs:
                kwargs[kw_high] = par._high
            # Round down the low bin
            low_bin = int((kwargs[kw_low] - par._low) / par.get_width())
            # Round up the high bin
            high_bin = numpy.ceil((kwargs[kw_high] - par._low) /
                                  par.get_width())
            # new_low is the new lower first bin edge
            new_low = par.round(par._low + low_bin * par.get_width())
            # new_high is the new upper last bin edge
            new_high = par.round(par._low + high_bin * par.get_width())
            # Correct floating point errors: If the difference between
            # input high/low and calculated new high/low is approximately
            # equal (<1%) to a bin width then assume user requested the bin
            # above/below to be cut.
            if numpy.fabs(new_high - kwargs[kw_high]) > (0.99 *
                                                         par.get_width()):
                print ("WARNING: Correcting possible floating point error in "
                       "spectra.Spectra.shrink\n%s was the input. %s is the "
                       "calculated value for %s" % (kwargs[kw_low],
                                                    new_low, kw_low))
                if (new_high - kwargs[kw_high]) > 0.0:
                    high_bin -= 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                else:
                    high_bin += 1
                    new_high = par.round(par._low + high_bin * par.get_width())
                print "Corrected %s to %s" % (kw_low, new_low)
            if numpy.fabs(new_low - kwargs[kw_low]) > (0.99 * par.get_width()):
                print ("WARNING: Correcting possible floating point error in "
                       "spectra.Spectra.shrink\n%s was the input. %s is the "
                       "calculated value for %s" % (kwargs[kw_low],
                                                    new_low, kw_low))
                if (new_low - kwargs[kw_low]) > 0.0:
                    low_bin -= 1
                    new_low = par._low + low_bin * par.get_width()
                else:
                    low_bin += 1
                    new_low = par._low + low_bin * par.get_width()
                print "Corrected %s to %s" % (kw_low, new_low)
            slices_high.append(high_bin)
            slices_low.append(low_bin)
            new_bins = high_bin - low_bin
            par.set_par(low=new_low, high=new_high, bins=new_bins)
        # First set up the command then evaluate. Hacky but hey ho.
        cmd = "self._data["
        for i in range(len(slices_low)):
            low = str(slices_low[i])
            high = str(slices_high[i])
            cmd += low+":"+high+","
        cmd = cmd[:-1]+"]"
        self._data = eval(cmd)

    def cut(self, **kwargs):
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
          \**kwargs (float): Named parameters to slice on; note that these
            must be of the form [name]_low or [name]_high where [name]
            is a dimension present in the SpectraConfig.
        """
        initial_count = self.sum()  # Store initial count
        self.shrink(**kwargs)
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
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum to add.

        Raises:
          ValueError: spectrum has different dimenstions to the current
            spectra.
          IndexError: spectrum does not contain a dimension(s) that is in the
            current spectra config.
          IndexError: The current spectra does not contain a dimension(s) that
            is in the spectrum config.
          ValueError: The upper bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The lower bounds of a parameter in the current spectra
            and spectra are not equal.
          ValueError: The number of bins of a parameter in the current spectra
            and spectra are not equal.
        """
        if self._data.shape != spectrum._data.shape:
            raise ValueError("The spectra have different dimensions.\n"
                             "Dimension of self: %s. Dimension of spectrum %s"
                             % (self._data.shape, spectrum._data.shape))
        for v in self._config.get_dims():
            if v not in spectrum.get_config().get_dims():
                raise IndexError("%s not present in new spectrum" % v)
        for v in spectrum.get_config().get_dims():
            if v not in self._config.get_dims():
                raise IndexError("%s not present in this spectrum" % v)
        # Dictionary containing dimensions which have different types in the
        # two spectra. The type of the dimension of spectrum is the value
        types = {}
        for v in spectrum.get_config().get_pars():
            if v not in self._config.get_pars():
                dim = spectrum.get_config().get_dim(v)
                dim_type = spectrum._config.get_dim_type(dim)
                types[dim] = dim_type
        for v in self._config.get_pars():
            dim = self._config.get_dim(v)
            if dim in types:
                v_spec = dim+'_'+types[dim]
            else:
                v_spec = v
            if self._config.get_par(v)._high != \
                    spectrum.get_config().get_par(v_spec)._high:
                raise ValueError("Upper %s bounds in spectra are not equal."
                                 % v)
            if self._config.get_par(v)._low != \
                    spectrum.get_config().get_par(v_spec)._low:
                raise ValueError("Lower %s bounds in spectra are not equal."
                                 % v)
            if self._config.get_par(v)._bins != \
                    spectrum.get_config().get_par(v_spec)._bins:
                raise ValueError("Number of %s bins in spectra are not equal."
                                 % v)
        self._data += spectrum._data
        self._raw_events += spectrum._raw_events
        self._num_decays += spectrum._num_decays

    def rebin(self, new_bins):
        """ Rebin spectra data into a smaller spectra of the same rank whose
        dimensions are factors of the original dimensions.

        Args:
          new_bins (tuple): new binning, this must match both the
            number and ordering of dimensions in the spectra config.

        Raises:
          ValueError: Shape mismatch. Number of dimenesions are different.
          ValueError: Old bins/ New bins must be integer
        """
        # Check all keys in kwargs are in the config parameters and visa versa
        if len(new_bins) != len(self._config.get_pars()):
            raise ValueError('Incorrect number of dimensions; need %s'
                             % len(self._config.get_pars()))
        # Now do the rebinning
        for i, v in enumerate(self._config.get_pars()):
            if self._config.get_par(v)._bins % new_bins[i] != 0:
                raise ValueError("Old bins/New bins must be integer old: %s"
                                 " new: %s for parameter %s"
                                 % (self._config.get_par(v)._bins,
                                    new_bins[i], v))
            self._config.get_par(v)._bins = new_bins[i]

        compression_pairs = [(d, c//d) for d, c in zip(new_bins,
                                                       self._data.shape)]
        flattened = [l for p in compression_pairs for l in p]
        self._data = self._data.reshape(flattened)
        for i in range(len(new_bins)):
            self._data = self._data.sum(-1*(i+1))
