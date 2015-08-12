import numpy as np
import itertools
import echidna.core.spectra as spectra


class Smear(object):
    """ This class smears the energy and radius of a spectra.

    The class can recieve energy and radius as individual data points or a
    1 dimensional numpy array to smear which is then returned. 2d and 3d
    arrays with linked energy, radius and time information is yet to be
    implemented.

    Attributes:
      _light_yield (float): Number of PMT hits expected for a
        MeV energy deposit in NHit/MeV
      _position_resolution (float): Sigma in mm
    """

    def __init__(self):
        """ Initialise the Smear class by seeding the random number generator
        """
        np.random.seed()
        self._num_sigma = 5.

    def calc_gaussian(self, x, mean, sigma):
        """ Calculates the value of a gaussian whose integral is equal to
          one at position x with a given mean and sigma.

          Args:
            x : Position to calculate the gaussian
            mean : Mean of the gaussian
            sigma : Sigma of the gaussian

          Returns:
            Value of the gaussian at the given position
        """
        return np.exp(-(x-mean)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    def get_bin_mean(self, low, bin, width):
        """
        """
        return low + (bin + 0.5)*width

    def get_num_sigma(self):
        """
        """
        return self._num_sigma

    def set_num_sigma(self, num_sigma):
        """
        """
        self._num_sigma = num_sigma

    def get_bounds(self, mean, sigma):
        """
        """
        low = mean - 0.5*self._num_sigma*sigma
        high = mean + 0.5*self._num_sigma*sigma
        return low, high


class EnergySmearLY(Smear):
    """
    """

    def __init__(self):
        """
        """
        super(EnergySmearLY, self).__init__()
        self._light_yield = 200.  # NHits/MeV

    def get_resolution(self):
        """
        """
        return self._light_yield

    def get_sigma(self, energy):
        """ Calculates sigma at a given energy.

        Args:
          energy (float): Energy value of data point(s)

        Returns:
          Sigma equivalent to sqrt(energy/_light_yield)
        """
        return np.sqrt(energy/self._light_yield)

    def set_resolution(self, light_yield):
        """
        """
        self._light_yield = light_yield

    def weighted_smear(self, spectrum, par="energy_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name+"_ly"+self._light_yield,
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                low, high = None
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma(mean)
                        low, high = Smear.get_bounds(mean, sigma)
                        low = spectrum.get_config().get_par(
                            par).round(low - 0.5 * widths[i]) + 0.5 * widths[i]
                        high = spectrum.get_config().get_par(
                            par).round(high + 0.5 * widths[i]) + \
                            0.5 * widths[i]
                        if low < spectrum.get_config().get_par(par)._low:
                            low = spectrum.get_config().get_par(par)._low + \
                                0.5 * widths[i]
                        if high > spectrum.get_config().get_par(par)._high:
                            high = spectrum.get_config().get_par(par)._high - \
                                0.5 * widths[i]
                        weights = []
                        for energy in np.arrange(low, high, widths[i]):
                            weights.append(Smear.calc_gaussian(energy,
                                                               mean,
                                                               sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for energy in np.arrange(low, high, widths[idx]):
                    data[par] = energy
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="energy_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name+"_ly"+self._light_yield,
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        data[par] = mean
                        sigma = self.get_sigma(mean)
                    else:
                        data[par_names[i]] = mean
                for i in range(entires):
                    data[par] = np.fabs(np.random.normal(data[par], sigma))
                    try:
                        smeared_spec.fill(**data)
                    except ValueError:
                        print "WARNING: Smeared energy out of bounds. Skipping"
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec


class EnergySmearRes(Smear):
    """ Allows you to smear directly by supplied energy resolution
      (in :math:`\sqrt{MeV}`).

    Inherits from :class:`Smear`

    Attributes:
      _energy_resolution (float): Energy resolution in :math:`\sqrt{MeV}`
        e.g. 0.05 for :math:`\sigma = 5\%/\sqrt{E[MeV]}`.
    """

    def __init__(self):
        """ Initialise the class
        """
        super(EnergySmearRes, self).__init__()
        self._resolution = 0.05  # 5%/sqrt(MeV)

    def get_resolution(self):
        """ Get the energy resolution

        Returns:
          float: Energy resolution in  :math:`\sqrt{MeV}`
          e.g. 0.05 for :math:`\sigma = 5\%/\sqrt{E[MeV]}`
        """
        return self._resolution

    def get_sigma(self, energy):
        """ Calculates sigma at a given energy.

        Args:
          energy (float): Energy value of data point(s)

        Returns:
          float: Sigma (MeV) equivalent to energy_resolution *
            :math:`\sqrt{energy}`
        """
        return self._resolution * np.power(energy, (1. / 2.))

    def set_resolution(self, resolution):
        """ Set the energy resolution in :math:`\sqrt{MeV}`
        e.g. 0.05 for :math:`\sigma = 5\%/\sqrt{E[MeV]}`.

        Args:
          resolution (float): Energy resolution in :math:`\sqrt{MeV}`
            e.g. 0.05 for :math:`\sigma = 5\%/\sqrt{E[MeV]}`.
        """
        self._resolution = resolution

    def weighted_smear(self, spectrum, par="energy_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       str(100.*self._resolution)+"%",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                low, high = None
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma(mean)
                        low, high = Smear.get_bounds(mean, sigma)
                        low = spectrum.get_config().get_par(
                            par).round(low - 0.5 * widths[i]) + 0.5 * widths[i]
                        high = spectrum.get_config().get_par(
                            par).round(high + 0.5 * widths[i]) + \
                            0.5 * widths[i]
                        if low < spectrum.get_config().get_par(par)._low:
                            low = spectrum.get_config().get_par(par)._low + \
                                0.5 * widths[i]
                        if high > spectrum.get_config().get_par(par)._high:
                            high = spectrum.get_config().get_par(par)._high - \
                                0.5 * widths[i]
                        weights = []
                        for energy in np.arrange(low, high, widths[i]):
                            weights.append(Smear.calc_gaussian(energy,
                                                               mean,
                                                               sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for energy in np.arrange(low, high, widths[idx]):
                    data[par] = energy
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="energy_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       str(100.*self._resolution)+"%",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        data[par] = mean
                        sigma = self.get_sigma(mean)
                    else:
                        data[par_names[i]] = mean
                for i in range(entires):
                    data[par] = np.fabs(np.random.normal(data[par], sigma))
                    try:
                        smeared_spec.fill(**data)
                    except ValueError:
                        print "WARNING: Smeared energy out of bounds. Skipping"
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec


class RadiusSmear(Smear):
    """
    """

    def __init__(self):
        """
        """
        super(EnergySmearRes, self).__init__()
        self._resolution = 100.  # mm

    def get_resolution(self):
        """Gets the position resolution.

        Returns:
          float: Position resolution.
        """
        return self._resolution

    def set_resolution(self, resolution):
        """Sets the position resolution:

        Args:
          resolution (float): Position resolution in mm.
        """
        self._resolution = resolution

    def get_sigma(self):
        """Sigma and resolution are equivalent for radial dimensions
        currently. This function calls self.get_resolution()

        Returns:
          float: Sigma in mm equivalent to resolution
        """
        return self.get_resolution()

    def weighted_smear(self, spectrum, par="radial_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       self._resolution + "mm",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                low, high = None
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma(mean)
                        low, high = Smear.get_bounds(mean, sigma)
                        low = spectrum.get_config().get_par(
                            par).round(low - 0.5 * widths[i]) + 0.5 * widths[i]
                        high = spectrum.get_config().get_par(
                            par).round(high + 0.5 * widths[i]) + \
                            0.5 * widths[i]
                        if low < spectrum.get_config().get_par(par)._low:
                            low = spectrum.get_config().get_par(par)._low + \
                                0.5 * widths[i]
                        if high > spectrum.get_config().get_par(par)._high:
                            high = spectrum.get_config().get_par(par)._high - \
                                0.5 * widths[i]
                        weights = []
                        for radius in np.arrange(low, high, widths[i]):
                            weights.append(Smear.calc_gaussian(radius,
                                                               mean,
                                                               sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for radius in np.arrange(low, high, widths[idx]):
                    data[par] = radius
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="radial_mc"):
        """
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_par(par).get_index()
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(spectrum.get_config().get_par(par_name)._bins)
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       self._resolution + "mm",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data_dict = {}
                for i in range(len(bin)):
                    mean = Smear.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        data[par] = mean
                        sigma = self.get_sigma(mean)
                    else:
                        data[par_names[i]] = mean
                for i in range(entires):
                    data[par] = np.fabs(np.random.normal(data[par], sigma))
                    try:
                        smeared_spec.fill(**data)
                    except ValueError:
                        print "WARNING: Smeared radius out of bounds. Skipping"
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec
