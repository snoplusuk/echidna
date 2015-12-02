"""
Examples:
  To smear an already smeared spectrum with a light yield of 200 to a
  a light yield of 190 then the following lines are required::

    >>> smearer = smear.SmearEnergySmearLY()
    >>> ly = smearer.calc_smear_ly(190., cur_ly=200.)
    >>> smearer.set_resolution(ly)
    >>> smeared_spec = smearer.weighted_smear(spectrum)

.. note:: Similar methods are available in all other smearing classes.
"""
import numpy as np
import itertools
import echidna.core.spectra as spectra


class Smear(object):
    """ The base class for smearing spectra.

    Args:
      name (string): The name of the smearing class.

    Attributes:
      _name (string): name of the smeaing class.
      _num_sigma (float): The width of the window in terms of number of sigma
        you wish to apply weights to.
    """

    def __init__(self, name):
        """ Initialise the Smear class by seeding the random number generator.
        """
        np.random.seed()
        self._name = name
        self._num_sigma = 5.

    def calc_gaussian(self, x, mean, sigma):
        """ Calculates the value of a gaussian whose integral is equal to
          one at position x with a given mean and sigma.

          Args:
            x : Position to calculate the gaussian
            mean : Mean of the gaussian
            sigma : Sigma of the gaussian

          Returns:
            float: Value of the gaussian at the given position
        """
        return np.exp(-(x-mean)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

    def get_bin_mean(self, low, bin, width):
        """ Calculates the mean value of a bin.

        Args:
          low (float): The lower bound value of the parameter.
          bin (int): The number of the bin you wish to calculate the mean of.
          width (float): The width of the bin.

        Returns:
          float: The mean value of the bin.
        """
        return low + (bin + 0.5)*width

    def get_num_sigma(self):
        """ Returns the width of the window in terms of number of sigma
          you wish to apply weights to.

        Returns:
          float: The number of sigma.
        """
        return self._num_sigma

    def set_num_sigma(self, num_sigma):
        """ Sets the width of the window in terms of number of sigma
          you wish to apply weights to.

        Args:
          num_sigma (float): The number of sigma you wish to apply weights to.
        Raises:
          ValueError: If the number of sigma is zero or negative.
        """
        if (num_sigma > 0.):
            self._num_sigma = float(num_sigma)
        else:
            raise ValueError("%s is an invalid num_sigma. Value must be "
                             "greater than zero." % num_sigma)

    def get_bounds(self, mean, sigma):
        """ Calculates the boundaries you wish to apply the smearing
          weights to.

        Args:
          mean (float): The mean value you are smearing.
          sigma (float): The sigma of the gaussian you are using to smear.

        Returns:
          tuple: First value of the tuple is the lower bound. The second is
            the upper bound.
        """
        low = mean - self._num_sigma*sigma
        high = mean + self._num_sigma*sigma
        return low, high


class EnergySmearLY(Smear):
    """ The class which smears energy. It accepts resolution in terms of light
      yield (LY) in units of NHit per MeV.

    Args:
      poisson (bool): If True, use poisson smearing.      

    Attributes:
      _light_yield (float): The light yield of the scintillator in NHits per
        MeV.
      _poisson_smear (Bool): True if poisson smearing is to be applied. False if
        gaussian smearing is to be applied. 
    """

    def __init__(self, poisson=True):
        """ Initialises the class.
        """
        super(EnergySmearLY, self).__init__("energy_light_yield")
        self._poisson_smear = poisson
        self._light_yield = 200  # Nhit/MeV
        self._log_factorial = {}

    def calc_poisson_energy(self, x, lamb):
        """ Calculates the value of a poisson whose integral is equal to
        one at position x with a given lambda value.

        Args:
          x : Number of events
          lamb : Lambda of the poisson

        Returns: 
          float: The value of the poisson at the given position
        """
        photons = int(x*self._light_yield)
        expected = lamb*self._light_yield
        if self._log_factorial.has_key(photons) == False:
            self._log_factorial[photons] = np.sum(np.log(np.arange(1,(photons+1))))
        logPois = photons*np.log(expected) - self._log_factorial[photons] - expected#*np.log(np.e)
        return np.exp(logPois)
    
    def calc_smear_ly(self, new_ly, cur_ly=None):
        """Calculates the value of light yield (ly) required to smear a
          data set which has already been smeared with a light yield of cur_ly
          to achieve a smeared data set with a new light yield of new_ly.

        Args:
          new_ly (float): The value of light yield wanted for the smeared PDF.
          cur_ly (float, optional): Current value of light yield the PDF
            has been convolved with from the true value PDF.

        Raises:
          ValueError: If new_ly is smaller than cur_sigma. Can't smear to
            higher light yields (smaller sigmas)

        Returns:
          float: The value of light yield needed to smear the current
            PDF to obtain a new light yield: new_ly.
        """
        if not cur_ly:
            cur_ly = self.get_resolution()
        if new_ly > cur_ly:
            raise ValueError("New light yield must be smaller than the"
                             "current light yield. cur_ly: %s. new_ly: %s."
                             % (cur_ly, new_ly))
        return new_ly*cur_ly/(cur_ly-new_ly)

    def get_resolution(self):
        """ Returns the light yield.

        Returns:
          float: The light yield.
        """
        return self._light_yield

    def get_sigma(self, energy):
        """ Calculates sigma at a given energy.

        Args:
          energy (float): Energy value of data point(s)

        Returns:
          float: Sigma equivalent to sqrt(energy/_light_yield)
        """
        return np.sqrt(energy/self._light_yield)

    def set_resolution(self, light_yield):
        """ Sets the light yield

        Args:
          light_yield (float): The value you wish to set the light yield to.

        Raises:
          ValueError: If the light yield is zero or negative.
        """
        if light_yield > 0.:
            self._light_yield = float(light_yield)
        else:
            raise ValueError("%s is an invalid light yield. Light yield "
                             "must be greater than zero.")

    def weighted_smear(self, spectrum, par="energy_mc"):
        """ Smears the energy of a :class:`echidna.core.spectra.Spectra` by
          calculating a Gaussian PDF for each bin. Weights are then applied
          to a window of width specified by the number of sigma depending on
          the value of the Gaussian PDF at the mean of the bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is energy_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name+"_ly" +
                                       str(self._light_yield),
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data = {}
                low = None
                high = None
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma(mean)
                        low, high = self.get_bounds(mean, sigma)
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
                        for energy in np.arange(low, high, widths[i]):
                            if self._poisson_smear == True:
                                weights.append(self.calc_poisson_energy(energy,
                                                                        mean))
                            else:
                                weights.append(self.calc_gaussian(energy,
                                                                  mean,
                                                                  sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for energy in np.arange(low, high, widths[idx]):
                    data[par] = energy
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="energy_mc"):
        """ Smears the energy of a :class:`echidna.core.spectra.Spectra` by
          generating a number of random points from  Gaussian PDF generated
          from that bins mean value and the corresponding sigma. The number
          of points generated is equivalent to the number of entries in that
          bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is energy_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name+"_ly" +
                                       str(self._light_yield),
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = int(spectrum._data[bin])
            if entries:
                data = {}
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        mean_e = mean
                        sigma = self.get_sigma(mean)
                    else:
                        data[par_names[i]] = mean
                for i in range(entries):
                    if self._poisson_smear == True:
                        data[par] = np.fabs(np.random.poisson(mean_e))
                    else:
                        data[par] = np.fabs(np.random.normal(mean_e, sigma))
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

    Args:
      poisson (bool): If True, use poisson smearing.      

    Attributes:
      _energy_resolution (float): Energy resolution in :math:`\sqrt{MeV}`
        e.g. 0.05 for :math:`\sigma = 5\%/\sqrt{E[MeV]}`.
      _poisson_smear (Bool): True if poisson smearing is to be applied. False if
        gaussian smearing is to be applied. 
    """

    def __init__(self, poisson=True):
        """ Initialise the class
        """
        super(EnergySmearRes, self).__init__("energy_resolution")
        self._poisson_smear = poisson
        self._light_yield = 200  # Nhit/MeV
        self._log_factorial = {}

    def calc_poisson_energy(self, x, lamb):
        """ Calculates the value of a poisson whose integral is equal to
        one at position x with a given lambda value.

        Args:
          x : Number of events
          lamb : Lambda of the poisson

        Returns: 
          float: The value of the poisson at the given position
        """
        photons = int(x*self._light_yield)
        expected = lamb*self._light_yield
        if self._log_factorial.has_key(photons) == False:
            self._log_factorial[photons] = np.sum(np.log(np.arange(1,(photons+1))))
        logPois = photons*np.log(expected) - self._log_factorial[photons] - expected#*np.log(np.e)
        return np.exp(logPois)

    def calc_smear_resoluton(self, new_res, cur_res=None):
        """Calculates the value of resolution required to smear a data set
          which has already been smeared with a resolution of cur_res to
          achieve a new resolution of new_res.

        Args:
          new_res (float): The value of resolution wanted for the smeared PDF.
          cur_res (float, optional): Current value of resolution the PDF
            has been convolved with from the true value PDF.

        Raises:
          ValueError: If new_res is smaller than cur_sigma. Can't smear to
            higher resolutions (smaller sigmas)

        Returns:
          float: The value of resolution needed to smear the current
            PDF to obtain a new resolution with sigma value new_res.
        """
        if not cur_res:
            cur_res = self.get_resolution()
        if new_res < cur_res:
            raise ValueError("New resolution must be larger than the"
                             "current resolution. cur_res: %s. new_res: %s."
                             % (cur_res, new_res))
        return numpy.fabs(numpy.sqrt(new_res**2 - cur_res**2))

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

        Raises:
          ValueError: If the resolution is not between 0 and 1.
        """
        if (resolution > 0. and resolution < 1.):
            self._resolution = resolution
        else:
            raise ValueError("%s is an invalid energy resolution. Value "
                             "must be between 0. and 1." % resolution)

    def weighted_smear(self, spectrum, par="energy_mc"):
        """ Smears the energy of a :class:`echidna.core.spectra.Spectra` by
          calculating a Gaussian PDF for each bin. Weights are then applied
          to a window of width specified by the number of sigma depending on
          the value of the Gaussian PDF at the mean of the bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is energy_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
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
                data = {}
                low = None
                high = None
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma(mean)
                        low, high = self.get_bounds(mean, sigma)
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
                        for energy in np.arange(low, high, widths[i]):
                            if self._poisson_smear == True:
                                weights.append(self.calc_poisson_energy(energy,
                                                                        mean))
                            else:
                                weights.append(self.calc_gaussian(energy,
                                                                  mean,
                                                                  sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for energy in np.arange(low, high, widths[idx]):
                    data[par] = energy
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="energy_mc"):
        """ Smears the energy of a :class:`echidna.core.spectra.Spectra` by
          generating a number of random points from  Gaussian PDF generated
          from that bins mean value and the corresponding sigma. The number
          of points generated is equivalent to the number of entries in that
          bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is energy_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       str(100.*self._resolution)+"%",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = int(spectrum._data[bin])
            if entries:
                data = {}
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        mean_e = mean
                        sigma = self.get_sigma(mean)
                    else:
                        data[par_names[i]] = mean
                for i in range(entries):
                    if self._poisson_smear == True:
                        data[par] = np.fabs(np.random.normal(mean_e))
                    else:
                        data[par] = np.fabs(np.random.normal(mean_e, sigma))
                    try:
                        smeared_spec.fill(**data)
                    except ValueError:
                        print "WARNING: Smeared energy out of bounds. Skipping"
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec


class RadialSmear(Smear):
    """ The class which smears the radius. It accepts resolution in terms of
      sigma in units of mm.

    Args:
      poisson (bool): If True, use poisson smearing.      

    Attributes:
      _resolution (float): The position resolution (mm).
      _poisson_smear (Bool): True if poisson smearing is to be applied. False if
        gaussian smearing is to be applied. 
    """

    def __init__(self):
        """ Initialises the class.
        """
        super(RadialSmear, self).__init__("radial")
        self._resolution = 100.  # mm

    def calc_smear_resoluton(self, new_res, cur_res=None):
        """Calculates the value of resolution required to smear a data set
          which has already been smeared with a resolution of cur_res to
          achieve a new resolution of new_res.

        Args:
          new_res (float): The value of resolution wanted for the smeared PDF.
          cur_res (float, optional): Current value of resolution the PDF
            has been convolved with from the true value PDF.

        Raises:
          ValueError: If new_res is smaller than cur_sigma. Can't smear to
            higher resolutions (smaller sigmas)

        Returns:
          float: The value of resolution needed to smear the current
            PDF to obtain a new resolution: new_res.
        """
        if not cur_res:
            cur_res = self.get_resolution()
        if new_res < cur_res:
            raise ValueError("New resolution must be larger than the"
                             "current resolution. cur_res: %s. new_res: %s."
                             % (cur_res, new_res))
        return numpy.fabs(numpy.sqrt(new_res**2 - cur_res**2))

    def get_resolution(self):
        """Gets the position resolution.

        Returns:
          float: Position resolution.
        """
        return self._resolution

    def set_resolution(self, resolution):
        """Sets the position resolution:

        Raises:
          ValueError: If resolution is zero or less.

        Args:
          resolution (float): Position resolution in mm.
        """
        if resolution > 0:
            self._resolution = resolution
        else:
            raise ValueError("%s is an incorrect position resolutioin. Value "
                             "must be greater than zero." % resolution)

    def get_sigma(self):
        """Sigma and resolution are equivalent for radial dimensions
        currently. This function calls self.get_resolution()

        Returns:
          float: Sigma in mm equivalent to resolution
        """
        return self.get_resolution()

    def weighted_smear(self, spectrum, par="radial_mc"):
        """ Smears the radius of a :class:`echidna.core.spectra.Spectra` by
          calculating a Gaussian PDF for each bin. Weights are then applied
          to a window of width specified by the number of sigma depending on
          the value of the Gaussian PDF at the mean of the bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is radial_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       str(self._resolution) + "mm",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data = {}
                low = None
                high = None
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        sigma = self.get_sigma()
                        low, high = self.get_bounds(mean, sigma)
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
                        for radius in np.arange(low, high, widths[i]):
                            weights.append(self.calc_gaussian(radius,
                                                              mean,
                                                              sigma))
                    else:
                        data[par_names[i]] = mean
                total_weight = sum(weights)
                i = 0
                for radius in np.arange(low, high, widths[idx]):
                    data[par] = radius
                    smeared_spec.fill(weight=entries*weights[i]/total_weight,
                                      **data)
                    i += 1
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec

    def random_smear(self, spectrum, par="radial_mc"):
        """ Smears the radius of a :class:`echidna.core.spectra.Spectra` by
          generating a number of random points from  Gaussian PDF generated
          from that bins mean value and the corresponding sigma. The number
          of points generated is equivalent to the number of entries in that
          bin.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum you wish
            to smear.
          par (string, optional): The name of the parameter you wish to smear.
            The default is radial_mc.

        Raises:
          IndexError: If par is not in the specta config.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The smeared spectrum
        """
        if par not in spectrum.get_config().get_pars():
            raise IndexError("%s is not a parameter in the spectrum" % par)
        idx = spectrum.get_config().get_index(par)
        bins = []
        lows = []
        widths = []
        par_names = []
        for par_name in spectrum.get_config().get_pars():
            bins.append(range(spectrum.get_config().get_par(par_name)._bins))
            lows.append(spectrum.get_config().get_par(par_name)._low)
            widths.append(spectrum.get_config().get_par(par_name).get_width())
            par_names.append(par_name)
        smeared_spec = spectra.Spectra(spectrum._name + "_" +
                                       str(self._resolution) + "mm",
                                       spectrum._num_decays,
                                       spectrum.get_config())
        for bin in itertools.product(*bins):
            entries = spectrum._data[bin]
            if entries:
                data = {}
                for i in range(len(bin)):
                    mean = self.get_bin_mean(lows[i], bin[i], widths[i])
                    if i == idx:
                        mean_r = mean
                        sigma = self.get_sigma()
                    else:
                        data[par_names[i]] = mean
                for i in range(entries):
                    data[par] = np.fabs(np.random.normal(mean_r, sigma))
                    try:
                        smeared_spec.fill(**data)
                    except ValueError:
                        print "WARNING: Smeared radius out of bounds. Skipping"
        smeared_spec._raw_events = spectrum._raw_events
        return smeared_spec
