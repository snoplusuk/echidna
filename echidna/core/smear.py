import numpy as np
import decimal
import spectra


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
    _light_yield = 200.  # NHit per MeV
    _position_resolution = 100.  # mm

    def __init__(self):
        """ Initialise the Smear class by seeding the random number generator
        """
        np.random.seed()

    def bin_1d_array(self, array, bins):
        """ Sorts a 1 dimensional array and bins it

        Args:
          array (:class:`numpy.array`): To sort and bin
          bins (list): Upper limit of bins

        Returns:
          A 1 dimensional numpy array, sorted and binned.
        """
        array = np.sort(array)
        split_at = array.searchsorted(bins)
        return np.split(array, split_at)

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

    def floor_to_bin(self, x, bin_size):
        """ Rounds down value bin content to lower edge of nearest bin.

        Args:
          x (float): Value to round down
          bin_size (float): Width of a bin

        Returns:
          Value of nearest lower bin edge
        """
        dp = abs(decimal.Decimal(str(bin_size)).as_tuple().exponent)
        coef = np.power(10, dp)
        return np.floor(coef*(x//bin_size)*bin_size)/coef

    def ceil_to_bin(self, x, bin_size):
        """ Rounds up value bin content to upper edge of nearest bin.

        Args:
          x (float): Value to round down
          bin_size (float): Width of a bin

        Returns:
          Value of nearest lower bin edge
        """
        dp = abs(decimal.Decimal(str(bin_size)).as_tuple().exponent)
        coef = np.power(10, dp)
        return np.ceil(coef*(bin_size+(x//bin_size)*bin_size))/coef

    def get_energy_sigma(self, energy):
        """ Calculates sigma at a given energy.

        Args:
          energy (float): Energy value of data point(s)

        Returns:
          Sigma equivalent to sqrt(energy/_light_yield)
        """
        return np.sqrt(energy/self._light_yield)

    def smear_energy_0d(self, energy):
        """ Smears a single energy value

        Args:
          energy (float): Value to smear

        Returns:
          Smeared energy value
        """
        sigma = self.get_energy_sigma(energy)
        return np.fabs(np.random.normal(energy, sigma))

    def smear_energy_1d(self, energies, bins, binned=False):
        """ Smears a 1 dimensional array of energy values

        Args:
          energies (:class:`numpy.array`): Values to smear
          bins (list): Upper edge of bins for array
          binned (bool): Is the array already binned? (True or False)

        Returns:
          Smeared and sorted 1 dimensional numpy array of energy values
        """
        if binned is False:
            energies = self.bin_1d_array(energies, bins)
        bin_size = bins[1]-bins[0]
        smeared_energies = []
        for energy in energies:
            if energy.any():
                energy_bin = self.floor_to_bin(energy[0], bin_size)+0.5*bin_size
                num_entries = len(energy)
                smeared_energies += self.smear_energy_bin(energy_bin,
                                                          num_entries)
        return np.array(smeared_energies)

    def smear_energy_bin(self, energy, entries):
        """ Smears one energy bin.

        Args:
          energy (float): Central value of energy of bin
          entries (int): Number of entries in the bin

        Returns:
          A list of smeared energies corresponding to the input bin.
        """
        sigma = self.get_energy_sigma(energy)
        smeared_energies = []
        for i in range(entries):
            smeared_energies.append(np.fabs(np.random.normal(energy, sigma)))
        return smeared_energies

    def smear_radius_0d(self, radius):
        """ Smears a single radius value

        Args:
          radius (float): Value to smear

        Returns:
          Smeared radius value
        """
        return np.fabs(np.random.normal(radius, self._position_resolution))

    def smear_radii_1d(self, radii, bins, binned=False):
        """ Smears a 1 dimensional array of radius values

        Args:
          radii (:class:`numpy.array`): Values to smear
          bins (list): Upper edge of bins for array
          binned (bool): Is the array already binned? (True or False)

        Returns:
          Smeared and sorted 1 dimensional numpy array of radius values
        """
        if binned is False:
            radii = self.bin_1d_array(radii, bins)
        bin_size = bins[1]-bins[0]
        smeared_radii = []
        for radius in radii:
            if radius.any():
                radius_bin = self.floor_to_bin(radius[0], bin_size)+0.5*bin_size
                num_entries = len(radius)
                smeared_radii += self.smear_radius_bin(radius_bin, num_entries)
        return np.array(smeared_radii)

    def smear_radius_bin(self, radius, entries):
        """ Smears one energy bin.

        Args:
          radius (float): Central value of radius of bin
          entries (int): Number of entries in the bin

        Returns:
          A list of smeared radii corresponding to the input bin.
        """
        smeared_radii = []
        for i in range(entries):
            smeared_radii.append(np.fabs(np.random.normal(radius,
                                                          self._position_resolution)))
        return smeared_radii

    def random_gaussian_energy_spectra(self, true_spectrum):
        """ Smears the energy of a spectra object by generating
          a number of random points from a Gaussian pdf generated
          for that bin. The number of points generated is equivalent
          to the number of entries in that bin.

        Args:
          true_spectrum (spectra): spectrum to be smeared

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+str(self._light_yield)+"_light_yield")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for radial_bin in range(true_spectrum._radial_bins):
                mean_radius = radial_bin*radial_step+0.5*radial_step
                for energy_bin in range(true_spectrum._energy_bins):
                    mean_energy = energy_bin*energy_step+0.5*energy_step
                    sigma = self.get_energy_sigma(mean_energy)
                    entries = true_spectrum._data[energy_bin,
                                                  radial_bin,
                                                  time_bin]
                    for i in range(int(entries)):
                        try:
                            smeared_spectrum.fill(np.fabs(np.random.normal(mean_energy,
                                                                           sigma)),
                                                  mean_radius,
                                                  mean_time)
                        except:
                            # Occurs when smeared energy is > max bin
                            print "Warning: Smeared energy out of bounds. Skipping."
                            continue
        return smeared_spectrum

    def weight_gaussian_energy_spectra(self, true_spectrum, num_sigma=5.):
        """ Smears the energy of a spectra object by calculating a Gaussian pdf
          for each bin and applying a weight to the bin and corresponding bins
          a default 5 sigma apart.

        Args:
          true_spectrum (spectra): spectrum to be smeared
          num_sigma (float): Width of window to apply the weight method.
            Default is 5.

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+str(self._light_yield)+"_light_yield")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for radial_bin in range(true_spectrum._radial_bins):
                mean_radius = radial_bin*radial_step+0.5*radial_step
                for energy_bin in range(true_spectrum._energy_bins):
                    mean_energy = energy_bin*energy_step+0.5*energy_step
                    sigma = self.get_energy_sigma(mean_energy)
                    entries = float(true_spectrum._data[energy_bin,
                                                        radial_bin,
                                                        time_bin])
                    if entries == 0:
                        continue  # Bin Empty
                    lower_bin = self.floor_to_bin(mean_energy-num_sigma*sigma,
                                                  energy_step)+0.5*energy_step
                    upper_bin = self.ceil_to_bin(mean_energy+num_sigma*sigma,
                                                 energy_step)-0.5*energy_step
                    if upper_bin > true_spectrum._energy_high:
                        upper_bin = true_spectrum._energy_high-0.5*energy_step
                    if lower_bin < true_spectrum._energy_low:
                        lower_bin = true_spectrum._energy_low+0.5*energy_step
                    weights = []
                    for energy in np.arange(lower_bin, upper_bin, energy_step):
                        weights.append(self.calc_gaussian(energy,
                                                          mean_energy,
                                                          sigma))
                    i = 0
                    tot_weight = np.array(weights).sum()
                    for energy in np.arange(lower_bin, upper_bin, energy_step):
                        smeared_spectrum.fill(energy,
                                              mean_radius,
                                              mean_time,
                                              entries*weights[i]/tot_weight)
                        i += 1
        return smeared_spectrum

    def random_gaussian_radius_spectra(self, true_spectrum):
        """ Smears the radius of a spectra object by generating a
          number of random points from a Gaussian pdf generated for
          that bin. The number of points generated is equivalent
          to the number of entries in that bin.

        Args:
          true_spectrum (spectra): spectrum to be smeared

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+str(self._positin_resolution)+"_position_resolution")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for energy_bin in range(true_spectrum._energy_bins):
                mean_energy = energy_bin*energy_step+0.5*energy_step
                for radial_bin in range(true_spectrum._radial_bins):
                    mean_radius = radial_bin*radial_step+0.5*radial_step
                    entries = true_spectrum._data[energy_bin,
                                                  radial_bin,
                                                  time_bin]
                    for i in range(int(entries)):
                        try:
                            smeared_spectrum.fill(mean_energy,
                                                  np.fabs(np.random.normal(mean_radius,
                                                                           self._position_resolution)),
                                                  mean_time)
                        except:
                            # Occurs when smeared radius is > max bin
                            print "Warning: Smeared radius out of bounds. Skipping."
                            continue
        return smeared_spectrum

    def weight_gaussian_radius_spectra(self, true_spectrum, num_sigma=5.):
        """ Smears the radius of a spectra object by calculating a Gaussian pdf
          for each bin and applies a weight to the bin and corresponding bins a
          default 5 sigma apart.

        Args:
          true_spectrum (spectra): spectrum to be smeared
          num_sigma (float): Width of window to apply the weight method.
            Default is 5.

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+str(self._position_resolution)+"_position_resolution")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for energy_bin in range(true_spectrum._energy_bins):
                mean_energy = energy_bin*energy_step+0.5*energy_step
                for radial_bin in range(true_spectrum._radial_bins):
                    mean_radius = radial_bin*radial_step+0.5*radial_step
                    entries = float(true_spectrum._data[energy_bin,
                                                        radial_bin,
                                                        time_bin])
                    if entries == 0:
                        continue  # Bin Empty
                    lower_bin = self.floor_to_bin(mean_radius-num_sigma*self._position_resolution,
                                                  radial_step)+0.5*radial_step
                    upper_bin = self.ceil_to_bin(mean_radius+num_sigma*self._position_resolution,
                                                 radial_step)-0.5*radial_step
                    if upper_bin > true_spectrum._radial_high:
                        upper_bin = true_spectrum._radial_high-0.5*energy_step
                    if lower_bin < true_spectrum._radial_low:
                        lower_bin = true_spectrum._radial_low+0.5*energy_step
                    weights = []
                    for radius in np.arange(lower_bin, upper_bin, radial_step):
                        weights.append(self.calc_gaussian(radius,
                                                          mean_radius,
                                                          self._position_resolution))
                    weight_tot = np.array(weights).sum()
                    i = 0
                    for radius in np.arange(lower_bin, upper_bin, radial_step):
                        smeared_spectrum.fill(mean_energy,
                                              radius,
                                              mean_time,
                                              entries*weights[i]/weight_tot)
                        i += 1
        return smeared_spectrum
