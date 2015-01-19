import numpy
import decimal
import spectra

class Smear(object):
    """ This class smears the energy and radius of a spectra.

    The class can recieve energy and radius as individual data points or a 
    1 dimensional numpy array to smear which is then returned. 2d and 3d 
    arrays with linked energy, radius and time information is yet to be
    implemented.

    Attributes:
      _light_yield (float): Number of PMT hits expected for a MeV energy deposit in NHit/MeV
      _position_resolution (float): Sigma in mm
    """
    _light_yield = 200. # NHit per MeV
    _position_resolution = 100. # mm

    def __init__(self):
        """ Initialise the Smear class by seeding the random number generator
        """
        numpy.random.seed()

    def bin_1d_array(self, array, bins):
        """ Sorts a 1 dimensional array and bins it
        
        Args:
          array (numpy array): To sort and bin
          bins (list): Upper limit of bins

        Returns:
          A 1 dimensional numpy array, sorted and binned.
        """
        array = numpy.sort(array)
        split_at = array.searchsorted(bins)
        return numpy.split(array,split_at)
        
    def floor_to_bin(self, x,bin_size):
        """ Rounds down value bin content to lower edge of nearest bin.
        
        Args:
          x (float): Value to round down
          bin_size (float): Width of a bin

        Returns:
          Value of nearest lower bin edge
        """
        dp = abs(decimal.Decimal(str(bin_size)).as_tuple().exponent)
        coef = numpy.power(10,dp)
        return numpy.floor(coef*(x//bin_size)*bin_size)/coef

    def get_energy_sigma(self,energy):
        """ Calculates sigma at a given energy.
        
        Args:
          energy (float): Energy value of data point(s)
        
        Returns:
          Sigma equivalent to sqrt(energy/_light_yield)
        """
        return numpy.sqrt(energy/self._light_yield)

    def smear_energy_0d(self,energy):
        """ Smears a single energy value
        
        Args:
          energy (float): Value to smear

        Returns:
          Smeared energy value
        """
        sigma = self.get_energy_sigma(energy)
        return numpy.fabs(numpy.random.normal(energy,sigma))   

    def smear_energy_1d(self,energies, bins, binned = False):
        """ Smears a 1 dimensional array of energy values
        
        Args:
          energies (numpy array): Values to smear
          bins (list): Upper edge of bins for array
          binned (bool): Is the array already binned? (True or False)

        Returns:
          Smeared and sorted 1 dimensional numpy array of energy values
        """
        if binned == False:
            energies = self.bin_1d_array(energies,bins)
        bin_size = bins[1]-bins[0]
        smeared_energies = []
        for entry in energies:
            if not entry.any():
                # Bin is empty
                continue
            energy_bin = self.floor_to_bin(entry[0],bin_size)+0.5*bin_size
            num_entries = len(entry)
            smeared_energies += self.smear_energy_bin(energy_bin,num_entries)
        return numpy.array(smeared_energies)

    def smear_energy_bin(self,energy, entries):
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
            smeared_energies.append(numpy.fabs(numpy.random.normal(energy,sigma)))
        return smeared_energies

    def smear_radius_0d(self,radius):
        """ Smears a single radius value
        
        Args:
          radius (float): Value to smear

        Returns:
          Smeared radius value
        """
        return numpy.fabs(numpy.random.normal(radius,self._position_resolution))

    def smear_radii_1d(self,radii, bins, binned = False):
        """ Smears a 1 dimensional array of radius values
        
        Args:
          radii (numpy array): Values to smear
          bins (list): Upper edge of bins for array
          binned (bool): Is the array already binned? (True or False)

        Returns:
          Smeared and sorted 1 dimensional numpy array of radius values
        """
        if binned == False:
            radii = self.bin_1d_array(radii,bins)
        bin_size = bins[1]-bins[0]
        smeared_radii = []
        for entry in radii:
            if not entry.any():
                #Bin is empty
                continue
            radius_bin = self.floor_to_bin(entry[0],bin_size)+0.5*bin_size
            num_entries = len(entry)
            smeared_radii += self.smear_radius_bin(radius_bin,num_entries)
        return numpy.array(smeared_radii)

    def smear_radius_bin(self,radius, entries):
        """ Smears one energy bin.

        Args:
          radius (float): Central value of radius of bin
          entries (int): Number of entries in the bin

        Returns:
          A list of smeared radii corresponding to the input bin.
        """
        smeared_radii = []
        for i in range(entries):
            smeared_radii.append(numpy.fabs(numpy.random.normal(radius,self._position_resolution)))
        return smeared_radii


    def gaussian_smear_spectra_energy(self,true_spectrum):
        """ Smears the energy of a spectra object using a gaussian.

        Args:
          true_spectrum (spectra): spectrum to be smeared

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+"_smeared_"+str(self._light_yield)+"_light_yield")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for radial_bin in range(true_spectrum._radial_bins):
                mean_radius = radial_bin*radial_step+0.5*radial_step
                for energy_bin in range(true_spectrum._energy_bins):
                    mean_energy = energy_bin*energy_step+0.5*energy_step
                    entries = true_spectrum._data[energy_bin, radial_bin, time_bin]
                    for i in range(int(entries)):
                        sigma = self.get_energy_sigma(mean_energy)
                        try:
                            smeared_spectrum.fill(numpy.fabs(numpy.random.normal(mean_energy,sigma)),mean_radius,mean_time)
                        except:
                            # Only occurs when smeared energy is greater than max bin
                            print "Warning: Smeared energy out of bounds. Skipping."
                            continue
        return smeared_spectrum

    def gaussian_smear_spectra_radius(self,true_spectrum):
        """ Smears the radius of a spectra object using a gaussian.

        Args:
          true_spectrum (spectra): spectrum to be smeared

        Returns:
          A smeared spectra object.
        """
        energy_step = (true_spectrum._energy_high-true_spectrum._energy_low)/true_spectrum._energy_bins
        time_step = (true_spectrum._time_high-true_spectrum._time_low)/true_spectrum._time_bins
        radial_step = (true_spectrum._radial_high-true_spectrum._radial_low)/true_spectrum._radial_bins
        smeared_spectrum = spectra.Spectra(true_spectrum._name+"_smeared_"+str(self._light_yield)+"_position_resolution")
        for time_bin in range(true_spectrum._time_bins):
            mean_time = time_bin*time_step+0.5*time_step
            for energy_bin in range(true_spectrum._energy_bins):
                mean_energy = energy_bin*energy_step+0.5*energy_step
                for radial_bin in range(true_spectrum._radial_bins):
                    mean_radius = radial_bin*radial_step+0.5*radial_step
                    entries = true_spectrum._data[energy_bin, radial_bin, time_bin]
                    for i in range(int(entries)):
                        try:
                            smeared_spectrum.fill(mean_energy, numpy.fabs(numpy.random.normal(mean_radius,self._position_resolution)),mean_time)
                        except:
                            # Only occurs when smeared radius is greater than max bin
                            print "Warning: Smeared radius out of bounds. Skipping."
                            continue
        return smeared_spectrum
