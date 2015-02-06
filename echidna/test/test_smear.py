import unittest
import echidna.core.smear as smear
import echidna.core.spectra as spectra
import numpy
from scipy.optimize import curve_fit


class TestSmear(unittest.TestCase):

    def gaussian(self, x, *p):
        """ A gaussian used for fitting.

        Args:
          x (float): Position the gaussian is calculated at.
          *p (list): List of parameters to fit

        Returns:
          Value of gaussian at x for given parameters (float)
        """
        A, mean, sigma = p
        A = numpy.fabs(A)
        mean = numpy.fabs(mean)
        sigma = numpy.fabs(sigma)
        return A*numpy.exp(-(x-mean)**2/(2.*sigma**2))

    def fit_gaussian_energy(self, spectra):
        """ Fits a gausian to the energy of a spectrum.

        Args:
          spectra (core.spectra): Spectrum to be smeared

        Returns:
          mean (float), sigma (float) and integral (float) of the spectrum.
        """
        entries = []
        energies = []
        energy_width = (spectra._energy_high-spectra._energy_low)/spectra._energy_bins
        spectra_proj = spectra.project(0)
        for i in range(len(spectra_proj)):
            entries.append(spectra_proj[i])
            energies.append(spectra._energy_low+energy_width*(i+0.5))
        pars0 = [300., 2.5, 0.1]
        coeff, var_mtrx = curve_fit(self.gaussian, energies, entries, p0=pars0)
        return coeff[1], coeff[2], numpy.array(entries).sum()

    def fit_gaussian_radius(self, spectra):
        """ Fits a gausian to the radius of a spectrum.

        Args:
          spectra (core.spectra): Spectrum to be smeared

        Returns:
          mean (float), sigma (float) and integral (float)
        """
        entries = []
        radii = []
        radial_width = (spectra._radial_high - spectra._radial_low)/spectra._radial_bins
        spectra_proj = spectra.project(1)
        for i in range(len(spectra_proj)):
            entries.append(spectra_proj[i])
            radii.append(spectra._radial_low+radial_width*(i+0.5))
        pars0 = [400., 1000., 100.]
        coeff, var_mtrx = curve_fit(self.gaussian, radii, entries, p0=pars0)
        return coeff[1], coeff[2], numpy.array(entries).sum()

    def test_weight_energy(self):
        """ Tests the weighted gaussian smearing method for energy.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        num_entries = 10000
        test_spectra = spectra.Spectra("Test", num_entries)
        energy = 2.5  # MeV
        for i in range(num_entries):
            test_spectra.fill(energy, 0, 0)
        smearing = smear.Smear()
        smearing._light_yield = 200.  # NHit per MeV
        smeared_spectra = smearing.weight_gaussian_energy_spectra(test_spectra,
                                                                  num_sigma=5.)
        expected_sigma = numpy.sqrt(energy/200.)
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.01*mean and energy > 0.99*mean)
        self.assertTrue(expected_sigma < 1.01*sigma and expected_sigma > 0.99*sigma)
        self.assertAlmostEqual(integral/float(num_entries), 1.0)

    def test_random_energy(self):
        """ Tests the random gaussian smearing method for energy.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        num_entries = 100000
        test_spectra = spectra.Spectra("Test",num_entries)
        energy = 2.5  # MeV
        for i in range(num_entries):
            test_spectra.fill(energy, 0, 0)
        smearing = smear.Smear()
        smearing._light_yield = 200.  # NHit per MeV
        smeared_spectra = smearing.random_gaussian_energy_spectra(test_spectra)
        expected_sigma = numpy.sqrt(energy/200.)
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.01*mean and energy > 0.99*mean)
        self.assertTrue(expected_sigma < 1.01*sigma and expected_sigma > 0.99*sigma)
        self.assertAlmostEqual(integral/float(num_entries), 1.0)

    def test_weight_radius(self):
        """ Tests the weighted gaussian smearing method for radius.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        num_entries = 10000
        test_spectra = spectra.Spectra("Test", num_entries)
        radius = 1000.  # mm
        for i in range(num_entries):
            test_spectra.fill(0, radius, 0)
        smearing = smear.Smear()
        smearing._position_resolution = 100.  # mm
        smeared_spectra = smearing.weight_gaussian_radius_spectra(test_spectra,
                                                                  num_sigma=5.)
        mean, sigma, integral = self.fit_gaussian_radius(smeared_spectra)
        self.assertTrue(radius < 1.01*mean and radius > 0.99*mean)
        self.assertTrue(smearing._position_resolution < 1.01*sigma and smearing._position_resolution > 0.99*sigma)
        self.assertAlmostEqual(integral/float(num_entries), 1.0)

    def test_random_radius(self):
        """ Tests the random gaussian smearing method for radius.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        num_entries = 100000
        test_spectra = spectra.Spectra("Test", num_entries)
        radius = 1000.  # mm
        for i in range(num_entries):
            test_spectra.fill(0, radius, 0)
        smearing = smear.Smear()
        smearing._position_resolution = 100.  # mm
        smeared_spectra = smearing.random_gaussian_radius_spectra(test_spectra)
        mean, sigma, integral = self.fit_gaussian_radius(smeared_spectra)
        self.assertTrue(radius < 1.01*mean and radius > 0.99*mean)
        self.assertTrue(smearing._position_resolution < 1.01*sigma and smearing._position_resolution > 0.99*sigma)
        self.assertAlmostEqual(integral/float(num_entries), 1.0)
