import numpy
from scipy.optimize import curve_fit

import echidna.core.scale as scale
from echidna.core.config import SpectraConfig
import echidna.core.spectra as spectra

import unittest


class TestScale(unittest.TestCase):

    def gaussian(self, x, *p):
        """ A gaussian used for fitting.

        Args:
          x (float): Position the gaussian is calculated at.
          *p (list): List of parameters to fit

        Returns:
          float: Value of gaussian at x for given parameters
        """
        A, mean, sigma = p
        A = numpy.fabs(A)
        mean = numpy.fabs(mean)
        sigma = numpy.fabs(sigma)
        return A*numpy.exp(-(x-mean)**2/(2.*sigma**2))

    def fit_gaussian_energy(self, spectrum):
        """ Fits a gausian to the energy of a spectrum.

        Args:
          spectra (:class:`spectra.Spectra`): Spectrum to be fitted

        Returns:
          tuple: mean (float), sigma (float) and
            integral (float) of the spectrum.
        """
        entries = []
        energies = []
        energy_width = spectrum.get_config().get_par("energy_mc").get_width()
        energy_low = spectrum.get_config().get_par("energy_mc").get_low()
        spectrum_proj = spectrum.project("energy_mc")
        for i in range(len(spectrum_proj)):
            entries.append(spectrum_proj[i])
            energies.append(energy_low+energy_width*(i+0.5))
        pars0 = [300., 2.5, 0.1]
        coeff, var_mtrx = curve_fit(self.gaussian, energies, entries, p0=pars0)
        return coeff[1], numpy.fabs(coeff[2]), numpy.array(entries).sum()

    def test_scale(self):
        """ Tests the variable scaling method.

        Creates a Gaussian spectra with mean energy 2.5 MeV and sigma 0.2 MeV.
        Radial values of the spectra have a uniform distribution.
        The "energy_mc" of the spectra is then scaled by a factor by 1.1.
        The scaled spectra is fitted with a Gaussian and the extracted
        mean and sigma are checked against expected values within 1 %.
        Integral of scaled spectrum is checked against original number of
        entries.
        """
        numpy.random.seed()
        test_decays = 10000
        config_path = "echidna/config/spectra_example.yml"
        config = SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        mean_energy = 2.5  # MeV
        sigma_energy = 0.2  # MeV
        for i in range(test_decays):
            energy = numpy.random.normal(mean_energy, sigma_energy)
            radius = numpy.random.random() * \
                test_spectra.get_config().get_par("radial_mc")._high
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
        mean_energy, sigma_energy, integral = self.fit_gaussian_energy(
            test_spectra)
        scaler = scale.Scale()
        self.assertRaises(ValueError, scaler.set_scale_factor, -1.1)
        scale_factor = 1.1
        scaler.set_scale_factor(scale_factor)
        scaled_spectra = scaler.scale(test_spectra, "energy_mc")
        mean, sigma, integral = self.fit_gaussian_energy(scaled_spectra)
        expected_mean = mean_energy*scale_factor
        expected_sigma = sigma_energy*scale_factor
        self.assertTrue(expected_mean < 1.01*mean and
                        expected_mean > 0.99*mean,
                        msg="Expected mean energy %s, fitted mean energy %s"
                        % (expected_mean, mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(scaled_spectra.sum()/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, scaled_spectra.sum()))
