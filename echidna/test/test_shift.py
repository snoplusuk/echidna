import unittest
import echidna.core.shift as shift
import echidna.core.spectra as spectra
import numpy as np
from scipy.optimize import curve_fit


class TestShift(unittest.TestCase):

    def gaussian(self, x, *p):
        """ A gaussian used for fitting.

        Args:
          x (float): Position the gaussian is calculated at.
          *p (list): List of parameters to fit

        Returns:
          float: Value of gaussian at x for given parameters
        """
        A, mean, sigma = p
        A = np.fabs(A)
        mean = np.fabs(mean)
        sigma = np.fabs(sigma)
        return A*np.exp(-(x-mean)**2/(2.*sigma**2))

    def fit_gaussian_energy(self, spectra):
        """ Fits a gausian to the energy of a spectrum.

        Args:
          spectra (core.spectra): Spectrum to be fitted

        Returns:
          tuple: mean (float), sigma (float) and
            integral (float) of the spectrum.
        """
        entries = []
        energies = []
        energy_width = spectra.get_config().get_par("energy_mc").get_width()
        energy_low = spectra.get_config().get_par("energy_mc")._low
        spectra_proj = spectra.project("energy_mc")
        for i in range(len(spectra_proj)):
            entries.append(spectra_proj[i])
            energies.append(energy_low+energy_width*(i+0.5))
        pars0 = [300., 2.5, 0.1]
        coeff, var_mtrx = curve_fit(self.gaussian, energies, entries, p0=pars0)
        return coeff[1], np.fabs(coeff[2]), np.array(entries).sum()

    def test_shift(self):
        """ Tests the variable shifting method.

        Creates a Gaussian spectra with mean energy 2.5 MeV and sigma 0.2 MeV.
        Radial values of the spectra have a uniform distribution.
        The "energy_mc" of the spectra is then shifted by 0.111 MeV.
        The shifted spectra is fitted with a Gaussian and the extracted
        mean and sigma are checked against expected values within 1 %.
        Integral of shifted spectrum is checked against original number of
        entries.
        This is then repeated for a shift of 0.2 MeV to test the shift_by_bin
        method.
        """
        np.random.seed()
        test_decays = 10000
        config_path = "echidna/config/spectra_example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        mean_energy = 2.5  # MeV
        sigma_energy = 0.2  # MeV
        for i in range(test_decays):
            energy = np.random.normal(mean_energy, sigma_energy)
            radius = np.random.random() * \
                test_spectra.get_config().get_par("radial_mc")._high
            test_spectra.fill(energy_mc=energy, radial_mc=radius)
        mean_energy, sigma_energy, integral = self.fit_gaussian_energy(
            test_spectra)
        # First test interpolation shift
        shifter = shift.Shift()
        shift_e = 0.111
        shifter.set_shift(shift_e)
        shifted_spectra = shifter.shift(test_spectra, "energy_mc")
        mean, sigma, integral = self.fit_gaussian_energy(shifted_spectra)
        expected_mean = mean_energy+shift_e
        expected_sigma = sigma_energy
        self.assertTrue(expected_mean < 1.01*mean and
                        expected_mean > 0.99*mean,
                        msg="Expected mean energy %s, fitted mean energy %s"
                        % (expected_mean, mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))
        # Now test shift by bin
        self.assertRaises(ValueError, shifter.shift_by_bin, test_spectra,
                          "energy_mc")
        shift_e = 0.2
        shifter.set_shift(shift_e)
        shifted_spectra = shifter.shift_by_bin(test_spectra, "energy_mc")
        mean, sigma, integral = self.fit_gaussian_energy(shifted_spectra)
        expected_mean = mean_energy+shift_e
        expected_sigma = sigma_energy
        self.assertTrue(expected_mean < 1.01*mean and
                        expected_mean > 0.99*mean,
                        msg="Expected mean energy %s, fitted mean energy %s"
                        % (expected_mean, mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))
