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
          float: Value of gaussian at x for given parameters (float)
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
        return coeff[1], numpy.fabs(coeff[2]), numpy.array(entries).sum()

    def fit_gaussian_radius(self, spectra):
        """ Fits a gausian to the radius of a spectrum.

        Args:
          spectra (core.spectra): Spectrum to be smeared

        Returns:
          tuple: mean (float), sigma (float) and integral (float)
        """
        entries = []
        radii = []
        radial_width = spectra.get_config().get_par("radial_mc").get_width()
        radial_low = spectra.get_config().get_par("radial_mc")._low
        spectra_proj = spectra.project("radial_mc")
        for i in range(len(spectra_proj)):
            entries.append(spectra_proj[i])
            radii.append(radial_low+radial_width*(i+0.5))
        pars0 = [400., 1000., 100.]
        coeff, var_mtrx = curve_fit(self.gaussian, radii, entries, p0=pars0)
        return coeff[1], numpy.fabs(coeff[2]), numpy.array(entries).sum()

    def test_weight_energy_ly(self):
        """ Tests the weighted gaussian smearing method for energy
        by light yield.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 10000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy = 2.5  # MeV
        for i in range(test_decays):
            test_spectra.fill(energy_mc=energy, radial_mc=0)
        smearing = smear.EnergySmearLY()
        self.assertRaises(ValueError, smearing.set_resolution, -1.)
        smearing.set_resolution(200.)
        # Test set and get resolution
        self.assertTrue(smearing.get_resolution() == 200.,
                        msg="Expected a resolution of 200 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        # Test set and get num_sigma
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.weighted_smear(test_spectra,
                                                  par="energy_mc")
        expected_sigma = numpy.sqrt(energy/200.)
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.01*mean and energy > 0.99*mean,
                        msg="Input energy %s, fitted energy %s" % (energy,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))

    def test_random_energy_ly(self):
        """ Tests the random gaussian smearing method for energy light yield.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 50000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy = 2.5  # MeV
        for i in range(test_decays):
            test_spectra.fill(energy_mc=energy, radial_mc=0)
        smearing = smear.EnergySmearLY()
        self.assertRaises(ValueError, smearing.set_resolution, -200.)
        smearing.set_resolution(200.)  # NHit per MeV
        # Test set and get resolution
        self.assertTrue(smearing.get_resolution() == 200.,
                        msg="Expected a resolution of 200 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        # Test set and get num_sigma
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.random_smear(test_spectra,
                                                par="energy_mc")
        expected_sigma = numpy.sqrt(energy/200.)
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.02*mean and energy > 0.98*mean,
                        msg="Input energy %s, fitted energy %s" % (energy,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))

    def test_weight_energy_res(self):
        """ Tests the weighted gaussian smearing method for energy resolution.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 10000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy = 2.5  # MeV
        for i in range(test_decays):
            test_spectra.fill(energy_mc=energy, radial_mc=0)
        smearing = smear.EnergySmearRes()
        self.assertRaises(ValueError, smearing.set_resolution, 200.)
        smearing.set_resolution(0.05)
        # Test set and get resolution
        self.assertTrue(smearing.get_resolution() == 0.05,
                        msg="Expected a resolution of 0.05 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        # Test set and get num_sigma
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.weighted_smear(test_spectra,
                                                  par="energy_mc")
        expected_sigma = numpy.sqrt(energy)*0.05
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.01*mean and energy > 0.99*mean,
                        msg="Input energy %s, fitted energy %s" % (energy,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))

    def test_random_energy_res(self):
        """ Tests the random gaussian smearing method for energy resolution.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 50000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        energy = 2.5  # MeV
        for i in range(test_decays):
            test_spectra.fill(energy_mc=energy, radial_mc=0)
        smearing = smear.EnergySmearRes()
        self.assertRaises(ValueError, smearing.set_resolution, 2.)
        smearing.set_resolution(0.05)  # NHit per MeV
        # Test set and get resolution
        self.assertTrue(smearing.get_resolution() == 0.05,
                        msg="Expected a resolution of 0.05 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        # Test set and get num_sigma
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.random_smear(test_spectra,
                                                par="energy_mc")
        expected_sigma = numpy.sqrt(energy)*0.05
        mean, sigma, integral = self.fit_gaussian_energy(smeared_spectra)
        self.assertTrue(energy < 1.01*mean and energy > 0.99*mean,
                        msg="Input energy %s, fitted energy %s" % (energy,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.02*sigma and
                        expected_sigma > 0.98*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))

    def test_weight_radius(self):
        """ Tests the weighted gaussian smearing method for radius.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 10000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        radius = 1000.  # mm
        for i in range(test_decays):
            test_spectra.fill(energy_mc=0, radial_mc=radius)
        smearing = smear.RadialSmear()
        self.assertRaises(ValueError, smearing.set_resolution, -1.)
        smearing.set_resolution(100.)  # mm
        self.assertTrue(smearing.get_resolution() == 100.,
                        msg="Expected a resolution of 100 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.weighted_smear(test_spectra,
                                                  par="radial_mc")
        expected_sigma = 100.
        mean, sigma, integral = self.fit_gaussian_radius(smeared_spectra)
        self.assertTrue(radius < 1.01*mean and radius > 0.99*mean,
                        msg="Input radius %s, fitted energy %s" % (radius,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))

    def test_random_radius(self):
        """ Tests the random gaussian smearing method for radius.

        Creates a delta function and fits the mean and sigma after smearing.
        Mean and sigma are checked against set values within 1 %.
        Integral of smeared spectrum is checked against original number of
        entries.
        """
        test_decays = 50000
        config_path = "echidna/config/example.yml"
        config = spectra.SpectraConfig.load_from_file(config_path)
        test_spectra = spectra.Spectra("Test", test_decays, config)
        radius = 1000.  # mm
        for i in range(test_decays):
            test_spectra.fill(energy_mc=0, radial_mc=radius)
        smearing = smear.RadialSmear()
        self.assertRaises(ValueError, smearing.set_resolution, -1.)
        smearing.set_resolution(100.)  # mm
        self.assertTrue(smearing.get_resolution() == 100.,
                        msg="Expected a resolution of 100 but got %s"
                        % smearing.get_resolution())
        self.assertRaises(ValueError, smearing.set_num_sigma, -1.)
        smearing.set_num_sigma(5.)
        self.assertTrue(smearing.get_num_sigma() == 5.,
                        "Expected num_sigma to be 5 but got %s"
                        % smearing.get_num_sigma())
        smeared_spectra = smearing.weighted_smear(test_spectra,
                                                  par="radial_mc")
        expected_sigma = 100.
        mean, sigma, integral = self.fit_gaussian_radius(smeared_spectra)
        self.assertTrue(radius < 1.01*mean and radius > 0.99*mean,
                        msg="Input radius %s, fitted energy %s" % (radius,
                                                                   mean))
        self.assertTrue(expected_sigma < 1.01*sigma and
                        expected_sigma > 0.99*sigma,
                        msg="Expected sigma %s, fitted sigma %s"
                        % (expected_sigma, sigma))
        self.assertAlmostEqual(integral/float(test_decays), 1.0,
                               msg="Input decays %s, integral of spectra %s"
                               % (test_decays, integral))
