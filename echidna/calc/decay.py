""" Double beta decay utility converter

Provides a useful tool for converting between different double beta
dacay parameters.

"""
import numpy

from echidna.calc import constants as const
import echidna.test.physics_tests as physics_tests


class DBIsotope(object):
    """ Class which calculates expected counts for a DBD isotope
      over a given experiment livetime.

    Args:
      name (string): Name of the isotope.
      atm_weight_iso (float): Atomic weight of isotope in g/mol.
      atm_weight_nat (float): Atomic weight of natural element in g/mol.
      abundance (float): Natural abundance of isotope with 0 to 1
        equivalent to 0% to 100%.
      phase_space (float): Phase space of the isotope.
      matrix_element (float): Matrix element of the isotope.
      loading (float, optional): Loading of isotope with 0 to 1
        equivalent to 0% to 100%. Default is stored in
        :class:`echidna.calc.constants`
      scint_density (float, optional): Density of liquid scintillator in
        kg/mm^3. Default is stored in :class:`echidna.calc.constants`
      outer_radius (float, optional): Radius of outer container
        containing fiducial volume, e.g. AV, in mm. Default is stored in
        :class:`echidna.calc.constants`

    Attributes:
      _name (string): Name of the isotope.
      _atm_weight_iso (float): Atomic weight of isotope in g/mol.
      _atm_weight_nat (float): Atomic weight of natural element in g/mol.
      _abundance (float): Natural abundance of isotope with 0 to 1
        equivalent to 0% to 100%.
      _phase_space (float): Phase space of the isotope.
      _matrix_element (float): Matrix element of the isotope.
      _loading (float): Loading of isotope with 0 to 1 equivalent to 0%
        to 100%. Default is stored in :class:`echidna.calc.constants`
      _scint_density (float): Density of liquid scintillator in
        kg/mm^3. Default is stored in :class:`echidna.calc.constants`
      _outer_radius (float): Radius of outer container containing
        fiducial volume, e.g. AV, in mm. Default is stored in
        :class:`echidna.calc.constants`

    Raises:
      ValueError: If abundance is < 0. or > 1.
      ValueError: If :obj:`outer_radius` is negative or zero.
    """
    def __init__(self, name, atm_weight_iso, atm_weight_nat, abundance,
                 phase_space, matrix_element, loading=None,
                 outer_radius=None, scint_density=None):
        if abundance < 0. or abundance > 1.:
            raise ValueError("Abundance ranges from 0 to 1")
        self._name = name
        self._atm_weight_iso = atm_weight_iso
        self._atm_weight_nat = atm_weight_nat
        self._abundance = abundance
        self._phase_space = phase_space
        self._matrix_element = matrix_element
        if loading:
            if loading < 0. or loading > 1.:
                raise ValueError("Loading ranges from 0 to 1")
            self._loading = loading
        else:
            # Default SNO+ Loading
            self._loading = const._loading
        if outer_radius:
            if outer_radius <= 0.:
                raise ValueError("Outer radius must be positive and non-zero")
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        if scint_density:
            self._scint_density = scint_density
        else:
            self._scint_density = const._scint_density

    def get_n_atoms(self, loading=None, scint_density=None,
                    target_mass=None, scint_mass=None, outer_radius=None):
        """ Calculates the number of atoms of the double-beta isotope.

          Set up to follow the full (SNO+-specific) calculation as per
          SNO+-doc-1728v2 but can look at other scenarios/detectors by
          overriding the default args.

        .. warning:: All args default to SNO+ specific values!

        Args:
          loading (float, optional): Loading of isotope with 0 to 1
            equivalent to 0% to 100%. Default is stored as a class
            variable.
          scint_density (float, optional): Density of liquid scintillator in
            kg/mm^3. Default is stored as a class variable.
          target_mass (float, optional): Target mass in kg. Calculates a
            value by default.
          scint_mass (float, optional): Mass of scintillator in kg.
            Calculates a value by default.
          outer_radius (float, optional): Radius of outer container
            containing fiducial volume, e.g. AV, in mm. Default is stored
            as a class variable.

        Raises:
          ValueError: If :obj:`loading` is not between zero and 1.
          ValueError: If :obj:`outer_radius` is negative or zero.

        Returns:
          float: Number of atoms.

        """
        # Set defaults
        if outer_radius is None:  # use class variable
            outer_radius = self._outer_radius
        if outer_radius <= 0.:
            raise ValueError("Outer radius must be positive and non-zero")
        if loading is None:  # use class variable
            loading = self._loading
        if loading < 0. or loading > 1.:
            raise ValueError("Loading ranges from 0 to 1")
        if scint_density is None:  # use class variable
            scint_density = self._scint_density
        if target_mass is None:  # Calculate target mass
            if scint_mass is None:  # Calculate scint_mass
                # Mass of scintillator
                volume = (4./3.) * numpy.pi * outer_radius**3  # mm^3
                scint_mass = scint_density * volume
            # Mass fraction
            mass_iso = self._atm_weight_iso*const._atomic_mass_unit  # kg/atom
            mass_nat = self._atm_weight_nat*const._atomic_mass_unit  # kg/atom
            mass_fraction = self._abundance*mass_iso/mass_nat

            target_mass = mass_fraction * loading * scint_mass

        n_atoms = (target_mass*const._n_avagadro) /\
            (self._atm_weight_iso*1.e-3)
        return n_atoms

    def half_life_to_activity(self, half_life, n_atoms=None):
        """ Calculates the activity for an isotope with a given half-life
          and number of atoms.

        Args:
          half_life (float): Half-life of an isotope in years.
          n_atoms (float, optional): Number of atoms of an isotope.

        Returns:
          float: Activity in decays per year.

        """
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        return (numpy.log(2)/half_life)*n_atoms

    def activity_to_half_life(self, activity, n_atoms=None):
        """ Calculates the half-life of an isotope with a given
        activity and number of atoms.

        Args:
          activity (float): Activity of the isotope in
          :math:`years^{-1}`.
          n_atoms (float, optional): Number of atoms of an isotope.

        Returns:
          float: Half-life in years.

        """
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        return numpy.log(2)*n_atoms/activity

    def eff_mass_to_half_life(self, eff_mass):
        """ Converts from effective majorana mass to :math:`0\\nu2\\beta`
        half-life.

        Args:
          eff_mass (float): Effective majorana mass, in eV.

        Raises:
          ValueError: If effective mass is not positive and non-zero.

        Returns:
          float: :math:`0\\nu2\\beta` half-life, in years.

        """
        if eff_mass <= 0.:
            raise ValueError("Effective mass should be positive and non-zero")
        sq_mass_ratio = eff_mass**2/const._electron_mass**2
        return 1/(self._phase_space*self._matrix_element**2*sq_mass_ratio)

    def half_life_to_eff_mass(self, half_life):
        """ Converts from :math:`0\\nu2\\beta` half-life to effective
          majorana mass.

        Args:
          half_life (float): :math:`0\\nu2\\beta` half-life, in years.

        Returns:
          float: Effective majorana mass, in eV.

        """
        return numpy.sqrt(const._electron_mass ** 2 /
                          (self._phase_space * self._matrix_element ** 2 *
                           half_life))

    def activity_to_counts(self, activity, livetime=5.):
        """ Converts activity to number of counts, assuming constant activity.

        Args:
          activity (float): Initial activity of the isotope in
            :math:`years^{-1}`.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Number of counts.

        """
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        return activity*livetime

    def counts_to_activity(self, counts, livetime=5.):
        """ Converts counts to activity, assuming constant activity.

        Args:
          counts (float): Number of counts.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Activity of the isotope in :math:`years^{-1}`.

        """
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        return counts/livetime

    def counts_to_eff_mass(self, counts, n_atoms=None, livetime=5.):
        """ Converts from signal counts to effective majorana mass.

        Args:
          counts (float): Number of signal counts within the livetime
            specified.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Effective majorana mass in eV.

        """
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        half_life = self.counts_to_half_life(counts, n_atoms, livetime)
        return self.half_life_to_eff_mass(half_life)

    def eff_mass_to_counts(self, eff_mass, n_atoms=None, livetime=5.):
        """ Converts from effective majorana mass to signal counts.

        Args:
          eff_mass (float): Effective majorana mass in eV.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If effective mass is not positive and non-zero.
          ValueError: If arg:`livetime` is not positive and non-zero.

        Returns:
          float: Expected number of signal counts within the livetime
            specified.

        """
        if eff_mass <= 0.:
            raise ValueError("Effective mass should be positive and non-zero")
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        half_life = self.eff_mass_to_half_life(eff_mass)
        return self.half_life_to_counts(half_life, n_atoms, livetime)

    def half_life_to_counts(self, half_life, n_atoms=None, livetime=5.):
        """ Converts from isotope's half-life to signal counts.

        Args:
          half_life (float): Isotope's :math:`0\\nu2\\beta` half-life in
            years.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Expected number of counts.

        """
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        activity = self.half_life_to_activity(half_life, n_atoms)
        return self.activity_to_counts(activity, livetime)

    def counts_to_half_life(self, counts, n_atoms=None, livetime=5.):
        """ Converts from signal count to isotope's half-life.

        Args:
          count (float): Number of signal counts within the livetime
            specified.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Isotope's :math:`0\\nu2\\beta` half-life in years.

        """
        if n_atoms is None:  # Calculate n_atoms from class variables
            n_atoms = self.get_n_atoms()
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        activity = self.counts_to_activity(counts, livetime)
        return self.activity_to_half_life(activity, n_atoms)


def test(args):
    """ Test function to show agreement with Andy's numbers.

    Args:
      args (dict): Command line arguments from :mod:`argparse`
    """
    # Load signal
    signal = store.load(args.signal)
    # Cut to 3.5m FV and 5 year livetime
    signal.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    # Shrink to ROI
    signal.shrink_to_roi(2.46, 2.68, 0)  # ROI used by Andy

    print "============"
    print "decay module"
    print "------------"
    # Check results of each function
    # Create instance of DBIsotope for Te130
    Te130_atm_weight = 129.906229  # SNO+-doc-1728v2
    TeNat_atm_weight = 127.6  # SNO+-doc-1728v2
    Te130_abundance = 0.3408  # SNO+-doc-1728v2
    phase_space = 3.69e-14  # PRC 85, 034316 (2012)
    matrix_element = 4.03  # IBM-2 PRC 87, 014315 (2013)

    te130_converter = DBIsotope("Te130", Te130_atm_weight,
                                TeNat_atm_weight, Te130_abundance,
                                phase_space, matrix_element)

    # Check get_n_atoms for 0.3% loading, no FV cut
    expected = 3.7573e27  # SNO+-doc-1728v2
    result, message = physics_tests.test_function_float(
        te130_converter.get_n_atoms, expected)
    print message, "(no FV cut)"

    # Check get_n_atoms with SNO+ defaults
    # calculated - A Back 2015-02-25, based on SNO+-doc-1728v2
    expected = 7.4694e26
    result, message = physics_tests.test_function_float(
        te130_converter.get_n_atoms, expected)
    print message

    # Create a DBIsotope instance for KLZ
    # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-05-07
    Xe136_atm_weight = 135.907219
    # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-06-03
    Xe134_atm_weight = 133.90539450
    # We want the atomic weight of the enriched Xenon
    XeEn_atm_weight = 0.9093*Xe136_atm_weight + 0.0889*Xe134_atm_weight
    Xe136_abundance = 0.9093  # PRC 86, 021601 (2012)
    phase_space = 1433.0e-17  # PRC 85, 034316 (2012)
    matrix_element = 3.33  # IBM-2 PRC 87, 014315 (2013)
    loading = 0.0244  # 2.44%, PRC 86, 021601 (2012)
    scint_density = 756.28e-9  # kg/mm^3 calculated A Back 2015-07-22
    outer_radius = 1540.  # mm, PRC 86, 021601 (2012)

    xe136_converter = DBIsotope("Xe136", Xe136_atm_weight, XeEn_atm_weight,
                                Xe136_abundance, phase_space, matrix_element,
                                loading, outer_radius, scint_density)

    expected = 5.3985e+26  # Calculated - A Back 2015-06-30
    result, message = physics_tests.test_function_float(
        xe136_converter.get_n_atoms, expected, loading=loading,
        scint_density=scint_density, outer_radius=outer_radius)
    print message, "(KamLAND-Zen)"

    # Check half_life_to_activity
    expected = 50.4  # /y, SNO+-doc-2593v8
    half_life = 5.17e25  # y, SNO+-doc-2593v8 (3 sigma FC limit @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.half_life_to_activity, expected, half_life=half_life,
        n_atoms=te130_converter.get_n_atoms())
    print message, "(no FV cut)"

    # Check activity_to_half_life
    expected = 5.17e25  # y, SNO+-doc-2593v8
    activity = 50.4  # /y, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.activity_to_half_life, expected, activity=activity,
        n_atoms=te130_converter.get_n_atoms())
    print message, "(no FV cut)"

    # Check eff_mass_to_half_life
    expected = 4.37e25  # y, SNO+-doc-2593v8 (90% CL @ 1 y livetime)
    eff_mass = 0.0999  # eV, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.eff_mass_to_half_life, expected, eff_mass=eff_mass)
    print message

    # Check half_life_to_eff_mass
    expected = 0.0999  # eV, SNO+-doc-2593v8
    half_life = 4.37e25  # y, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.half_life_to_eff_mass, expected, half_life=half_life)
    print message

    # Check activity_to_counts
    livetime = 5.0
    # ROI counts, SNO+-doc-2593v8 (3 sigma FC limit @ 5 y livetime)
    expected = 31.2
    # /y SNO+-doc-2593v8 - adjusted to FV
    activity = 50.4
    result, message = physics_tests.test_function_float(
        te130_converter.activity_to_counts, expected, activity=activity,
        livetime=livetime)
    print message

    # Check counts_to_activity
    # /y SNO+-doc-2593v8 - adjusted to FV
    expected = 50.4
    counts = 31.2  # ROI counts, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_activity, expected, counts=counts,
        livetime=livetime)
    print message

    # Check counts_to_eff_mass
    # eV, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    expected = te130_converter.half_life_to_eff_mass(5.17e25)
    counts = 31.2  # ROI counts, SNO+-doc-2593v8 (3 sigma CL @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_eff_mass,
        expected, counts=counts)
    print message

    # Check eff_mass_to_counts
    expected = 31.2  # ROI counts, SNO+-doc-2593v8 (3 sigma CL @ 5 y livetime)
    # eV, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    eff_mass = te130_converter.half_life_to_eff_mass(5.17e25)
    result, message = physics_tests.test_function_float(
        te130_converter.eff_mass_to_counts,
        expected, eff_mass=eff_mass)
    print message

    # Check half_life_to_counts
    expected = 31.2  # ROI counts, SNO+-doc-2593v8
    half_life = 5.17e25  # y, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.half_life_to_counts,
        expected, half_life=half_life)
    print message

    # Check counts_to_half_life
    expected = 5.17e25  # y, SNO+-doc-2593v8
    counts = 31.2  # ROI counts, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_half_life,
        expected, counts=counts)
    print message

    print "============"


# Matrix elements - dictionary with Spectra name as key and matrix element as
#                   value.
matrix_elements = {
    # REF: F. Simkovic et al. Phys. Rev. C. 79, 055501 1-10 (2009)
    # Averaged over min and max values from columns 2, 4 & 6 in Table III
    "Xe136_0n2b_n1": 2.205,
    "Xe136_0n2b_n2": None,
    # REF: M. Hirsh et al. Phys. Lett. B. 372, 8-14 (1996) - Table 2
    # Assuming two Majorons emitted i.e. only type IE or IID modes
    "Xe136_0n2b_n3": 1.e-3,
    # REF: M. Hirsh et al. Phys. Lett. B. 372, 8-14 (1996) - Table 2
    "Xe136_0n2b_n7": 1.e-3
    }

# Phase space factors - dictionary with Spectra name as key and phase space
#                       factor as value.
phase_spaces = {
    # REF: Suhonen, J. & Civitarese, O. Physics Reports, Elsevier BV, 300
    # 123-214 (1998)
    # Table 6
    "Xe136_0n2b_n1": 6.02e-16,
    "Xe136_0n2b_n2": None,
    # Assuming two Majorons emitted i.e. only type IE or IID modes
    # REF: M. Hirsh et al. Phys. Lett. B. 372, 8-14 (1996) - Table 3
    "Xe136_0n2b_n3": 1.06e-17,
    # REF: M. Hirsh et al. Phys. Lett. B. 372, 8-14 (1996) - Table 3
    "Xe136_0n2b_n7": 4.54e-17
    }


if __name__ == "__main__":
    import argparse

    from echidna.scripts.zero_nu_limit import ReadableDir
    import echidna.output.store as store

    parser = argparse.ArgumentParser(description="Example DBIsotpe calculator "
                                     "script and validation.")
    parser.add_argument("-s", "--signal", action=ReadableDir,
                        help="Supply path for signal hdf5 file")
    args = parser.parse_args()

    test(args)
