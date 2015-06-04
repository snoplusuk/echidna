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
      name (string): Name of the isotope
      atm_weight_iso (float): Atomic weight of isotope in g/mol.
      atm_weight_nat (float): Atomic weight of natural element in g/mol.
      abundance (float): Natural abundance of isotope with 0 to 1
        equivalent to 0% to 100%.
      phase_space (float): Phase space of the isotope.
      matrix_element (float): Matrix element of the isotope.

    Raises:
      ValueError: If abundance is < 0. or > 1.
      ValueError: If loading is < 0. or > 1.
    """
    def __init__(self, name, atm_weight_iso, atm_weight_nat, abundance,
                 phase_space, matrix_element):
        if abundance < 0. or abundance > 1.:
            raise ValueError("Abundance ranges from 0 to 1")
        self._name = name
        self._atm_weight_iso = atm_weight_iso
        self._atm_weight_nat = atm_weight_nat
        self._abundance = abundance
        self._phase_space = phase_space
        self._matrix_element = matrix_element
        self._roi_factor = 0.62465  # integral of roi/integral of full spectrum

    def get_n_atoms(self, fv_radius=None, loading=None, scint_density=None,
                    target_mass=None, scint_mass=None, outer_radius=None):
        """ Calculates the number of atoms of an isotope within the
          fiducial volume.

          Set up to follow the full (SNO+-specific) calculation as per
          SNO+-doc-1728v2 but can look at other scenarios/detectors by
          overriding the default args.

        .. warning:: All args default to SNO+ specific values!

        Args:
          fv_radius (float, optional): Radius of fiducial volume in mm.
          loading (float, optional): Loading of isotope with 0 to 1
            equivalent to 0% to 100%.
          scint_density (float): Density of liquid scintillator in
            kg/mm^3.
          target_mass (float, optional): Target mass in kg.
          scint_mass (float, optional): Mass of scintillator in kg.
          outer_radius (float, optional): Radius of outer container
            containing fiducial volume, e.g. AV, in mm.

        Returns:
          float: Number of atoms.
        """
        # Set defaults
        if fv_radius is None:  # use default from constants
            fv_radius = const._fv_radius
        if loading is None:  # use default from constants
            loading = const._loading
        if loading < 0. or loading > 1.:
            raise ValueError("Loading ranges from 0 to 1")
        if scint_density is None:  # use default from constants
            scint_density = const._scint_density
        if outer_radius is None:  # use default from constants
            outer_radius = const._av_radius
        if target_mass is None:  # Calculate target mass
            if scint_mass is None:  # Calculate scint_mass
                # Mass of scintillator
                scint_mass = (scint_density/const._d2o_density) * const._d2o_mass
            # Mass fraction
            mass_iso = self._atm_weight_iso*const._atomic_mass_unit  # kg/atom
            mass_nat = self._atm_weight_nat*const._atomic_mass_unit  # kg/atom
            mass_fraction = self._abundance*mass_iso/mass_nat

            # Volume fraction
            volume_fraction = fv_radius**3 / outer_radius**3
            target_mass = mass_fraction * volume_fraction * loading * scint_mass

        n_atoms = (target_mass*const._n_avagadro) / (self._atm_weight_iso*1.e-3)
        return n_atoms

    def half_life_to_activity(self, half_life, n_atoms):
        """ Calculates the activity for an isotope with a given half-life
          and number of atoms.

        Args:
          half_life (float): Half-life of an isotope in years.
          n_atoms (float): Number of atoms of an isotope.

        Returns:
          float: Activity in decays per year.

        """
        return (numpy.log(2)/half_life)*n_atoms

    def activity_to_half_life(self, activity, n_atoms):
        return numpy.log(2)*n_atoms/activity

    def eff_mass_to_half_life(self, eff_mass):
        """ Calculates the 0n2b half-life of an isotope given a
          phase space, matrix element and an effective Majorana
          mass.

        Args:
          eff_mass (float): Effective majorana mass

        Returns:
          float: Zero neutrino half-life.
        """
        sq_mass_ratio = eff_mass**2/const._electron_mass**2
        return 1/(self._phase_space*self._matrix_element**2*sq_mass_ratio)

    def half_life_to_eff_mass(self, half_life):
        return numpy.sqrt(const._electron_mass**2 /
                          (self._phase_space*self._matrix_element**2*half_life))

    def activity_to_counts(self, activity, livetime, **kwargs):
        """ Converts activity to number of counts assuming constant activity.

        Args:
          activity (float): Initial activity of the isotope in
            :math:`years^{-1}`.
          livetime (float): Amount of years of data taking.

        Returns:
          float: Number of counts.

        .. note::

          keyword arguments include:

            * roi_cut (*bool*): if true counts in roi is used
        """
        if kwargs.get("roi_cut"):
            return activity*livetime*self._roi_factor
        else:
            return activity*livetime

    def counts_to_activity(self, counts, livetime=5., **kwargs):
        """ Converts activity to number of counts assuming constant activity.

        Args:
          activity (float): Initial activity of the isotope in
            :math:`years^{-1}`.
          livetime (float): Amount of years of data taking.

        Returns:
          float: Number of counts.

            .. note::

          keyword arguments include:

            * roi_cut (*bool*): if true counts in roi is used
        """
        if kwargs.get("roi_cut"):
            return counts/(livetime*self._roi_factor)
        else:
            return counts/livetime

    def counts_to_eff_mass(self, counts, n_atoms, livetime=5., **kwargs):
        """
        .. note::

          keyword arguments include:

            * roi_cut (*bool*): if true counts in roi is used
        """
        activity = self.counts_to_activty(counts, livetime, **kwargs)
        half_life = self.counts_to_half_life(count, n_atoms)
        return self.half_life_to_eff_mass(half_life)

    def eff_mass_to_counts(self, eff_mass, livetime=5., **kwargs):
        """ Calculates the 0n2b counts of an isotope given a
          phase space, matrix element and an effective Majorana
          mass.

        Args:
          eff_mass (float): Effective majorana mass in eV

        Returns:
          float: 0 neutrino half-life.
        """
        if eff_mass <= 0.:
            raise ValueError("Effective mass should be positive and non zero")
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        half_life = self.eff_mass_to_half_life(eff_mass)
        return self.half_life_to_counts(half_life, livetime, **kwargs)

    def half_life_to_counts(self, half_life, livetime=5., **kwargs):
        """ Converts a double beta decay isotopes half-life
        and mass into counts in years.

        Args:
          half_life (float): Isotope's half-life in years.
          livetime (float): Number of years of data taking.

        Raises:
          ValueError: If abundance is < 0. or > 1.

        Returns:
          float: Number of expected counts.

        .. note::

          keyword arguments include:

            * roi_cut (*bool*): if true counts in roi is used
        """
        n_atoms = self.get_n_atoms()
        activity = self.half_life_to_activity(half_life, n_atoms)
        return self.activity_to_counts(activity, livetime, **kwargs)

    def counts_to_half_life(self, counts, livetime=5.):
        """ Converts a double beta decay isotopes half-life
        and mass into counts in years.

        Args:
          half_life (float): Isotope's half-life in years.
          livetime (float): Number of years of data taking.

        Raises:
          ValueError: If abundance is < 0. or > 1.

        Returns:
          float: Number of expected counts.
        """
        n_atoms = self.get_n_atoms()
        activity = self.counts_to_activity(counts, livetime)
        return n_atoms/(numpy.log(2)*activity)


def main():
    """ Test function to show agreement with Andy's numbers.
    """
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

    te130_converter = DBIsotope("Te130", Te130_atm_weight, TeNat_atm_weight,
                                Te130_abundance, phase_space, matrix_element)

    # Check get_n_atoms for 0.3% loading, no FV cut
    fv_radius = 5997.  # radius of AV in mm, calculated - A Back 2015-02-25

    expected = 3.7573e27  # SNO+-doc-1728v2
    result, message = physics_tests.test_function_float(te130_converter.get_n_atoms,
                                                        expected,
                                                        fv_radius=fv_radius)
    print message, "(no FV cut)"

    # Check get_n_atoms with SNO+ defaults
    expected = 7.4694e26  # calculated - A Back 2015-02-25, based on SNO+-doc-1728v2
    result, message = physics_tests.test_function_float(te130_converter.get_n_atoms,
                                                        expected)
    print message

    # Create a DBIsotope instance for KLZ
    Xe136_atm_weight = 135.907219  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-05-07
    Xe134_atm_weight = 133.90539450  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-06-03
    # We want the atomic weight of the enriched Xenon
    XeEn_atm_weight = 0.9093*Xe136_atm_weight + 0.0889*Xe134_atm_weight
    Xe136_abundance = 0.9093  # PRC 86, 021601 (2012)
    phase_space = 1433.0e-27  # PRC 85, 034316 (2012)
    matrix_element = 3.33  # IBM-2 PRC 87, 014315 (2013)

    xe136_converter = DBIsotope("Xe136", Xe136_atm_weight, XeEn_atm_weight,
                                Xe136_abundance, phase_space, matrix_element)
    # Check get_n_atoms with 2.44% loading in KLZ
    fv_radius = 1200.  # mm, PRC 86, 021601 (2012)
    loading = 0.0244  # 2.44%, PRC 86, 021601 (2012)
    scint_mass = 13.0e3  # kg (13 tonnes), PRC 86, 021601 (2012)
    outer_radius = 3080.  # mm, PRC 86, 021601 (2012)
    target_mass = 125.  # kg, PRC 86, 021601 (2012)

    expected = 5.5388e26  # Calculated - A Back 2015-06-03
    result, message = physics_tests.test_function_float(xe136_converter.get_n_atoms,
                                                        expected,
                                                        target_mass=target_mass)
    print message, "(KamLAND-Zen)"

    # Check
    print "============"


if __name__ == "__main__":
    main()
