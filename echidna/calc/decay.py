import numpy
from echidna.calc import constants as const

class DBIsotope(object):
    """ Class which calculates expected counts for a DBD isotope
      over a given experiment livetime.

    Args:
      name (string): Name of the isotope
      loading (float): Loading of isotope with 0 to 1 equivalent
        to 0% to 100%.
      atm_weight_iso (float): Atomic weight of isotope in g/mol.
      atm_weight_nat (float): Atomic weight of natural element in g/mol.
      abundance (float): Natural abundance of isotope with 0 to 1
        equivalent to 0% to 100%.
      phase_space (float): Phase space of the isotope.
      matrix_element (float): Matrix element of the isotope.
      fv_radius (float): Fiducial volume radius. Defaults to SNO+ values.
      scint_density (float): Density of liquid scintillator. Defaults to
        SNO+ values.

    Raises:
      ValueError: If abundance is < 0. or > 1.
      ValueError: If loading is < 0. or > 1.
    """
    def __init__(self, name, loading, atm_weight_iso, atm_weight_nat,
                 abundance, phase_space, matrix_element, g_A,
                 fv_radius=None, scint_density=None):

        if loading < 0. or loading > 1.:
            raise ValueError("Loading ranges from 0 to 1")
        if abundance < 0. or abundance > 1.:
            raise ValueError("Abundance ranges from 0 to 1")
        self._name = name
        self._loading = loading
        self._atm_weight_iso = atm_weight_iso
        self._atm_weight_nat = atm_weight_nat
        self._abundance = abundance
        self._phase_space = phase_space
        self._matrix_element = matrix_element
        self._g_A = g_A
        self._fv_radius = fv_radius
        self._scint_density = scint_density

    def get_n_atoms(self):
        """ Calculates the number of atoms of an isotope within the
          fiducial volume.

        Returns:
          float: Number of atoms.
        """
        if self._loading < 0. or self._loading > 1.:
            raise ValueError("Loading ranges from 0 to 1")
        if self._abundance < 0. or self._abundance > 1.:
            raise ValueError("Abundance ranges from 0 to 1")
        mass_iso = self._atm_weight_iso*const._atomic_mass_unit  # kg/atom
        mass_nat = self._atm_weight_nat*const._atomic_mass_unit  # kg/atom
        mass_fraction = self._abundance*mass_iso/mass_nat
        if not self._scint_density:
            self._scint_density = const._scint_density
        if not self._fv_radius:
            self._fv_radius = const._fv_radius
        scint_mass = (4./3.)*numpy.pi*self._fv_radius**3*self._scint_density
        active_mass = scint_mass*mass_fraction*self._loading
        return active_mass/mass_iso

    def get_activity(self, half_life, n_atoms):
        """ Calculates the activity for an isotope with a given half-life
          and number of atoms.

        Args:
          half_life (float): Half-life of an isotope in years.
          n_atoms (float): Number of atoms of an isotope.

        Returns:
          float: Activity in decays per year.
        """
        return (numpy.log(2)/half_life)*n_atoms

    def get_0n2b_half_life(self, eff_mass):
        """ Calculates the 0n2b half-life of an isotope given a
          phase space, matrix element and an effective Majorana
          mass.

        Args:
          eff_mass (float): Effective majorana mass

        Returns:
          float: Zero neutrino half-life.
        """
        sq_mass_ratio = eff_mass**2/const._electron_mass**2
        return 1/(self._g_A**4*self._phase_space*self._matrix_element**2*sq_mass_ratio)

    def activity_to_counts(self, activity, livetime):
        """ Converts activity to number of counts assuming constant activity.

        Args:
          activity (float): Initial activity of the isotope in
            :math:`years^{-1}`.
          livetime (float): Amount of years of data taking.

        Returns:
          float: Number of counts.
        """
        return activity*livetime

    def counts_to_activty(self, counts, livetime=5.):
        return counts/livetime

    def activity_to_half_life(self, activity, n_atoms):
        return 1/((activity/n_atoms)*numpy.log(2))

    def half_life_to_mass(self, half_life):
        return numpy.sqrt(const._electron_mass**2/(self.g_A**4*self._phase_space*self._matrix_element**2*half_life))

    def counts_to_mass(self, counts, n_atoms, livetime=5.):
        activity = counts_to_activty(self, counts, livetime)
        half_life = activity_to_half_life(self, n_atoms, activity)
        return half_life_to_mass(self, half_life)

    def eff_mass_to_counts(self, eff_mass, livetime=5.):
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
        half_life = self.get_0n2b_half_life(eff_mass)
        return self.half_life_to_counts(half_life, livetime)

    def half_life_to_counts(self, half_life, livetime=5.):
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
        activity = self.get_activity(half_life, n_atoms)
        return self.activity_to_counts(activity, livetime)
