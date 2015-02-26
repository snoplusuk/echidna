import numpy

_avagadros = 6.02214129e23  # PDG Avagadro's constant
_electron_mass = 0.510998910e6  # PDG eV/c^2


def get_n_atoms(mass, atm_weight, abundance):
    """ Calculates the number of atoms of an isotope.
    Args:
      mass (float): Mass of isotope in kg.
      atm_weight (float): Atomic weight of isotope in g/mol.
      abundance (float): Natural abundance of isotope with 0 to 1 equivalent
      to 0% to 100%.

    Raises:
      ValueError: If abundance is < 0. or > 1.

    Returns:
      float: Number of atoms.
    """
    if abundance < 0. or abundance > 1.:
        raise ValueError("Abundance ranges from 0 to 1")
    return (mass*1000.*_avagadros*abundance)/atm_weight


def get_activity(halflife, n_atoms):
    """ Calculates the activity for an isotope with a given half-life
      and number of atoms.

    Args:
      halflife (float): Half-life of an isotope in years.
      n_atoms (float): Number of atoms of an isotope.
    """
    return (numpy.log(2)/halflife)*n_atoms


def get_0n2b_half_life(phase_space, matrix_elem, eff_mass):
    """ Calculates the 0n2b of an isotope given a phase space,
      matrix element and an effective Majorana mass.

    Args:
      phase_space (float): Phase space of the isotope
      matrix_elem (float): Matrix element of the isotope
      eff_mass (float): Effective majorana mass

    Returns:
      float: 0 neutrino half-life.
    """
    return 1/(phase_space*matrix_elem**2*(eff_mass**2/_electron_mass**2))


def dbd_activity_to_counts(activity, years):
    """ Converts activity to number of counts assuming constant activity.

    Args:
      activity (float): Initial activity of the isotope in
        :math:`years^{-1}`.
      years (float): Amount of years of data taking.

    Returns:
      float: Number of counts.
    """
    return activity*years


def dbd_halflife_to_counts(halflife, mass, atm_weight,
                           abundance, years=5.):
    """ Converts a double beta decay isotopes halflife
      and mass into counts in years.

    Args:
      halflife (float): Isotope's halflife in years.
      mass (float): Mass of isotope in kg.
      atm_weight (float): Atomic weight of isotope in g/mol.
      abundance (float): Natural abundance of isotope with 0 to 1
      equivalent to 0% to 100%.
      years (float): Number of years of data taking.

    Raises:
      ValueError: If abundance is < 0. or > 1.

    Returns:
      float: Number of expected counts.
    """
    if abundance < 0. or abundance > 1.:
        raise ValueError("Abundance ranges from 0 to 1")
    n_atoms = get_n_atoms(mass, atm_weight, abundance)
    activity = get_activity(half_life, n_atoms)
    return dbd_activity_to_counts(activity, years)
