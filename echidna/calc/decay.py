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
      fv_radius (float, optional): Radius of fiducial volume in mm.
        Default is stored in :class:`echidna.calc.constants`
      scint_density (float, optional): Density of liquid scintillator in
        kg/mm^3. Default is stored in :class:`echidna.calc.constants`
      outer_radius (float, optional): Radius of outer container
        containing fiducial volume, e.g. AV, in mm. Default is stored in
        :class:`echidna.calc.constants`
      roi_efficiency (float, optional): Efficiency factor of ROI.
        Calculated by dividing the integral of the spectrum, shrunk to
        the ROI, by the integral of the full spectrum. Default is
        0.62465 (-0.5 to 1.5 sigma integral of a standard gaussian)

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
      _fv_radius (float): Radius of fiducial volume in mm. Default is
        stored in :class:`echidna.calc.constants`
      _scint_density (float): Density of liquid scintillator in
        kg/mm^3. Default is stored in :class:`echidna.calc.constants`
      _outer_radius (float): Radius of outer container containing
        fiducial volume, e.g. AV, in mm. Default is stored in
        :class:`echidna.calc.constants`
      _roi_efficiency (float): Efficiency factor of ROI. Calculated by
        dividing the integral of the spectrum, shrunk to the ROI, by
        the integral of the full spectrum. Default is 0.62465 (-0.5 to
        1.5 sigma integral of a standard gaussian)

    Raises:
      ValueError: If abundance is < 0. or > 1.

    """
    def __init__(self, name, atm_weight_iso, atm_weight_nat, abundance,
                 phase_space, matrix_element, loading=None, fv_radius=None,
                 outer_radius=None, scint_density=None,
                 roi_efficiency=0.62465):
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
        if fv_radius:
            if fv_radius <= 0. or fv_radius > self._outer_radius:
                raise ValueError("FV radius must be between zero and outer "
                                 "radius")
            self._fv_radius = fv_radius
        else:
            self._fv_radius = const._fv_radius
        if scint_density:
            self._scint_density = scint_density
        else:
            self._scint_density = const._scint_density
        # Defaults to standard Gaussian efficiency for
        # -1/2 sigma to +3/2 sigma ROI
        self._roi_efficiency = roi_efficiency
        if roi_efficiency != 0.62465:
            print ("Warning: using calculated ROI efficiency %.4f "
                   "not default (0.62465)" % roi_efficiency)

    def get_n_atoms(self, fv_radius=None, loading=None, scint_density=None,
                    target_mass=None, scint_mass=None, outer_radius=None):
        """ Calculates the number of atoms of the double-beta isotope.

          Set up to follow the full (SNO+-specific) calculation as per
          SNO+-doc-1728v2 but can look at other scenarios/detectors by
          overriding the default args.

        .. warning:: All args default to SNO+ specific values!

        Args:
          fv_radius (float, optional): Radius of fiducial volume in mm.
            Default is stored as a class variable.
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
          ValueError: If :obj:`fv_radius` is not between zero and
            :obj:`outer_radius`.

        Returns:
          float: Number of atoms.

        """
        # Set defaults
        if outer_radius is None:  # use class variable
            outer_radius = self._outer_radius
        if outer_radius <= 0.:
            raise ValueError("Outer radius must be positive and non-zero")
        if fv_radius is None:  # use class variable
            fv_radius = self._fv_radius
        if fv_radius <= 0. or fv_radius > outer_radius:
            raise ValueError("FV radius must be between zero and outer radius")
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

            # Volume fraction
            volume_fraction = fv_radius**3 / outer_radius**3
            target_mass = mass_fraction * volume_fraction * loading * scint_mass

        n_atoms = (target_mass*const._n_avagadro) / (self._atm_weight_iso*1.e-3)
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
        return numpy.sqrt(const._electron_mass**2 /
                          (self._phase_space*self._matrix_element**2*half_life))

    def activity_to_counts(self, activity, roi_cut=True, livetime=5.):
        """ Converts activity to number of counts, assuming constant activity.

        Args:
          activity (float): Initial activity of the isotope in
            :math:`years^{-1}`.
          roi_cut (bool, optional): If True (default) calculates counts
            in the ROI, not counts in the full spectrum.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Number of counts.

        """
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        if roi_cut:
            return activity*livetime*self._roi_efficiency
        else:
            return activity*livetime

    def counts_to_activity(self, counts, roi_cut=True, livetime=5.):
        """ Converts counts to activity, assuming constant activity.

        Args:
          counts (float): Number of counts.
          roi_cut (bool, optional): If True (default) assumes counts
            in the ROI, not counts in the full spectrum.
          livetime (float, optional): Amount of years of data taking.
            Default is 5 years.

        Raises:
          ValueError: If :obj:`livetime` is not positive and non-zero.

        Returns:
          float: Activity of the isotope in :math:`years^{-1}`.

        """
        if livetime <= 0.:
            raise ValueError("Livetime should be positive and non zero")
        if roi_cut:
            return counts/(livetime*self._roi_efficiency)
        else:
            return counts/livetime

    def counts_to_eff_mass(self, counts, n_atoms=None,
                           roi_cut=True, livetime=5.):
        """ Converts from signal counts to effective majorana mass.

        Args:
          counts (float): Number of signal counts within the livetime
            specified.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          roi_cut (bool, optional): If True (default) assumes counts
            in the ROI, not counts in the full spectrum.
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
        half_life = self.counts_to_half_life(counts, n_atoms,
                                             roi_cut, livetime)
        return self.half_life_to_eff_mass(half_life)

    def eff_mass_to_counts(self, eff_mass, n_atoms=None,
                           roi_cut=True, livetime=5.):
        """ Converts from effective majorana mass to signal counts.

        Args:
          eff_mass (float): Effective majorana mass in eV.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          roi_cut (bool, optional): If True (default) calculates counts
            in the ROI, not counts in the full spectrum.
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
        return self.half_life_to_counts(half_life, n_atoms, roi_cut, livetime)

    def half_life_to_counts(self, half_life, n_atoms=None,
                            roi_cut=True, livetime=5.):
        """ Converts from isotope's half-life to signal counts.

        Args:
          half_life (float): Isotope's :math:`0\\nu2\\beta` half-life in
            years.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          roi_cut (bool, optional): If True (default) calculates counts
            in the ROI, not counts in the full spectrum.
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
        return self.activity_to_counts(activity, roi_cut, livetime)

    def counts_to_half_life(self, counts, n_atoms=None,
                            roi_cut=True, livetime=5.):
        """ Converts from signal count to isotope's half-life.

        Args:
          count (float): Number of signal counts within the livetime
            specified.
          n_atoms (float, optional): Number of isotope atoms/nuclei that could
            potentially decay to produce signal.
          roi_cut (bool, optional): If True (default) assumes counts
            in the ROI, not counts in the full spectrum.
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
        activity = self.counts_to_activity(counts, roi_cut, livetime)
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

    if args.roi_efficiency:
        te130_converter = DBIsotope("Te130", Te130_atm_weight,
                                    TeNat_atm_weight, Te130_abundance,
                                    phase_space, matrix_element,
                                    signal.get_roi(0).get("efficiency"))
    else:
        te130_converter = DBIsotope("Te130", Te130_atm_weight,
                                    TeNat_atm_weight, Te130_abundance,
                                    phase_space, matrix_element)

    # Check get_n_atoms for 0.3% loading, no FV cut
    expected = 3.7573e27  # SNO+-doc-1728v2
    result, message = physics_tests.test_function_float(
        te130_converter.get_n_atoms, expected, fv_radius=const._av_radius)
    print message, "(no FV cut)"

    # Check get_n_atoms with SNO+ defaults
    expected = 7.4694e26  # calculated - A Back 2015-02-25, based on SNO+-doc-1728v2
    result, message = physics_tests.test_function_float(
        te130_converter.get_n_atoms, expected)
    print message

    # Create a DBIsotope instance for KLZ
    Xe136_atm_weight = 135.907219  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-05-07
    Xe134_atm_weight = 133.90539450  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-06-03
    # We want the atomic weight of the enriched Xenon
    XeEn_atm_weight = 0.9093*Xe136_atm_weight + 0.0889*Xe134_atm_weight
    Xe136_abundance = 0.9093  # PRC 86, 021601 (2012)
    phase_space = 1433.0e-17  # PRC 85, 034316 (2012)
    matrix_element = 3.33  # IBM-2 PRC 87, 014315 (2013)
    fv_radius = 1200.  # mm, PRC 86, 021601 (2012)
    loading = 0.0244  # 2.44%, PRC 86, 021601 (2012)
    scint_density = 756.28e-9  # kg/mm^3 calculated A Back 2015-07-22
    outer_radius = 1540.  # mm, PRC 86, 021601 (2012)

    xe136_converter = DBIsotope("Xe136", Xe136_atm_weight, XeEn_atm_weight,
                                Xe136_abundance, phase_space, matrix_element,
                                loading, fv_radius, outer_radius,
                                scint_density)

    expected = 5.3985e+26  # Calculated - A Back 2015-06-30
    result, message = physics_tests.test_function_float(
        xe136_converter.get_n_atoms, expected,
        fv_radius=fv_radius, loading=loading,
        scint_density=scint_density, outer_radius=outer_radius)
    print message, "(KamLAND-Zen)"

    # Check half_life_to_activity
    expected = 50.4  # /y, SNO+-doc-2593v8
    half_life = 5.17e25  # y, SNO+-doc-2593v8 (3 sigma FC limit @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.half_life_to_activity, expected, half_life=half_life,
        n_atoms=te130_converter.get_n_atoms(fv_radius=const._av_radius))
    print message, "(no FV cut)"

    # Check activity_to_half_life
    expected = 5.17e25  # y, SNO+-doc-2593v8
    activity = 50.4  # /y, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.activity_to_half_life, expected, activity=activity,
        n_atoms=te130_converter.get_n_atoms(fv_radius=const._av_radius))
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
    expected = 31.2  # ROI counts, SNO+-doc-2593v8 (3 sigma FC limit @ 5 y livetime)
    activity = 50.4 * (const._fv_radius**3/const._av_radius**3) # /y SNO+-doc-2593v8 - adjusted to FV
    result, message = physics_tests.test_function_float(
        te130_converter.activity_to_counts, expected, activity=activity,
        livetime=livetime, roi_cut=True)
    print message

    # Check counts_to_activity
    expected = 50.4 * (const._fv_radius**3/const._av_radius**3) # /y SNO+-doc-2593v8 - adjusted to FV
    counts = 31.2  # ROI counts, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_activity, expected, counts=counts,
        livetime=livetime, roi_cut=True)
    print message

    # Check counts_to_eff_mass
    expected = te130_converter.half_life_to_eff_mass(5.17e25)  # eV, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    counts = 31.2  # ROI counts, SNO+-doc-2593v8 (3 sigma CL @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_eff_mass,
        expected, counts=counts, roi_cut=True)
    print message

    # Check eff_mass_to_counts
    expected = 31.2  # ROI counts, SNO+-doc-2593v8 (3 sigma CL @ 5 y livetime)
    eff_mass = te130_converter.half_life_to_eff_mass(5.17e25)  # eV, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.eff_mass_to_counts,
        expected, eff_mass=eff_mass, roi_cut=True)
    print message

    # Check half_life_to_counts
    expected = 31.2  # ROI counts, SNO+-doc-2593v8
    half_life = 5.17e25  # y, SNO+-doc-2593v8 (3 sigma @ 5 y livetime)
    result, message = physics_tests.test_function_float(
        te130_converter.half_life_to_counts,
        expected, half_life=half_life, roi_cut=True)
    print message

    # Check counts_to_half_life
    expected = 5.17e25  # y, SNO+-doc-2593v8
    counts = 31.2  # ROI counts, SNO+-doc-2593v8
    result, message = physics_tests.test_function_float(
        te130_converter.counts_to_half_life,
        expected, counts=counts, roi_cut=True)
    print message

    print "============"


if __name__ == "__main__":
    import argparse

    from echidna.scripts.zero_nu_limit import ReadableDir
    import echidna.output.store as store

    parser = argparse.ArgumentParser(description="Example DBIsotpe calculator "
                                     "script and validation.")
    parser.add_argument("-s", "--signal", action=ReadableDir,
                        help="Supply path for signal hdf5 file")
    parser.add_argument("-e", "--roi_efficiency", action="store_true",
                        help="If ROI efficiency should be calculated to "
                        "override default value")
    args = parser.parse_args()

    test(args)
