# Physical Constants
_electron_mass = 0.510998910e6  # PDG eV/c^2
_atomic_mass_unit = 1.660538921e-27  # kg
_n_avagadro = 6.02214129e23  # /mol PDG 2012

# Detector Constants
_av_radius = 5997.  # mm
_d2o_mass = 1.0e6  # kg, mass of D2O in SNO - Wilson 2004
_d2o_density = 1.107e-6  # kg/mm^3
_scint_density = 862.8e-9  # kg/mm^3

# SNO+ defaults
_fv_radius = 3500.  # mm
_loading = 0.003  # 0.3%

# KamLAND-Zen detector information
klz_detector = {
    # REF: Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-05-07
    "Xe136_atm_weight": 135.907219,
    # REF: Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 2015-06-03
    "Xe134_atm_weight": 133.90539450,
    # We want the atomic weight of the enriched Xenon
    "XeEn_atm_weight": 0.9093*135.907219 + 0.0889*133.90539450,
    # REF: Xenon @ Periodic Table of Chemical Elements,
    #   http://www/webqc.org/periodictable-Xenon-Xe.html, 05/07/2015
    "Xe136_abundance": 0.089,
    # REF: A. Gando et al. (KamLAND-Zen Collaboration) Phys. Rev. C. 86,
    #   021601 (2012) - both values.
    "loading": 0.0244,
    "ib_radius": 1540.,   # mm
    "scint_density": 7.5628e-7,  # kg/mm^3, calculated by A Back 2015-07-28
    # REF: A. Gando et al. (KamLAND-Zen Collaboration) Phys. Rev. C. 86,
    #   021601 (2012) - both values.
    "livetime": 112.3 / 365.25,  # y, (112.3 live days)
    "fv_radius": 1200.
}
