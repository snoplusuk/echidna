""" SNO+ Majoron limit setting script

This script sets 90% confidence limit on the Majoron-emitting
neutrinoless double beta decay modes (with spectral indices n = 1, 2, 3
and 7), with SNO+.

Examples:
  To use simply run the script::

    $ python zero_nu_limit.py -s /path/to/signal.hdf5 -t /path/to/2n2b.hdf5 -b /path/to/B8_Solar.hdf5

.. note:: Use the -v option to print out progress and timing information
"""
import numpy

import echidna
import echidna.output.store as store
import echidna.limit.limit_config as limit_config
import echidna.limit.limit_setting as limit_setting
import echidna.limit.chi_squared as chi_squared
import echidna.calc.decay as decay
import echidna.calc.constants as constants


def main():
    """ Script to set 90% CL on all four Majoron-emitting modes.
    """
    # Load signal spectra
    signal_hdf5s = ["/data/Xe136_0n2b_n1_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Xe136_0n2b_n2_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Xe136_0n2b_n3_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Xe136_0n2b_n7_r1.ntuple.root_mc_smeared.hdf5"]
    signals = []
    for signal_hdf5 in signal_hdf5s:
        spectrum = store.load(echidna.__echidna_base__+signal_hdf5)
        print spectrum._name
        signals.append(spectrum)

    # Load background spectra
    backgrounds = []
    Xe136_2n2b = store.load(echidna.__echidna_base__ +
                            "/data/Xe136_2n2b_mc_smeared.hdf5")
    print Xe136_2n2b._name
    backgrounds.append(Xe136_2n2b)
    B8_Solar = store.load(echidna.__echidna_base__ +
                          "/data/B8_Solar_KLZ_mc_smeared.hdf5")
    print B8_Solar._name
    B8_Solar._num_decays = 1044770.0
    backgrounds.append(B8_Solar)

    # Apply FV and livetime cuts
    fv_radius = 1200.0  # 1.2m PRC 86, 021601 (2012)
    #livetime = 0.3077  # 112.3 days PRC 86, 021601 (2012)
    livetime = 1.0
    for spectrum in signals:
        spectrum.shrink(0.0, 10.0, 0.0, fv_radius, 0.0, livetime)
    for spectrum in backgrounds:
        spectrum.shrink(0.0, 10.0, 0.0, fv_radius, 0.0, livetime)

    # Signal configuration
    signal_configs = []
    prior = 0.0
    Xe136_0n2b_n1_counts = numpy.linspace(signals[0]._num_decays, 0.0, 100,
                                          False)
    # endpoint=False in linspace arrays
    Xe136_0n2b_n1_config = limit_config.LimitConfig(prior,
                                                    Xe136_0n2b_n1_counts)
    signal_configs.append(Xe136_0n2b_n1_config)
    Xe136_0n2b_n2_counts = numpy.linspace(signals[1]._num_decays, 0.0, 100,
                                          False)
    Xe136_0n2b_n2_config = limit_config.LimitConfig(prior,
                                                    Xe136_0n2b_n2_counts)
    signal_configs.append(Xe136_0n2b_n2_config)
    Xe136_0n2b_n3_counts = numpy.linspace(signals[2]._num_decays, 0.0, 100,
                                          False)
    Xe136_0n2b_n3_config = limit_config.LimitConfig(prior,
                                                    Xe136_0n2b_n3_counts)
    signal_configs.append(Xe136_0n2b_n3_config)
    Xe136_0n2b_n7_counts = numpy.linspace(signals[3]._num_decays, 0.0, 100,
                                          False)
    Xe136_0n2b_n7_config = limit_config.LimitConfig(prior,
                                                    Xe136_0n2b_n7_counts)
    signal_configs.append(Xe136_0n2b_n7_config)

    # Background configuration
    # Xe136_2n2b
    Xe136_2n2b_prior = 11.32e6  # Based on KLZ T_1/2, for 10 years
    Xe136_2n2b_counts = numpy.linspace(0.947*Xe136_2n2b_prior,
                                       1.053*Xe136_2n2b_prior, 50)
    # to use in penalty term (5.3% PRC 86, 021601 (2012))
    sigma = 0.053 * Xe136_2n2b_prior
    Xe136_2n2b_config = limit_config.LimitConfig(Xe136_2n2b_prior,
                                                 Xe136_2n2b_counts, sigma)

    # B8_Solar
    # from integrating whole spectrum scaled to Valentina's number
    B8_Solar_prior = 12529.9691
    B8_Solar_counts = numpy.linspace(0.96*B8_Solar_prior,
                                     1.04*B8_Solar_prior, 10)
    sigma = 0.04 * B8_Solar_prior  # 4% To use in penalty term
    B8_Solar_config = limit_config.LimitConfig(B8_Solar_prior,
                                               B8_Solar_counts, sigma)

    loading = 0.0244  # PRC 86, 021601 (2012)
    Xe_136_atm_weight = 135.907219  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 05/07/2015
    Xe_nat_atm_weight = 131.293  # Molar Mass Calculator, http://www.webqc.org/mmcalc.php, 05/07/2015
    Xe_136_abundance = 0.089  # Xenon @ Periodic Table of Chemical Elements, http://www/webqc.org/periodictable-Xenon-Xe.html, 05/07/2015
    phase_space = 1433.0e-27  # PRC 85, 034316 (2012)
    matrix_element = 3.33  # IBM-2 PRC 87, 014315 (2013)
    fv_radius = 1.2  # PRC 86, 021601 (2012)
    scint_density = 13.0e3 / ((4.0/3.0) * numpy.pi * 1.54)  # 13 tons in 3.08m-diameter IB - PRC 86, 021601 (2012)

    te130_converter = decay.DBIsotope("Xe136", loading, Xe_136_atm_weight,
                                      Xe_nat_atm_weight, Xe_136_abundance,
                                      phase_space, matrix_element, fv_radius,
                                      scint_density)
    print te130_converter.get_n_atoms()
    # Phase space and matrix element won't have any effect here

    # chi squared calculator
    calculator = chi_squared.ChiSquared()

    signal_num = 0
    for signal, signal_config in zip(signals, signal_configs):
        print signal._name
        # Create limit setter
        set_limit = limit_setting.LimitSetting(signal, backgrounds,
                                               verbose=True)
        # Configure signal
        set_limit.configure_signal(signal_config)
        # Configure 2n2b
        set_limit.configure_background(Xe136_2n2b._name, Xe136_2n2b_config,
                                       plot_systematic=True)
        # Configure B8
        set_limit.configure_background(B8_Solar._name, B8_Solar_config,
                                       plot_systematic=True)
        # Set chi squared calculator
        set_limit.set_calculator(calculator)

        # Get limit
        try:
            limit = set_limit.get_limit()
            print "-----------------------------------"
            print "90% CL at " + str(limit) + " counts"
            activity = te130_converter.counts_to_activty(limit, livetime)
            half_life = te130_converter.activity_to_half_life(
                activity, te130_converter.get_n_atoms())
            print "90% CL at " + str(half_life) + " yr"
            print "-----------------------------------"
        except IndexError as detail:
            print "-----------------------------------"
            print detail
            print "-----------------------------------"

        # Dump SystAnalysers to hdf5
        for syst_analyser in set_limit._syst_analysers.values():
            store.dump_ndarray(syst_analyser._name+str(signal_num)+".hdf5",
                               syst_analyser)
        signal_num += 1

    # Dump configs to hdf5
    for i, signal_config in enumerate(signal_configs):
        store.dump_ndarray(signals[i]._name+"_config.hdf5", signal_config)
    store.dump_ndarray("Xe136_2n2b_config.hdf5", Xe136_2n2b_config)
    store.dump_ndarray("B8_Solar_config.hdf5", B8_Solar_config)

if __name__ == "__main__":
    import argparse
    from echidna.scripts.zero_nu_limit import ReadableDir

    parser = argparse.ArgumentParser(description="Example limit setting "
                                     "script for SNO+ Majoron limits")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress and timing information")
    parser.add_argument("-s", "--signals", action=ReadableDir, nargs="+",
                        help="Supply path for signal hdf5 file")
    parser.add_argument("-t", "--two_nu", action=ReadableDir,
                        help="Supply paths for Te130_2n2b hdf5 files")
    parser.add_argument("-b", "--b8_solar", action=ReadableDir,
                        help="Supply paths for B8_Solar hdf5 files")
    args = parser.parse_args()
    main()
