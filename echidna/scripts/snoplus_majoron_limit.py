""" SNO+ Majoron limit setting script

This script sets 90% confidence limit on the Majoron-emitting
neutrinoless double beta decay modes (with spectral indices n = 1, 2, 3
and 7), with SNO+.

Examples:
  To use simply run the script::

    $ python snoplus_majoron_limit.py -s /path/to/majoron_mode.hdf5
      -t /path/to/2n2b.hdf5 -b /path/to/B8_Solar.hdf5

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


def main(args):
    """ Script to set 90% CL on all four Majoron-emitting modes.
    """
    # Load signal spectra
    signals = []
    for signal_hdf5 in args.signals:
        spectrum = store.load(signal_hdf5)
        print spectrum._name
        print "Num decays:", spectrum._num_decays
        print "events:", spectrum.sum()
        signals.append(spectrum)

    # Load background spectra
    floating_backgrounds = []
    Te130_2n2b = store.load(args.two_nu)
    print Te130_2n2b._name
    Te130_2n2b._num_decays = Te130_2n2b.sum()  # Sum not raw events
    print "Num decays:", Te130_2n2b._num_decays
    print "events:", Te130_2n2b.sum()
    floating_backgrounds.append(Te130_2n2b)
    B8_Solar = store.load(args.b8_solar)
    print B8_Solar._name
    B8_Solar._num_decays = B8_Solar.sum()  # Sum not raw events
    print "Num decays:", B8_Solar._num_decays
    print "events:", B8_Solar.sum()
    floating_backgrounds.append(B8_Solar)

    # Apply FV and livetime cuts
    fv_radius = constants._fv_radius
    livetime = 1.0
    for spectrum in signals:
        spectrum.cut(time_low=0.0, time_high=livetime)  # cut to livetime
        spectrum.shrink(radial_low=0.0, radial_high=fv_radius)  # shrink to FV
        spectrum.shrink_to_roi(0.5, 3.0, "energy_mc")  # shrink to ROI
    for spectrum in floating_backgrounds:
        spectrum.cut(time_low=0.0, time_high=livetime)  # cut to livelime
        spectrum.shrink(radial_low=0.0, radial_high=fv_radius)  # shrink to FV
        spectrum.shrink_to_roi(0.5, 3.0, "energy_mc")  # shrink to ROI

    # Signal configuration
    signal_configs_np = []  # no penalty term
    signal_configs = []
    prior = 0.0

    Te130_0n2b_n1_counts = numpy.linspace(signals[0]._num_decays,
                                          0.0, 100, False)
    # endpoint=False in linspace arrays
    Te130_0n2b_n1_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n1_counts)
    Te130_0n2b_n1_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n1_counts)
    signal_configs_np.append(Te130_0n2b_n1_config_np)
    signal_configs.append(Te130_0n2b_n1_config)

    Te130_0n2b_n2_counts = numpy.linspace(signals[1]._num_decays,
                                          0.0, 100, False)
    Te130_0n2b_n2_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n2_counts)
    Te130_0n2b_n2_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n2_counts)
    signal_configs_np.append(Te130_0n2b_n2_config_np)
    signal_configs.append(Te130_0n2b_n2_config)

    Te130_0n2b_n3_counts = numpy.linspace(signals[2]._num_decays,
                                          0.0, 100, False)
    Te130_0n2b_n3_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n3_counts)
    Te130_0n2b_n3_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n3_counts)
    signal_configs_np.append(Te130_0n2b_n3_config_np)
    signal_configs.append(Te130_0n2b_n3_config)

    Te130_0n2b_n7_counts = numpy.linspace(signals[3]._num_decays,
                                          0.0, 100, False)
    Te130_0n2b_n7_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n7_counts)
    Te130_0n2b_n7_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n7_counts)
    signal_configs_np.append(Te130_0n2b_n7_config_np)
    signal_configs.append(Te130_0n2b_n7_config)

    # Background configuration
    # Te130_2n2b
    # Based on NEMO-3 T_1/2, for 1 year livetime.
    # Since we used cut method to cut to livetime
    Te130_2n2b_prior = 3.7396e6

    # No penalty term
    Te130_2n2b_counts_np = numpy.array([Te130_2n2b_prior])
    Te130_2n2b_config_np = limit_config.LimitConfig(Te130_2n2b_prior,
                                                    Te130_2n2b_counts_np)
    # With penalty term
    Te130_2n2b_counts = numpy.linspace(0.8*Te130_2n2b_prior,
                                       1.2*Te130_2n2b_prior, 51)
    # 51 bins to make sure midpoint (no variation from prior) is included
    # to use in penalty term (20%, Andy's document on systematics)
    sigma = 0.2 * Te130_2n2b_prior
    Te130_2n2b_config = limit_config.LimitConfig(Te130_2n2b_prior,
                                                 Te130_2n2b_counts, sigma)

    # B8_Solar
    # from integrating whole spectrum scaled to Valentina's number
    B8_Solar_prior = 1252.99691
    # No penalty term
    B8_Solar_counts_np = numpy.array([B8_Solar_prior])
    B8_Solar_config_np = limit_config.LimitConfig(B8_Solar_prior,
                                                  B8_Solar_counts_np)
    # With penalty term
    B8_Solar_counts = numpy.linspace(0.96*B8_Solar_prior,
                                     1.04*B8_Solar_prior, 11)
    # 11 bins to make sure midpoint (no variation from prior) is included
    sigma = 0.04 * B8_Solar_prior  # 4% To use in penalty term
    B8_Solar_config = limit_config.LimitConfig(B8_Solar_prior,
                                               B8_Solar_counts, sigma)

    # DBIsotope converter information - constant across modes
    isotope_name = "Te130"
    atm_weight_iso = 129.906229
    atm_weight_nat = 127.6
    abundance = 0.3408

    # Make a list of associated nuclear physics info
    nuclear_params = []
    # n=1:
    phase_space = 5.94e-16
    matrix_element = 3.97  # Averaged
    nuclear_params.append((phase_space, matrix_element))
    # n=2:
    phase_space = None
    matrix_element = None
    nuclear_params.append((phase_space, matrix_element))
    # n=1:
    phase_space = 1.06e-17  # Assuming two Majorons emitted
    matrix_element = 1.e-3
    nuclear_params.append((phase_space, matrix_element))
    # n=1:
    phase_space = 4.83e-17
    matrix_element = 1.e-3
    nuclear_params.append((phase_space, matrix_element))

    # chi squared calculator
    calculator = chi_squared.ChiSquared()

    # Set output location
    output_dir = echidna.__echidna_base__ + "/results/snoplus/"

    for signal, signal_config_np, nuclear_param in zip(signals,
                                                       signal_configs_np,
                                                       nuclear_params):
        print signal._name
        # Create no penalty limit setter
        set_limit_np = limit_setting.LimitSetting(
            signal, floating_backgrounds=floating_backgrounds)
        # Configure signal
        set_limit_np.configure_signal(signal_config_np)
        # Configure 2n2b
        set_limit_np.configure_background(Te130_2n2b._name,
                                          Te130_2n2b_config_np)
        # Configure B8
        set_limit_np.configure_background(B8_Solar._name, B8_Solar_config_np)

        # Set converter
        phase_space, matrix_element = nuclear_param
        roi_efficiency = signal.get_roi(0).get("efficiency")
        converter = decay.DBIsotope(
            isotope_name, atm_weight_iso, atm_weight_nat, abundance,
            phase_space, matrix_element, roi_efficiency=roi_efficiency)

        # Set chi squared calculator
        set_limit_np.set_calculator(calculator)

        # Get limit
        try:
            limit = set_limit_np.get_limit()
            print "-----------------------------------"
            print "90% CL at " + str(limit) + " counts"
            half_life = converter.counts_to_half_life(limit, livetime=livetime)
            print "90% CL at " + str(half_life) + " yr"
            print "-----------------------------------"
        except IndexError as detail:
            print "-----------------------------------"
            print detail
            print "-----------------------------------"

    for i, signal_config_np in enumerate(signal_configs_np):
        store.dump_ndarray(output_dir+signals[i]._name+"_np.hdf5",
                           signal_config_np)
    raw_input("RETURN to continue")

    signal_num = 0
    for signal, signal_config, nuclear_param in zip(signals, signal_configs,
                                                    nuclear_params):
        print signal._name
        # Create limit setter
        set_limit = limit_setting.LimitSetting(
            signal, floating_backgrounds=floating_backgrounds)
        # Configure signal
        set_limit.configure_signal(signal_config)
        # Configure 2n2b
        set_limit.configure_background(Te130_2n2b._name, Te130_2n2b_config,
                                       plot_systematic=True)
        # Configure B8
        set_limit.configure_background(B8_Solar._name, B8_Solar_config,
                                       plot_systematic=True)

        # Set converter
        phase_space, matrix_element = nuclear_param
        roi_efficiency = signal.get_roi(0).get("efficiency")
        converter = decay.DBIsotope(
            isotope_name, atm_weight_iso, atm_weight_nat, abundance,
            phase_space, matrix_element, roi_efficiency=roi_efficiency)

        # Set chi squared calculator
        set_limit.set_calculator(calculator)

        # Get limit
        try:
            limit = set_limit.get_limit()
            print "-----------------------------------"
            print "90% CL at " + str(limit) + " counts"
            half_life = converter.counts_to_half_life(limit, livetime=livetime)
            print "90% CL at " + str(half_life) + " yr"
            print "-----------------------------------"
        except IndexError as detail:
            print "-----------------------------------"
            print detail
            print "-----------------------------------"

        # Dump SystAnalysers to hdf5
        for syst_analyser in set_limit._syst_analysers.values():
            store.dump_ndarray(output_dir + syst_analyser._name +
                               str(signal_num)+".hdf5",
                               syst_analyser)
        signal_num += 1

    # Dump configs to hdf5
    for i, signal_config in enumerate(signal_configs):
        store.dump_ndarray(output_dir+signals[i]._name+".hdf5", signal_config)
    store.dump_ndarray(output_dir+"Te130_2n2b_config.hdf5", Te130_2n2b_config)
    store.dump_ndarray(output_dir+"B8_Solar_config.hdf5", B8_Solar_config)


if __name__ == "__main__":
    import argparse
    from echidna.scripts.zero_nu_limit import ReadableDir

    parser = argparse.ArgumentParser(description="Example limit setting "
                                     "script for SNO+ Majoron limits")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress and timing information")
    parser.add_argument("-s", "--signals", action=ReadableDir, nargs=4,
                        help="Supply path for signal hdf5 file")
    parser.add_argument("-t", "--two_nu", action=ReadableDir,
                        help="Supply paths for Te130_2n2b hdf5 files")
    parser.add_argument("-b", "--b8_solar", action=ReadableDir,
                        help="Supply paths for B8_Solar hdf5 files")
    args = parser.parse_args()
    main(args)
