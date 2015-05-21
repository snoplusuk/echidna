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


def main():
    """ Script to set 90% CL on all four Majoron-emitting modes.
    """
    # Load signal spectra
    signal_hdf5s = ["/data/Te130_0n2b_n1_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Te130_0n2b_n2_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Te130_0n2b_n3_r1.ntuple.root_mc_smeared.hdf5",
                    "/data/Te130_0n2b_n7_r1.ntuple.root_mc_smeared.hdf5"]
    signals = []
    for signal_hdf5 in signal_hdf5s:
        spectrum = store.load(echidna.__echidna_base__+signal_hdf5)
        print spectrum._name
        print "Num decays:", spectrum._num_decays
        print "events:", spectrum.sum()
        signals.append(spectrum)

    # Load background spectra
    backgrounds = []
    Te130_2n2b = store.load(echidna.__echidna_base__ +
                            "/data/Te130_2n2b_mc_smeared.hdf5")
    print Te130_2n2b._name
    Te130_2n2b._num_decays = Te130_2n2b.sum()  # Sum not raw events
    print "Num decays:", Te130_2n2b._num_decays
    print "events:", Te130_2n2b.sum()
    backgrounds.append(Te130_2n2b)
    B8_Solar = store.load(echidna.__echidna_base__ +
                          "/data/B8_Solar_mc_smeared.hdf5")
    print B8_Solar._name
    B8_Solar._num_decays = B8_Solar.sum()  # Sum not raw events
    print "Num decays:", B8_Solar._num_decays
    print "events:", B8_Solar.sum()
    backgrounds.append(B8_Solar)

    # Apply FV and livetime cuts
    fv_radius = 3500.0
    livetime = 5.0
    for spectrum in signals:
        spectrum.shrink(0.0, 10.0, 0.0, fv_radius, 0.0, livetime)
    for spectrum in backgrounds:
        spectrum.shrink(0.0, 10.0, 0.0, fv_radius, 0.0, livetime)

    # Signal configuration
    signal_configs_np = []
    signal_configs = []
    prior = 0.0
    Te130_0n2b_n1_counts = numpy.linspace(signals[0]._num_decays, 0.0, 100,
                                          False)
    # endpoint=False in linspace arrays
    Te130_0n2b_n1_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n1_counts)
    Te130_0n2b_n1_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n1_counts)
    signal_configs_np.append(Te130_0n2b_n1_config_np)
    signal_configs.append(Te130_0n2b_n1_config)

    Te130_0n2b_n2_counts = numpy.linspace(signals[1]._num_decays, 0.0, 100,
                                          False)
    Te130_0n2b_n2_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n2_counts)
    Te130_0n2b_n2_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n2_counts)
    signal_configs_np.append(Te130_0n2b_n2_config_np)
    signal_configs.append(Te130_0n2b_n2_config)

    Te130_0n2b_n3_counts = numpy.linspace(signals[2]._num_decays, 0.0, 100,
                                          False)
    Te130_0n2b_n3_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n3_counts)
    Te130_0n2b_n3_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n3_counts)
    signal_configs_np.append(Te130_0n2b_n3_config_np)
    signal_configs.append(Te130_0n2b_n3_config)

    Te130_0n2b_n7_counts = numpy.linspace(signals[3]._num_decays, 0.0, 100,
                                          False)
    Te130_0n2b_n7_config_np = limit_config.LimitConfig(prior,
                                                       Te130_0n2b_n7_counts)
    Te130_0n2b_n7_config = limit_config.LimitConfig(prior,
                                                    Te130_0n2b_n7_counts)
    signal_configs_np.append(Te130_0n2b_n7_config_np)
    signal_configs.append(Te130_0n2b_n7_config)

    # Background configuration
    # Te130_2n2b
    Te130_2n2b_prior = 37.396e6  # Based on NEMO-3 T_1/2, for 10 years
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
    B8_Solar_prior = 12529.9691
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

    te130_converter = decay.DBIsotope("Te130", 0.003, 129.906229, 127.6,
                                      0.3408, 3.69e-14, 4.03)
    # Phase space and matrix element won't have any effect here

    # chi squared calculator
    calculator = chi_squared.ChiSquared()

    # Set output location
    output_dir = echidna.__echidna_base__ + "/results/snoplus/"

    for signal, signal_config_np in zip(signals, signal_configs_np):
        print signal._name
        # Create no penalty limit setter
        set_limit_np = limit_setting.LimitSetting(signal, backgrounds,
                                                  verbose=True)
        # Configure signal
        set_limit_np.configure_signal(signal_config_np)
        # Configure 2n2b
        set_limit_np.configure_background(Te130_2n2b._name,
                                          Te130_2n2b_config_np)
        # Configure B8
        set_limit_np.configure_background(B8_Solar._name, B8_Solar_config_np)
        # Set chi squared calculator
        set_limit_np.set_calculator(calculator)

        # Get limit
        try:
            limit = set_limit_np.get_limit()
            print "-----------------------------------"
            print "90% CL at " + str(limit) + " counts"
            activity = te130_converter.counts_to_activty(limit)
            half_life = te130_converter.activity_to_half_life(
                activity, te130_converter.get_n_atoms())
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
    for signal, signal_config in zip(signals, signal_configs):
        print signal._name
        # Create limit setter
        set_limit = limit_setting.LimitSetting(signal, backgrounds,
                                               verbose=True)
        # Configure signal
        set_limit.configure_signal(signal_config)
        # Configure 2n2b
        set_limit.configure_background(Te130_2n2b._name, Te130_2n2b_config,
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
            activity = te130_converter.counts_to_activty(limit)
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
            store.dump_ndarray(output_dir+syst_analyser._name+str(signal_num)+".hdf5",
                               syst_analyser)
        signal_num += 1

    # Dump configs to hdf5
    for i, signal_config in enumerate(signal_configs):
        store.dump_ndarray(output_dir+signals[i]._name+".hdf5", signal_config)
    store.dump_ndarray(output_dir+"Te130_2n2b_config.hdf5", Te130_2n2b_config)
    store.dump_ndarray(output_dir+"B8_Solar_config.hdf5", B8_Solar_config)


if __name__ == "__main__":
    main()
