""" Example limit setting script

This script provides an example of how to use the limit setting tools,
built into echidna, to set a 90% confidence limit on neutrinoless double
beta decay.

The numbers used in scaling the signal/backgrounds should set a
reasonable limit, but are not necessariy the optimum choice of
parameters.

Examples:
  To use simply run the script::

    $ python zero_nu_limit.py -s /path/to/signal.hdf5 -t /path/to/2n2b.hdf5
      -b /path/to/B8_Solar.hdf5

.. note:: Use the -v option to print out progress and timing information
"""
import numpy

import echidna
import echidna.output.store as store
import echidna.limit.limit_config as limit_config
import echidna.limit.limit_setting as limit_setting
import echidna.limit.chi_squared as chi_squared
import echidna.output.plot_chi_squared as plot_chi_squared
from echidna.calc import decay

import argparse
import os


class ReadableDir(argparse.Action):
    """ Custom argparse action

    Adapted from http://stackoverflow.com/a/11415816

    Checks that hdf5 files supplied via command line exist and can be read
    """
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dirs = []
        if type(values) is str:
            prospective_dirs.append(values)
        elif type(values) is list:
            prospective_dirs = values
        else:
            raise TypeError("Invalid type for arg.")
        for prospective_dir in prospective_dirs:
            if not os.path.isfile(prospective_dir):
                raise argparse.ArgumentTypeError(
                    "ReadableDir:{0} not a valid path".format(prospective_dir))
            if not os.access(prospective_dir, os.R_OK):
                raise argparse.ArgumentTypeError(
                    "ReadableDir:{0} is not readable".format(prospective_dir))
        setattr(namespace, self.dest, values)  # keeps original format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example limit setting "
                                     "script.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress and timing information")
    parser.add_argument("-s", "--signal", action=ReadableDir,
                        help="Supply path for signal hdf5 file")
    parser.add_argument("-t", "--two_nu", action=ReadableDir,
                        help="Supply paths for Te130_2n2b hdf5 files")
    parser.add_argument("-b", "--b8_solar", action=ReadableDir,
                        help="Supply paths for B8_Solar hdf5 files")
    args = parser.parse_args()

    # Create signal spectrum
    Te130_0n2b = store.load(args.signal)
    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_0n2b.scale(200.)
    unshrunk = Te130_0n2b.sum()
    Te130_0n2b = store.load(args.signal)
    Te130_0n2b.shrink(2.46, 2.68, 0.0, 3500.0, 0.0, 5.0)
    Te130_0n2b.scale(200.)
    shrunk = Te130_0n2b.sum()
    scaling = shrunk/unshrunk

    Te130_0n2b = store.load(args.signal)

    # Create background spectra
    Te130_2n2b = store.load(args.two_nu)
    B8_Solar = store.load(args.b8_solar)

    Te130_0n2b._num_decays = Te130_0n2b.sum()
    Te130_0n2b._raw_events = 200034
    Te130_2n2b._num_decays = Te130_2n2b.sum()
    print Te130_2n2b._num_decays
    Te130_2n2b._raw_events = 75073953
    B8_Solar._num_decays = B8_Solar.sum()
    print B8_Solar._num_decays
    B8_Solar._raw_events = 106228

    # Shrink spectra to 5 years - livetime used by Andy
    # And make 3.5m fiducial volume cut
    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_2n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    B8_Solar.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)

    # 1/ Set limit with no penalty term
    # Create dictionary of backgrounds and priors
    Te130_2n2b_prior = 37.396e6  # Based on NEMO-3 T_1/2, for 10 years
    # from integrating whole spectrum scaled to Valentina's number
    B8_Solar_prior = 12529.9691
    fixed_backgrounds = {Te130_2n2b: Te130_2n2b_prior,
                         B8_Solar: B8_Solar_prior}
    # Create fixed spectrum. Pre-shrink here if pre-shrinking in LimitSetting
    roi = (2.46, 2.68)  # Define ROI - as used by Andy
    fixed = limit_setting.make_fixed_background(fixed_backgrounds,
                                                pre_shrink=True,
                                                roi=roi)

    # Initialise limit setting class
    set_limit = limit_setting.LimitSetting(Te130_0n2b, fixed_background=fixed,
                                           roi=roi, pre_shrink=True,
                                           verbose=args.verbose)

    # Configure Te130_0n2b
    Te130_0n2b_counts = numpy.arange(5.0, 1000.0, 5.0, dtype=float)
    Te130_0n2b_prior = 0.  # Setting a 90% CL so no signal in observed
    Te130_0n2b_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                 Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_config)

    # Set chi squared calculator
    calculator = chi_squared.ChiSquared()
    set_limit.set_calculator(calculator)

    # Calculate confidence limit
    sig_num_decays = set_limit.get_limit_no_float()

    # Set decay converter
    atm_weight_iso = 129.9062244
    atm_weight_nat = 127.603
    abundance = 0.3408
    phase_space = 3.69e-14
    matrix_element = 4.03

    converter = decay.DBIsotope("Te130", atm_weight_iso, atm_weight_nat,
                                abundance, phase_space, matrix_element,
                                Te130_0n2b.get_roi(0).get("efficiency"))

    half_life = converter.counts_to_half_life(sig_num_decays, roi_cut=True)
    print "90% CL with no peanalty at: " + str(sig_num_decays) + " ROI counts"
    print "90% CL with no peanalty at: " + str(half_life) + " y"


    # 2/ Now try fixing B8_Solar and floating Te130_2n2b
    Te130_0n2b = store.load(args.signal)

    # Reload background spectra
    Te130_2n2b = store.load(args.two_nu)
    B8_Solar = store.load(args.b8_solar)

    # Shrink spectra to 5 years - livetime used by Andy
    # And make 3.5m fiducial volume cut
    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_2n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    B8_Solar.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)

    fixed_backgrounds = {B8_Solar: B8_Solar_prior}
    fixed = limit_setting.make_fixed_background(fixed_backgrounds,
                                                pre_shrink=True,
                                                roi=roi)

    # List of backgrounds to float
    floating = [Te130_2n2b]
    # Reinitialise limit setting
    set_limit = limit_setting.LimitSetting(Te130_0n2b, fixed_background=fixed,
                                           floating_backgrounds=floating,
                                           roi=roi, pre_shrink=True,
                                           verbose=args.verbose)
    # Configure Te130_0n2b
    Te130_0n2b_penalty_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                         Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_penalty_config)

    # Set config for Te130_2n2b
    # Floating range:
    Te130_2n2b_counts = numpy.linspace(0.797*Te130_2n2b_prior,
                                       1.203*Te130_2n2b_prior, 51)
    # Sigma of rate:
    sigma = 0.203 * Te130_2n2b_prior  # Used in penalty term (20.3%, Andy's doc on systematics)
    Te130_2n2b_penalty_config = limit_config.LimitConfig(
        Te130_2n2b_prior, Te130_2n2b_counts, sigma)
    set_limit.configure_background(Te130_2n2b._name,
                                   Te130_2n2b_penalty_config,
                                   plot_systematic=True)

    # Set chi squared calculator
    set_limit.set_calculator(calculator)

    # Calculate confidence limit
    sig_num_decays = set_limit.get_limit()
    half_life = converter.counts_to_half_life(sig_num_decays)
    print ("90% CL with Te130_2n2b floating at: " +
           str(sig_num_decays) + " ROI counts")
    print "90% CL with Te130_2n2b floating at: " + str(half_life) + " y"
    fig1 = plot_chi_squared.chi_squared_vs_signal(
        Te130_0n2b_config, converter, fig_num=1,
        penalty=Te130_0n2b_penalty_config,
        show=True, roi_cut=True)
    for syst_analyser in set_limit._syst_analysers.values():
        store.dump_ndarray(syst_analyser._name+"_2.hdf5", syst_analyser)

    # 3/ Fix no backgrounds and float all#
    Te130_0n2b = store.load(args.signal)
    # Reload background spectra
    Te130_2n2b = store.load(args.two_nu)
    B8_Solar = store.load(args.b8_solar)

    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_2n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    B8_Solar.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)

    # List of backgrounds to float
    floating = [Te130_2n2b, B8_Solar]

    # Reinitialise limit setting
    set_limit = limit_setting.LimitSetting(Te130_0n2b,
                                           floating_backgrounds=floating,
                                           roi=roi, pre_shrink=True,
                                           verbose=args.verbose)
    # Configure Te130_0n2b
    Te130_0n2b_penalty_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                         Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_penalty_config)

    # Set config for Te130_2n2b
    Te130_2n2b_counts = numpy.linspace(0.797*Te130_2n2b_prior,
                                       1.203*Te130_2n2b_prior, 51)
    # Sigma of rate:
    sigma = 0.203 * Te130_2n2b_prior # Used in penalty term (20.3%, Andy's doc on systematics)
    Te130_2n2b_penalty_config = limit_config.LimitConfig(
        Te130_2n2b_prior, Te130_2n2b_counts, sigma)
    set_limit.configure_background(Te130_2n2b._name,
                                   Te130_2n2b_penalty_config,
                                   plot_systematic=True)
    # Set config for B8_Solar
    B8_Solar_counts = numpy.linspace(0.96*B8_Solar_prior,
                                     1.04*B8_Solar_prior, 11)
    # 11 bins to make sure midpoint (no variation from prior) is included
    sigma = 0.04 * B8_Solar_prior  # 4% To use in penalty term
    B8_Solar_penalty_config = limit_config.LimitConfig(B8_Solar_prior,
                                                       B8_Solar_counts, sigma)
    set_limit.configure_background(B8_Solar._name, B8_Solar_penalty_config,
                                   plot_systematic=True)
    # Set chi squared calculator
    set_limit.set_calculator(calculator)

    # Calculate confidence limit
    sig_num_decays = set_limit.get_limit()
    half_life = converter.counts_to_half_life(sig_num_decays)
    print ("90% CL, with all backgrounds floating, at: " +
           str(sig_num_decays) + " ROI counts")
    print "90% CL, with all backgrounds floating, at: " + str(half_life) + " y"
    fig2 = plot_chi_squared.chi_squared_vs_signal(
        Te130_0n2b_config, converter, fig_num=2,
        penalty=Te130_0n2b_penalty_config,
        show=True, roi_cut=True)

    for syst_analyser in set_limit._syst_analysers.values():
        store.dump_ndarray(syst_analyser._name+"_3.hdf5", syst_analyser)
    store.dump_ndarray("Te130_0n2b_config.hdf5", Te130_0n2b_config)
    store.dump_ndarray("Te130_0n2b_penalty_config.hdf5",
                       Te130_0n2b_penalty_config)
