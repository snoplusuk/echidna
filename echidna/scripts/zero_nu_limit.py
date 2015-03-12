""" Example limit setting script

This script provides an example of how to use the limit setting tools,
built into echidna, to set a 90% confidence limit on neutrinoless double
beta decay.

The numbers used in scaling the signal/backgrounds should set a
reasonable limit, but are not necessariy the optimum choice of
parameters.

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
import echidna.output.plot_chi_squared as plot_chi_squared

import argparse
import os


class ReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isfile(prospective_dir):
            raise argparse.ArgumentTypeError("ReadableDir:{0} is not a valid "
                                             "path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("ReadableDir:{0} is not readable"
                                             .format(prospective_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example limit setting script")
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

    # Create background spectra
    Te130_2n2b = store.load(args.two_nu)
    B8_Solar = store.load(args.b8_solar)

    Te130_0n2b._num_decays = Te130_0n2b.sum()
    Te130_0n2b._raw_events = 200034
    Te130_2n2b._num_decays = Te130_2n2b.sum()
    Te130_2n2b._raw_events = 75073953
    B8_Solar._num_decays = B8_Solar.sum()
    B8_Solar._raw_events = 106228

    # Shrink spectra to 5 years - livetime used by Andy
    # And make 3.5m fiducial volume cut
    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_2n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    B8_Solar.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)

    # Create list of backgrounds
    backgrounds = []
    backgrounds.append(Te130_2n2b)
    backgrounds.append(B8_Solar)

    # Initialise limit setting class
    roi = (2.46, 2.68)  # Define ROI - as used by Andy
    set_limit = limit_setting.LimitSetting(Te130_0n2b, backgrounds, roi=roi,
                                           pre_shrink=True,
                                           verbose=args.verbose)

    # Configure Te130_0n2b
    Te130_0n2b_counts = numpy.arange(5.0, 500.0, 5.0, dtype=float)
    Te130_0n2b_prior = 0.0       #262.0143 Based on T_1/2 = 9.94e25 y @ 90% CL
                                 # (SNO+-doc-2593-v8) for 5 year livetime
                                 # Note extrapolating here to 10 years
    Te130_0n2b_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                 Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_config)

    # Configure Te130_2n2b
    Te130_2n2b_counts = numpy.arange(37.5e6, 38.5e6,
                                     1.0e6, dtype=float)
    # no penalty term to start with so just an array containing one value
    Te130_2n2b_prior = 37.5e6  # Based on T_1/2 = 2.3e21 y for 10 years
    Te130_2n2b_config = limit_config.LimitConfig(Te130_2n2b_prior,
                                                 Te130_2n2b_counts)
    set_limit.configure_background(Te130_2n2b._name, Te130_2n2b_config)
    # configs should have same name as background

    # Configure B8_Solar
    B8_Solar_counts = numpy.arange(12529.9691, 12530.9691, 1.0, dtype=float)
    # again, no penalty term for now
    B8_Solar_prior = 12529.9691  # from integrating whole spectrum scaled to
                                 # Valentina's number
    B8_Solar_config = limit_config.LimitConfig(B8_Solar_prior, B8_Solar_counts)
    set_limit.configure_background(B8_Solar._name, B8_Solar_config)

    # Set chi squared calculator
    calculator = chi_squared.ChiSquared()
    set_limit.set_calculator(calculator)

    # Calculate confidence limit
    print "90% CL at: " + str(set_limit.get_limit()) + " counts"

    # Now try with a penalty term
    # Configure Te130_0n2b
    Te130_0n2b_counts = numpy.arange(5.0, 500.0, 5.0, dtype=float)
    Te130_0n2b_prior = 0.0       # 262.0143 Based on T_1/2 = 9.94e25 y @ 90% CL
                                 # (SNO+-doc-2593-v8) for 5 year livetime
                                 # Note extrapolating here to 10 years
    Te130_0n2b_penalty_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                         Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_penalty_config)

    # Set new configs this time with more counts
    Te130_2n2b_counts = numpy.arange(30.0e6, 45.0e6, 0.1e6, dtype=float)
    sigma = 7.5e6  # To use in penalty term (20%, Andy's document on
                      # systematics)
    Te130_2n2b_penalty_config = limit_config.LimitConfig(Te130_2n2b_prior,
                                                         Te130_2n2b_counts,
                                                         sigma)
    set_limit.configure_background(Te130_2n2b._name, Te130_2n2b_penalty_config,
                                   plot_systematic=True)

    B8_Solar_counts = numpy.arange(12.0e3, 13.0e3, 0.1e3, dtype=float)
    sigma = 501.1988  # To use in penalty term
    B8_Solar_penalty_config = limit_config.LimitConfig(B8_Solar_prior,
                                                       B8_Solar_counts, sigma)
    set_limit.configure_background(B8_Solar._name, B8_Solar_penalty_config,
                                   plot_systematic=True)

    # Calculate confidence limit
    print "90% CL at: " + str(set_limit.get_limit()) + " counts"
    plot_chi_squared.chi_squared_vs_signal(Te130_0n2b_config,
                                           penalty=Te130_0n2b_penalty_config)

    for syst_analyser in set_limit._syst_analysers.values():
        store.dump_ndarray(syst_analyser._name+".hdf5", syst_analyser)
