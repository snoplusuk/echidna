""" Example limit setting script

This script provides an example of how to use the limit setting tools,
built into echidna, to set a 90% confidence limit on neutrinoless double
beta decay.

The numbers used in scaling the signal/backgrounds should set a
reasonable limit, but are not necessariy the optimum choice of
parameters.

Examples:
  To use simply run the script::

    $ python zero_nu_limit.py

.. note:: The script assumes that hdf5 files have already been generated
  for the signal and both backgrounds and are saved in ``echidna/data``
"""
import numpy

import echidna
import echidna.output.store as store
import echidna.limit.limit_config as limit_config
import echidna.limit.limit_setting as limit_setting
import echidna.limit.chi_squared as chi_squared


if __name__ == "__main__":
    # Create signal spectrum
    Te130_0n2b = store.load(echidna.__echidna_home__ +
                            "data/TeLoadedTe130_0n2b.ntuple_reco.hdf5")

    # Create background spectra
    Te130_2n2b = store.load(echidna.__echidna_home__ +
                            "data/TeLoadedTe130_2n2b.ntuple_reco.hdf5")
    B8 = store.load(echidna.__echidna_home__ +
                    "data/TeLoadedB8.ntuple_reco.hdf5")

    # Shrink spectra to 5 years - livetime used by Andy
    # And make 3.5m fiducial volume cut
    Te130_0n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    Te130_2n2b.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)
    B8.shrink(0.0, 10.0, 0.0, 3500.0, 0.0, 5.0)

    # Create list of backgrounds
    backgrounds = []
    backgrounds.append(Te130_2n2b)
    backgrounds.append(B8)

    # Initialise limit setting class
    roi = (2.46, 2.68)  # Define ROI - as used by Andy
    set_limit = limit_setting.LimitSetting(Te130_0n2b, backgrounds, roi=roi)

    # Configure Te130_0n2b
    Te130_0n2b_counts = numpy.arange(10.0, 6.0e3, 10.0, dtype=float)
    Te130_0n2b_prior = 4.2077507e3  # Based on T_1/2 = 6.2e24 y for 10 years
    Te130_0n2b_config = limit_config.LimitConfig(Te130_0n2b_prior,
                                                 Te130_0n2b_counts)
    set_limit.configure_signal(Te130_0n2b_config)

    # Configure Te130_2n2b
    Te130_2n2b_counts = numpy.arange(11.342632e6, 11.342633e6,
                                     1.0, dtype=float)
    # no penalty term to start with so just an array containing one value
    Te130_2n2b_prior = 11.342632e6  # Based on T_1/2 = 2.3e21 y for 10 years
    Te130_2n2b_config = limit_config.LimitConfig(Te130_2n2b_prior,
                                                 Te130_2n2b_counts)
    set_limit.configure_background(Te130_2n2b._name, background_config)
    # configs should have same name as background

    # Configure B8
    B8_counts = numpy.arange(12529.9691, 12530.9691, 1.0, dtype=float)
    # again, no penalty term for now
    B8_prior = 12529.9691  # from integrating whole spectrum scaled to Valentina's number
    B8_config = limit_config.LimitConfig(B8_prior, B8_counts)

    # Set chi squared calculator
    calculator = chi_squared.ChiSquared()
    set_limit.set_calculator(calculator)

    # Calculate confidence limit
    print "90% CL at: " + str(set_limit.get_limit()) + " counts"

    # Now try with a penalty term
    # Set new configs this time with more counts
    Te130_2n2b_counts = numpy.arange(5.5e6, 16.5e6, 0.01e6, dtype=float)
    sigma = 50.0  # To use in penalty term
    Te130_2n2b_config = limit_config.LimitConfig(Te130_2n2b_prior,
                                                 Te130_2n2b_counts, sigma)
    set_limit.configure_background(Te130_2n2b._name, Te130_2n2b_config)

    B8_counts = numpy.arange(6.0e3, 18.0e3, 0.01e3, dtype=float)
    sigma = 50.0  # To use in penalty term
    B8_config = limit_config.LimitConfig(B8_prior, B8_counts, sigma)
    set_limit.configure_background(B8._name, B8_config)

    # Calculate confidence limit
    print "90% CL at: " + str(set_limit.get_limit()) + " counts"
