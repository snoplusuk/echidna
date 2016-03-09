""" KamLAND-Zen (plot-grab) Majoron limits script

This script:

  * Sets 90% confidence limit on the Majoron-emitting neutrinoless
    double beta decay modes (with spectral indices n = 1, 2, 3 and 7),
    using plot-grabbed data from KamLAND-Zen.

Examples:
  To use simply run the script and supply a YAML file detailing the
  spectra (data, fixed, floating) to load::

        $ python zero_nu_limit.py --from_file klz_majoron_limits_config.yaml

The ``--upper_bound`` and ``--lower_bound`` flags from the command, can
be used to return an estimate on the error introduced through the
plot-grabbing process.

.. note:: An example config would be::

        data:
            data/klz/v1.0.0/klz_data.hdf5

    ::

        fixed:
            {data/klz/v1.0.0/total_b_g_klz.hdf5: 26647.1077395}

    ::

        floating:
            [data/klz/v1.0.0/Xe136_2n2b_fig2.hdf5]

    ::

        signals:
            {
                klz_n1: data/klz/v1.0.0/Xe136_0n2b_n1_fig2.hdf5,
                klz_n2: data/klz/v1.0.0/Xe136_0n2b_n2_fig2.hdf5,
                klz_n3: data/klz/v1.0.0/Xe136_0n2b_n3_fig2.hdf5,
                klz_n7: data/klz/v1.0.0/Xe136_0n2b_n7_fig2.hdf5}

    ::

        roi:
            energy:
                !!python/tuple [1.0, 3.0]

    ::

        per_bin:
            true

    ::

        store_summary:
            true

"""
import numpy

import echidna.output as output
import echidna.utilities as utilities
import echidna.fit.test_statistic as test_statistic
from echidna.core.config import GlobalFitConfig
import echidna.output.store as store
import echidna.fit.fit as fit
from echidna.errors.custom_errors import CompatibilityError
import echidna.calc.decay as decay
import echidna.calc.constants as constants
import echidna.limit.limit as limit

import yaml
import json
import logging
from collections import OrderedDict
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
            if not os.access(prospective_dir, os.R_OK):
                raise argparse.ArgumentTypeError(
                    "ReadableDir:{0} is not readable".format(prospective_dir))
        setattr(namespace, self.dest, values)  # keeps original format


def main(args):
    """ The limit setting script.

    Args:
      args (:class:`argparse.Namespace`): Arguments passed via command-
        line
    """
    logger = utilities.start_logging()

    if args.save_path is not None:
        logger.warning("Overriding default save path!")
        logging.getLogger("extra").warning(
            " --> all output will be saved to %s" %
            output.__default_save_path__)

    args_config = yaml.load(open(args.from_file, "r"))
    name = args_config.get("name")
    logger.info("Configuration name: %s" % name)
    logging.getLogger("extra").debug("\n\n%s\n" % yaml.dump(args_config))

    # Set plot-grab error if required
    if args.upper_bound or args.lower_bound:
        pixel_err = 0.005
        n_pixel = 4
        plot_grab_err = numpy.sqrt(3 * (n_pixel*pixel_err)**2)

    # Set ROI from config
    if args_config.get("roi") is not None:
        roi = args_config.get("roi")
        logger.info("Set ROI")
        logging.getLogger("extra").info("\n%s\n" % json.dumps(roi))
        if not isinstance(roi, dict):
            raise TypeError("roi should be a dictionary, "
                            "not type %s" % type(roi))
    else:
        logger.warning("No ROI found, spectra will not be shrunk")

    # Set per_bin, as required
    if args_config.get("per_bin") is not None:
        per_bin = args_config.get("per_bin")
        logger.info("Storing per-bin information: %s" % per_bin)
    else:
        logger.warning("No per-bin flag found - setting per_bin to False")
        per_bin = False

    # Set store_summary, as required
    if args_config.get("store_summary") is not None:
        store_summary = args_config.get("store_summary")
        logger.info("Storing Summary information: %s" % store_summary)
    else:
        logger.warning("No store_summary flag found - "
                       "setting store_summary to False")
        store_summary = False

    # Set test_statistic
    # This is fixed
    chi_squared = test_statistic.BakerCousinsChi(per_bin=True)
    logger.info("Using test statisitc: BakerCousinsChi")

    # Set fit_config
    if args_config.get("fit_config") is not None:
        fit_config = GlobalFitConfig.load_from_file(
            args_config.get("fit_config"))
    else:  # Don't have any global fit parameters here - make blank config
        logger.warning("No fit_config path found - creating blank config")
        parameters = OrderedDict({})
        # The name set here will be the same name given to the GridSearch
        # created by the fitter, and the Summary class saved to hdf5.
        fit_config = GlobalFitConfig(name, parameters)
    logger.info("Using GlobalFitConfig with the following parameters:")
    logging.getLogger("extra").info(fit_config.get_pars())

    # Set data
    if args_config.get("data") is not None:
        logger.info("Using data spectrum %s" % args_config.get("data"))
        data = store.load(args_config.get("data"))
    else:
        logger.error("No data path found")
        logging.getLogger("extra").warning(
            " --> echidna can use total background as data, "
            "but a blank data spectrum should still be supplied.")
        raise ValueError("No data path found")

    # Apply plot-grab errors as appropriate
    if args.upper_bound:
        data_neg_errors = utilities.get_array_errors(
            data._data, lin_err=-plot_grab_err, log10=True)
        data._data = data._data + data_neg_errors
    if args.lower_bound:
        data_pos_errors = utilities.get_array_errors(
            data._data, lin_err=plot_grab_err, log10=True)
        data._data = data._data + data_pos_errors

    # Set fixed backgrounds
    # Create fixed_backgrounds dict with Spectra as keys and priors as values
    fixed_backgrounds = {}
    if args_config.get("fixed") is not None:
        if not isinstance(args_config.get("fixed"), dict):
            raise TypeError(
                "Expecting dictionary with paths to fixed backgrounds as keys "
                "and num_decays for each background as values")
        for filename, num_decays in args_config.get("fixed").iteritems():
            logger.info("Using fixed spectrum: %s (%.4f decays)" %
                        (filename, num_decays))
            spectrum = store.load(filename)

            # Add plot-grab errors as appropriate
            if args.upper_bound:
                spectrum_neg_errors = utilities.get_array_errors(
                    spectrum._data, lin_err=-plot_grab_err, log10=True)
                spectrum._data = spectrum._data + spectrum_neg_errors
            if args.lower_bound:
                spectrum_pos_errors = utilities.get_array_errors(
                    spectrum._data, lin_err=plot_grab_err, log10=True)
                spectrum._data = spectrum._data + spectrum_pos_errors

            fixed_backgrounds[spectrum] = num_decays
    else:
        logger.warning("No fixed spectra found")

    # Set floating backgrounds
    floating_backgrounds = []
    # Add any floating backgrounds passed directly
    for background in args.floating:
        floating_backgrounds.append(background)
    if args_config.get("floating") is not None:
        if not isinstance(args_config.get("floating"), list):
            raise TypeError("Expecting list of paths to floating backgrounds")
        for filename in args_config.get("floating"):
            logger.info("Using floating background: %s" % filename)
            spectrum = store.load(filename)

            floating_backgrounds.append(spectrum)
    else:
        logger.warning("No floating backgrounds found")

    # Add plot-grab errors as appropriate
    for background in floating_backgrounds:
        if args.upper_bound:
            spectrum_neg_errors = utilities.get_array_errors(
                background._data, lin_err=-plot_grab_err, log10=True)
            background._data = background._data + spectrum_neg_errors
        if args.lower_bound:
            spectrum_pos_errors = utilities.get_array_errors(
                background._data, lin_err=plot_grab_err, log10=True)
            background._data = background._data + spectrum_pos_errors

    # Using default minimiser (GridSearch) so let Fit class handle this

    # Create fitter
    # No convolutions here --> use_pre_made = False
    fitter = fit.Fit(roi, chi_squared, fit_config, data=data,
                     fixed_backgrounds=fixed_backgrounds,
                     floating_backgrounds=floating_backgrounds,
                     per_bin=per_bin, use_pre_made=False)
    logger.info("Created fitter")

    # Make data if running sensitivity study
    if args.sensitivity:
        data = fitter.get_data()  # Already added blank spectrum
        # Add fixed background
        data.add(fitter.get_fixed_background())
        # Add floating backgrounds - scaled to prior
        for background in fitter.get_floating_backgrounds():
            prior = background.get_fit_config().get_par("rate").get_prior()
            background.scale(prior)
            data.add(background)
        # Re-set data
        fitter.set_data(data)

    # Fit with no signal
    stat_zero = fitter.fit()
    fit_results = fitter.get_fit_results()
    logger.info("Calculated stat_zero: %.4f" % stat_zero)
    logger.info("Fit summary:")
    logging.getLogger("extra").info("\n%s\n" %
                                    json.dumps(fit_results.get_summary()))

    # Load signals
    signals = []
    # Add any signals parsed directly
    for signal in args.signals:
        signals.append(signal)
    if args_config.get("signals") is not None:
        for name, filename in args_config.get("signals").iteritems():
            logger.info("Using signal spectrum: %s" % filename)
            signal = store.load(filename)
            signals.append(signal)
    else:
        logger.error("No signal spectra found")
        raise CompatibilityError("Must have at least one signal to set limit")

    # Add plot-grab errors as appropriate
    # For signal we want to swap negative and positive fluctuations
    # The lower bound on the limit, is when all our backgrounds have
    # fluctuated down (through plot-grabbing) but the signal has
    # fluctuated up. Then the reverse is true for the upper bound,
    # backgrounds are fluctuated up and signal is fluctuated down
    for signal in signals:
        if args.upper_bound:
            signal_pos_errors = utilities.get_array_errors(
                signal._data, lin_err=plot_grab_err, log10=True)
            signal._data = signal._data + signal_pos_errors
        if args.lower_bound:
            signal_neg_errors = utilities.get_array_errors(
                signal._data, lin_err=-plot_grab_err, log10=True)
            signal._data = signal._data + signal_neg_errors

    # KamLAND-Zen limits
    klz_limits = {"Xe136_0n2b_n1": 2.6e24,
                  "Xe136_0n2b_n2": 1.0e24,
                  "Xe136_0n2b_n3": 4.5e23,
                  "Xe136_0n2b_n7": 1.1e22}

    # KamLAND-Zen detector info
    klz_detector = constants.klz_detector

    # Loop through signals and set limit for each
    for signal in signals:
        # Reset GridSearch - with added signal rate parameter
        fitter.get_fit_results().reset_grids()

        # Create converter
        converter = decay.DBIsotope(
            signal._name, klz_detector.get("Xe136_atm_weight"),
            klz_detector.get("XeEn_atm_weight"),
            klz_detector.get("Xe136_abundance"),
            decay.phase_spaces.get(signal._name),
            decay.matrix_elements.get(signal._name),
            loading=klz_detector.get("loading"),
            outer_radius=klz_detector.get("fv_radius"),
            scint_density=klz_detector.get("scint_density"))
        klz_limit = klz_limits.get(signal._name)

        # Create limit setter
        limit_setter = limit.Limit(signal, fitter, per_bin=per_bin)

        limit_scaling = limit_setter.get_limit(store_summary=store_summary)
        signal.scale(limit_scaling)
        half_life = converter.counts_to_half_life(
            limit_scaling,
            n_atoms=converter.get_n_atoms(
                target_mass=klz_detector.get("target_mass")),
            livetime=klz_detector.get("livetime"))

        logging.getLogger("extra").info(
            "\n########################################\n"
            "Signal: %s\n"
            "Calculated limit scaling of %.4g\n"
            " --> equivalent to %.4f events\n" %
            (signal._name, limit_scaling, signal.sum()))
        logging.getLogger("extra").info(
            "Calculated limit half life of %.4g y\n"
            " --> KamLAND-Zen equivalent limit: %.4g y\n"
            "########################################\n" %
            (half_life, klz_limit))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KamLAND-Zen (plot-grab) Majoron limits script")
    parser.add_argument("--from_file", action=ReadableDir,
                        help="Path to config file containing arg values")
    parser.add_argument("-s", "--save_path", action=ReadableDir,
                        help="Path to save all ouput files to. "
                        "Overrides default from output module.")
    parser.add_argument("--floating", nargs="*", help="Parse floating "
                        "background spectra directly")
    parser.add_argument("--signals", nargs="*", help="Parse signal spectra "
                        "directly")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Use expected background as data. Note a blank "
                        "'data' spectrum must still be supplied, which will "
                        "then be filled with the appropriate expected "
                        "background spectrum.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lower_bound", action="store_true",
                       help="Estimate lower bound on limit "
                       "due to plot-grab errors")
    group.add_argument("--upper_bound", action="store_true",
                       help="Estimate upper bound on limit "
                       "due plot-grab errors")
    args = parser.parse_args()

    if args.save_path is not None:
        output.__default_save_path__ = args.save_path

    main(args)
