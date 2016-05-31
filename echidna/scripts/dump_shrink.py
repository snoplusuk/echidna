""" Example shrinking script

This script:
  * Reads in spectrum from hdf5.
  * Shrinks spectrum with user input pars and low high values.
  * Saves shrunk spectrum to given output or, by default, to the same
    directory with _shrink added to the filename.

Examples:
  To shrink hdf5 file ``example.hdf5`` that has dimensions ``energy_mc`` and
  ``radial3_mc`` to a range in energy of 1. to 4. MeV and radial3 from
  0. to 0.5 and save the spectrum to ``example2.hdf5`` then the following
  command is required::

    $ python dump_shrink.py -i /path/to/example.hdf5 -o /path/to/example2.hdf5
      -p energy_mc radial3_mc -l 1. 0. -u 4. 0.5

  This will create the shrunk hdf5 file ``/path/to/example2.hdf5``.

"""

from echidna.output import store
import echidna.utilities as utilities
import os

_logger = utilities.start_logging()


def main(args):
    """Smears energy and dumps spectra.

    Args:
      args (Namespace): Container for arguments. See::
        python dump_smeared_energy.py -h
    """
    if not args.input:
        parser.print_help()
        raise ValueError("No input file provided")
    if not args.pars:
        parser.print_help()
        raise ValueError("No parameters provided")
    if not args.low and not args.up:
        parser.print_help()
        raise ValueError("Must provide lower and/or upper bounds to"
                         "shrink to.")
    spectrum = store.load(args.input)
    num_pars = len(args.pars)
    if args.low and args.up:
        if len(args.low) != num_pars or len(args.up) != num_pars:
            raise ValueError("Must have the same number of pars as bounds")
        shrink = {}
        for i, par in enumerate(args.pars):
            par_low = par+"_low"
            par_high = par+"_high"
            shrink[par_low] = args.low[i]
            shrink[par_high] = args.up[i]
        spectrum.shrink(**shrink)
    elif args.low:
        if len(args.low) != num_pars:
            raise ValueError("Must have the same number of pars as bounds")
        shrink = {}
        for i, par in enumerate(args.pars):
            par_low = par+"_low"
            shrink[par_low] = args.low[i]
        spectrum.shrink(**shrink)
    else:
        if len(args.up) != num_pars:
            raise ValueError("Must have the same number of pars as bounds")
        shrink = {}
        for i, par in enumerate(args.pars):
            par_high = par+"_high"
            shrink[par_high] = args.up[i]
        spectrum.shrink(**shrink)
    f_out = args.output
    if not f_out:
        directory = os.path.dirname(args.input)
        filename = os.path.splitext(os.path.basename(args.input))[0]
        f_out = directory + "/" + filename + "_shrunk.hdf5"
    store.dump(f_out, spectrum)
    _logger.info("Shrunk "+str(args.input)+", saved to "+str(f_out))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Input spectra to rebin")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Name of output. Default adds _rebin to name")
    parser.add_argument("-p", "--pars", nargs='+', type=str,
                        help="List of parameters to shrink")
    parser.add_argument("-l", "--low", nargs='+', type=float,
                        help="List of lower bounds to shrink to.")
    parser.add_argument("-u", "--up", nargs='+', type=float,
                        help="List of upper  bounds to shrink to.")
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        _logger.exception("echidna terminated because of the following error.")
