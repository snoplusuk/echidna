""" Example scaling script

This script:
  * Reads in mc spectra from hdf5
  * Scales spectra
  * Scaled spectrum is saved to the same directory with ``_scaled``
    added to the file name

Examples:
  To scale energy_mc of the hdf5 file ``example.hdf5`` by a factor 1.1::

    $ python dump_scaled.py -p energy_mc -f /path/to/example.hdf5 -s 1.1
      -d /another/path

  This will create the scaled hdf5 file ``/another/path/example_1.1sc.hdf5``.
"""

import echidna.output.store as store
import echidna.core.scale as scale
import echidna.utilities as utilities

import os

_logger = utilities.start_logging()


def main(args):
    """ Scales and dumps spectra.

    Args:
      args (Namespace): Container for arguments. See::

        $ python dump_scaled.py -h

    Raises:
      IOError: If no file is given to scale
      ValueError: If no scale factor is given
      ValueError: If no parameter is given to scale.
      ValueError: If destination directory does not exits.
    """
    if not args.file:
        parser.print_help()
        raise IOError("No file given in command line to scale")
    if not args.scale:
        parser.print_help()
        raise ValueError("No scale factor given")
    if not args.par:
        parser.print_help()
        raise ValueError("No parameter to scale given")
    if args.dest:
        if os.path.isdir(args.dest):
            directory = args.dest
            if directory[-1] != "/":
                directory += "/"
        else:
            raise ValueError("%s does not exist." % args.dest)
    else:
        directory = os.path.dirname(args.file) + "/"
    filename = os.path.splitext(os.path.basename(args.file))[0]
    scaler = scale.Scale()
    scaler.set_scale_factor(args.scale)
    spectrum = store.load(args.file)
    scaled_spectrum = scaler.scale(spectrum, args.par)
    str_sc = str(args.scale)
    if str_sc[-2:] == '.0':
        str_sc = str_sc[:-2]
    str_sc.rstrip('0')
    filename = directory + filename + "_" + str_sc + "sc.hdf5"
    store.dump(filename, scaled_spectrum)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--par", default=None, type=str,
                        help="specify parameter to scale e.g. energy_mc"
                        "e.g. 0.05 for 5 percent")
    parser.add_argument("-s", "--scale", default=None, type=float,
                        help="specify scale factor"
                        "e.g. 0.05 for 5 percent")
    parser.add_argument("-f", "--file", default=None, type=str,
                        help="specify path to hdf5 file")
    parser.add_argument("-d", "--dest", default=None, type=str,
                        help="specify path to dump hdf5 file")
    args = parser.parse_args()
    try:
        main(args)
    except:
        _logger.exception("echidna terminated because of the following error.")
