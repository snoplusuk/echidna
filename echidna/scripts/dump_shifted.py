""" Example shifting script

This script:
  * Reads in mc spectra from hdf5
  * Shifts spectra
  * Shifted spectrum is saved to the same directory with ``_shifted``
    added to the file name

Examples:
  To shift energy_mc of the hdf5 file ``example.hdf5`` by 0.1 MeV::

    $ python dump_shifted.py -d energy_mc -f /path/to/example.hdf5 -s 0.1

  This will create the shifted hdf5 file ``/path/to/example_shifted.hdf5``.
"""

import echidna.output.store as store
import echidna.core.shift as shift

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", default=None, type=str,
                        help="specify dimension to shift e.g. energy_mc")
    parser.add_argument("-s", "--shift", default=None, type=float,
                        help="specify shift value")
    parser.add_argument("-f", "--file", default=None, type=str,
                        help="specify path to hdf5 file")
    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        raise IOError("No file given in command line to shift")
    if not args.shift:
        parser.print_help()
        raise ValueError("No shift value given")
    if not args.dimension:
        parser.print_help()
        raise ValueError("No dimension to shift given")

    directory = args.file[:args.file.rfind("/")+1]  # strip filename
    # strip directory and extension
    filename = args.file[args.file.rfind("/")+1:args.file.rfind(".")]
    shifter = shift.Shift()
    shifter.set_shift(args.shift)
    spectrum = store.load(args.file)
    shifted_spectrum = shifter.shift(spectrum, args.dimension)
    filename = directory + filename + "_shifted.hdf5"
    store.dump(filename, shifted_spectrum)
