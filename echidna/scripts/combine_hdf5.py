""" Example smearing script

This script:
  * Reads in multiple spectra from hdf5
  * Combines the spectra into one spectrum
  * Dumps spectrum to ``combined.hdf5``

Examples:
  To combine hdf5 files ``example1.hdf5`` and ``example2.hdf5`` ::

    $ python echidna/scripts/combine_hdf5.py -f /path/to/example1.hdf5
      /path/to/example2.hdf5

  This will create the hdf5 file ``combined.hdf5``.
  There is no limit to the number of files you can combine.
"""

from echidna.output import store
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs='+', type=str,
                        help="Space seperated hdf5 files to combine.")
    args = parser.parse_args()
    if not args.files:
        parser.print_help()
        parser.error("Must pass more than 1 file to combine")
    if len(args.files) < 2:
        parser.print_help()
        parser.error("Must pass more than 1 file to combine")
    first = True
    for hdf5 in args.files:
        if first:
            spectrum1 = store.load(hdf5)
            first = False
        else:
            spectrum2 = store.load(hdf5)
            spectrum1.add(spectrum2)
    store.dump("combined.hdf5", spectrum1)
