""" Example smearing script

This script:
  * Reads in two spectra from hdf5
  * Combines the spectra into one spectrum
  * Dumps spectrum to ``combined.hdf5``

Examples:
  To combine hdf5 files ``example1.hdf5``and ``example2.hdf5`` ::

    $ python echidna/scripts/combine_hdf5.py /path/to/example1.hdf5
      /path/to/example2.hdf5

  This will create the hdf5 file ``combined.hdf5``.
"""

import echidna.output.store as store
import sys

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    if len(sys.argv) != 3:
        print "ERROR: Must provide paths to two hdf5 to combine"
        sys.exit(1)
    spectrum1 = store.load(file1)
    spectrum2 = store.load(file1)
    spectrum1.add(spectrum2)
    store.dump("combined.hdf5",spectrum1)
