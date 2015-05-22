""" Example smearing script

This script:
  * Reads in spectrum from hdf5
  * Rebins spectrum with user input number of bins
  * Saves rebinned spectrum to given output or, by default, to the same
    directory with _rebin added to the filename.

Examples:
  To rebin hdf5 file ``example.hdf5`` that has 3 dimensions with 1000 bins in
  each dimension to 500 bins in the first dimension and save the spectrum to
  ``example2.hdf5`` then the following command is required::

    $ python rebin.py -i /path/to/example.hdf5 -o /path/to/example2.hdf5
      -b 500 1000 1000

  This will create the smeared hdf5 file ``/path/to/example_smeared.hdf5``.

"""

from echidna.output import store
from echidna.core import spectra

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Input spectra to rebin")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Name of output. Default adds _rebin to name")
    parser.add_argument("-b", "--bins", nargs='+', type=int,
                        help="Number of bins for each dimension")
    args = parser.parse_args()
    directory = args.input[:args.input.rfind("/")+1]  # strip filename
    # strip directory and extension
    filename = args.input[args.input.rfind("/")+1:args.input.rfind(".")]
    spectrum = store.load(args.input)
    new_bins = args.bins
    print spectrum._data.shape
    print "sum pre bin", spectrum.sum()
    spectrum.rebin(new_bins)
    print 'Sum post bin:', spectrum.sum()
    print spectrum._data.shape
    f_out = args.output
    if not f_out:
        f_out = directory + filename + "_rebin" + ".hdf5"
    print "Rebinned", args.input, ", saved to", f_out
    store.dump(f_out, spectrum)
