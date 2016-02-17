""" Example scaling script

This script:
  * Reads in mc spectra from hdf5
  * Scales spectra
  * Scaled spectrum is saved to the same directory with ``_scaled``
    added to the file name

Examples:
  To scale energy_mc of the hdf5 file ``example.hdf5`` by a factor 1.1::

    $ python dump_scaled.py -d energy_mc -f /path/to/example.hdf5 -s 1.1

  This will create the scaled hdf5 file ``/path/to/example_scaled.hdf5``.
"""

import echidna.output.store as store
import echidna.core.scale as scale

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension", default=None, type=str,
                        help="specify dimension to scale e.g. energy_mc"
                        "e.g. 0.05 for 5 percent")
    parser.add_argument("-s", "--scale", default=None, type=float,
                        help="specify scale factor"
                        "e.g. 0.05 for 5 percent")
    parser.add_argument("-f", "--file", default=None, type=str,
                        help="specify path to hdf5 file")
    args = parser.parse_args()

    if not args.file:
        parser.print_help()
        raise IOError("No file given in command line to scale")
    if not args.scale:
        parser.print_help()
        raise ValueError("No scale factor given")
    if not args.dimension:
        parser.print_help()
        raise ValueError("No dimension to scale given")

    directory = args.file[:args.file.rfind("/")+1]  # strip filename
    # strip directory and extension
    filename = args.file[args.file.rfind("/")+1:args.file.rfind(".")]
    scaler = scale.Scale()
    scaler.set_scale_factor(args.scale)
    spectrum = store.load(args.file)
    scaled_spectrum = scaler.scale(spectrum, args.dimension)
    filename = directory + filename + "_scaled.hdf5"
    store.dump(filename, scaled_spectrum)
