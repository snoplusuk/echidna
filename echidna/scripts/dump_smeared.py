""" Example smearing script

This script:
  * Reads in mc spectra from hdf5
  * Smears spectra, default is to use weighted Gaussian method, but can
    also use specify random Gaussian method via command line
  * Smeared spectrum is saved to the same directory with ``_smeared``
    added to the file name

Examples:
  To smear hdf5 file ``example.hdf5`` using the random Gaussian method::
  
    $ python dump_smeared.py --smear_method "random" /path/to/example.hdf5

  This will create the smeared hdf5 file ``/path/to/example_smeared.hdf5``.
"""

import echidna.output.store as store
import echidna.core.smear as smear

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--smear_method", type=str,
                        help="specify the smearing method to use")
    parser.add_argument("path", type=str,
                        help="specify path to hdf5 file")
    args = parser.parse_args()

    directory = args.path[:args.path.rfind("/")+1]  # strip filename
    # strip directory and extension
    filename = args.path[args.path.rfind("/")+1:args.path.rfind(".")]

    smearer = smear.Smear()
    spectrum = store.load(args.path)

    if (args.smear_method == "random"):
        smeared_spectrum = smearer.random_gaussian_energy_spectra(spectrum)
        smeared_spectrum = smearer.random_gaussian_radius_spectra(
            smeared_spectrum)
    else:  # Use default smear method
        smeared_spectrum = smearer.weight_gaussian_energy_spectra(spectrum)
        smeared_spectrum = smearer.weight_gaussian_radius_spectra(
            smeared_spectrum)
    
    filename = directory + filename + "_smeared" + ".hdf5"
    store.dump(filename, smeared_spectrum)
