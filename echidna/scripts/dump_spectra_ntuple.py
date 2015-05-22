""" Example script to create spectrum objects from ntuple file and store
    in hdf5 format.

This script:
  * Reads in ntuple file of background / signal isotope
  * Creates and fills spectra objects with mc and reconstructed information
  * Plots Energy, radius and time dimensions of spectra object
  * Saves spectra objects to file in hdf5 format

Examples:
  To read rat_ds root file "example.ntuple.root"::

    $ python dump_spectra_ntuple.py /path/to/config.yml /path/to/example.ntuple.root

  This will create the smeared hdf5 file ``./example.hdf5``.
  To specify a save directory, include a -s flag followed by path to
  the required save destination.
"""
import numpy
import argparse
import csv
import echidna.output.store as store
import echidna.core.spectra as spectra
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot as plot

def read_and_dump_ntuple(fname, config_path, spectrum_name, save_path):
    """ Creates both mc and reco spectra from ntuple files, dumping the results as a
    spectrum object in a hdf5 file

    Args:
      fname (str): The file to be evaluated
      spectrum_name (str): Name to be applied to the spectrum
      save_path (str): Path to a directory where the hdf5 files will be dumped

    Returns:
      None
    """
    mc_config = spectra.SpectraConfig.load_from_file(config_path)
    reco_config = spectra.SpectraConfig.load_from_file(config_path)
    mc_spec = fill_spectrum.fill_mc_ntuple_spectrum(fname,
                                                    spectrumname = "%s_mc" % (spectrum_name),
                                                    config = mc_config)
    reco_spec = fill_spectrum.fill_reco_ntuple_spectrum(fname,
                                                        spectrumname = "%s_reco" % (spectrum_name),
                                                        config = reco_config)

    # Plot
    plot_spectrum(mc_spec, mc_config)
    plot_spectrum(reco_spec, reco_config)

    # Dump to file
    store.dump("%s/%s_mc.hdf5" % (save_path, spectrum_name), mc_spec)
    store.dump("%s/%s_reco.hdf5" % (save_path, spectrum_name), reco_spec)

def plot_spectrum(spec, config):
    """ Plot spectra for each of the spectrum dimensions (e.g. energy)

    Args: 
      Spec (:class:`echidna.core.spectra.Spectra`): Spectrum object to be plotted

    Returns:
      None
    """
    for v in config.getpars():
        plot.plot_projection(spec, v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path",
                      type=str,
                      default="./",
                      help="Enter destination path for .hdf5 spectra files.")
    parser.add_argument("config",
                        type=str,
                        help="Path to config file")
    parser.add_argument("fname",
                      type=str,
                      help="Path to root file to be read.")
    args = parser.parse_args()

    # If args passed directly, deal with them
    fname = args.fname
    spectrum_name = fname[fname.rfind('/', 0, -1)+1:]  
    read_and_dump_ntuple(fname, args.config, spectrum_name, args.save_path)
