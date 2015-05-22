""" Example script to create a sinlge spectrum file from multiple ntuple files

This script:
  * Reads all ntuple files in a directory
  * For each ntuple, both mc and reconstructed spectra are created and
    saved as hdf5 files in dedicated mc / reco directories 
    (themselves automatically generated)
  * Summed spectra containing the information from all ntuples are
    created for both mc and reconstructed data sets in dedicated
    "Summed" directory. Spectra saved in hdf5 format.

Examples:
  To read all ntuples in a directory and save both individual and summed 
  spectra to file::

    $ python multi_ntuple_spectrum.py /path/to/config.yml /path/to/ntuple/direc/

  This will create hdf5 spectra files in diectory structure:
  ./Summed/ ./mc/ ./reco/
  To specify a save directory, include a -s flag followed by path to
  the required save destination.
"""
import numpy
import os
import optparse
import echidna.output.store as store
import echidna.core.spectrum as spectrum
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot as plot

def create_combined_ntuple_spectrum(data_path, config_path, bkgnd_name, save_path):
    """ Creates both mc and reco spectra from directory containing background ntuples, 
    dumping the results as a spectrum object in a hdf5 file.

    Args:
      data_path (str): Path to directory containing the ntuples to be evaluated
      bkgnd_name (str): Name of the background being processed
      save_path (str): Path to a directory where the hdf5 files will be dumped      

    Returns:
      None
    """
    mc_config = spectra.SpectraConfig.load_from_file(config_path)
    reco_config = spectra.SpectraConfig.load_from_file(config_path)
    file_list = os.listdir(data_path)
    for idx, fname in enumerate(file_list):
        file_path = "%s/%s" % (data_path, fname)
        if idx == 0:
            mc_spec = fill_spectrum.fill_mc_ntuple_spectrum(file_path, "%s_mc" % bkgnd_name, config = mc_config)
            reco_spec = fill_spectrum.fill_reco_ntuple_spectrum(file_path, "%s_reco" % bkgnd_name, config = reco_config)
        else: 
            mc_spec = fill_spectrum.fill_mc_ntuple_spectrum(file_path, spectrum = mc_spec)
            reco_spec = fill_spectrum.fill_reco_ntuple_spectrum(file_path, spectrum = reco_spec)

    # Plot
    plot_spectrum(mc_spec)
    plot_spectrum(reco_spec)

    # Dump to file
    store.dump("%s%s_mc.hdf5" % (save_path, bkgnd_name), mc_spec)
    store.dump("%s%s_reco.hdf5" % (save_path, bkgnd_name), reco_spec)

def plot_spectrum(spec):
    """ Plot spectra for each of the three spectrum dimensions: Energy, radius and time

    Args:
      Spec (:class:`echidna.core.spectra.Spectra`): Spectrum object to be plotted

    Returns:
      None
    """
    plot.plot_projection(spec, 0)
    plot.plot_projection(spec, 1)
    plot.plot_projection(spec, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path",
                      type=str,
                      default="./",
                      help="Enter destination path for .hdf5 spectra files.")
    parser.add_argument("-n", "--bkgnd_name",
                      type=str,
                      help="Name of background (to be used as file and spectrum name)")
    parser.add_argument("config",
                        type=str,
                        help="Path to config file")
    parser.add_argument("path",
                      type=str,
                      help="Path to ntuple directory")
    args = parser.parse_args()

    # Take data_path from arg input
    data_path = args.path

    # Define name for spectrum
    if args.bkgnd_name:
        bkgnd_name = options.bkgnd_name
    else:
        bkgnd_name = data_path[data_path.rfind('/', 0, -2)+1:-1]

    ###############################################################################
    # Set path to folder created when grabbing ntuples from grid.
    # All files contained should be read and filled into a single specturm object.
    ############################################################################### 
    create_combined_ntuple_spectrum(data_path, args.config,
                                    bkgnd_name, args.save_path)
