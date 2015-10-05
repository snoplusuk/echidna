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
  spectra to file with config cnfg.yml::

    $ python multi_ntuple_spectrum.py /path/to/cnfg.yml /path/to/ntuple/direc/

  To specify a save directory, include a -s flag followed by path to
  the required save destination.
"""
import os
import argparse
import echidna.output.store as store
import echidna.core.spectra as spectra
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot_root as plot


def create_combined_ntuple_spectrum(data_path, config_path, bkgnd_name,
                                    save_path, bipo, fv_radius, outer_radius):
    """ Creates both mc, truth and reco spectra from directory containing
    background ntuples, dumping the results as a spectrum object in an
    hdf5 file.

    Args:
      data_path (str): Path to directory containing the ntuples to be evaluated
      config_path (str): Path to config file
      bkgnd_name (str): Name of the background being processed
      save_path (str): Path to a directory where the hdf5 files will be dumped
      bipo (bool): Apply Bi*Po* cuts when extracting data if True.
      fv_radius (float): Cut events outside the fiducial volume of this radius.
      outer_radius (float): Used for calculating the radial3 parameter. 
        See :class:`echidna.core.dsextract` for details.
    """
    config = spectra.SpectraConfig.load_from_file(config_path)
    file_list = os.listdir(data_path)
    if outer_radius:
        if "radial3" not in config.get_dims():
            raise ValueError("Outer radius passed as an command line arg "
                             "but no radial3 in the config file.")
        for idx, fname in enumerate(file_list):
            file_path = "%s/%s" % (data_path, fname)
            if idx == 0:
                spec = fill_spectrum.fill_from_ntuple(
                    file_path, spectrum_name="%s" % bkgnd_name, config=config,
                    bipo=bipo, fv_radius=fv_radius, outer_radius=outer_radius)
            else:
                spec = fill_spectrum.fill_from_ntuple(
                    file_path, spectrum=spec, bipo=bipo, fv_radius=fv_radius,
                    outer_radius=outer_radius)
    else:
        for idx, fname in enumerate(file_list):
            file_path = "%s/%s" % (data_path, fname)
            if idx == 0:
                spec = fill_spectrum.fill_from_ntuple(
                    file_path, spectrum_name="%s" % bkgnd_name, config=config,
                    bipo=bipo, fv_radius=fv_radius)
            else:
                spec = fill_spectrum.fill_from_ntuple(
                    file_path, spectrum=spec, bipo=bipo, fv_radius=fv_radius)
    # Plot
    plot_spectrum(spec, config)
    # Dump to file
    store.dump("%s%s.hdf5" % (save_path, bkgnd_name), spec)


def plot_spectrum(spec, config):
    """ Plot spectra for each of the spectrum dimensions (e.g energy)

    Args:
      Spec (:class:`echidna.core.spectra.Spectra`): Spectrum object to
        be plotted.

    Returns:
      None
    """
    for var in config.get_pars():
        plot.plot_projection(spec, var)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default="./",
                        help="Enter destination path for .hdf5 spectra files.")
    parser.add_argument("-n", "--bkgnd_name", type=str,
                        help="Name of background (to be used as file and "
                        "spectrum name)")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to config file")
    parser.add_argument("-p", "--path", type=str,
                        help="Path to ntuple directory")
    parser.add_argument("--bipo", dest="bipo", action="store_true",
                        help="Apply bipo cut")
    parser.add_argument("--no-bipo", dest="bipo", action="store_false",
                        help="Don't apply bipo cut (default)")
    parser.add_argument("-v", "--fv_radius", type=float,
                        help="Radius for fiducial volume cut", default=None)
    parser.add_argument("-o", "--outer_radius", type=float,
                        help="Outer radius for filling spectra with the"
                        "parameter radial3.", default=None)
    parser.set_defaults(bipo=False)
    args = parser.parse_args()

    # Take data_path from arg input
    data_path = args.path

    # Define name for spectrum
    if args.bkgnd_name:
        bkgnd_name = args.bkgnd_name
    else:
        bkgnd_name = data_path[data_path.rfind('/', 0, -2)+1:-1]

    ##########################################################################
    # Set path to folder created when grabbing ntuples from grid.
    # All files contained should be read and filled into a single specturm
    # object.
    ##########################################################################
    create_combined_ntuple_spectrum(data_path, args.config, bkgnd_name,
                                    args.save_path, args.bipo, args.fv_radius,
                                    args.outer_radius)
