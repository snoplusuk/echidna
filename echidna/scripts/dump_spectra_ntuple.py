""" Example script to create spectrum objects from ntuple file and store
    in hdf5 format.

This script:
  * Reads in ntuple file of background / signal isotope
  * Creates and fills spectra objects with mc and reconstructed information
  * Plots Energy, radius and time dimensions of spectra object
  * Saves spectra objects to file in hdf5 format

Examples:
  To read rat_ds root file "file.ntuple.root" with config file cnfg.yml::

    $ python dump_spectra_ntuple.py /path/to/cnfg.yml /path/to/file.ntuple.root

  This will create the smeared hdf5 file ``./example.hdf5``.
  To specify a save directory, include a -s flag followed by path to
  the required save destination.
"""
import argparse
import csv
import echidna.output.store as store
import echidna.core.spectra as spectra
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot_root as plot


def read_and_dump_ntuple(fname, config_path, spectrum_name, save_path, bipo,
                         fv_radius, outer_radius):
    """ Creates both mc and reco spectra from ntuple files, dumping the
        results as a spectrum object in a hdf5 file

    Raises:
      ValueError: If outer_radius not None and radial3 is not in the config.

    Args:
      fname (str): The file to be evaluated
      config_path (str): Path to the config file
      spectrum_name (str): Name to be applied to the spectrum
      save_path (str): Path to a directory where the hdf5 files will be dumped
      bipo (bool): Apply Bi*Po* cuts when extracting data if True.
      fv_radius (float): Cut events outside the fiducial volume of this radius.
      outer_radius (float): Used for calculating the radial3 parameter.
        See :class:`echidna.core.dsextract` for details.
    """
    config = spectra.SpectraConfig.load_from_file(config_path)
    if outer_radius:
        if "radial3" not in config.get_dims():
            raise ValueError("Outer radius passed as an command line arg "
                             "but no radial3 in the config file.")
        spectrum = fill_spectrum.fill_from_ntuple(
            fname, spectrum_name="%s_mc" % (spectrum_name), config=config,
            bipo=bipo, fv_radius=fv_radius, outer_radius=outer_radius)
    else:
        spectrum = fill_spectrum.fill_from_ntuple(
            fname, spectrum_name="%s_mc" % (spectrum_name), config=config,
            bipo=bipo, fv_radius=fv_radius)

    # Plot
    plot_spectrum(spectrum, config)

    # Dump to file
    store.dump("%s/%s.hdf5" % (save_path, spectrum_name), spectrum)


def plot_spectrum(spec, config):
    """ Plot spectra for each of the spectrum dimensions (e.g. energy)

    Args:
      Spec (:class:`echidna.core.spectra.Spectra`): Spectrum object
        to be plotted
      config (:class:`echidna.core.spectra.Config`): configuration object
    """
    for v in config.get_pars():
        plot.plot_projection(spec, v)


def read_tab_delim_file(fname):
    """ Read file paths from text file.

    Args:
      fname (str): Name of file to be read.

    Returns:
      list: List of file paths read from file
    """
    file_paths = []
    with open(fname, 'r') as f:
        # next(f) # skip headings
        reader = csv.reader(f, delimiter='\t')
        for path in reader:
            file_paths.append(path)
    return file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--read_text_file", type=str,
                        help="Pass path to list .txt of filepaths.")
    parser.add_argument("-s", "--save_path", type=str, default="./",
                        help="Enter destination path for .hdf5 spectra files.")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to config file")
    parser.add_argument("-f", "--fname", type=str,
                        help="Path to root file to be read.")
    parser.add_argument("--bipo", dest="bipo", action="store_true",
                        help="Apply bipo cut")
    parser.add_argument("--no-bipo", dest="bipo", action="store_false",
                        help="Don't apply bipo cut")
    parser.add_argument("-v", "--fv_radius", type=float,
                        help="Radius for fiducial volume cut", default=None)
    parser.add_argument("-o", "--outer_radius", type=float,
                        help="Outer radius for filling spectra with the"
                        "parameter radial3.", default=None)
    parser.set_defaults(bipo=False)
    args = parser.parse_args()



    if args.read_text_file:      # If passed text file: read, format and dump
        path_list, half_life_list = read_tab_delim_file(args.read_text_file)
        for fname, half_life in zip(path_list, half_life_list):
            spectrum_name = fname[fname.rfind('/', 0, -1)+1:]
            read_and_dump_ntuple(fname, config, spectrum_name,
                                 args.save_path, args.bipo, args.fv_radius,
                                 args.outer_radius)
    else:  # If args passed directly, deal with them
        fname = args.fname
        spectrum_name = fname[fname.rfind('/', 0, -1)+1:]
        read_and_dump_ntuple(fname, args.config, spectrum_name,
                             args.save_path, args.bipo, args.fv_radius,
                             args.outer_radius)
