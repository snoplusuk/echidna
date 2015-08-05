""" Example script to create spectrum objects from rat_ds root files
  and store in hdf5 format.

This script:
  * Reads in rat_ds root file of background / signal isotope
  * Creates and fills spectra objects with mc and reconstructed information
  * Plots Energy, radius and time dimensions of spectra object
  * Saves spectra objects to file in hdf5 format

Examples:
  To read rat_ds root file "example.root"::

    $ python dump_spectra_ntuple.py /path/to/config.yml /path/to/example.root

  This will create the smeared hdf5 file ``./example.hdf5``.
  To specify a save directory, include a -s flag followed by path to
  the required save destination.
"""
import argparse
import csv
import echidna.output.store as store
import echidna.core.spectra as spectra
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot as plot


def read_and_dump_root(fname, config_path, spectrum_name, save_path):
    """ Creates both mc and reco spectra from ROOT files, dumping the
    results as a spectrum object in a hdf5 file

    Args:
      fname (str): The file to be evaluated
      config_path (str): Path to the config file
      spectrum_name (str): Name to be applied to the spectrum
      save_path (str): Path to a directory where the hdf5 files will be dumped

    Returns:
      None
    """
    mc_config = spectra.SpectraConfig.load_from_file(config_path)
    reco_config = spectra.SpectraConfig.load_from_file(config_path)
    truth_config = spectra.SpectraConfig.load_from_file(config_path)
    mc_spec = fill_spectrum.fill_mc_spectrum(
        fname, spectrumname="%s_mc" % (spectrum_name), config=mc_config)
    reco_spec = fill_spectrum.fill_reco_spectrum(
        fname, spectrumname="%s_reco" % (spectrum_name), config=reco_config)
    truth_spec = fill_spectrum.fill_truth_spectrum(
        fname, spectrumname="%s_truth" % (spectrum_name), config=truth_config)
    # Plot
    plot_spectrum(mc_spec, mc_config)
    plot_spectrum(reco_spec, reco_config)
    plot_spectrum(truth_spec, truth_config)

    # Dump to file
    store.dump("%s/%s_mc.hdf5" % (save_path, spectrum_name), mc_spec)
    store.dump("%s/%s_reco.hdf5" % (save_path, spectrum_name), reco_spec)
    store.dump("%s/%s_truth.hdf5" % (save_path, spectrum_name), truth_spec)


def plot_spectrum(spec, config):
    """ Plot spectra for each of the spectrum dimensions (e.g. energy)

    Args:
      Spec (:class:`echidna.core.spectra.Spectra`): Spectrum object to
        be plotted
      config (:class:`echidna.core.spectra.Config`): configuration object
    """
    for v in config.getpars():
        plot.plot_projection(spec, v)


def read_tab_delim_file(fname):
    """ Read file paths from text file

    Args:
      fname (str): Name of file to be read.

    Returns:
      file_paths (list): List of file paths read from file
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
                        help="Pass path to list .txt of file paths")
    parser.add_argument("-s", "--save_path", type=str, default="./",
                        help="Enter destination path for .hdf5 spectra files.")
    parser.add_argument("config", type=str,
                        help="Path to config file")
    parser.add_argument("fname", type=str,
                        help="Path to root file to be read.")
    args = parser.parse_args()

    if args.read_text_file:  # If passed text file: read, format and dump
        path_list = read_tab_delim_file(args.read_text_file)
        for fname in path_list:
            spectrum_name = fname[fname.rfind('/', 0, -1)+1:]
            read_and_dump_root(fname, args.config, spectrum_name,
                               args.save_path)
    else:  # If args passed directly, deal with them
        fname = args.fname
        spectrum_name = fname[fname.rfind('/', 0, -1)+1:]
        read_and_dump_root(fname, args.config,
                           spectrum_name, args.save_path)
