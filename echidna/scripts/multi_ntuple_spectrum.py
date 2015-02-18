import numpy
import os
import optparse
import echidna.output.store as store
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot as plot

def create_combined_ntuple_spectrum(data_path, half_life, bkgnd_name, save_path):
    """ Creates both mc and reco spectra from directory containing background ntuples, 
    dumping the results as a spectrum object in a hdf5 file.

    Args:
      data_path (str): Path to directory containing the ntuples to be evaluated
      half_life (float): Half-life of isotope
      bkgnd_name (str): Name of the background being processed
      save_path (str): Path to a directory where the hdf5 files will be dumped      

    Returns:
      None
    """
    file_list = os.listdir(data_path)
    for idx, fname in enumerate(file_list):
        file_path = "%s/%s" % (data_path, fname)
        if idx == 0:
            mc_spec = fill_spectrum.fill_mc_ntuple_spectrum(file_path, half_life, spectrumname = "%s_mc" % bkgnd_name)
            reco_spec = fill_spectrum.fill_reco_ntuple_spectrum(file_path, half_life, spectrumname = "%s_reco" % bkgnd_name)
        else: 
            mc_spec = fill_spectrum.fill_mc_ntuple_spectrum(file_path, half_life, spectrum = mc_spec)
            reco_spec = fill_spectrum.fill_reco_ntuple_spectrum(file_path, half_life, spectrum = reco_spec)

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
    parser = optparse.OptionParser("Usage: python multi_ntuple_spectrum.py [option] <data_path> <half_life>")
    parser.add_option("-s", "--Save",
                      dest="Savepath",
                      default=False,
                      help="Enter destination path for .hdf5 spectra files.")
    parser.add_option("-n", "--Name",
                      dest="bkgnd_name",
                      default=False,
                      help="Name of background (to be used as file and spectrum name)")
    (options, args) = parser.parse_args()

    # Take data_path from arg input
    data_path = args[0]

    # Create save path
    if options.Savepath is False:
        save_path = "./"
    else:
        save_path = options.Savepath

    # Define name for spectrum
    if options.bkgnd_name is False:
        bkgnd_name = data_path[data_path.rfind('/', 0, -2)+1:-1]
    else: 
        bkgnd_name = options.bkgnd_name

    ###############################################################################
    # Set path to folder created when grabbing ntuples from grid.
    # All files contained should be read and filled into a single specturm object.
    ############################################################################### 
    create_combined_ntuple_spectrum(data_path, float(args[1]), bkgnd_name, data_path)
