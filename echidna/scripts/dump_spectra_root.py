import numpy
import optparse
import csv
import echidna.output.store as store
import echidna.core.fill_spectrum as fill_spectrum
import echidna.output.plot as plot

def read_and_dump_root(fname, half_life, spectrum_name, save_path):
    """ Creates both mc and reco spectra from ROOT files, dumping the results as a
    spectrum object in a hdf5 file

    Args:
      fname (str): The file to be evaluated
      half_life (float): Half-life of isotope
      spectrum_name (str): Name to be applied to the spectrum
      save_path (str): Path to a directory where the hdf5 files will be dumped

    Returns:
      None
    """
    mc_spec = fill_spectrum.fill_mc_spectrum(fname, half_life, spectrumname = "%s_mc" % (spectrum_name) )
    reco_spec = fill_spectrum.fill_reco_spectrum(fname, half_life, spectrumname = "%s_reco" % (spectrum_name) )

    # Plot
    plot_spectrum(mc_spec)
    plot_spectrum(reco_spec)

    # Dump to file
    store.dump("%s/%s_mc.hdf5" % (save_path, spectrum_name), mc_spec)
    store.dump("%s/%s_reco.hdf5" % (save_path, spectrum_name), reco_spec)

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

def read_tab_delim_file(fname):
    """ Read file paths and respective half lives from tab delimited text file.

    Args: 
      fname (str): Name of file to be read.

    Returns:
      file_paths (list): List of file paths read from file
      half_lives (list): List of half lives read from file
    """
    file_paths, half_lives = [], []
    with open(path_to_file, 'r') as f:
        #next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for path, half  in reader:
            file_paths.append(path)
            half_lives.append(half) 
    return file_paths, half_lives

if __name__ == "__main__":
    parser = optparse.OptionParser("Usage: python dump_spectra.py [option] <file_path> <half_life>")
    parser.add_option("-t", "--text",
                      dest="Read_from_txt_file",
                      default=False,
                      help="Pass path to list .txt of tab separated filepaths and half_lives.")
    parser.add_option("-s", "--Save",
                      dest="Savepath",
                      default=False,
                      help="Enter destination path for .hdf5 spectra files.")
    (options, args) = parser.parse_args()

    # Set-up save path
    if options.Savepath is False:
        save_path = "./"
    else:
        save_path = options.Savepath

    # If args passed directly, deal with them
    if len(args) % 2 == 0:
        for i in range(0, len(args)/2+1, 2):
            fname = args[i]
            spectrum_name = fname[fname.rfind('/', 0, -1)+1:]  
            read_and_dump_root(fname, float(args[i+1]), spectrum_name, save_path)
    elif len(args) % 2 == 1:
        raise Exception("Expecting pairs of file paths and respective half lifes as arguments!")

    # If passed text file: read, format and dump 
    if options.Read_from_txt_file is not False:
        path_list, half_life_list = read_tab_delim_file(options.Read_from_txt_file)
        for idx, fname in enumerate(path_list):
            spectrum_name = fname[fname.rfind('/', 0, -1)+1:]
            read_and_dump_root(path, half_life_list[idx], spectrum_name, save_path)
            
