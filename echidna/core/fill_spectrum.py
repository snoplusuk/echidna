import ROOT
import rat
import math
import echidna.core.spectra as spectra

def _scint_weights(T):
    """This method applies to the scintillator backgrounds.
    It produces the list of time periods to study the time 
    dependence. For each time period it is calculated the
    weight, based on radioactive decay rate formula, and
    stored in a list. Both, studied time and Half-life must
    be written in the same units. 
 
    Args:
      initial_time (int): The beginning of time period to study
      final_time (int): The end time of time period to study
      T (float): The Half-life of a studied background

    Returns:
      (*tuple*). Times (*list* of *int*) and Weights (*list* of *float*)
    """
    times = [0]
    weights = [0]
    for time_step in range(0, spectra.Spectra._time_bins):
        time = time_step * spectra.Spectra._time_width + spectra.Spectra._time_low 
        weights.append( math.exp(-time/T) )
        times.append(time)
    return (times, weights)

def fill_spectrum(filename, spectrumname, T):
    """This function fills in the ndarray of energies, radii, times 
    and weights. It takes the reconstructed energies and positions
    of the events from the root file. In order to keep the statistics, 
    the time dependence is performed via adding weights to every event
    depending on the time period. Both, studied time and Half-life must 
    be written in the same units.  
    
    Args:
      filename (str): A root file to study 
      spectrum (spectra.Spectra): Ndarray to be filled
      initial_time (int): The beginning of time period to study
      final_time (int): The end time of time period to study
      T (float): The Half-life of a studied background
    """
    print filename
    print spectrumname
    spectrum = spectra.Spectra( str(spectrumname) )
 
    if 'AV' in spectrumname:
        print "AV timimg scaling is currently unavailale"
    else:
        times, weights = _scint_weights(T)

    total_events = 0
    reconstructed_events = 0
    for ds, run in rat.dsreader(filename):
        total_events += 1
        if ds.GetEVCount() == 0:
            continue
        ev = ds.GetEV(0)       
        vertex = ev.GetFitResult("scintFitter").GetVertex(0)
        if not vertex.ContainsEnergy() or not vertex.ValidEnergy():
            continue
        if not vertex.ContainsPosition() or not vertex.ValidPosition():
            continue
        reconstructed_events += 1

        if vertex.GetPosition().Mag() > 4000.0:
            continue 
        
        for time, weight in zip(times, weights):
            spectrum.fill(vertex.GetEnergy(), vertex.GetPosition().Mag(), time, weight)

    return spectrum
    
