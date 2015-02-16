import rat
from ROOT import RAT
from ROOT import TChain
import math
import echidna.core.spectra as spectra


def _scint_weights(times, T):
    """This method applies to the scintillator backgrounds.
    It produces the list of weights relative to each time period.
    The calculation of weights is based on radioactive decay formula.

    Args:
      times (*list* of *int*): Time periods
      T (float): The Half-life of a studied background

    Returns:
      Weights (*list* of *float*)
    """
    weights = []
    for time in times:
        weights.append(math.exp(-time/T))
    return (weights)


def _av_weights(times, T):
    """This method applies to the backgrounds due to AV leaching.
    It produces the list of weights relative to each time period.
    The calculation of weights is based on radioactive decay formula.

    Args:
      times (*list* of *int*): Time periods
      T (float): The Half-life of a studied background

    Returns:
      Weights (*list* of *float*)
    """
    weights = []
    for time in times:
        weights.append(1.0)
    return (weights)


def fill_reco_spectrum(filename, T, spectrumname="", spectrum=None):
    """This function fills in the ndarray of energies, radii, times
    and weights. It takes the reconstructed energies and positions
    of the events from the root file. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    depending on the time period. Both, studied time and Half-life must
    be written in the same units.

    Args:
      filename (str): A root file to study
      T (float): The Half-life of a studied background
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.

    Raises:
      ValueError: If spectrumname is not set when creating a new
        spectrum.

    Returns:
      spectrum (:class:`echidna.core.spectra.Spectra`)
    """
    print filename
    print spectrumname
    dsreader = RAT.DU.DSReader(filename)
    if spectrum is None:
        if spectrumname == "":
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), dsreader.GetEntryCount())

    times = [0]
    for time_step in range(0, spectrum._time_bins):
        time = time_step * spectrum._time_width + spectrum._time_low
        times.append(time)

    if 'AV' in spectrumname:
        print "AV WEIGHTS ARE CURRENTLY UNAVAILABLE"
        weights = _av_weights(times, T)
    else:
        weights = _scint_weights(times, T)

    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        for ievent in range(0, ds.GetEVCount()):
            ev = ds.GetEV(ievent)
            if not ev.DefaultFitVertexExists() or not ev.GetDefaultFitVertex().ContainsEnergy() or not ev.GetDefaultFitVertex().ValidEnergy():
                continue
            energy = ev.GetDefaultFitVertex().GetEnergy()
            position = ev.GetDefaultFitVertex().GetPosition().Mag()
            for time, weight in zip(times, weights):
                try:
                    spectrum.fill(energy, position, time, weight)
                except ValueError:
                    pass

    return spectrum


def fill_mc_spectrum(filename, T, spectrumname="", spectrum=None):
    """This function fills in the ndarray of true energies, radii, times
    and weights. It takes the true energies and positions of the events
    from the root file. In order to keep the statistics, the time
    dependence is performed via adding weights to every event depending
    on the time period. Both, studied time and Half-life must be
    written in the same units.

    Args:
      filename (str): A root file to study
      T (float): The Half-life of a studied background
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.

    Raises:
      ValueError: If spectrumname is not set when creating a new
        spectrum.

    Returns:
      spectrum (:class:`echidna.core.spectra.Spectra`)
    """
    print filename
    print spectrumname
    dsreader = RAT.DU.DSReader(filename)
    if spectrum is None:
        if spectrumname == "":
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), dsreader.GetEntryCount())

    times = []
    for time_step in range(0, spectrum._time_bins):
        time = time_step * spectrum._time_width + spectrum._time_low
        times.append(time)

    if 'AV' in spectrumname:
        print "AV WEIGHTS ARE CURRENTLY UNAVAILABLE"
        weights = _av_weights(times, T)
    else:
        weights = _scint_weights(times, T)

    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        mc = ds.GetMC()
        if mc.GetMCParticleCount() > 0:
            energy = mc.GetScintQuenchedEnergyDeposit()
            position = mc.GetMCParticle(0).GetPosition().Mag()
            for time, weight in zip(times, weights):
                try:
                    spectrum.fill(energy, position, time, weight)
                except ValueError:
                    pass

    return spectrum


def fill_reco_ntuple_spectrum(filename, T, spectrumname="", spectrum=None):
    """This function fills in the ndarray of energies, radii, times
    and weights. It takes the reconstructed energies and positions
    of the events from the ntuple. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    depending on the time period. Both, studied time and Half-life must
    be written in the same units.

    Args:
      filename (str): The ntuple to study
      T (float): The Half-life of a studied background
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.

    Raises:
      ValueError: If spectrumname is not set when creating a new
        spectrum.

    Returns:
      spectrum (:class:`echidna.core.spectra.Spectra`)
    """
    print filename
    print spectrumname
    chain = TChain("output")
    chain.Add(filename)
    if spectrum is None:
        if spectrumname == "":
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), chain.GetEntries())

    times = []
    for time_step in range(0, spectrum._time_bins):
        time = time_step * spectrum._time_width + spectrum._time_low
        times.append(time)

    if 'AV' in spectrumname:
        print "AV WEIGHTS ARE CURRENTLY UNAVAILABLE"
        weights = _av_weights(times, T)
    else:
        weights = _scint_weights(times, T)

    for event in chain:
        if event.scintFit == 0:
            continue
        energy = event.energy
        position = math.fabs(math.sqrt((event.posx)**2 +
                                       (event.posy)**2 + (event.posz)**2))
        for time, weight in zip(times, weights):
            try:
                spectrum.fill(energy, position, time, weight)
            except ValueError:
                pass

    return spectrum


def fill_mc_ntuple_spectrum(filename, T, spectrumname="", spectrum=None):
    """This function fills in the ndarray of energies, radii, times
    and weights. It takes the reconstructed energies and positions
    of the events from ntuple. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    depending on the time period. Both, studied time and Half-life must
    be written in the same units.

    Args:
      filename (str): The ntuple to study
      T (float): The Half-life of a studied background
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.

    Raises:
      ValueError: If spectrumname is not set when creating a new
        spectrum.

    Returns:
      spectrum (:class:`echidna.core.spectra.Spectra`)
    """
    print filename
    print spectrumname
    chain = TChain("output")
    chain.Add(filename)
    if spectrum is None:
        if spectrumname == "":
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), chain.GetEntries())

    times = []
    for time_step in range(0, spectrum._time_bins):
        time = time_step * spectrum._time_width + spectrum._time_low
        times.append(time)

    if 'AV' in spectrumname:
        print "AV WEIGHTS ARE CURRENTLY UNAVAILABLE"
        weights = _av_weights(times, T)
    else:
        weights = _scint_weights(times, T)

    for event in chain:
        energy = event.mcEdepQuenched
        position = math.fabs(math.sqrt((event.mcPosx)**2 +
                                       (event.mcPosy)**2 + (event.mcPosz)**2))
        for time, weight in zip(times, weights):
            try:
                spectrum.fill(energy, position, time, weight)
            except ValueError:
                pass

    return spectrum
