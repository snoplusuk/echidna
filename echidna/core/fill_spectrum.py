""" Provides code for creating echidna spectra and populating with data
from RAT Root files/ntuples.
"""

from echidna.util import root_help
import rat
from ROOT import RAT
from ROOT import TChain
import math
import echidna.core.spectra as spectra
import echidna.core.dsextract as dsextract


def fill_reco_spectrum(filename, spectrumname="", config=None, spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the reconstructed energies and positions of the
    events from the root file. In order to keep the statistics, the time
    dependence is performed via adding weights to every event depending on
    the time period. Both, studied time and Half-life must be written in the
    same units.

    Args:
      filename (str): A root file to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
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
        if spectrumname == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname),
                                   dsreader.GetEntryCount(),
                                   config)
    else:
        spectrum._num_decays += dsreader.GetEntryCount()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        for ievent in range(0, ds.GetEVCount()):
            ev = ds.GetEV(ievent)
            # Check to see if all parameters are valid
            if False in [e.ev_get_valid(ev) for e in extractors]:
                continue

            # All OK, fill the spectrum
            kwargs = {}
            for e in extractors:
                kwargs[e.name] = e.ev_get_value(ev)
            spectrum.fill(**kwargs)
            spectrum._raw_events += 1

    return spectrum


def fill_mc_spectrum(filename, spectrumname="", config=None, spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the true energies and positions of the events
    from the root file. In order to keep the statistics, the time
    dependence is performed via adding weights to every event depending
    on the time period. Both, studied time and Half-life must be
    written in the same units.


    Args:
      filename (str): A root file to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
      spectrum (:class:`echidna.core.spectra.Spectra`, optional): Spectrum
        you wish to append. Not required when creating a new spectrum.

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
        spectrum = spectra.Spectra(str(spectrumname),
                                   dsreader.GetEntryCount())
    else:
        spectrum._num_decays += dsreader.GetEntryCount()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        mc = ds.GetMC()
        if mc.GetMCParticleCount() > 0:

            if False in [e.mc_get_valid(mc) for e in extractors]:
                continue

            kwargs = {}
            for e in extractors:
                kwargs[e.name] = e.ev_get_value(ev)
            spectrum.fill(**kwargs)
            spectrum._raw_events += 1

    return spectrum


def fill_truth_spectrum(filename, spectrumname="", config=None, spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray of true energies, radii, times
    and weights. It takes the true (non-quenched) energies and
    positions of the events from the root file. In order to keep the
    statistics, the time dependence is performed via adding weights to
    every event depending on the time period. Both, studied time and
    Half-life must be written in the same units.

    Args:
      filename (str): A root file to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
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
        if spectrumname == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname),
                                   dsreader.GetEntryCount(),
                                   config)
    else:
        spectrum._num_decays += dsreader.GetEntryCount()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        mc = ds.GetMC()
        if mc.GetMCParticleCount() > 0:

            if False in [e.truth_get_valid(mc) for e in extractors]:
                continue

            kwargs = {}
            for e in extractors:
                kwargs[e.name] = e.truth_get_value(mc)
            spectrum.fill(**kwargs)
            spectrum._raw_events += 1

    return spectrum


def fill_reco_ntuple_spectrum(filename, spectrumname="", config=None,
                              spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the reconstructed energies and positions
    of the events from the ntuple. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    depending on the time period. Both, studied time and Half-life must
    be written in the same units.

    Args:
      filename (str): The ntuple to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
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
    chain = TChain("output")
    chain.Add(filename)
    if spectrum is None:
        if spectrumname == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), chain.GetEntries(),
                                   config)
    else:
        spectrum._num_decays += chain.GetEntries()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for event in chain:

        if False in [e.ntuple_ev_get_valid(event) for e in extractors]:
            continue

        kwargs = {}
        for e in extractors:
            kwargs[e.name] = e.ntuple_ev_get_value(event)
        spectrum.fill(**kwargs)
        spectrum._raw_events += 1

    return spectrum


def fill_mc_ntuple_spectrum(filename, spectrumname="", config=None,
                            spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the reconstructed energies and positions
    of the events from ntuple. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    written in the same units.

    Args:
      filename (str): The ntuple to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
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
    chain = TChain("output")
    chain.Add(filename)
    if spectrum is None:
        if spectrumname == "":
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), chain.GetEntries())
    else:
        spectrum._num_decays += chain.GetEntries()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for event in chain:

        if False in [e.ntuple_mv_get_valid(event) for e in extractors]:
            continue

        kwargs = {}
        for e in extractors:
            kwargs[e.name] = e.ntuple_mc_get_value(event)
        spectrum.fill(**kwargs)
        spectrum._raw_events += 1

    return spectrum


def fill_truth_ntuple_spectrum(filename, T, spectrumname="", config=None,
                               spectrum=None):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the reconstructed energies and positions
    of the events from ntuple. In order to keep the statistics,
    the time dependence is performed via adding weights to every event
    written in the same units.

    Args:
      filename (str): The ntuple to study
      spectrumname (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
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
    chain = TChain("output")
    chain.Add(filename)
    if spectrum is None:
        if spectrumname == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrumname), chain.GetEntries(),
                                   config)
    else:
        spectrum._num_decays += chain.GetEntries()
        spectrumname = spectrum._name
    print spectrumname

    extractors = []
    for var in spectrum.get_config().getpars():
        extractors.append(dsextract.function_factory(var))

    for event in chain:

        if False in [e.ntuple_truth_get_valid(event) for e in extractors]:
            continue

        kwargs = {}
        for e in extractors:
            kwargs[e.name] = e.ntuple_truth_get_value(event)
        spectrum.fill(**kwargs)
        spectrum._raw_events += 1

    return spectrum
