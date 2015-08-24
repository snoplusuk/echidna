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


def _root_mix(spectrum, dsreader, extractors, bipo):
    """ Internal function for filling a spectrum whose config has a mixture of
      mc (and/or truth) and reco paremeters.

    Args:
      spectrum (:class:`echidna.core.spectra.Spectra`): The spectrum which is
        being filled.
      dsreder (ROOT.RAT.DU.DSReader): rat's data structure reader
        for the root file.
      extractors (dict): Keys are the variable names and the keys are their
        respective extractors.
      bipo (bool, optional): Applies the bipo cut if set to True.
    """
    for entry in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        fill_kwargs = {}
        # Note mc will be the same for all evs in loop below:
        mc = ds.GetMC()
        if bipo and ds.GetEVCount() != 1:
            # Only bipos with 1 ev survive bipo cut
            continue
        for ievent in range(0, ds.GetEVCount()):
            ev = ds.GetEV(ievent)
            fill = True
            for var, extractor in extractors.iteritems():
                var_type = var.split("_")[-1]
                if var_type == "reco":
                    if extractor.get_valid_root(ev):
                        fill_kwargs[extractor._name] = \
                            extractor.get_value_root(ev)
                    else:
                        fill = False
                        break
                else:  # mc or truth
                    if extractor.get_valid_root(mc):
                        fill_kwargs[extractor._name] = \
                            extractor.get_value_root(mc)
                    else:
                        fill = False
                        break
            if fill:
                try:
                    spectrum.fill(**fill_kwargs)
                    spectrum._raw_events += 1
                except ValueError:
                    pass


def _root_ev(spectrum, dsreader, extractors, bipo):
    """ Internal function for filling a spectrum whose config only has
      reco paremeters.

    Args:
      spectrum (:class:`echidna.core.spectra.Spectra`): The spectrum which is
        being filled.
      dsreder (ROOT.RAT.DU.DSReader): rat's data structure reader
        for the root file.
      extractors (dict): Keys are the variable names and the keys are their
        respective extractors.
      bipo (bool, optional): Applies the bipo cut if set to True.
    """
    for entry in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        if bipo and ds.GetEVCount() != 1:
            # Only bipos with 1 ev survive bipo cut
            continue
        for ievent in range(0, ds.GetEVCount()):
            ev = ds.GetEV(ievent)
            fill_kwargs = {}
            fill = True
            for var, extractor in extractors.iteritems():
                if extractor.get_valid_root(ev):
                    fill_kwargs[extractor._name] = extractor.get_value_root(ev)
                else:
                    fill = False
                    break
            if fill:
                try:
                    spectrum.fill(**fill_kwargs)
                    spectrum._raw_events += 1
                except ValueError:
                    pass


def _root_mc(spectrum, dsreader, extractors, bipo):
    """ Internal function for filling a spectrum whose config only has
      mc or truth paremeters.

    Args:
      spectrum (:class:`echidna.core.spectra.Spectra`): The spectrum which is
        being filled.
      dsreder (ROOT.RAT.DU.DSReader): rat's data structure reader
        for the root file.
      extractors (dict): Keys are the variable names and the keys are their
        respective extractors.
      bipo (bool, optional): Applies the bipo cut if set to True.
    """
    for entry in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        mc = ds.GetMC()
        fill = True
        fill_kwargs = {}
        if bipo and ds.GetEVCount() != 1:
            # Only bipos with 1 ev survive bipo cut
            continue
        for var, extractor in extractors.iteritems():
            if extractor.get_valid_root(mc):
                fill_kwargs[extractor._name] = extractor.get_value_root(mc)
            else:
                fill = False
                break
        if fill:
            try:
                spectrum.fill(**fill_kwargs)
                spectrum._raw_events += 1
            except ValueError:
                pass


def fill_from_root(filename, spectrum_name="", config=None, spectrum=None,
                   bipo=False, **kwargs):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the parameter specified in the config from the
    events in the root file.

    Args:
      filename (str): A root file to study
      spectrum_name (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`echidna.core.spectra.SpectrumConfig`, optional):
        The config for the spectrum. Not requried when appending a spectrum.
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.
      bipo (bool, optional): Applies the bipo cut if set to True.
        Default is False.
      \**kwargs (dict): Passed to and checked by the dsextractor.

    Raises:
      ValueError: If spectrum_name is not set when creating a new
        spectrum.
      IndexError: Unknown dimension type (not mc, truth or reco).

    Returns:
      :class:`echidna.core.spectra.Spectra`: The filled spectrum.
    """
    if spectrum is None:
        if spectrum_name == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrum_name),
                                   dsreader.GetEntryCount(),
                                   config)
    else:
        spectrum._num_decays += dsreader.GetEntryCount()
        spectrum_name = spectrum._name
    print "Filling", spectrum_name, "with", filename
    extractors = {}
    mc_fill = False
    ev_fill = False
    for var in spectrum.get_config().get_pars():
        var_type = var.split("_")[-1]
        if var_type == "mc" or var_type == "truth":
            mc_fill = True
        elif var_type == "reco":
            ev_fill = True
        else:
            raise IndexError("Unknown paramer type %s" % var_type)
        extractors[var] = dsextract.function_factory(var, **kwargs)
    dsreader = RAT.DU.DSReader(filename)
    if bipo:
        spectrum._bipo = 1  # Flag to indicate bipo cuts are applied
    if mc_fill and ev_fill:
        _root_mix(spectrum, dsreader, extractors, bipo)
    elif mc_fill:
        _root_mc(spectrum, dsreader, extractors, bipo)
    else:
        _root_ev(spectrum, dsreader, extractors, bipo)
    return spectrum


def fill_from_ntuple(filename, spectrum_name="", config=None, spectrum=None,
                     **kwargs):
    """**Weights have been disabled.**
    This function fills in the ndarray (dimensions specified in the config)
    with weights. It takes the parameters specified in the config from
    the events in the ntuple.

    Args:
      filename (str): The ntuple to study
      spectrum_name (str, optional): A name of future spectrum. Not
        required when appending a spectrum.
      config (:class:`spectra.SpectrumConfig`, optional): The config for
        the spectrum
      spectrum (:class:`echidna.core.spectra.Spectra`, optional):
        Spectrum you wish to append. Not required when creating a
        new spectrum.
      \**kwargs (dict): Passed to and checked by the dsextractor.

    Raises:
      ValueError: If spectrum_name is not set when creating a new
        spectrum.

    Returns:
      :class:`echidna.core.spectra.Spectra`: The filled spectrum.
    """
    chain = TChain("output")
    chain.Add(filename)
    entries = chain.GetEntries()
    if spectrum is None:
        if spectrum_name == "" or not config:
            raise ValueError("Name not set when creating new spectra.")
        spectrum = spectra.Spectra(str(spectrum_name), chain.GetEntries(),
                                   config)
    else:
        spectrum._num_decays += chain.GetEntries()
        spectrum_name = spectrum._name
    print "Filling", spectrum_name, "with", filename
    extractors = []
    for var in spectrum.get_config().get_pars():
        extractors.append(dsextract.function_factory(var, **kwargs))
    if bipo:
        spectrum._bipo = 1  # Flag to indicate bipo cuts are applied
    for ievent in range(0, entries):
        fill = True
        fill_kwargs = {}
        event = chain.GetEvent(ievent)
        # Apply bipo cut here:
        if bipo:
            # Only want the first triggered event:
            if event.evIndex != 1:
                continue
            # Make sure you dont go out of scope·
            if ievent + 1 != entires:
                event2 = chain.GetEvent(ievent + 1)
                # First triggered event but check if it has trailing events:
                elif event2.evIndex > 1:
                    continue
        # Check to see if all parameters are valid and extract values
        for e in extractors:
            if e.get_valid_ntuple(event):
                fill_kwargs[e.name] = e.get_value_ntuple(event)
            else:
                fill = False
                break
        # If all OK, fill the spectrum
        if fill:
            try:
                spectrum.fill(**fill_kwargs)
                spectrum._raw_events += 1
            except ValueError:
                pass
    return spectrum
