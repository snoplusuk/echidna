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


def fill_from_root(filename, spectrum_name="", config=None, spectrum=None,
                   **kwargs):
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
      \**kwargs (dict): Passed to and checked by the dsextractor.

    Raises:
      ValueError: If spectrum_name is not set when creating a new
        spectrum.
      IndexError: Unknown dimension type (not mc, truth or reco).

    Returns:
      :class:`echidna.core.spectra.Spectra`: The filled spectrum.
    """
    dsreader = RAT.DU.DSReader(filename)
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
        extractors[var] = {"type": var_type,
                           "extractor": dsextract.function_factory(var,
                                                                   **kwargs)}
    for ievent in range(0, dsreader.GetEntryCount()):
        ds = dsreader.GetEntry(ievent)
        for ievent in range(0, ds.GetEVCount()):
            if ev_fill:
                ev = ds.GetEV(ievent)
            if mc_fill:
                mc = ds.GetMC()
            # Check to see if all parameters are valid and extract values
            fill = True
            fill_kwargs = {}
            for var in extractors:
                extractor = extractors[var]["extractor"]
                if extractors[var]["type"] == "reco":
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
            # If all OK, fill the spectrum
            if fill:
                try:
                    spectrum.fill(**fill_kwargs)
                    spectrum._raw_events += 1
                except ValueError:
                    pass
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
    for event in chain:
        fill = True
        fill_kwargs = {}
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
