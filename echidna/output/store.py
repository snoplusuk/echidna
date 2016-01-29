from echidna.core.spectra import (Spectra, SpectraConfig, SpectraParameter,
                                  SpectraFitConfig, GlobalFitConfig)
from echidna.limit.summary import Summary, ReducedSummary

import logging
import h5py
import sys
import collections
import json


_logger = logging.getLogger("store")


def dict_to_string(in_dict):
    """ *** DEPRECIATED ***
    Converts a dicionary to a string so it can be saved in a hdf5 file.

    Args:
      in_dict (dict): Dictionary to convert.

    Raises:
      TypeError: If the type of a value of in_dict is not supported currently.
        Supported types are string, float and int.

    Returns:
      string: The converted dictionary.
    """
    out_string = ""
    for key, value in in_dict.iteritems():
        out_string += key+":"+str(value)+";"
        if type(value) is str:
            out_string += "str,"
        elif type(value) is float:
            out_string += "float,"
        elif type(value) is int:
            out_string += "int,"
        else:
            raise TypeError("%s has the unsupported type %s" % (value,
                                                                type(value)))
    return out_string[:-1]


def string_to_dict(in_string):
    """ *** DEPRECIATED ***
    Converts a string to a dictionary so it can be loaded from the hdf5 file.

    Args:
      in_string (string): The string to convert into a dictionary.

    Raises:
      TypeError: If the type of a value of in_dict is not supported currently.
        Supported types are string, float and int.

    Returns:
      dict: The converted string.
    """
    out_dict = {}
    keys_values = in_string.split(',')
    for entry in keys_values:
        key = entry.split(":")[0]
        value = entry.split(":")[1].split(";")[0]
        typ = entry.split(";")[1]
        if typ == "str":
            out_dict[key] = value
        elif typ == "int":
            out_dict[key] = int(value)
        elif typ == "float":
            out_dict[key] = float(value)
        else:
            raise TypeError("%s has the unsupported type %s" % (value, typ))
    return out_dict


def dump(file_path, spectrum):
    """ Dump the spectrum to the file_path.

    Args:
      file_path (string): Location to save to.
      spectrum (:class:`spectra.Spectra`): The spectrum to save
    """
    with h5py.File(file_path, "w") as file_:
        file_.attrs["name"] = spectrum.get_name()
        file_.attrs["config_name"] = spectrum.get_config().get_name()
        file_.attrs["config"] = json.dumps(spectrum.get_config().dump())
        if spectrum.get_fit_config():
            file_.attrs["fit_config_name"] = spectrum.get_fit_config().\
                get_name()
            file_.attrs["fit_config"] = json.dumps(
                spectrum.get_fit_config().dump())
        file_.attrs["num_decays"] = spectrum.get_num_decays()
        file_.attrs["raw_events"] = spectrum._raw_events
        file_.attrs["bipo"] = spectrum.get_bipo()
        if len(spectrum.get_style()) == 0:
            file_.attrs["style"] = ""
        else:
            file_.attrs["style"] = json.dumps(spectrum.get_style())
        if len(spectrum._rois) == 0:
            file_.attrs["rois"] = ""
        else:
            file_.attrs["rois"] = json.dumps(spectrum._rois)
        file_.create_dataset("data", data=spectrum._data, compression="gzip")
    _logger.info("Saved spectrum %s to %s" % (spectrum.get_name(), file_path))


def dump_ndarray(file_path, ndarray_object):
    """ Dump any other class, mostly containing numpy arrays.

    Args:
      file_path (string): Location to save to.
      ndarray_object (object): Any class instance mainly consisting of
        numpy array(s).

    Raises:
      AttributeError: If attribute in not an ndarray and is larger than
        64k - h5py limit for attribute sizes.
    """
    with h5py.File(file_path, "w") as file_:
        for attr_name, attribute in ndarray_object.__dict__.iteritems():
            if attribute is None:  # Can't save to hdf5, skip --> continue
                continue
            elif type(attribute).__name__ == "ndarray":
                file_.create_dataset(attr_name, data=attribute,
                                     compression="gzip")
            elif sys.getsizeof(attribute) < 65536:  # 64k
                file_.attrs[attr_name] = attribute
            else:
                raise AttributeError("attribute " + str(attr_name) + " is not "
                                     "an 'ndarray' and is too large to be "
                                     "saved as an h5py attribute.")
    _logger.info("Saved %s to %s" % (str(ndarray_object), file_path))


def dump_summary(file_path, summary):
    """ Dump the limit setting summary to the file_path.

    Args:
      file_path (string): Location to save to.
      summary (:class:`echdina.limit.summary.Summary`): The summary to save
    """
    with h5py.File(file_path, "w") as file_:
        if isinstance(summary, ReducedSummary):
            reduced = True
            file_.attrs["reduced"] = json.dumps(reduced)
        else:
            reduced = False
            file_.attrs["reduced"] = json.dumps(reduced)

        file_.attrs["name"] = summary._name
        file_.attrs["num_scales"] = summary._num_scales
        file_.attrs["spectra_config"] = json.dumps(
            summary._spectra_config.dump())
        file_.attrs["spectra_config_name"] = (
            summary._spectra_config.get_name())
        file_.attrs["fit_config"] = json.dumps(
            summary._fit_config.dump())
        file_.attrs["fit_config_name"] = summary._fit_config.get_name()

        for parameter in summary.get_fit_config().get_pars():
            par = summary.get_fit_config().get_par(parameter)
            if par._values is not None:
                file_.create_dataset(parameter+"_values", data=par._values,
                                     compression="gzip")

        file_.create_dataset("best_fits", data=summary._best_fits,
                             compression="gzip")
        file_.create_dataset("penalty_terms", data=summary._penalty_terms,
                             compression="gzip")
        file_.create_dataset("scales", data=summary._scales,
                             compression="gzip")
        file_.create_dataset("stats", data=summary._stats, compression="gzip")
        file_.create_dataset("priors", data=summary._priors,
                             compression="gzip")
        file_.create_dataset("sigmas", data=summary._sigmas,
                             compression="gzip")

        file_.attrs["limit"] = json.dumps(summary._limit)
        file_.attrs["limit_idx"] = json.dumps(summary._limit_idx)

    _logger.info("Saved summary %s to %s" % (summary.get_name(), file_path))


def load(file_path):
    """ Load a spectrum from file_path.

    Args:
      file_path (string): Location to save to.

    Returns:
      Loaded spectrum (:class:`spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        spec_name = file_.attrs["name"]
        num_decays = file_.attrs["num_decays"]
        config_name = file_.attrs["config_name"]
        config = SpectraConfig.load(json.loads(file_.attrs["config"]),
                                    name=config_name)
        try:
            fit_config_name = file_.attrs["fit_config_name"]
            fit_config = SpectraFitConfig.load(
                json.loads(file_.attrs["fit_config"]),
                spectra_name=spec_name, name=fit_config_name)
        except KeyError as detail:
            _logger.warning("Handling run-time error: %s" % detail)
            logging.getLogger("extra").warning(" --> setting to None")
            fit_config = None

        # Create spectrum
        spec = Spectra(spec_name, num_decays, config, fit_config=fit_config)
        spec._raw_events = file_.attrs["raw_events"]
        try:
            spec._bipo = file_.attrs["bipo"]
        except KeyError as detail:
            _logger.warning("Handling run-time error: %s" % detail)
            logging.getLogger("extra").warning(" --> setting to 0")
            spec._bipo = 0
        style_dict = file_.attrs["style"]
        if len(style_dict) > 0:
            spec._style = json.loads(style_dict)
        rois_dict = file_.attrs["rois"]
        if len(rois_dict) > 0:
            spec._rois = json.loads(rois_dict)
        # else the default values of Spectra __init__ are kept

        spec._data = file_["data"].value

    _logger.info("Loaded spectrum %s" % spec.get_name())
    return spec


def load_old(file_path):
    """ Load a spectra from file_path.
    Args:
      file_path (string): Location to save to.
    Returns:
      Loaded spectra (:class:`spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        parameters = collections.OrderedDict()
        parameters["energy"] = SpectraParameter("energy",
                                                file_.attrs["energy_low"],
                                                file_.attrs["energy_high"],
                                                file_.attrs["energy_bins"])
        parameters["radial"] = SpectraParameter("radial",
                                                file_.attrs["radial_low"],
                                                file_.attrs["radial_high"],
                                                file_.attrs["radial_bins"])
        parameters["time"] = SpectraParameter("time",
                                              file_.attrs["time_low"],
                                              file_.attrs["time_high"],
                                              file_.attrs["time_bins"])
        spectra_config = SpectraConfig(parameters)
        spec = Spectra(file_.attrs["name"],
                       file_.attrs["num_decays"],
                       spectra_config)
        # The following may not be present in old, old hdf5s
        if file_.attrs["raw_events"]:
            spec._raw_events = file_.attrs["raw_events"]
        if file_.attrs["style"]:
            spec._style = file_.attrs["style"]
        if file_.attrs["rois"]:
            spec._rois = file_.attrs["rois"]
        spec._data = file_["data"].value
    return spec


def load_ndarray(file_path, ndarray_object):
    """ Dump any other class, mostly containing numpy arrays.

    Args:
      file_path (string): Location to load class attributes from.
      array_object (object): Any class instance mainly consisting of
        numpy array(s).
    """
    with h5py.File(file_path, "r") as file_:
        for attr_name, attribute in ndarray_object.__dict__.iteritems():
            try:
                if type(attribute).__name__ == "ndarray":
                    setattr(ndarray_object, attr_name, file_[attr_name].value)
                else:
                    setattr(ndarray_object, attr_name, file_.attrs[attr_name])
            except KeyError as detail:  # unable to locate attribute, skip
                _logger.warning("Handling run-time error: %s" % detail)
                logging.getLogger("extra").warning(" --> skipping")
                continue
    _logger.info("Loaded object %s" % str(ndarray_object))
    return ndarray_object


def load_summary(file_path):
    """ Load a limit setting summary from file_path.

    Args:
      file_path (string): Location to load from.

    Returns:
      :class:`Summary`: The Summary object.
    """
    with h5py.File(file_path, "r") as file_:
        if json.loads(file_.attrs["reduced"]):  # use ReducedSummary
            reduced = True
        else:
            reduced = False

        name = file_.attrs["name"]
        num_scales = file_.attrs["num_scales"]
        if not reduced:  # only required for full summary
            spectra_config_name = file_.attrs["spectra_config_name"]
            spectra_config = SpectraConfig.load(
                json.loads(file_.attrs["spectra_config"]),
                name=spectra_config_name)
        fit_config_name = file_.attrs["fit_config_name"]
        fit_config = GlobalFitConfig.load(
            json.loads(file_.attrs["fit_config"])[0],
            spectral_config=json.loads(file_.attrs["fit_config"])[1],
            name=fit_config_name)
        for parameter in fit_config.get_pars():
            par = fit_config.get_par(parameter)
            try:
                par._values = file_[parameter+"_values"].value
            except KeyError as detail:  # unable to locate attribute, skip
                _logger.warning("Handling run-time error: %s" % detail)
                logging.getLogger("extra").warning(" --> skipping")

        if reduced:
            summary = ReducedSummary(name, num_scales, fit_config)
        else:
            summary = Summary(name, num_scales, spectra_config, fit_config)

        summary.set_best_fits(file_["best_fits"].value)
        summary.set_penalty_terms(file_["penalty_terms"].value)
        summary.set_priors(file_["priors"].value)
        summary.set_scales(file_["scales"].value)
        summary.set_sigmas(file_["sigmas"].value)
        summary.set_stats(file_["stats"].value)

        summary.set_limit(json.loads(file_.attrs["limit"]))
        summary.set_limit_idx(json.loads(file_.attrs["limit_idx"]))

    _logger.info("Loaded summary %s" % summary.get_name())
    return summary
