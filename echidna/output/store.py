import numpy

from echidna.core.spectra import Spectra
from echidna.core.config import (SpectraConfig, SpectraParameter,
                                 SpectraFitConfig, GlobalFitConfig)
from echidna.limit.summary import Summary, ReducedSummary
from echidna.fit.fit_results import FitResults, LimitResults
from echidna.fit.minimise import GridSearch

from collections import OrderedDict
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


def dump(file_path, spectrum, append=False, overwrite=False):
    """ Dump the spectrum to the file_path.

    Args:
      file_path (string): Location to save to.
      spectrum (:class:`spectra.Spectra`): The spectrum to save
    """
    group_name = "spectrum"
    if append:
        file_opt = "a"
    else:
        file_opt = "w"
    with h5py.File(file_path, file_opt) as file_:
        if overwrite and group_name in file_.keys():  # Delete existing group
            _logger.warning("Removing existing group %s" % group_name)
            del file_[group_name]
        group = file_.create_group(group_name)
        group.attrs["name"] = spectrum.get_name()
        group.attrs["config_name"] = spectrum.get_config().get_name()
        group.attrs["config"] = json.dumps(spectrum.get_config().dump())
        if spectrum.get_fit_config():
            group.attrs["fit_config_name"] = spectrum.get_fit_config().\
                get_name()
            group.attrs["fit_config"] = json.dumps(
                spectrum.get_fit_config().dump())
        group.attrs["num_decays"] = spectrum.get_num_decays()
        group.attrs["raw_events"] = spectrum._raw_events
        group.attrs["bipo"] = spectrum.get_bipo()
        if len(spectrum.get_style()) == 0:
            group.attrs["style"] = ""
        else:
            group.attrs["style"] = json.dumps(spectrum.get_style())
        if len(spectrum._rois) == 0:
            group.attrs["rois"] = ""
        else:
            group.attrs["rois"] = json.dumps(spectrum._rois)
        group.create_dataset("data", data=spectrum._data, compression="gzip")
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


def dump_summary(file_path, summary, append=False):
    """ Dump the limit setting summary to the file_path.

    Args:
      file_path (string): Location to save to.
      summary (:class:`echdina.limit.summary.Summary`): The summary to save
    """
    group_name = "summary"
    if append:
        file_opt = "a"
    else:
        file_opt = "w"
    with h5py.File(file_path, file_opt) as file_:
        group = file_.create_group(group_name)
        if isinstance(summary, ReducedSummary):
            reduced = True
            group.attrs["reduced"] = reduced
        else:
            reduced = False
            group.attrs["reduced"] = reduced

        group.attrs["name"] = summary._name
        group.attrs["num_scales"] = summary._num_scales
        group.attrs["spectra_config"] = json.dumps(
            summary._spectra_config.dump())
        group.attrs["spectra_config_name"] = summary._spectra_config.get_name()
        group.attrs["fit_config"] = json.dumps(
            summary._fit_config.dump())
        group.attrs["fit_config_name"] = summary._fit_config.get_name()

        for parameter in summary.get_fit_config().get_pars():
            par = summary.get_fit_config().get_par(parameter)
            if par._values is not None:
                group.create_dataset(parameter+"_values", data=par._values,
                                     compression="gzip")

        # Write best fits array if it exists and is not empty
        if (summary._best_fits is not None and
                numpy.any(summary._best_fits)):
            group.create_dataset("best_fits", data=summary._best_fits,
                                 compression="gzip")

        # Write penalty terms array if it exists and is not empty
        if (summary._penalty_terms is not None and
                numpy.any(summary._penalty_terms)):
            group.create_dataset("penalty_terms", data=summary._penalty_terms,
                                 compression="gzip")

        group.create_dataset("scales", data=summary._scales,
                             compression="gzip")
        group.create_dataset("stats", data=summary._stats, compression="gzip")

        # Write priors array if it exists and is not empty
        if (summary._priors is not None and
                numpy.any(summary._priors)):
            group.create_dataset("priors", data=summary._priors,
                                 compression="gzip")

        # Write sigmas array if it exists and is not empty
        if (summary._sigmas is not None and
                numpy.any(summary._sigmas)):
            group.create_dataset("sigmas", data=summary._sigmas,
                                 compression="gzip")

        group.attrs["limit"] = summary._limit
        group.attrs["limit_idx"] = summary._limit_idx

    _logger.info("Saved summary %s to %s" % (summary.get_name(), file_path))


def dump_fit_results(file_path, fit_results, append=False):
    """ Dump the fit results to the specified file_path.

    Args:
      file_path (string): Location to save to.
      summary (:class:`echdina.fit.fit_results.FitResults`): The
        FitResults to save.
      append (bool, optional): Append to existing hdf5 file.
    """
    group_name = "fit_results"
    if append:
        file_opt = "a"
    else:
        file_opt = "w"
    with h5py.File(file_path, file_opt) as file_:
        group = file_.create_group(group_name)
        group.attrs["name"] = fit_results._name
        group.attrs["spectra_config"] = json.dumps(
            fit_results._spectra_config.dump())
        group.attrs["spectra_config_name"] = (
            fit_results._spectra_config.get_name())
        group.attrs["fit_config"] = json.dumps(
            fit_results._fit_config.dump())
        group.attrs["fit_config_name"] = fit_results._fit_config.get_name()

        group.create_dataset("penalty_terms", data=fit_results._penalty_terms,
                             compression="gzip")
        group.create_dataset("stats", data=fit_results._stats,
                             compression="gzip")

        if fit_results._minimum_value:
            group.attrs["minimum_value"] = fit_results._minimum_value
        if fit_results._minimum_position:
            group.attrs["minimum_position"] = fit_results._minimum_position
        group.attrs["resets"] = fit_results._resets

    _logger.info("Saved fit results %s to %s" %
                 (fit_results.get_name(), file_path))


def dump_limit_results(file_path, limit_results, append=False):
    """ Dump the limit results to the specified file_path.

    Args:
      file_path (string): Location to save to.
      limit_results (:class:`echdina.fit.fit_results.LimitResults`): The
        LimitResults to save.
      append (bool, optional): Append to existing hdf5 file.
    """
    group_name = "limit_results"
    if append:
        file_opt = "a"
    else:
        file_opt = "w"
    with h5py.File(file_path, file_opt) as file_:
        group = file_.create_group(group_name)
        group.attrs["name"] = limit_results._name
        group.attrs["spectra_config"] = json.dumps(
            limit_results._spectra_config.dump())
        group.attrs["spectra_config_name"] = (
            limit_results._spectra_config.get_name())
        group.attrs["fit_config"] = json.dumps(
            limit_results._fit_config.dump())
        group.attrs["fit_config_name"] = limit_results._fit_config.get_name()
        group.attrs["limit_config"] = json.dumps(
            limit_results._limit_config.dump())
        group.attrs["limit_config_name"] = \
            limit_results._limit_config.get_name()
        group.attrs["limit_config_spectra_name"] = \
            limit_results._limit_config._spectra_name
        if limit_results._penalty_terms.any():
            group.create_dataset("penalty_terms",
                                 data=limit_results._penalty_terms,
                                 compression="gzip")
        if limit_results._best_fits.any():
            group.create_dataset("best_fits",
                                 data=limit_results._best_fits,
                                 compression="gzip")
        group.create_dataset("stats", data=limit_results._stats,
                             compression="gzip")
        if limit_results._fit_results.any():
            group.attrs["fits_exist"] = 1.
            for i, fit_result in enumerate(limit_results._fit_results):
                sub_group = group.create_group(str(i))
                if type(fit_result) is GridSearch:
                    sub_group.attrs["exists"] = 1.
                    sub_group.create_dataset("penalty_terms",
                                             data=fit_result._penalty_terms,
                                             compression="gzip")
                    sub_group.create_dataset("stats", data=fit_result._stats,
                                             compression="gzip")
                    sub_group.attrs["resets"] = fit_result._resets
                else:
                    sub_group.attrs["exists"] = 0.
        else:
            group.attrs["fits_exist"] = 0.
    _logger.info("Saved limit results %s to %s" %
                 (limit_results.get_name(), file_path))


def load(file_path):
    """ Load a spectrum from file_path.

    Args:
      file_path (string): Location to save to.

    Returns:
      Loaded spectrum (:class:`spectra.Spectra`).
    """
    group_name = "spectrum"
    try:
        with h5py.File(file_path, "r") as file_:
            group = file_[group_name]
            spec_name = group.attrs["name"]
            num_decays = group.attrs["num_decays"]
            config_name = group.attrs["config_name"]
            config = SpectraConfig.load(
                json.loads(group.attrs["config"],
                           object_pairs_hook=OrderedDict),
                name=config_name)
            try:
                fit_config_name = group.attrs["fit_config_name"]
                fit_config = SpectraFitConfig.load(
                    json.loads(
                        group.attrs["fit_config"],
                        object_pairs_hook=OrderedDict),
                    spectra_name=spec_name, name=fit_config_name)
            except KeyError as detail:
                _logger.warning("Handling run-time error: %s" % detail)
                logging.getLogger("extra").warning(" --> setting to None")
                fit_config = None

            # Create spectrum
            spec = Spectra(spec_name, num_decays,
                           config, fit_config=fit_config)
            spec._raw_events = group.attrs["raw_events"]
            try:
                spec._bipo = group.attrs["bipo"]
            except KeyError as detail:
                _logger.warning("Handling run-time error: %s" % detail)
                logging.getLogger("extra").warning(" --> setting to 0")
                spec._bipo = 0
            style_dict = group.attrs["style"]
            if len(style_dict) > 0:
                spec._style = json.loads(
                    style_dict, object_pairs_hook=OrderedDict)
            rois_dict = group.attrs["rois"]
            if len(rois_dict) > 0:
                spec._rois = json.loads(
                    rois_dict, object_pairs_hook=OrderedDict)
            # else the default values of Spectra __init__ are kept

            spec._data = group["data"].value
            spec._location = file_path
        _logger.info("Loaded spectrum %s" % spec.get_name())
        return spec
    except KeyError as detail:
        _logger.warning("Recieved KeyError: %s" % detail)
        logging.getLogger("extra").warning(" --> attempting to load old-style")
        return _load_old(file_path)


def _load_old(file_path):
    """ Load a spectra from file_path.

    Args:
      file_path (string): Location to save to.

    Returns:
      Loaded spectra (:class:`spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        parameters = collections.OrderedDict()
        for v in file_.attrs:
            if v.startswith("pars:"):
                [_, par, val] = v.split(":")
                if par not in parameters:
                    parameters[str(par)] = SpectraParameter(par, 1., 1., 1)
                parameters[str(par)].set_par(**{val: float(file_.attrs[v])})
        spec = Spectra(file_.attrs["name"],
                       file_.attrs["num_decays"],
                       SpectraConfig("spectral_config", parameters))
        spec._raw_events = file_.attrs["raw_events"]
        try:
            spec._bipo = file_.attrs["bipo"]
        except:
            spec._bipo = 0
        style_dict = file_.attrs["style"]
        if len(style_dict) > 0:
            spec._style = string_to_dict(style_dict)
        rois_dict = file_.attrs["rois"]
        if len(rois_dict) > 0:
            spec._rois = string_to_dict(rois_dict)
        # else the default values of Spectra __init__ are kept
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
    group_name = "summary"
    with h5py.File(file_path, "r") as file_:
        group = file_[group_name]
        if json.loads(group.attrs["reduced"], object_pairs_hook=OrderedDict):
            reduced = True
        else:
            reduced = False

        name = group.attrs["name"]
        num_scales = group.attrs["num_scales"]
        spectra_config_name = group.attrs["spectra_config_name"]
        spectra_config = SpectraConfig.load(
            json.loads(group.attrs["spectra_config"],
                       object_pairs_hook=OrderedDict),
            name=spectra_config_name)
        fit_config_name = group.attrs["fit_config_name"]
        fit_config = GlobalFitConfig.load(
            json.loads(group.attrs["fit_config"],
                       object_pairs_hook=OrderedDict)[0],
            spectral_config=json.loads(group.attrs["fit_config"],
                                       object_pairs_hook=OrderedDict)[1],
            name=fit_config_name)
        for parameter in fit_config.get_pars():
            par = fit_config.get_par(parameter)
            try:
                par._values = group[parameter+"_values"].value
            except KeyError as detail:  # unable to locate attribute, skip
                _logger.warning("Handling run-time error: %s" % detail)
                logging.getLogger("extra").warning(" --> skipping")

        if reduced:
            summary = ReducedSummary(name, num_scales,
                                     spectra_config, fit_config)
        else:
            summary = Summary(name, num_scales, spectra_config, fit_config)

        summary.set_best_fits(group["best_fits"].value)
        summary.set_penalty_terms(group["penalty_terms"].value)
        summary.set_priors(group["priors"].value)
        summary.set_scales(group["scales"].value)
        summary.set_sigmas(group["sigmas"].value)
        summary.set_stats(group["stats"].value)

        summary.set_limit(json.loads(group.attrs["limit"],
                                     object_pairs_hook=OrderedDict))
        summary.set_limit_idx(json.loads(group.attrs["limit_idx"],
                                         object_pairs_hook=OrderedDict))

    _logger.info("Loaded summary %s" % summary.get_name())
    return summary


def load_fit_results(file_path):
    """ Load a :class:`FitResults` object from file.

    Args:
      file_path (string): Location from which to load :class:`FitResults`.

    Returns:
      :class:`FitResults`: The loaded fit results object.
    """
    group_name = "fit_results"
    with h5py.File(file_path, "r") as file_:
        group = file_[group_name]

        name = group.attrs["name"]
        spectra_config_name = group.attrs["spectra_config_name"]
        spectra_config = SpectraConfig.load(
            json.loads(group.attrs["spectra_config"],
                       object_pairs_hook=OrderedDict),
            name=spectra_config_name)
        fit_config_name = group.attrs["fit_config_name"]
        fit_config = GlobalFitConfig.load(
            json.loads(group.attrs["fit_config"],
                       object_pairs_hook=OrderedDict)[0],
            spectral_config=json.loads(group.attrs["fit_config"],
                                       object_pairs_hook=OrderedDict)[1],
            name=fit_config_name)
        fit_results = GridSearch(fit_config=fit_config,
                                 spectra_config=spectra_config, name=name)

        fit_results.set_penalty_terms(group["penalty_terms"].value)
        fit_results.set_stats(group["stats"].value)

        try:
            fit_results.set_minimum_position(group.attrs["minimum_position"])
        except:
            fit_results.set_minimum_position(None)
        try:
            fit_results.set_minimum_value(group.attrs["minimum_value"])
        except:
            fit_results.set_minimum_value(None)
        fit_results._resets = group.attrs["resets"]

    _logger.info("Loaded FitResults %s" % fit_results.get_name())
    return fit_results


def load_limit_results(file_path):
    """ Load a :class:`LimitResults` object from file.

    Args:
      file_path (string): Location from which to load :class:`FitResults`.

    Returns:
      :class:`FitResults`: The loaded fit results object.
    """
    group_name = "limit_results"
    with h5py.File(file_path, "r") as file_:
        group = file_[group_name]
        name = group.attrs["name"]
        spectra_config_name = group.attrs["spectra_config_name"]
        spectra_config = SpectraConfig.load(
            json.loads(group.attrs["spectra_config"],
                       object_pairs_hook=OrderedDict),
            name=spectra_config_name)
        fit_config_name = group.attrs["fit_config_name"]
        global_config = json.loads(group.attrs["fit_config"],
                                   object_pairs_hook=OrderedDict)[0]
        spectral_config = json.loads(group.attrs["fit_config"],
                                     object_pairs_hook=OrderedDict)[1]
        if spectral_config["spectral_fit_parameters"]:
            fit_config = GlobalFitConfig.load(global_config,
                                              spectral_config=spectral_config,
                                              name=fit_config_name)
        else:
            fit_config = GlobalFitConfig.load(global_config,
                                              name=fit_config_name)
        limit_config = json.loads(group.attrs["limit_config"],
                                  object_pairs_hook=OrderedDict)
        limit_config_name = group.attrs["limit_config_name"]
        limit_config_spectra_name = group.attrs["limit_config_spectra_name"]
        limit_config = SpectraFitConfig.load(limit_config,
                                             limit_config_spectra_name,
                                             name=limit_config_name)
        limit_results = LimitResults(fit_config, spectra_config, limit_config,
                                     name)
        try:
            limit_results._penalty_terms = group["penalty_terms"].value
        except:
            pass
        limit_results._stats = group["stats"].value
        try:
            limit_results._best_fits = group["best_fits"].value
        except:
            pass
        if group.attrs["fits_exist"] == 1.:
            for i in range(len(limit_results._stats)):
                sub_group_name = group_name+"/"+str(i)
                sub_group = file_[sub_group_name]
                if sub_group.attrs["exists"] == 1.:
                    fit_results = GridSearch(fit_config=fit_config,
                                             spectra_config=spectra_config,
                                             name=fit_config_name)
                    fit_results._stats = sub_group["stats"].value
                    fit_results._resets = sub_group.attrs["resets"]
                    fit_results._penalty_terms = sub_group[
                        "penalty_terms"].value
                    limit_results._fit_results[i] = fit_results

    _logger.info("Loaded LimitResults %s" % limit_results.get_name())
    return limit_results
