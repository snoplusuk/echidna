import echidna.core.spectra as spectra

import h5py
import sys
import collections


def dict_to_string(in_dict):
    """ Converts a dicionary to a string so it can be saved in a hdf5 file.

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
    """ Converts a string to a dictionary so it can
      be loaded from the hdf5 file.

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


def dump(file_path, spectra):
    """ Dump the spectra to the file_path.

    Args:
      file_path (string): Location to save to.
      spectra (:class:`spectra.Spectra`): The spectra to save
    """
    with h5py.File(file_path, "w") as file_:
        file_.attrs["name"] = spectra._name
        # Store parameters with a key word 'pars'
        for v in spectra.get_config().get_pars():
            file_.attrs["pars:%s:low" % v] = \
                spectra.get_config().get_par(v)._low
            file_.attrs["pars:%s:high" % v] = \
                spectra.get_config().get_par(v)._high
            file_.attrs["pars:%s:bins" % v] = \
                spectra.get_config().get_par(v)._bins
        if spectra.get_fit_config():
            for par in spectra.get_fit_config().get_pars():
                file_.attrs["fit_pars:%s:prior" % par] = \
                    spectra.get_fit_config().get_par(par)._prior
                file_.attrs["fit_pars:%s:sigma" % par] = \
                    spectra.get_fit_config().get_par(par)._sigma
                file_.attrs["fit_pars:%s:low" % par] = \
                    spectra.get_fit_config().get_par(par)._low
                file_.attrs["fit_pars:%s:high" % par] = \
                    spectra.get_fit_config().get_par(par)._high
                file_.attrs["fit_pars:%s:bins" % par] = \
                    spectra.get_fit_config().get_par(par)._bins
                file_.attrs["fit_pars:%s:logscale" % par] = \
                    spectra.get_fit_config().get_par(par)._logscale
                file_.attrs["fit_pars:%s:base" % par] = \
                    spectra.get_fit_config().get_par(par)._base
        file_.attrs["num_decays"] = spectra._num_decays
        file_.attrs["raw_events"] = spectra._raw_events
        file_.attrs["bipo"] = spectra._bipo
        if len(spectra._style) == 0:
            file_.attrs["style"] = ""
        else:
            file_.attrs["style"] = dict_to_string(spectra._style)
        if len(spectra._rois) == 0:
            file_.attrs["rois"] = ""
        else:
            file_.attrs["rois"] = dict_to_string(spectra._rois)
        file_.create_dataset("data", data=spectra._data, compression="gzip")


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


def load(file_path):
    """ Load a spectra from file_path.

    Args:
      file_path (string): Location to save to.

    Returns:
      Loaded spectra (:class:`spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        parameters = collections.OrderedDict()
        fit_parameters = collections.OrderedDict()
        for key, value in file_.attrs.iteritems():
            if key.startswith("pars:"):
                [_, par, attr] = key.split(":")
                if par not in parameters:
                    parameters[str(par)] = spectra.SpectraParameter(
                        par, 1., 1., 1)
                parameters[str(par)].set_par(**{attr: float(value)})
            if key.startswith("fit_pars:"):
                [_, par, attr] = key.split(":")
                if par not in fit_parameters:
                    # create RateParameter instance with all values as 0
                    fit_parameters[str(par)] = spectra.RateParameter(
                        par, 0., 0., 0., 0., 0.)
                # Fill correct values
                fit_parameters[str(par)].set_par(**{attr: value})
        spec_name = file_.attrs["name"]
        spec = spectra.Spectra(
            name=spec_name, num_decays=file_.attrs["num_decays"],
            spectra_config=spectra.SpectraConfig(parameters),
            fit_config=spectra.SpectraFitConfig(fit_parameters, spec_name))
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


def load_old(file_path):
    """ Load a spectra from file_path.
    Args:
      file_path (string): Location to save to.
    Returns:
      Loaded spectra (:class:`spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        parameters = collections.OrderedDict()
        parameters["energy"] = \
            spectra.SpectraParameter("energy",
                                     file_.attrs["energy_low"],
                                     file_.attrs["energy_high"],
                                     file_.attrs["energy_bins"])
        parameters["radial"] = \
            spectra.SpectraParameter("radial",
                                     file_.attrs["radial_low"],
                                     file_.attrs["radial_high"],
                                     file_.attrs["radial_bins"])
        parameters["time"] = spectra.SpectraParameter("time",
                                                      file_.attrs["time_low"],
                                                      file_.attrs["time_high"],
                                                      file_.attrs["time_bins"])
        spectra_config = spectra.SpectraConfig(parameters)
        spec = spectra.Spectra(file_.attrs["name"],
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
                print detail
                print " --> skipping!"
                continue
    return ndarray_object
