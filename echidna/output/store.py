import numpy

import echidna.core.spectra as spectra
import echidna.limit.limit_setting as limit_setting

import h5py
import sys
import collections


def dump(file_path, spectra):
    """ Dump the spectra to the file_path.

    Args:
      file_path (string): Location to save to.
      spectra (:class:`spectra.Spectra`): The spectra to save
    """

    with h5py.File(file_path, "w") as file_:
        file_.attrs["name"] = spectra._name
        # Store parameters with a key word 'pars'
        for v in spectra.get_config().getpars():
            file_.attrs["pars:%s:low" % v]  = spectra.get_config().getpar(v).low  
            file_.attrs["pars:%s:high" % v]  = spectra.get_config().getpar(v).high
            file_.attrs["pars:%s:bins" % v]  = spectra.get_config().getpar(v).bins
        file_.attrs["num_decays"] = spectra._num_decays

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
            print attr_name
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
        for v in file_.attrs:
            print v
            if v.startswith("pars:"):
                [_, par, val] = v.split(":")
                if par not in parameters:
                    parameters[par] = spectra.SpectraParameter(par, 1, 1, 1)
                parameters[par].setvar(**{val: file_.attrs[v]})
                
        spec = spectra.Spectra(file_.attrs["name"],
                               file_.attrs["num_decays"],
                               spectra.SpectraConfig(parameters))
        spec._data = file_["data"].value
    print spec.get_config().getpars()
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
        parameters["energy"] = spectra.SpectraParameter("energy",
                                                        file_.attrs["energy_low"],
                                                        file_.attrs["energy_high"],
                                                        file_.attrs["energy_bins"])
        parameters["radial"] = spectra.SpectraParameter("radial",
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
            print attr_name
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
