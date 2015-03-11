import numpy

import echidna.core.spectra
import echidna.limit.limit_setting as limit_setting

import h5py
import sys


def dump(file_path, spectra):
    """ Dump the spectra to the file_path.

    Args:
      file_path (string): Location to save to.
      spectra (:class:`echidna.core.spectra.Spectra`): The spectra to save
    """

    with h5py.File(file_path, "w") as file_:
        file_.attrs["name"] = spectra._name
        file_.attrs["energy_low"] = spectra._energy_low
        file_.attrs["energy_high"] = spectra._energy_high
        file_.attrs["energy_bins"] = spectra._energy_bins
        file_.attrs["energy_width"] = spectra._energy_width
        file_.attrs["radial_low"] = spectra._radial_low
        file_.attrs["radial_high"] = spectra._radial_high
        file_.attrs["radial_bins"] = spectra._radial_bins
        file_.attrs["radial_width"] = spectra._radial_width
        file_.attrs["time_low"] = spectra._time_low
        file_.attrs["time_high"] = spectra._time_high
        file_.attrs["time_bins"] = spectra._time_bins
        file_.attrs["time_width"] = spectra._time_width
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
            if type(attribute).__name__ == "ndarray":
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
      Loaded spectra (:class:`echidna.core.spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        spectra = echidna.core.spectra.Spectra(file_.attrs["name"],
                                               file_.attrs["num_decays"])
        spectra._energy_low = file_.attrs["energy_low"]
        spectra._energy_high = file_.attrs["energy_high"]
        spectra._energy_bins = file_.attrs["energy_bins"]
        spectra._energy_width = file_.attrs["energy_width"]
        spectra._radial_low = file_.attrs["radial_low"]
        spectra._radial_high = file_.attrs["radial_high"]
        spectra._radial_bins = file_.attrs["radial_bins"]
        spectra._radial_width = file_.attrs["radial_width"]
        spectra._time_low = file_.attrs["time_low"]
        spectra._time_high = file_.attrs["time_high"]
        spectra._time_bins = file_.attrs["time_bins"]
        spectra._time_width = file_.attrs["time_width"]

        spectra._data = file_["data"].value
    return spectra


def load_ndarray(file_path, ndarray_object):
    """ Dump any other class, mostly containing numpy arrays.

    Args:
      file_path (string): Location to load class attributes from.
      array_object (object): Any class instance mainly consisting of
        numpy array(s).
    """
    with h5py.File(file_path, "r") as file_:
        for attr_name, attribute in ndarray_object.__dict__.iteritems():
            if type(attribute).__name__ == "ndarray":
                setattr(ndarray_object, attr_name, file_[attr_name].value)
            else:
                setattr(ndarray_object, attr_name, file_.attrs[attr_name])
    return ndarray_object
