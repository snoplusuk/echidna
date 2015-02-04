import h5py
import echidna.core.spectra


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

        file_.create_dataset("data", data=spectra._data, compression="gzip")


def load(file_path):
    """ Load a spectra from file_path.

    Args:
      file_path (string): Location to save to.

    Returns:
      Loaded spectra (:class:`echidna.core.spectra.Spectra`).
    """
    with h5py.File(file_path, "r") as file_:
        spectra = echidna.core.spectra.Spectra(file_.attrs["name"])
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
