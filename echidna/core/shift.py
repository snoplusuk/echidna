import echidna.core.spectra as spectra


class Shift(object):
    """ A class for shifting the parameter space of a spectra.

    Attributes:
      _shift (float): The factor you want to shift a parameter by.
    """

    def __init__(self):
        """ Initialise the Shift class.
        """
        self._shift = 0.

    def get_shift(self):
        """ Returns the shift factor.

        Returns:
          float: The shift factor.
        """
        return self._shift

    def set_shift(self, shift):
        """ Sets the shift factor.

        Args:
          shift (float): Value you wish to set the shift factor to.
        """
        self._shift = float(shift)

    def shift(self, spectrum, dimension, **kwargs):
        """ Shifts a given spectrum's dimension using interpolation.

        Args:
          spectrum (float): The spectrum you want to shift.
          dimension (string): The dimension of the spectrum you want to shift.
          kwargs (dict): To passed to the interpolation function in
            :class:`echidna.core.spectra.Spectra`

        Returns:
          :class:`echidna.core.spectra.Spectra`: The shifted spectrum.
        """
        shift = self.get_shift()
        step = spectrum.get_config().get_par(dimension).get_width()
        if shift % step == 0.:
            # shift size multiple of step size. Interpolation not required.
            return self.shift_by_bin(spectrum, dimension)
        preshift_sum = spectrum.sum()
        interpolation = spectrum.interpolate1d(dimension, **kwargs)
        shifted_spec = spectra.Spectra(spectrum._name+"_shift" +
                                       str(shift),
                                       spectrum._num_decays,
                                       spectrum.get_config())
        n_dim = len(spectrum._data.shape)
        axis = spectrum.get_config().get_index(dimension)
        low = spectrum.get_config().get_par(dimension)._low
        high = spectrum.get_config().get_par(dimension)._high
        n_bins = spectrum.get_config().get_par(dimension)._bins
        for bin in range(n_bins):
            x = low + (bin + 0.5) * step
            if (x - shift) < low or (x - shift) > high:
                continue  # Trying to shift values outside range (Unknown)
            old_bin = spectrum.get_config().get_par(dimension).get_bin(x - 
                                                                       shift)
            y = interpolation(x - shift)
            # Prepare array split. Is there a better way to do this not using
            # eval and exec?
            cur_slice = "["
            old_slice = "["
            for dim in range(n_dim):
                if dim == axis:
                    if bin < n_bins - 1:
                        cur_slice += str(bin) + ":" + str(bin + 1) + ","
                    else:
                        cur_slice += str(bin) + ":,"
                    if old_bin < n_bins - 1:
                        old_slice += str(old_bin) + ":" + str(old_bin + 1)+","
                    else:
                        old_slice += str(old_bin) + ":,"
                else:
                    cur_slice += ":,"
                    old_slice += ":,"
            cur_slice = cur_slice[:-1] + "]"
            old_slice = old_slice[:-1] + "]"
            old_data = eval("spectrum._data"+old_slice)
            unshifted_sum = float(old_data.sum())
            # Check to see if there is data to shift and counts is positive
            if unshifted_sum > 0. and y > 0.:
                fill_cmd = ("shifted_spec._data" + cur_slice + "+= old_data * "
                            "(y / unshifted_sum)")
                exec(fill_cmd)
        # renormalise to prescale number of counts
        shifted_spec._num_decays = shifted_spec.sum()
        shifted_spec.scale(preshift_sum)
        shifted_spec._num_decays = spectrum._num_decays
        return shifted_spec

    def shift_by_bin(self, spectrum, dimension):
        """ Shifts a given spectrum's dimension by shifting bins.

        Args:
          spectrum (float): The spectrum you want to shift.
          dimension (string): The dimension of the spectrum you want to shift.
          kwargs (dict): To passed to the interpolation function in
            :class:`echidna.core.spectra.Spectra`

        Returns:
          :class:`echidna.core.spectra.Spectra`: The shifted spectrum.
        """
        shift = self.get_shift()
        step = spectrum.get_config().get_par(dimension).get_width()
        if shift % step != 0.:
            raise ValueError("Shift (%s) must be a multiple of bin width (%s)"
                             % (shift, step))
        shifted_spec = spectra.Spectra(spectrum._name+"_shift" +
                                       str(shift),
                                       spectrum._num_decays,
                                       spectrum.get_config())
        n_dim = len(spectrum._data.shape)
        axis = spectrum.get_config().get_index(dimension)
        low = spectrum.get_config().get_par(dimension)._low
        high = spectrum.get_config().get_par(dimension)._high
        n_bins = spectrum.get_config().get_par(dimension)._bins
        for bin in range(n_bins):
            x = low + (bin + 0.5) * step
            if (x - shift) < low or (x - shift) > high:
                continue  # Trying to shift values outside range (Unknown)
            old_bin = spectrum.get_config().get_par(dimension).get_bin(x - 
                                                                       shift)
            # Prepare array split. Is there a better way to do this not using
            # eval and exec?
            cur_slice = "["
            old_slice = "["
            for dim in range(n_dim):
                if dim == axis:
                    if bin < n_bins - 1:
                        cur_slice += str(bin) + ":" + str(bin + 1) + ","
                    else:
                        cur_slice += str(bin) + ":,"
                    if old_bin < n_bins - 1:
                        old_slice += str(old_bin) + ":" + str(old_bin + 1)+","
                    else:
                        old_slice += str(old_bin) + ":,"
                else:
                    cur_slice += ":,"
                    old_slice += ":,"
            cur_slice = cur_slice[:-1] + "]"
            old_slice = old_slice[:-1] + "]"
            old_data = eval("spectrum._data"+old_slice)
            unshifted_sum = float(old_data.sum())
            # Check to see if there is data
            if unshifted_sum > 0.:
                fill_cmd = "shifted_spec._data" + cur_slice + "+= old_data"
                exec(fill_cmd)
        return shifted_spec
