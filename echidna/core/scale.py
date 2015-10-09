import echidna.core.spectra as spectra


class Scale(object):
    """ A class for scaling the parameter space of a spectra.

    Attributes:
      _scale_factor (float): The factor you want to scale a parameter by.
    """

    def __init__(self):
        """ Initialise the Scale class.
        """
        self._scale_factor = 1.

    def get_scale_factor(self):
        """ Returns the scale factor.

        Returns:
          float: The scale factor.
        """
        return self._scale_factor

    def set_scale_factor(self, scale_factor):
        """ Sets the scale factor.

        Args:
          scale_factor (float): Value you wish to set the scale factor to.

       Raises:
         ValueError: If scale_factor is zero or below.
        """
        if scale_factor <= 0.:
            raise ValueError("Scale factor must be positive and non-zero.")
        self._scale_factor = scale_factor

    def scale(self, spectrum, dimension, **kwargs):
        """ Scales a given spectrum's dimension.

        Args:
          spectrum (float): The spectrum you want to scale.
          dimension (string): The dimension of the spectrum you want to scale.
          kwargs (dict): To passed to the interpolation function in
            :class:`echidna.core.spectra.Spectra`

        Returns:
          :class:`echidna.core.spectra.Spectra`: The scaled spectrum.
        """
        prescale_sum = spectrum.sum()
        interpolation = spectrum.interpolate1d(dimension, **kwargs)
        sf = self.get_scale_factor()
        scaled_spec = spectra.Spectra(spectrum._name+"_sf" +
                                      str(sf),
                                      spectrum._num_decays,
                                      spectrum.get_config())
        n_dim = len(spectrum._data.shape)
        axis = spectrum.get_config().get_index(dimension)
        low = spectrum.get_config().get_par(dimension)._low
        n_bins = spectrum.get_config().get_par(dimension)._bins
        step = spectrum.get_config().get_par(dimension).get_width()
        for bin in range(n_bins):
            x = low + (bin + 0.5) * step
            old_bin = spectrum.get_config().get_par(dimension).get_bin(x/sf)
            y = interpolation(x/sf)
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
            unscaled_sum = float(old_data.sum())
            # Check to see if there is data to scale and counts is positive
            if unscaled_sum > 0. and y > 0.:
                fill_cmd = ("scaled_spec._data" + cur_slice + "+= old_data * "
                            "(y / unscaled_sum)")
                exec(fill_cmd)
        # renormalise to prescale number of counts
        scaled_spec._num_decays = scaled_spec.sum()
        scaled_spec.scale(prescale_sum)
        scaled_spec._num_decays = spectrum._num_decays
        return scaled_spec
