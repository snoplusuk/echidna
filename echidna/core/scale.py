import echidna.core.spectra as spectra
import numpy


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
        par = spectrum.get_config().get_par(dimension)
        low = par._low
        n_bins = par._bins
        step = par.get_width()
        for bin in range(n_bins):
            x = par.get_bin_centre(bin)
            if x/sf < low or x/sf > par._high:
                continue  # Trying to scale values outside range (Unknown)
            y = interpolation(x/sf)
            if y <= 0.:
                continue
            old_bin1 = par.get_bin(x/sf)
            old_bin_centre1 = par.get_bin_centre(old_bin1)
            if par.get_bin_centre(old_bin1) > x/sf:
                old_bin2 = old_bin1 - 1
                if old_bin2 >= 0:
                    x_low1 = old_bin_centre1 - 0.5*step  # Equals x_high2
                    x_high1 = x/sf + 0.5*step
                    area1 = numpy.fabs(0.5 * (x_high1 - x_low1) *
                                       (interpolation(x_high1) +
                                        interpolation(x_low1)))
                    x_low2 = x/sf - 0.5*step
                    area2 = numpy.fabs(0.5 * (x_low1 - x_low2) *
                                       (interpolation(x_low1) +
                                        interpolation(x_low2)))
                else:
                    old_bin2 = 0
                    area2 = 0.  # This will set scale2 == 0
                    area1 = 1.
            else:
                old_bin2 = old_bin1 + 1
                if old_bin2 < n_bins:
                    x_low1 = x/sf - 0.5*step
                    x_high1 = old_bin_centre1 + 0.5*step  # Equals x_low2
                    area1 = numpy.fabs(0.5 * (x_high1 - x_low1) *
                                       (interpolation(x_high1) +
                                        interpolation(x_low1)))
                    x_high2 = x/sf + 0.5*step
                    area2 = numpy.fabs(0.5 * (x_high2 - x_high1) *
                                       (interpolation(x_high2) +
                                        interpolation(x_high1)))
                else:
                    old_bin2 = n_bins - 1
                    area2 = 0.  # This will set scale2 == 0
                    area1 = 1.
            scale1 = area1 / (area1 + area2)
            scale2 = area2 / (area1 + area2)
            # Prepare array split. Is there a better way to do this not using
            # eval and exec?
            cur_slice = "["
            old_slice1 = "["
            old_slice2 = "["
            for dim in range(n_dim):
                if dim == axis:
                    if bin < n_bins - 1:
                        cur_slice += str(bin) + ":" + str(bin + 1) + ","
                    else:
                        cur_slice += str(bin) + ":,"
                    if old_bin1 < n_bins - 1:
                        old_slice1 += (str(old_bin1) + ":" +
                                       str(old_bin1 + 1) + ",")
                    else:
                        old_slice1 += str(old_bin1) + ":,"
                    if old_bin2 < n_bins - 1:
                        old_slice2 += (str(old_bin2) + ":" +
                                       str(old_bin2 + 1) + ",")
                    else:
                        old_slice2 += str(old_bin2) + ":,"
                else:
                    cur_slice += ":,"
                    old_slice1 += ":,"
                    old_slice2 += ":,"
            cur_slice = cur_slice[:-1] + "]"
            old_slice1 = old_slice1[:-1] + "]"
            old_slice2 = old_slice2[:-1] + "]"
            old_data1 = eval("spectrum._data"+old_slice1)
            unscaled_sum1 = float(old_data1.sum())
            old_data2 = eval("spectrum._data"+old_slice2)
            unscaled_sum2 = float(old_data2.sum())
            # Check to see if there is data to scale and counts is positive
            if unscaled_sum1 <= 0. and unscaled_sum2 <= 0.:
                continue
            elif unscaled_sum1 <= 0.:
                fill_cmd = ("scaled_spec._data" + cur_slice + "+= old_data2 * "
                            "(y / unscaled_sum2)")
                exec(fill_cmd)
            elif unscaled_sum2 <= 0.:
                fill_cmd = ("scaled_spec._data" + cur_slice + "+= old_data1 * "
                            "(y / unscaled_sum1)")
                exec(fill_cmd)

            else:
                fill_cmd = ("scaled_spec._data" + cur_slice + "+= old_data1 * "
                            "scale1 * (y / unscaled_sum1) + old_data2 * "
                            "scale2 * (y / unscaled_sum2)")
                exec(fill_cmd)
        # renormalise to prescale number of counts
        scaled_spec._num_decays = scaled_spec.sum()
        scaled_spec.scale(prescale_sum)
        scaled_spec._num_decays = spectrum._num_decays
        return scaled_spec
