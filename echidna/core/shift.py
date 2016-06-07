import numpy
import copy


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
          kwargs (dict): To passed to
            :meth:`spectra.Spectra.interpolation1d`

        Returns:
          :class:`spectra.Spectra`: The shifted spectrum.
        """
        shift = self.get_shift()
        step = spectrum.get_config().get_par(dimension).get_width()
        if numpy.isclose(shift % step, 0.):
            # shift size multiple of step size. Interpolation not required.
            return self.shift_by_bin(spectrum, dimension)
        preshift_sum = spectrum.sum()
        interpolation = spectrum.interpolate1d(dimension, **kwargs)
        shifted_spec = copy.copy(spectrum)
        shifted_spec._name = spectrum._name + "_shift" + str(shift)
        shifted_spec._data = numpy.zeros(spectrum._data.shape)
        n_dim = len(spectrum._data.shape)
        axis = spectrum.get_config().get_index(dimension)
        par = spectrum.get_config().get_par(dimension)
        low = par._low
        high = par._high
        n_bins = par._bins
        for bin in range(n_bins):
            x = low + (bin + 0.5) * step
            current = x - shift
            if (current) < low or (current) >= high:
                continue  # Trying to shift values outside range (Unknown)
            elif current < low + 0.5*step:
                current = low + 0.5*step
            elif current > high - 0.5*step:
                current = high - 0.5*step - 1e-6  # floating point issue
            y = interpolation(current)
            if y <= 0.:  # Cant have negative num_events
                continue
            old_bin1 = par.get_bin(current)
            old_bin_centre1 = par.get_bin_centre(old_bin1)
            if old_bin_centre1 > current:
                old_bin2 = old_bin1 - 1
                if old_bin2 >= 0:
                    x_low1 = old_bin_centre1 - 0.5*step  # Equals x_high2
                    x_high1 = current + 0.5*step
                    if x_high1 > high - 0.5*step:
                        x_high1 = high - 0.5*step - 1e-6
                    area1 = numpy.fabs(0.5 * (x_high1 - x_low1) *
                                       (interpolation(x_high1) +
                                        interpolation(x_low1)))
                    x_low2 = current - 0.5*step
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
                    x_low1 = current - 0.5*step
                    if x_low1 < low + 0.5*step:
                        x_low1 = low + 0.5*step
                    x_high1 = old_bin_centre1 + 0.5*step  # Equals x_low2
                    area1 = numpy.fabs(0.5 * (x_high1 - x_low1) *
                                       (interpolation(x_high1) +
                                        interpolation(x_low1)))
                    x_high2 = current + 0.5*step
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
            unshifted_sum1 = float(old_data1.sum())
            old_data2 = eval("spectrum._data"+old_slice2)
            unshifted_sum2 = float(old_data2.sum())
            # Check to see if there is data to shift
            if unshifted_sum1 <= 0. and unshifted_sum2 <= 0.:
                continue
            elif unshifted_sum1 <= 0.:
                fill_cmd = ("shifted_spec._data" + cur_slice + " += "
                            "old_data2 * (y / unshifted_sum2)")
                exec(fill_cmd)
            elif unshifted_sum2 <= 0.:
                fill_cmd = ("shifted_spec._data" + cur_slice + " += "
                            "old_data1 * (y / unshifted_sum1)")
                exec(fill_cmd)

            else:
                fill_cmd = ("shifted_spec._data" + cur_slice + "+="
                            "old_data1 * scale1 * (y / unshifted_sum1) +"
                            "old_data2 * scale2 * (y / unshifted_sum2)")
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
          kwargs (dict): To passed to
            :meth:`spectra.Spectra.interpolation1d`

        Returns:
          :class:`spectra.Spectra`: The shifted spectrum.
        """
        shift = self.get_shift()
        step = spectrum.get_config().get_par(dimension).get_width()
        if not numpy.isclose(shift % step, 0.):
            raise ValueError("Shift (%s) must be a multiple of bin width (%s)"
                             % (shift, step))
        shifted_spec = copy.copy(spectrum)
        shifted_spec._name = spectrum._name + "_shift" + str(shift)
        shifted_spec._data = numpy.zeros(spectrum._data.shape)
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
