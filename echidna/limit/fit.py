import numpy
import copy

from echidna.errors.custom_errors import CompatibilityError


class Fit(object):
    """ Class to handle fitting.

    Args:
      roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      method (:class:`TBC`): Method for calculating test statistics.
      data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      fixed_background (:class:`echidna.core.spectra.Spectra`, optional):
        A spectrum containing all fixed backgrounds.
      floating_backgrounds (list, optional): one
        :class:`echidna.core.spectra.Spectra` for each background to float.
      shrink (bool, optional): If set to True (default),
        :meth:`shrink` method is called on all spectra shrinking them to
        the ROI.

    Attributes:
      _roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      _method (:class:`TBC`): Method for calculating test statistics.
      _data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      _fixed_background (:class:`echidna.core.spectra.Spectra`):
        A spectrum containing all fixed backgrounds.
      _floating_backgrounds (list): one :class:`echidna.core.spectra.Spectra`
        for each background to float.
      _signal (:class:`echidna.core.spectra.Spectra`):
        A spectrum of the signal that you are fitting.
    """
    def __init__(self, roi, method, data, fixed_background=None,
                 floating_backgrounds=None, shrink=True):
        self.set_roi(roi)
        self._method = method
        self._data = data
        self._fixed_background = fixed_background
        self._floating_backgrounds = floating_backgrounds
        self._signal = None
        self._data = data
        if shrink:
            self.shrink_all()
        self.check_all_spectra()

    def append_fixed_background(self, spectra_dict, shrink=True)
        ''' Appends the fixed background with more spectra.

        Args:
          spectra_dict (dict): Dictionary containing spectra as keys and
            prior counts as values.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        '''
        for spectrum, scaling in spectra_dict.iteritems():
            # Copy so original spectra is unchanged
            spectrum = copy.deepcopy(spectrum)
            if shrink:
                self.shrink_spectra(spectrum)
            spectrum.scale(scaling)
            self._fixed_background.add(spectrum)

    def check_all_spectra(self):
        """ Ensures that all spectra can be used for fitting.
        """
        self.check_spectra(self._data)
        if self._fixed_background:
            self.check_spectra(self._fixed_background)
        if self._signal:
            self.check_spectra(self._signal)
        if self._floating_backgrounds:
            for background in self._floating_backgrounds:
                self.check_fit_config(background)
                self.check_spectra(background)

    def check_fit_config(self, spectra):
        """ Checks that a spectra has a fit config.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            check.

        Raises:
          CompatibilityError: If spectra has no fit config
        """
        if not spectra.get_fit_config():
            raise CompatibilityError("%s has no fit config" % spectra._name)

    def check_roi(self, roi):
        """ Checks the ROI used to fit.

        Args:
          roi (dict): roi you want to check.

        Raises:
          TypeError: If roi is not a dict
          TypeError: If value in roi dict is not a list or a tuple
          CompatibilityError: If the length of a value in the roi dict is
            not 2.
        """
        if not isinstance(roi, dict):
            raise TypeError("roi must be a dictionary of parameter values")
        for dim in roi:
            if not isinstance(roi[dim], (tuple, list)):
                raise TypeError("roi must be a dictionary of tuples or lists")
            if len(roi[dim]) != 2:
                raise CompatibilityError("%s entry (%s) in roi must contain a"
                                         " low and high value in a tuple or"
                                         " list" % (dim, self._roi[dim]))
            if self._roi[dim][0] > self._roi[dim][1]:  # Make sure low is first
                self._roi[dim] = self._roi[dim][::-1]  # Reverses list/tuple

    def check_spectra(self, spectra):
        """ Checks the spectra you want to fit.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            check.

        Raises:
          ValueError: If roi low value and spectra low value are not equal.
          ValueError: If roi high value and spectra high value are not equal.
        """
        for dim in self._roi:
            dim_type = self._data.get_config().get_dim_type(dim)
            par = dim + "_" + dim_type
            if not numpy.isclose(self._roi[dim][0],
                                 spectra.get_config().get_par(par)._low):
                raise ValueError("roi %s low (%s) not equal to spectra %s"
                                 " %s low (%s)"
                                 % (dim, self.roi[dim][0], dim,
                                    spectra.get_config().get_par(par)._low))
            if not numpy.isclose(self._roi[dim][1],
                                 spectra.get_config().get_par(par)._high):
                raise ValueError("roi %s high (%s) not equal to spectra %s"
                                 " %s high (%s)"
                                 % (dim, self.roi[dim][0], dim,
                                    spectra.get_config().get_par(par)._high))

    def get_data(self):
        """ Gets the data you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The data you are fitting.
        """
        return self._data

    def get_fixed_background(self):
        """ Gets the fixed background you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The fixed background you are
            fitting.
        """
        return self._fixed_background

    def get_floating_backgrounds(self):
        """ Gets the floating backgrounds you are fitting.

        Returns:
          list: The floating backgrounds you are fitting.
        """
        return self._floating_backgrounds

    def get_method(self):
        """ Gets the method you are using to calculate the test statistic you
          are using to fit.

        Returns:
          :class:`TBC`: The method used to calculate test statistics.
        """
        return self._method

    def get_roi(self):
        """ Gets the region of interest (roi)

        Returns:
          dict: The region of interest
        """
        return self._roi

    def get_roi_pars(self, spectra):
        """ Get the parameters of a spectra that contain the roi.

        Args:
          :class:`echidna.core.spectra.Spectra`: The spectra you want to obtain
            the roi parameters for.

        Returns:
          list: Of the names of the spectra parameters which contain the roi.
        """
        pars = []
        for dim in self._roi:
            dim_type = spectra.get_config().get_dim_type(dim)
            par = dim + "_" + dim_type
            pars.append(par)
        return pars

    def get_signal(self):
        """ Gets the signal you are fitting.

        Returns:
          :class:`echidna.core.spectra.Spectra`: The fixed background you are
            fitting.
        """
        return self._signal

    def get_statistic(self, data_pars=None, fixed_pars=None,
                      floating_pars=None, signal_pars=None):
        """ Gets the value of the test statistic used for fitting.

        Args:
          data_pars (list, optional): Of data spectra parameters which
            contain the roi.
          fixed_pars (list, optional): Of fixed background spectra parameters
            which contain the roi.
          floating_pars (list, optional): Of lists each of which contains the
            spectra parameters which contain the roi for the background to
            float. The order of the list must be the same order as the
            floating_backgrounds.
        """
        if not data_pars:
            data_pars = self.get_roi_pars(self._data)
        if not fixed_pars:
            fixed_pars = self.get_roi_pars(self._fixed_background)
        if not self._floating_backgrounds:
            observed = self._data.nd_project(data_pars)
            expected = self._fixed_background.nd_project(fixed_pars)
            if self._signal:
                if not signal_pars:
                    signal_pars = self.get_roi_pars(self._signal)
                expected += self._signal.nd_project(signal_pars)
            return self._method.compute_statistic(observed.ravel(),
                                                  expected.ravel())
        if not floating_pars:
            floating_pars = []
            for background in self._floating_background:
                floating_pars.append(self.get_roi_pars(background))
        for background in self._floating_background:
            for systematic in background.get_fit_config().get_pars():
                return None

    def make_fixed_background(self, spectra_dict, shrink=True)
        ''' Makes a spectrum for fixed backgrounds and stores it in the class.

        Args:
          spectra_dict (dict): Dictionary containing spectra as keys and
            prior counts as values.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        '''
        first = True
        for spectrum, scaling in spectra_dict.iteritems():
            # Copy so original spectra is unchanged
            spectrum = copy.deepcopy(spectrum)
            if first:
                first = False
                if shrink:
                    self.shrink_spectra(spectrum)
                spectrum.scale(scaling)
                total_spectrum = spectrum
                total_spectrum._name = "Fixed Background"
            else:
                if shrink:
                    self.shrink_spectra(spectrum)
                spectrum.scale(scaling)
                total_spectrum.add(spectrum)
        if shrink:
            self._fixed_background = total_spectrum  # No need to check
        else:
            self.set_fixed_background(total_spectrum)

    def remove_signal(self):
        """
        """
        self._signal = None

    def set_data(self, data, shrink=True):
        """ Sets the data you want to fit.

        Args:
          data (:class:`echidna.core.spectra.Spectra`): Data spectrum you
            want to fit.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        if shrink:
            self.shrink_spectra(data)
        else:
            self.check_spectra(data)
        self._data = data

    def set_fixed_background(self, fixed_background, shrink=True):
        """ Sets the fixed background you want to fit.

        Args:
          fixed_background (:class:`echidna.core.spectra.Spectra`):
            The fixed background spectrum you want to fit.
          shrink (bool, optional): If set to True (default) :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        if shrink:
            self.shrink_spectra(fixed_background)
        else:
            self.check_spectra(fixed_background)
        self._fixed_background = fixed_background

    def set_floating_backgrounds(self, floating_background, shrink=True):
        """ Sets the floating backgrounds you want to fit.

        Args:
          floating_backgrounds (list): List of backgrounds you want to float
            in the fit.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        for background in floating_backgrounds:
            self.check_fit_config(background)
            if shrink:
                self.shrink_spectra(background)
            else:
                self.check_spectra(background)
        self._floating_backgrounds = floating_backgrounds

    def set_method(self, method):
        """ Sets the method you want to use to calculate test statistics in
          the fit.

        Args:
          method (:class:`TBC`): The method you want to calculate test
            statistics with in the fit.
        """
        self._method = method

    def set_roi(self, roi):
        """ Sets the region of interest you want to fit in.

        Args:
          roi (dictionary): The Region Of Interest you want to fit in.
            The format of roi is
            e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
        """
        self.check_roi(roi)
        self._roi = roi

    def set_signal(self, signal, shrink=True):
        """ Sets the signal you want to fit.

        Args:
          signal (:class:`echidna.core.spectra.Spectra`):
            The signal spectrum you want to fit.
          shrink (bool, optional): If set to True (default) :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        if shrink:
            self.shrink_spectra(signal)
        else:
            self.check_spectra(signal)
        self._signal = signal

    def shrink_all(self):
        """ Shrinks all the spectra used in the fit to the roi.
        """
        self.shrink_spectra(self._data)
        if self._fixed_background:
            self.shrink_spectra(self._fixed_background)
        if self._signal:
            self.shrink_spectra(self._signal)
        if self._floating_backgrounds:
            for background in self._floating_backgrounds:
                self.shrink_spectra(background)

    def shrink_spectra(self, spectra):
        """ Shrinks the spectra used in the fit to the roi.

        Args:
          spectra (:class:`echidna.core.spectra.Spectra`): Spectra you want to
            shrink to the roi.
        """
        shrink = {}
        for dim in self._roi:
            dim_type = spectra.get_config().get_dim_type(dim)
            par_low = dim + "_" + dim_type + "_low"
            par_high = dim + "_" + dim_type + "_high"
            shrink[par_low], shrink[par_high] = self._roi[dim]
        spectra.shrink(**shrink)


