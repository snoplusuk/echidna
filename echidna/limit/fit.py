import numpy

import echidna.output.store as store
from echidna.limit.fit_results import FitResults
from echidna.limit.minimise import GridSearch
from echidna.errors.custom_errors import CompatibilityError
from echidna.core.spectra import SpectraFitConfig

import copy
import os


class Fit(object):
    """ Class to handle fitting.

    Args:
      roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`): An
        appropriate class for calculating test statistics.
      data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      fixed_background (:class:`echidna.core.spectra.Spectra`, optional):
        A spectrum containing all fixed backgrounds.
      floating_backgrounds (list, optional): one
        :class:`echidna.core.spectra.Spectra` for each background to float.
      signal (:class:`echidna.core.spectra.Spectra`):
        A spectrum of the signal that you are fitting.
      shrink (bool, optional): If set to True (default),
        :meth:`shrink` method is called on all spectra shrinking them to
        the ROI.
      minimiser (:class:`echidna.limit.minimiser.Minimiser`): Object to
        handle the minimisation.
      use_pre_made (bool): Flag whether to load a pre-made spectrum
        for each systematic value, or apply convolutions on the fly.
      pre_made_dir (string): Directory in which pre-made convolved
        spectra are stored.

    Attributes:
      _roi (dictionary): Region Of Interest you want to fit in. The format of
        roi is e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
      _test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`): An
        appropriate class for calculating test statistics.
      _fit_config ():
      _data (:class:`echidna.core.spectra.Spectra`): Data spectrum you want to
        fit.
      _fixed_background (:class:`echidna.core.spectra.Spectra`):
        A spectrum containing all fixed backgrounds.
      _floating_backgrounds (list): one :class:`echidna.core.spectra.Spectra`
        for each background to float.
      _signal (:class:`echidna.core.spectra.Spectra`):
        A spectrum of the signal that you are fitting.
      _minimiser (:class:`echidna.limit.minimiser.Minimiser)`: Object to
        handle the minimisation.
      _checked (bool): If True then the fit class is ready to be used.
      _use_pre_made (bool): Flag whether to load a pre-made spectrum
        for each systematic value, or apply convolutions on the fly.
      _pre_made_dir (string): Directory in which pre-made convolved
        spectra are stored.
    """
    def __init__(self, roi, test_statistic, fit_config=None, data=None,
                 fixed_backgrounds=None, floating_backgrounds=None,
                 signal=None, shrink=True, minimiser=None, fit_results=None,
                 use_pre_made=True, pre_made_base_dir=None):
        self._checked = False
        self.set_roi(roi)
        self._test_statistic = test_statistic
        self._fit_config = fit_config
        if not self._fit_config:
            self._fit_config = SpectraFitConfig({}, "")
        self._data = data
        if self._data:
            self._data_pars = self.get_roi_pars(self._data)
        else:
            self._data_pars = None
        self._fixed_background = None
        if fixed_backgrounds:  # spectra_dict in expected form
            self.make_fixed_background(fixed_backgrounds)  # sets attribute
        if self._fixed_background:
            self._fixed_pars = self.get_roi_pars(self._fixed_background)
        else:
            self._fixed_pars = None
        self._floating_backgrounds = floating_backgrounds
        if self._floating_backgrounds:
            floating_pars = []
            for background in self._floating_backgrounds:
                self._floating_pars.append(self.get_roi_pars(background))
            self._floating_pars = floating_pars
        else:
            self._floating_pars = None
        self._signal = signal
        if self._signal:
            self._signal_pars = self.get_roi_pars(self._signal)
        else:
            self._signal_pars = None
        if shrink:
            self.shrink_all()
        self.check_all_spectra()
        if minimiser is None:  # Use default (GridSearch)
            minimiser = GridSearch(self._fit_config,
                                   self._fit_config.get_name())
        self._minimiser = minimiser
        # create fit results object - not required if using GridSearch.
        if (fit_results is None and
                not isinstance(self._minimiser, GridSearch)):
            fit_results = FitResults(fit_config, fit_config.get_name())
        self._fit_results = fit_results  # Still None for GridSearch
        self._use_pre_made = use_pre_made
        self._pre_made_base_dir = pre_made_base_dir

    def append_fixed_background(self, spectra_dict, shrink=True):
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

        Raises:
          CompatibilityError: If the data spectra exists and its roi pars have
            not been set.
          CompatibilityError: If the fixed background spectra exists and its
            roi pars have not been set.
          CompatibilityError: If the signal spectra exists and its
            roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            their roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            their roi pars have not been set.
          CompatibilityError: If the floating backgrounds spectra exists and
            length of their roi pars is different to the number of floating
            backgrounds.
        """
        if self._data:
            if not self._data_pars:
                raise CompatibilityError("data roi pars have not been set.")
            self.check_spectra(self._data)
        if self._fixed_background:
            if not self._fixed_pars:
                raise CompatibilityError("fixed background roi pars have not "
                                         "been set.")
            self.check_spectra(self._fixed_background)
        if self._signal:
            if not self._signal_pars:
                raise CompatibilityError("signal roi pars have not been set.")
            self.check_spectra(self._signal)
        if self._floating_backgrounds:
            if not self._floating_pars:
                raise CompatibilityError("floating background roi pars have "
                                         "not been set.")
            if len(self._floating_pars) != len(self._floating_backgrounds):
                raise CompatibilityError("Different number of sets of roi "
                                         "pars as the number of floating "
                                         "backgrounds.")
            for background in self._floating_backgrounds:
                self.check_fit_config(background)
                self.check_spectra(background)
        if self._signal:
            self.check_spectra(self._signal)

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

    def check_fitter(self):
        """ Checks that the Fit class is ready to be used for fitting.

        Raises:
          CompatibilityError: If no data spectrum is present.
          CompatibilityError: If no fixed or floating backgrounds spectra are
            present.
        """
        if not self._data:
            raise CompatibilityError("No data spectrum exists in the fitter.")
        if not self._fixed_background and not self._floating_backgrounds:
            raise CompatibilityError("No fixed or floating backgrounds exist "
                                     "in the fitter.")
        self.check_all_spectra()
        self._checked = True

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
            if roi[dim][0] > roi[dim][1]:  # Make sure low is first
                roi[dim] = roi[dim][::-1]  # Reverses list/tuple

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

    def get_test_statistic(self):
        """ Gets the class instance you are using to calculate the test
        statistic used in the fit.

        Returns:
          (:class:`echidna.limit.test_statistic.TestStatistic`): The class
            instance used to calculate test statistics.
        """
        return self._test_statistic

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

    def fit(self):
        """ Gets the value of the test statistic used for fitting.

        Returns:
          float or :class:`numpy.array`: The resulting test statisic(s)
            dependent upon what method is used to compute the statistic.
        """
        if not self._checked:
            self.check_fitter()
        if not self._floating_backgrounds:  # Use fixed only
            observed = self._data.nd_project(self._data_pars)
            expected = self._fixed_background.nd_project(self._fixed_pars)
            if self._signal:
                expected += self._signal.nd_project(self._signal_pars)
            return self._test_statistic.compute_statistic(observed.ravel(),
                                                          expected.ravel())
        else:  # Pass to minimiser
            if self._minimiser is None:
                raise AttributeError("Minimiser is not set.")
            return self._minimiser.minimise(self._funct)

    def _funct(self, *args):
        """ Callable to pass to minimiser.

        Args:
          args (list): List of fit parameter values to test in the
            current iteration.

        Returns:
          float: Value of the test statistic given the current values
            of the fit parameters.

        Raises:
          ValueError: If :attr:`_floating_backgrounds` is None. This
            method requires at least one floating background.

        .. note:: This method should not be called directly, it is
          intended to be passed to an appropriate minimiser and
          called from within the minimisation algorithm.

        .. note:: This method should not be used if there are no
          floating backgrounds.
        """
        if self._floating_backgrounds is None:
            raise ValueError("The _funct method can only be used " +
                             "with at least one floating background")

        # Update parameter current values
        for index, value in enumerate(args):
            par = self._fit_config.get_par_by_index(index)
            par.set_current_value(value)

        # Loop over all floating backgrounds
        observed = self._data.nd_project(self._data_pars)
        expected = self._fixed_background.nd_project(self._fixed_pars)
        global_pars = self._fit_config.get_gloal_pars()
        for spectrum in self._floating_backgrounds:
            # Apply global parameters first
            if self._use_pre_made:  # Load pre-made spectrum from file
                spectrum = self.load_pre_made(spectrum, global_pars)
            else:
                for parameter in global_pars:
                    par = self._fit_congfig.get_par()
                    spectrum = par.apply_to(spectrum)

            # Apply spectrum-specific parameters
            for parameter in spectrum.get_fit_config().get_pars():
                par = spectrum.get_fit_config().get_par(parameter)
                spectrum = par.apply_to(spectrum)

            # Spectrum should now be fully convolved/scaled
            # Shrink to roi
            self.shrink_spectra(spectrum)
            # and then add projection to expected spectrum
            expected += spectrum.nd_project(self._floating_pars)

        # Add signal, if required
        if self._signal:
            expected += self._signal.nd_project(self._signal_pars)

        # Calculate value of test statistic
        test_statistic = self._test_statistic.compute_statistic(
            observed.ravel(), expected.ravel())

        # Add penalty terms
        for parameter in self._fit_config.get_pars():
            par = self._fit_config.get_par(parameter)
            current_value = par.get_current_value()
            prior = par.get_prior()
            sigma = par.get_sigma()
            test_statistic += self._test_statistic.get_penalty_term(
                current_value, prior, sigma)
        return test_statistic

    def load_pre_made(self, spectrum, global_pars):
        """ Load pre-made convolved spectra.

        This method is used to load a pre-made spectra convolved with
        certain resolution, energy-scale or shift values, or a
        combination of two or more at given values.

        The method loads the loads the correct spectra from HDF5s,
        stored in the given directory.

        Args:
          spectrum (:class:`echidna.core.spectra.Spectra`): Spectrum
            to convolve.

        Returns:
          spectrum (:class:`echidna.core.spectra.Spectra`): Convolved
            spectrum, ready for applying further systematics or fitting.
        """
        # Locate spectrum to load from HDF5
        # Base directory should be set beforehand
        if self._pre_made_base_dir is None:
            raise AttributeError("Pre-made directory is not set.")

        # Start with base spectrum name
        filename = spectrum.get_name()
        directory = self._pre_made_base_dir

        # Add current value of each global parameter
        for parameter in global_pars:
            par = self._fit_config.get_par(parameter)
            directory, filename = par.get_pre_convolved(directory, filename)
        # Load spectrum from hdf5
        spectrum = store.load(directory + filename + ".hdf5")

        return spectrum

    def make_fixed_background(self, spectra_dict, shrink=True):
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
            self._fixed_pars = self.get_roi_pars(total_spectrum)
        else:
            self.set_fixed_background(total_spectrum)

    def remove_signal(self):
        """ Removes the signal spectra from the class.
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
        self._data_pars = self.get_roi_pars(data)

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
        self._fixed_pars = self.get_roi_pars(fixed_background)

    def set_floating_backgrounds(self, floating_backgrounds, shrink=True):
        """ Sets the floating backgrounds you want to fit.

        Args:
          floating_backgrounds (list): List of backgrounds you want to float
            in the fit.
          shrink (bool, optional): If set to True (default), :meth:`shrink`
            method is called on the spectra shrinking it to the ROI.
        """
        floating_pars = []
        for background in floating_backgrounds:
            self.check_fit_config(background)
            if shrink:
                self.shrink_spectra(background)
            else:
                self.check_spectra(background)
            floating_pars.append(self.get_roi_pars(background))
        self._floating_backgrounds = floating_backgrounds
        self._floating_pars = floating_pars

    def set_test_statistic(self, test_statistic):
        """ Sets the method you want to use to calculate test statistics in
          the fit.

        Args:
          test_statistic (:class:`echidna.limit.test_statistic.TestStatistic`):
            An appropriate class for calculating test statistics.
        """
        self._test_statistic = test_statistic

    def set_roi(self, roi):
        """ Sets the region of interest you want to fit in.

        Args:
          roi (dictionary): The Region Of Interest you want to fit in.
            The format of roi is
            e.g. {"energy": (2.4, 2.6), "radial3": (0., 0.2)}
        """
        self.check_roi(roi)
        self._roi = roi
        self._checked = False  # Must redo checks for a new roi

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
        self._signal_pars = self.get_roi_pars(signal)

    def set_pre_made_dir(self, directory):
        """ Sets the directory in which pre-made convolved spectra are
        stored.

        Args:
          directory (string): Directory in which pre-made spectra are
            located.
        """
        # Check that it is a valid directory
        if not os.path.isdir(directory):
            raise ValueError("Supplied directory %s is not a valid path."
                             % directory)
        self._pre_made_dir = directory

    def shrink_all(self):
        """ Shrinks all the spectra used in the fit to the roi.
        """
        if self._data:
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
