""" Module containing classes that act as minimisers in a fit.
"""
import numpy

from echidna.fit.fit_results import FitResults

import copy
import abc


class Minimiser(object):
    """ Base class for minimiser objects.

    Args:
      name (string): Name of minimiser.
      per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.

    Attributes:
      _name (string): Name of minimiser.
      _per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
      _type (string): Type of minimiser, e.g. GridSearch
    """
    __metaclass__ = abc.ABCMeta  # Only required for python 2

    def __init__(self, name, min_type, per_bin):
        self._name = name
        self._type = min_type
        self._per_bin = per_bin

    @abc.abstractmethod
    def minimise(self, funct, test_statistic):
        """ Abstract base class method to override.

        Args:
          funct (callable): Callable function to calculate the value
            of the test statistic you wish to minimise, for each
            combination of parameter values tested. The function must
            only accept, as arguments, a variable-sized array of
            parameter values. E.g. ``def funct(*args)``. Within the
            echidna framework, the :meth:`echidna.limit.fit.Fit.funct`
            method is the recommened callable to use here.
          test_statistic (:class:`echidna.limit.test_statistic`): The
            test_statistic object used to calcualte the test statistics.

        Returns:
          float: Minimum value found during minimisation.
        """
        raise NotImplementedError("The minimise method can only be used "
                                  "when overridden in a derived class")


class GridSearch(FitResults, Minimiser):
    """ A grid-search minimisation algorithm.

    Although this is minimisation, it makes more sense for the class to
    inherit from :class:`FitResults`, as the :attr:`_stats` attribute is
    the same in both classes, as is the :attr:`_fit_config`.

    Args:
      fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): Configuration
        for fit. This should be a direct copy of the ``FitConfig``
        in :class:`echidna.limit.fit.Fit`.
      spectra_config (:class:`echidna.core.spectra.SpectraConfig`): The
        for spectra configuration. The recommended spectrum config to
        include here is the one from the data spectrum, to which you
        are fitting.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".
      per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
      use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.

    Attributes:
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _penalty_terms (:class:`numpy.ndarray`): Array of values of the
        penalty terms calculated during the fit.
      _resets (int): Number of times the grid has been reset.
      _use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.
    """
    def __init__(self, fit_config, spectra_config,
                 name=None, per_bin=False, use_numpy=True):
        super(GridSearch, self).__init__(fit_config, spectra_config, name=name)
        Minimiser.__init__(self, name, GridSearch, per_bin)
        if per_bin:
            stats_shape = fit_config.get_shape() + spectra_config.get_shape()
        else:
            stats_shape = fit_config.get_shape()
        self._resets = 0.
        self._stats = numpy.zeros(stats_shape)
        self._penalty_terms = numpy.zeros(fit_config.get_shape())
        self._use_numpy = use_numpy

    def get_penalty_at(self, **kwargs):
        """ Get penalty term at given fit parameter values

        Args:
          kwargs (dict): Dict with par names as keys and par values as values.

        Returns:
          float: The value of the penalty term
        """
        bins = []
        for par_name in self._fit_config.get_pars():
            par = self._fit_config.get_par(par_name)
            bins.append(par.get_bin(kwargs[par_name]))
        return self._penalty_terms[tuple(bins)]

    def get_penalty_term(self, indices):
        """ Gets the array of penalty terms.

        .. note:: Unlike the :class:`echidna.fit.summary.Summary` class
          individual penalty contributions from each fit parameter are
          not stored here, only the total penalty term value.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the total
            penalty term value.

        Returns:
          (:class:`numpy.ndarray`): Array stored in :attr:`_penalty_terms`.
            Values of the penalty term calculated during the fit.

        Raises:
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        return self._penalty_terms[indices]

    def get_penalty_terms(self, par):
        """ Gets the array of penalty terms.

        Returns:
          (:class:`numpy.ndarray`): Array stored in :attr:`_penalty_terms`.
            Values of the penalty term calculated during the fit.
        """
        if len(self._fit_config.get_pars()) == 1.:
            return self._penalty_terms
        par_idx = self._fit_config.get_index(par)
        return self._penalty_terms[par_idx]

    def get_raw_stat(self, indices):
        """ Gets the raw test statistic(s) from array at the given indices.

        .. warning:: This has no penalty term contributions added.

        .. note:: Unlike :meth:`get_stat`, here you can specify indices
          for any number of fit parameters dimensions, so to get a
          slice of the raw array.

        Args:
          indices (tuple): Index along each fit parameter (dimension)
            specifiying the coordinates in the array.

        Returns:
          (float or :class:`numpy.ndarray`): The raw test statistic(s)
            at the given indices.
        """
        return self._stats[indices]

    def get_raw_stats(self, **kwargs):
        """ Gets the raw test statistics array.

        .. warning:: This has no penalty term contributions added.

        Args:
          kwargs (dict): Fit par names as keys and fit par values as values.

        Returns:
          :class:`numpy.array`: The raw test statistics values at each
            combination of fit parameter values.
        """
        stats = copy.copy(self._stats)
        if self._stats.shape == self._fit_config.get_shape() +\
                self._spectra_config.get_shape():
            for i in range(len(self._spectra_config.get_shape())):
                stats = stats.sum(-1)
        if kwargs:
            cmd = "stats["
            for par in self._fit_config.get_pars():
                if par in kwargs:
                    idx = par.get_bin(kwargs[par])
                    cmd += str(idx)+","
                else:
                    cmd += ":,"
            cmd = cmd[:-1] + "]"
            stats = eval(cmd)
        return stats

    def get_raw_stats_at(self, **kwargs):
        """ Get stats with no penalty added at given fit parmeters.

        Args:
          kwargs (dict): Dict with par names as keys and par values as values.

        Returns:
          float or :class:`numpy.ndarray`: Raw stats.
        """
        bins = []
        for par_name in self._fit_config.get_pars():
            par = self._fit_config.get_par(par_name)
            bins.append(par.get_bin(kwargs[par_name]))
        stats = copy.copy(self._stats[tuple(bins)])
        while stats.shape[0] == 1:
            stats = stats.sum(0)
        if type(stats) is float:
            return stats
        while stats.shape[-1] == 1:
            stats = stats.sum(-1)
        return stats

    def get_resets(self):
        """
        Returns:
          int: Number of times the grid has been reset (:attr:`_resets`).
        """
        return self._resets

    def get_scales(self, par):
        """Gets the parameter scales used in the fit

        Args:
          par (string): Name of parameter

        Returns:
          numpy.ndarray: Parameter scales.
        """
        return self._fit_config.get_par(par).get_values()

    def get_stat(self, indices):
        """ Combines the test-statistic array (collapsed to the parameter
          values grid - i.e. summed over spectral bins) with the penalty
          term grid of the same shape, for a single bin, specified by
          indices.

        .. warning:: Penalty term contributions **are** included here.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the test
            statistic value.

        Returns:
          (float): Combination of the value of the test statistic
            calculated during the fit and the penalty term value.

        Raises:
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        combined = copy.copy(self._stats[indices])

        if self._per_bin:
            # Collapse by summing over spectral dimensions
            for dim_size in self._spectra_config.get_shape():
                combined = numpy.sum(combined, axis=-1)  # always last axis

        # Add penalties
        combined = combined + self._penalty_terms[indices]
        return combined

    def get_stats(self):
        """ Combines the test-statistic array (collapsed to the parameter
          values grid - i.e. summed over spectral bins) with the penalty
          term grid of the same shape.

        .. warning:: Penalty term contributions **are** included here.

        Returns:
          (:class:`numpy.ndarray`): Array combining the values of the
            test statistic calculated during the fit and the penalty
            term values.
        """
        combined = copy.copy(self._stats)

        # Collapse by summing over spectral dimensions
        if self._per_bin:
            for dim_size in self._spectra_config.get_shape():
                combined = numpy.sum(combined, axis=-1)  # always last axis

        # Add penalties
        combined = combined + self._penalty_terms
        return combined

    def minimise(self, funct, test_statistic):
        """ Method to perform the minimisation.

        Args:
          funct (callable): Callable function to calculate the value
            of the test statistic you wish to minimise, for each
            combination of parameter values tested. The function must
            only accept, as arguments, a variable-sized array of
            parameter values. E.g. ``def funct(*args)``. Within the
            echidna framework, the :meth:`echidna.limit.fit.Fit.funct`
            method is the recommened callable to use here.
          test_statistic (:class:`echidna.limit.test_statistic`): The
            test_statistic object used to calcualte the test statistics.

        Returns:
          float: Minimum value found during minimisation.
        """
        # Loop over all possible combinations of fit parameter values
        for values, indices in self._get_fit_par_values():
            # Call funct and pass array to it
            result, penalty = funct(*values)

            # Check result is of correct form
            if self._per_bin:  # expecting numpy.ndarray
                if not isinstance(result, numpy.ndarray):
                    raise TypeError("Expecting result of type numpy.ndarray "
                                    "(not %s), for per_bin enabled" %
                                    type(result))
                    expected_shape = self._spectra_config.get_shape()
                    if result.shape != expected_shape:
                        raise ValueError(
                            "Expecting result to be numpy array with shape "
                            "%s (not %s), for per_bin enabled" %
                            (str(expected_shape), str(result.shape)))
            else:
                result = numpy.array(result)
            self.set_stat(result, tuple(indices))
            self.set_penalty_term(penalty, tuple(indices))

        # Now grid is filled minimise
        minimum = self.get_stats()

        if self._use_numpy:
            # Set best_fit values
            # This is probably not the most efficient way of doing this
            position = numpy.argmin(minimum)
            position = numpy.unravel_index(position, minimum.shape)
            minimum = numpy.nanmin(minimum)
        else:  # Use find_minimum method
            minimum, position = self.find_minimum(minimum)

        for index, par in zip(position, self._fit_config.get_pars()):
            parameter = self._fit_config.get_par(par)
            best_fit = parameter.get_value_at(index)
            sigma = parameter.get_sigma()
            prior = parameter.get_prior()
            parameter.set_best_fit(parameter.get_value_at(index))
            if sigma is not None:
                parameter.set_penalty_term(
                    test_statistic.get_penalty_term(best_fit, prior, sigma))
            else:  # penalty term = 0
                parameter.set_penalty_term(0.)

        # Return minimum to fitting
        return self.get_raw_stat(position), self.get_penalty_term(position)

    def nd_project_stat(self, indices, *parameters):
        """ Projects the test statistic values, at given the given
          indices, onto the axes specified by fit and spectral parameters.

        .. warning:: If only **fit** parameters are specified all
          spectral dimensions are collapsed and penalty term
          contributions **are** included. If any **spectral**
          parameters are provided penalty term contributions **are
          not** included.

        Args:
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to retrieve the test
            statistic value.
          *parameters (string): Names of a valid fit or spectral
            parameters onto which to project the test statistic values.

        Returns:
          :class:`numpy.ndarray`: Projection of :attr:`_stats` array,
            at the given indices, onto the given parameter axes.

        Raises:
          IndexError: If the parameter names supplied do not match
            any of those stored in the fit or spectra configs.
        """
        for parameter in parameters:
            if parameter not in itertools.chain(
                    self._fit_config.get_pars(),
                    self._spectra_config.get_pars()):
                raise IndexError("Unknown parameter %s" % parameter)
        if parameters in self._fit_config.get_pars():
            # Can apply penalty term contributions
            projection = self.get_stat(indices)
            for axis, parameter in self._fit_config.get_pars():
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        else:  # No penalty terms, use raw stats
            projection = copy.copy(self.get_raw_stat(indices))
            for axis, parameter in enumerate(
                    itertools.chain(self._fit_config.get_pars(),
                                    self._spectra_config.get_pars())):
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        return projection

    def nd_project_stats(self, *parameters):
        """ Projects the test statistic values onto the axes specified
          by fit and spectral parameters.

        .. warning:: If only **fit** parameters are specified all
          spectral dimensions are collapsed and penalty term
          contributions **are** included. If any **spectral**
          parameters are provided penalty term contributions **are
          not** included.

        Args:
          *parameters (string): Names of a valid fit or spectral
            parameters onto which to project the test statistic values.

        Returns:
          :class:`numpy.ndarray`: Projection of :attr:`_stats` array
            onto the given parameter axes.

        Raises:
          IndexError: If the parameter names supplied do not match
            any of those stored in the fit or spectra configs.
        """
        for parameter in parameters:
            if parameter not in itertools.chain(
                    self._fit_config.get_pars(),
                    self._spectra_config.get_pars()):
                raise IndexError("Unknown parameter %s" % parameter)
        if parameters in self._fit_config.get_pars():
            # Can apply penalty term contributions
            projection = self.get_stats()
            for axis, parameter in self._fit_config.get_pars():
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=axis)
        else:  # No penalty terms, use raw stats
            projection = copy.copy(self.get_raw_stats())
            counter = 0
            for axis, parameter in enumerate(
                    itertools.chain(self._fit_config.get_pars(),
                                    self._spectra_config.get_pars())):
                if parameter not in parameters:
                    projection = numpy.sum(projection, axis=(axis-counter))
                    counter = counter + 1  # compensate for earlier sums
        return projection

    def reset_grids(self):
        """ Resets the grids stored in :attr:`_stats` and
          :attr:`_penalty_terms`, including shape.

        .. warning:: If fit parameters have been added/removed, calling
          this method will increase/decrease the dimensions of the grid
          to compensate for this change.
        """
        if self._resets == 0:
            self._resets = 1
            self._name += "_%d" % self._resets
        else:
            new_name = self._name.split("_")[0]
            for part in self._name.split("_")[1:-1]:
                new_name += "_" + part
            self._resets += 1
            self._name = new_name + "_" + str(self._resets)
        stats_shape = (self._fit_config.get_shape() +
                       self._spectra_config.get_shape())
        self._stats = numpy.zeros(stats_shape)
        self._penalty_terms = numpy.zeros(self._fit_config.get_shape())

    def set_penalty_terms(self, penalty_terms):
        """ Sets the array containing penalty term values.

        Args:
          penalty_terms (:class:`numpy.ndarray`): The array of penalty
            term values

        Raises:
          TypeError: If penalty_terms is not an :class:`numpy.ndarray`
          ValueError: If the penalty_terms array does not have the required
            shape.
        """
        if not isinstance(penalty_terms, numpy.ndarray):
            raise TypeError("penalty_terms must be a numpy array")
        if penalty_terms.shape != self._fit_config.get_shape():
            raise ValueError("penalty_terms array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(penalty_terms.shape),
                              str(self._fit_config.get_shape())))
        self._penalty_terms = penalty_terms

    def set_penalty_term(self, penalty_term, indices):
        """ Sets the total penalty term value at the point in the array
        specified by indices.

        Args:
          penalty_term (float): Best fit value of a fit parameter.
          indices (tuple): The index along each fit parameter dimension
            specifying the coordinates from which to set the total
            penalty term value.

        Raises:
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
        """
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        self._penalty_terms[indices] = penalty_term

    def set_stat(self, stat, indices):
        """ Sets the test statistic values in array at the point
        specified by indices

        Args:
          stat (:class:`numpy.ndarray`): Values of the test statistic.
          indices (tuple): Position in the array.

        Raises:
          IndexError: If the indices supplied are out of bounds for
            the fit dimensions
          TypeError: If stat is not a :class:`numpy.ndarray`.
        """
        if indices > self._fit_config.get_shape():
            raise IndexError(
                "indices %s out of bounds for fit with dimensions %s" %
                (str(indices), str(self._fit_config.get_shape())))
        if not isinstance(stat, numpy.ndarray):
            raise TypeError("stat must be a numpy array")
        self._stats[indices] = stat

    def set_stats(self, stats):
        """ Sets the total test statistics array.

        Args:
          stats (:class:`numpy.ndarray`): The total test statistics array.

        Raises:
          TypeError: If stats is not a :class:`numpy.ndarray`.
          ValueError: If the stats array has incorrect shape.
        """
        if not isinstance(stats, numpy.ndarray):
            raise TypeError("stats must be a numpy array")
        if stats.shape != self._stats.shape:
            raise ValueError("stats array has incorrect shape (%s), "
                             "expected shape is %s" %
                             (str(stats.shape), str(self._stats.shape)))
        self._stats = stats

    def _update_coords(self, coords, new_coords):
        """ Internal method called by :meth:`find_minimum` to update the
        stored co-ordinates of minima.

        This method takes the new co-ordinates in ``new_coords`` and
        works out the indices to select the correct (previously-
        calculated) co-ordinates for the positions of minima, in the
        inner dimensions.

        Args:
          coords (:class:`numpy.ndarray`): 2D array containing the
            previously calculated co-ordinates for the inner dimensions.
          new_coords (:class:`numpy.ndarray`): Array containing the
            co-ordinates of each minima calculated for the current
            dimension.

        Returns:
          (:class:`numpy.ndarray`): 2D array containing the updated
            arrays of co-ordinates, for all dimensions processed so far.
        """
        new_coords = new_coords.ravel()
        multiplier = 0  # to calculate indices
        product = 1  # product of dimensions
        for dim in new_coords.shape:
            product *= dim
            multiplier += len(coords) / product

        # Calculate indices
        indices = [i * multiplier + j for i, j in enumerate(new_coords)]

        # Index current co-ordinates
        coords = coords[indices]

        # Append new co-ordinates
        coords = numpy.concatenate((new_coords.reshape(len(new_coords), 1),
                                    coords), axis=1)
        return coords

    def find_minimum(self, array):
        """ Alternative method for finding the minimum.

        Starting from the innermost dimension, locates the minima
        along the axis - effectively minimising over lots of 1D arrays.
        Once the minima are located, this axis is collapsed down to
        next innermost, storing just the values at the minima. The
        position of each minima is also stored in the ``coords`` array.

        This process is then repeated for the next innermost array.
        However now when we locate the position of each minima, we
        also wish to get the corresponding position calculated at the
        previous dimension - the :meth:`_update_coords` does this,
        concatenating the current locations with the previous ones
        so we start to build up an array of co-ordinates.

        As the code works outwards through the dimensions, the
        number of co-ordinates are added, but the choice of co-ordinates
        is reduced. Until the outermost dimension is processed and then
        *ideally* only one minimum value and one set of co-ordinates
        remains.

        Args:
          array (:class:`numpy.ndarray`): Array to minimise.

        Returns:
          float: Minimum value in array.
          tuple: Location (co-ordinates) of minimum.

        .. warning:: If the exact minimum value is repeated more than
          once in the array, the location with the lowest coordinate
          values (starting from the outermost dimension) will be
          returned. E.g. if two equal minima are at (4, 10, 15) and
          (41, 2, 12), the location (4, 10, 15) would be returned
        """
        dims = len(array.shape)
        minimum = copy.copy(array)
        coords = None

        # Loop over dimensions, working outwards
        for i, dim in enumerate(reversed(list(range(dims)))):
            # Work out coordinates
            if coords is None:  # Create array
                coords = numpy.argmin(minimum, axis=dim).ravel()
                coords = coords.reshape(len(coords), 1)
            elif dim > 0:  # Update existing array
                new_coords = numpy.argmin(minimum, axis=dim)
                coords = self._update_coords(coords, new_coords)
            else:  # Last dimension - new_coords is a single float
                new_coords = numpy.argmin(minimum, axis=dim)
                coords = coords[new_coords]
                if coords is not numpy.array:
                    coords = numpy.array([coords])
                coords = numpy.insert(coords, 0, new_coords)

            # Collapse minima into outer dimension
            minimum = numpy.nanmin(minimum, axis=dim)

        coords = coords.ravel()
        coords = tuple(coords)
        return minimum, coords

    def _get_fit_par_values(self):
        """ Internal method. Gets next array of parameter values to
        test in fit.

        Yields:
          (:class:`numpy.array`): Next array of parameter values to
            test in fit.
          (:class:`numpy.array`): Indices of these parameter values.
        """
        index = 0
        values = numpy.zeros((len(self._fit_config.get_pars())))
        indices = numpy.zeros((len(self._fit_config.get_pars())))
        for values, indices in self._get_values(index, values, indices):
            yield values, indices

    def _get_values(self, index, values, indices):
        """ Internal method. Called recursively to populate the array
        of current parameter values.

        Yields:
          (:class:`numpy.array`): Next array of parameter values to
            test in fit.
          (:class:`numpy.array`): Indices of these parameter values.
        """
        if index < len(values):
            parameter = self._fit_config.get_par_by_index(index)
            for value in parameter.get_values():
                values[index] = value
                indices[index] = parameter.get_value_index(value)
                for values, indices in self._get_values(index+1,
                                                        values, indices):
                    yield values, indices
        else:
            yield values, indices
