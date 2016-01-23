""" Module containing classes that act as minimisers in a fit.
"""
import numpy

from echidna.limit.fit_results import FitResults

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

    def __init__(self, name, per_bin=False):
        self._name = name
        self._type = None  # No type for base class
        self._per_bin = per_bin

    @abc.abstractmethod
    def minimise(self, funct):
        """ Abstract base class method to override.

        Args:
          funct (callable): Callable function to calculate the value
            of the test statistic you wish to minimise, for each
            combination of parameter values tested. The function must
            only accept, as arguments, a variable-sized array of
            parameter values. E.g. ``def funct(*args)``. Within the
            echidna framework, the :meth:`echidna.limit.fit.Fit.funct`
            method is the recommened callable to use here.

        Returns:
          float: Minimum value found during minimisation.
        """
        pass


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
      _fit_config (:class:`echidna.core.spectra.GlobalFitConfig`): The
        configuration for fit. This should be a direct copy of the
        ``FitConfig`` in :class:`echidna.limit.fit.Fit`.
      _spectra_config (:class:`echidna.core.spectra.SpectraConfig`): The
        for spectra configuration. The recommended spectrum config to
        include here is the one from the data spectrum, to which you
        are fitting.
      _name (string): Name of this :class:`GridSearch` class instance.
      _stats (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _penalties (:class:`numpy.ndarray`): Array of values of the
        penalty terms calculated during the fit.
      _minimum_value (float): Minimum value of the array returned by
        :meth:`get_fit_data`.
      _minimum_position (tuple): Position of the test statistic minimum
        value. The tuple contains the indices along each fit parameter
        (dimension), acting as coordinates of the position of the
        minimum.
      _resets (int): Number of times the grid has been reset.
      _type (string): Type of minimiser, e.g. GridSearch
      _per_bin (bool, optional): Flag if minimiser should expect a
        test statistic value per-bin.
      _use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.
    """
    def __init__(self, fit_config, spectra_config,
                 name=None, per_bin=False, use_numpy=True):
        # FitResults
        super(GridSearch, self).__init__(fit_config, spectra_config, name=None)
        # Minimiser __init__ won't be called, so replicate functionality
        self._name = name
        self._type = GridSearch
        self._per_bin = per_bin
        self._use_numpy = use_numpy

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

        Attributes:
          _minimum_value (float): Minimum value of test statistic found.
          _minimum_position (tuple): Position of minimum.

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
            parameter.set_penalty_term(
                test_statistic.get_penalty_term(best_fit, prior, sigma))

        self.set_minimum_value(minimum)
        self.set_minimum_position(position)  # save position of minimum
        # Return minimum to fitting
        return minimum

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
