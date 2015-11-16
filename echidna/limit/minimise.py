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

    Attributes:
      _name (string): Name of minimiser.
      _type (string): Type of minimiser, e.g. GridSearch
    """
    __metaclass__ = abc.ABCMeta  # Only required for python 2

    def __init__(self, name):
        self._name = name
        self._type = None  # No type for base class

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
    inherit from :class:`FitResults`, as the :attr:`_data` attribute is
    the same in both classes, as is the :attr:`_fit_config`.

    Args:
      fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      name (str, optional): Name of this :class:`FitResults` class
        instance. If no name is supplied, name from fit_results will be
        taken and appended with "_results".
      use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.

    Attributes:
      _fit_config (:class:`echidna.limit.fit.FitConfig`): Configuration
        for fit. This should be a direct copy of the
        :class:`echidna.limit.fit.FitConfig` object in
        :class:`echidna.limit.fit.Fit`.
      _name (string): Name of this :class:`GridSearch` class instance.
      _data (:class:`numpy.ndarray`): Array of values of the test
        statistic calculated during the fit.
      _use_numpy (bool, optional): Flag to indicate whether to use the
        built-in numpy functions for minimisation and locating the
        minimum, or use the :meth:`find_minimum` method. Default is to
        use numpy.
    """
    def __init__(self, fit_config, name=None, use_numpy=True):
        super(GridSearch, self).__init__(fit_config, name=None)  # FitResults
        # Minimiser __init__ won't be called, so replicate functionality
        self._name = name
        self._type = GridSearch
        self._use_numpy = use_numpy

    def minimise(self, funct):
        """ Method to perform the minimisation.

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
        # Loop over all possible combinations of fit parameter values
        for values, indices in self._get_fit_par_values():
            # Call funct and pass array to it
            result = funct(*values)
            # Fill result into grid
            self._data[tuple(indices)] = result

        # Now grid is filled minimise
        self._minimum = copy.copy(self._data)

        if self._use_numpy:
            self._minimum = numpy.nanmin(self._minimum)
            # Set best_fit values
            # This is probably not the most efficient way of doing this
            best_fit = numpy.argmin(self._data)
            best_fit = numpy.unravel_index(best_fit, self._data.shape)
        else:  # Use find_minimum method
            self._minimum, best_fit = self.find_minimum(self._minimum)

        for index, par in zip(best_fit, self._fit_config.get_pars()):
            parameter = self._fit_config.get_par(par)
            parameter.set_best_fit(parameter.get_value_at(index))

        # Return minimum to limit setting
        return self._minimum

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
