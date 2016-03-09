""" Echidna's config module.

Contains the :class:`Config` class and all classes that inherit from it.
"""
from echidna.core.parameter import (RateParameter, ScaleParameter,
                                    ShiftParameter, ResolutionParameter,
                                    SpectraParameter)

import abc
import yaml
import collections


class Config(object):
    """ The base class for creating config classes.

    Args:
      name (string): The name of the config.

    Attributes:
      _name (string): The name of the config.
      _type (string): The type of the config, this affects it's
        parameter types
      _parameters (:class:`collections.OrderedDict`): Dictionary of
        parameters.
    """

    def __init__(self, name, parameters):
        """ Initialise config class
        """
        self._name = name
        self._type = "general"
        self._parameters = parameters

    def add_par(self, par):
        """ Add parameter to the config.

        Args:
          par (:class:`echidna.core.spectra.Parameter`): The parameter you want
            to add.
        """
        self._parameters[par._name] = par

    @abc.abstractmethod
    def dump(self):
        """ Abstract base class method to override.

        Dumps the config to a config dictionary, containing all
        parameters. The dictionary has the form specified in the
        :meth:`Config.load` method.

        Returns:
          dict: Dictionary containing all the information on the
            parameters.
        """
        raise NotImplementedError("The dump method can only be used "
                                  "when overriden in a derived class.")

    @abc.abstractmethod
    def dump_to_file(self, path="", filename=None):
        """ Abstract base class method to override.

        Write config to YAML file.

        Args:
          path (string, optional): Location to save yaml file to,
            default is the current directory.
          filename (string, optional): Filename for yaml file. If no
            filename is supplied, the default is "spectra_config.yml".
            If a blank filename "" is given the config's name is used.
        """
        raise NotImplementedError("The dump_to_file method can only be used "
                                  "when overriden in a derived class.")

    def get_index(self, parameter):
        """Return the index of a parameter within the existing set

        Args:
          parameter (string): Name of the parameter.

        Raises:
          IndexError: parameter is not in the config.

        Returns:
          int: Index of the parameter
        """
        for i, p in enumerate(self.get_pars()):
            if p == parameter:
                return i
        raise IndexError("Unknown parameter %s" % parameter)

    def get_name(self):
        """
        Returns:
          string: Name of :class:`Config` class instance - stored in
            :attr:`_name`.
        """
        return self._name

    def get_par(self, name):
        """Get a named FitParameter.

        Args:
          name (string): Name of the parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Named parameter.
        """
        return self._parameters[name]

    def get_par_by_index(self, index):
        """ Get parameter corresponding to given index

        Args:
          index (int): Index of parameter.

        Returns:
          :class:`echidna.core.spectra.Parameter`: Corresponding
            parameter.
        """
        name = self.get_pars()[index]
        return self.get_par(name)

    def get_pars(self):
        """Get list of all parameter names in the config.

        Returns:
          list: List of parameter names
        """
        return self._parameters.keys()

    def get_shape(self):
        """ Get the shape of the parameter space.

        Returns:
          tuple: A tuple constructed of the number of bins for each
            parameter in the config - this can be thought of as the
            full shape of the parameter space, whether it is the shape
            of the parameter space for the fit, or the shape of the
            spectral dimensions.
        """
        return tuple([self.get_par(par).get_bins() for par in self.get_pars()])

    def get_type(self):
        """
        Returns:
          string: Type of :class:`Config` class instance - stored in
            :attr:`_name`.
        """
        return self._name

    @classmethod
    @abc.abstractmethod
    def load(cls, config, name="config"):
        """ Abstract base class method to override.

        Initialise Config class from a config dictionary (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          name (string, optional): Name to assign to the
            :class:`Config`. If no name is supplied the default
            'spectra_config' will be used.

        Returns:
          (:class:`Config`): A config object containing the parameters
            from the config dictionary.

        Raises:
          KeyError: If the :obj:`config` dictionary has the wrong format.

        .. warning:: :obj:`config` dict must have valid format.

        Valid format is::

            {"parameters": {
                "<parameter>": {
                    "low": <low>,
                    "high": <high>,
                    "bins": <bins>}}}

        """
        raise NotImplementedError("The load method can only be used "
                                  "when overriden in a derived class.")

    @classmethod
    @abc.abstractmethod
    def load_from_file(cls, filename, name=None):
        """ Abstract base class method to override.

        Initialise Config class from a config file (classmethod).

        Args:
          filename (str): path to config file
          name (string, optional): Assign a name to the :class:`Config`
            created. If no name is supplied, the default is 'config'.
            If a blank string is supplied, the name of the file will
            be used.

        Returns:
          (:class:`Config`): A config object containing the parameters
            in the file.
        """
        raise NotImplementedError("The load_from_file method can only be used "
                                  "when overriden in a derived class.")


class GlobalFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      config_name (string): Name of config
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
    """

    def __init__(self, config_name, parameters):
        """Initialise GlobalFitConfig class
        """
        super(GlobalFitConfig, self).__init__(config_name, parameters)
        self._type = "global_fit"

    def add_config(self, config):
        """ Add pars from a :class:`echidna.core.spectra.Config` to this
          :class:`echidna.core.spectra.GlobalFitConfig`

        Args:
          config (:class:`echidna.core.spectra.Config`): Config to be added.
        """
        if config._type == "spectra_fit":
            spectra_name = config._spectra_name
            for par_name in config.get_pars():
                name = spectra_name + "_" + par_name
                par = config.get_par(par_name)
                par._name = name
                self.add_par(par, "spectra")
        elif config._type == "global_fit":
            for par in config.get_global_pars():
                self.add_par(par, "global")
            for par in config.get_spectra_pars():
                self.add_par(par, "spectra")
        else:
            raise ValueError("Cannot add %s-type config to a config "
                             "of type %s" % (config._type, self._type))

    def add_par(self, par, par_type):
        """ Add parameter to the global fit config.

        Args:
          par (:class:`echidna.core.spectra.FitParameter`): Parameter you want
            to add.
          par_type (string): The type of parameter (global or spectra).
        """
        if par_type != 'global' and par_type != 'spectra':
            raise IndexError("%s is an invalid par_type. Must be 'global' or "
                             "'spectra'." % par_type)
        self._parameters[par._name] = {'par': par, 'type': par_type}

    def dump(self, basic=False):
        """ Dumps the config to a global fit config dictionary,
        containing all the 'global' parameters, and a spectral fit
        comfig dictionary (if required), containing any 'spectral'
        parameters that have been added. The dictionaries have,
        respectively, the forms specified in the
        :meth:`GlobalFitConfig.load` and
        :meth:`echidna.core.spectra.SpectralFitConfig.load` methods.

        Returns:
          dict: Dictionary containing all the information on the
            'global' parameters.
          dict: Dictionary containing all the information on the
            'spectral' parameters.
        """
        # Global fit parameters
        main_key = "global_fit_parameters"
        global_fit_config = {main_key: {}}

        for par in self.get_global_pars():
            dimension = par.get_dimension()

            # Make entry for dimensions - as required
            if dimension not in global_fit_config[main_key].keys():
                global_fit_config[main_key][dimension] = {}

            name = par.get_name()
            # Remove dimension from name, if required
            if dimension in name:
                name = name.replace(dimension+"_", "")

            # Get parameter dict from par
            global_fit_config[main_key][dimension][name] = par.to_dict(basic)

        # Spectral fit parameters
        main_key = "spectral_fit_parameters"
        spectral_fit_config = {main_key: {}}

        for par in self.get_spectra_pars():
            # No dimesnions required here
            name = par.get_name()

            # Get parameter dict from par
            spectral_fit_config[main_key][name] = par.to_dict(basic)

        return global_fit_config, spectral_fit_config

    def dump_to_file(self, path="", global_fname=None,
                     spectral_fname=None, basic=False):
        """ Write config(s) to YAML file. Separate files are created
        for global and spectral parameters.

        Args:
          path (string, optional): Location to save yaml file(s) to,
            default is the current directory.
          global_fname (string, optional): Filename for global
            parameters yaml file. If no filename is supplied, the
            default is "global_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_global").
          spectral_fname (string, optional): Filename for spectral
            parameters yaml file. If no filename is supplied, the
            default is "spectral_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_spectral").
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.
        """
        global_fit_config, spectral_fit_config = self.dump(basic)
        if global_fname is None:
            global_fname = "global_fit_config"
        elif global_fname == "":
            global_fname = self.get_name()
        if ".yml" not in global_fname:
            global_fname += ".yml"
        with open(path+global_fname, "w") as stream:
            yaml.dump(global_fit_config, stream=stream, indent=8)

        if spectral_fname is None:
            spectral_fname = "spectral_fit_config"
        elif spectral_fname == "":
            spectral_fname = self.get_name()
        if ".yml" not in spectral_fname:
            spectral_fname += ".yml"
        with open(path+spectral_fname, "w") as stream:
            yaml.dump(spectral_fit_config, stream=stream, indent=8)

    def get_par(self, name):
        """ Get requested parameter:

        Args:
          name (string): Name of the parameter

        Returns:
          :class:`echidna.core.spectra.FitParameter`: The requested parameter.
        """
        return self._parameters[name]['par']

    def get_global_pars(self):
        """ Gets the parameters which are applied to all spectra
          simultaneously.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'global':
                pars.append(self._parameters[name]['par'])
        return pars

    def get_spectra_pars(self):
        """ Gets the parameters that are applied to individual spectra.

        Returns:
          list: Of :class:`echidna.core.spectra.FitParameter` objects.
        """
        pars = []
        for name in self._parameters:
            if self._parameters[name]['type'] == 'spectra':
                pars.append(self._parameters[name]['par'])
        return pars

    @classmethod
    def load(cls, global_config, spectral_config=None,
             name="global_fit_config"):
        """Initialise GlobalFitConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          spectral_config (dict): Dictionary of spectral fit parameters
            to create config out of.
          name (string, optional): Name to assign to the
            :class:`GlobalFitConfig`. If no name is supplied the
            default 'global_fit_config' will be used.

        Returns:
          (:class:`echidna.core.spectra.GlobalFitConfig`): A config object
            containing the parameters in the file called filename.

        Raises:
          KeyError: If the :obj:`global_config` dictionary does not
            start with the key 'global_fit_parameters' as this suggests
            the dictionary has the wrong format.
          IndexError: If an invalid global fit parameter name is
            encountered.
          KeyError: If the :obj:`spectral_config` dictionary does not
            start with the key 'spectral_fit_parameters' as this
            suggests the dictionary has the wrong format.
          IndexError: If an invalid spectral fit parameter name is
            encountered.

        .. warning:: :obj:`config` dict must have valid format.

        Valid format is::

            {"gloabal_fit_parameters": {
                "<spectral_dimension>": {
                    "<parameter_name>": {
                        "prior": <prior>,
                        "sigma": <sigma>,
                        "low": <low>,
                        "high": <high>,
                        "bins": <bins>}}}}

        For spectral config see :meth:`SpectralFitConfig.load`.

        """
        main_key = "global_fit_parameters"
        parameters = collections.OrderedDict()
        if main_key not in global_config.keys():
            raise KeyError("Cannot read global_config dictionary. "
                           "Please check it has the correct form")
        for dim in global_config[main_key]:
            for syst in global_config[main_key][dim]:
                name = dim + "_" + syst
                if syst == 'resolution' or syst == 'resolution_ly':
                    parameters[name] = {
                        'par': ResolutionParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                elif syst == 'shift':
                    parameters[name] = {
                        'par': ShiftParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                elif syst == 'scale':
                    parameters[name] = {
                        'par': ScaleParameter(
                            name, dimension=dim,
                            **global_config[main_key][dim][syst]),
                        'type': 'global'}
                else:
                    raise IndexError("%s is not a valid global fit parameter."
                                     % syst)
        if spectral_config is None:
            return cls(name, parameters)

        # Add spectral fit parameters:
        main_key = "spectral_fit_parameters"
        if not spectral_config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        for syst in spectral_config[main_key]:
            if "rate" in syst:
                parameters[syst] = {
                    'par': RateParameter(
                        syst, **spectral_config[main_key][syst]),
                    'type': 'spectral'}
            else:
                raise IndexError("Unknown systematic in config: %s" % syst)

        return cls(name, parameters)

    @classmethod
    def load_from_file(cls, filename, sf_filename=None, name=None):
        """Initialise GlobalFitConfig class from a config file (classmethod).

        Args:
          filename (string): path to config file
          sf_filename (string, optional): path to a separate spectral
            fit config file, to include.
          name (string, optional): Assign a name to the
            :class:`GlobalFitConfig` created. If no name is supplied,
            the default is 'global_fit_config'. If a blank string is
            supplied, the name of the file will be used.

        Returns:
          (:class:`echidna.core.spectra.GlobalFitConfig`): A config object
            containing the parameters in the file called filename.
        """
        config = yaml.load(open(filename, 'r'))
        if sf_filename:
            spectral_fit_config = yaml.load(open(sf_filename, "r"))
        else:
            spectral_fit_config = None
        if not name:
            return cls.load(config, spectral_config=spectral_fit_config)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, spectral_config=spectral_fit_config, name=name)


class SpectraFitConfig(Config):
    """Configuration container for floating systematics and fitting Spectra
      objects.  Able to load directly with a set list of FitParameters or
      from yaml configuration files.

    Args:
      config_name (string): Name of config
      parameters (:class:`collections.OrderedDict`): List of
        FitParameter objects
      spectra_name (string): Name of the spectra associated with the
         :class:`echidna.core.spectra.SpectraFitConfig`

    Attributes:
      _spectra_name (string): Name of the spectra associated with the
        :class:`echidna.core.spectra.SpectraFitConfig`
    """

    def __init__(self, config_name, parameters, spectra_name):
        """Initialise SpectraFitConfig class
        """
        super(SpectraFitConfig, self).__init__(config_name, parameters)
        self._type = "spectra_fit"
        self._spectra_name = spectra_name

    def dump(self, basic=False):
        """ Dumps the config to a spectral fit comfig dictionary,
        containing all 'spectral' fit parameters. The dictionary has
        the form specified in the :meth:`SpectralFitConfig.load`
        method.

        Returns:
          dict: Dictionary containing all the information on the
            'spectral' parameters.
        """
        # Spectral fit parameters
        main_key = "spectral_fit_parameters"
        spectral_fit_config = {main_key: {}}

        for parameter in self.get_pars():
            par = self.get_par(parameter)

            # Get parameter dict from par
            spectral_fit_config[main_key][parameter] = par.to_dict(basic)

        return spectral_fit_config

    def dump_to_file(self, path="", spectral_fname=None, basic=False):
        """ Write config(s) to YAML file. Separate files are created
        for global and spectral parameters.

        Args:
          path (string, optional): Location to save yaml file(s) to,
            default is the current directory.
          spectral_fname (string, optional): Filename for spectral
            parameters yaml file. If no filename is supplied, the
            default is "spectral_fit_config.yml". If a blank filename ""
            is given the config's name is used (+ "_spectral").
          basic (bool, optional): If True, only the basic properties:
            prior, sigma, low, high and bins are included.
        """
        spectral_fit_config = self.dump(basic)
        if spectral_fname is None:
            spectral_fname = "spectral_fit_config"
        elif spectral_fname == "":
            spectral_fname = self.get_name()
        if ".yml" not in spectral_fname:
            spectral_fname += ".yml"
        with open(path+spectral_fname, "w") as stream:
            yaml.dump(spectral_fit_config, stream=stream, indent=8)

    @classmethod
    def load(cls, config, spectra_name, name="spectral_fit_config"):
        """Initialise SpectraFitConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create config out of.
          name (string, optional): Name to assign to the
            :class:`SpectraFitConfig`. If no name is supplied the
            default 'spectral_fit_config' will be used.

        Returns:
          (:class:`SpectraFitConfig`): A config object containing the
            parameters from the config dictionary.

        Raises:
          KeyError: If the :obj:`config` dictionary does not start with
            the key 'spectral_fit_parameters' as this suggests the
            dictionary has the wrong format.
          IndexError: If an invalid spectral fit parameter name is
            encountered.

        .. warning:: :obj:`config` dict must have valid format.

        Valid format is::

            {"spectral_fit_parameters": {
                "<parameter_name>": {
                    "prior": <prior>,
                    "sigma": <sigma>,
                    "low": <low>,
                    "high": <high>,
                    "bins": <bins>}}}
        """
        main_key = "spectral_fit_parameters"
        if not config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        parameters = collections.OrderedDict()
        for syst in config[main_key]:
            if "rate" in syst:
                parameters[syst] = RateParameter(syst,
                                                 **config[main_key][syst])
            else:
                raise IndexError("Unknown systematic in config: %s" % syst)
        return cls(name, parameters, spectra_name)

    @classmethod
    def load_from_file(cls, filename, spectra_name, name=None):
        """Initialise SpectraFitConfig class from a config file (classmethod).

        Args:
          filename (str): path to config file
          spectra_name (string): Name of the spectra associated with the
            :class:`echidna.core.spectra.SpectraFitConfig`
          name (string, optional): Assign a name to the
            :class:`SpectraFitConfig` created. If no name is supplied,
            the default is 'spectral_fit_config'. If a blank string is
            supplied, the name of the file will be used.

        Returns:
          (:class:`SpectraFitConfig`): A config object containing the
            parameters in the file.
        """
        config = yaml.load(open(filename, 'r'))
        if not name:
            return cls.load(config, spectra_name)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, spectra_name, name=name)


class SpectraConfig(Config):
    """Configuration container for Spectra objects.  Able to load
    directly with a set list of SpectraParameters or from yaml
    configuration files.

    Args:
      parameters (:class:`collections.OrderedDict`): List of
        SpectraParameter objects
    """

    def __init__(self, config_name, parameters):
        """Initialise SpectraConfig class
        """
        super(SpectraConfig, self).__init__(config_name, parameters)
        self._type = "spectra"

    def dump(self):
        """ Dumps the spectra config to a config dictionary, containing
        all spectra parameters. The dictionary has the form specified
        in the :meth:`SpectraConfig.load` method.

        Returns:
          dict: Dictionary containing all the information on the
            spectra parameters.
        """
        # Spectral parameters
        main_key = "parameters"
        config = {main_key: {}}

        for parameter in self.get_pars():
            par = self.get_par(parameter)

            # Get parameter dict from par
            config[main_key][parameter] = par.to_dict()

        return config

    def dump_to_file(self, path="", filename=None):
        """ Write spectra config to YAML file.

        Args:
          path (string, optional): Location to save yaml file to,
            default is the current directory.
          filename (string, optional): Filename for yaml file. If no
            filename is supplied, the default is "spectra_config.yml".
            If a blank filename "" is given the config's name is used.
        """
        config = self.dump()
        if filename is None:
            filename = "spectra_config"
        elif filename == "":
            filename = self.get_name()
        if ".yml" not in filename:
            filename += ".yml"
        with open(path+filename, "w") as stream:
            yaml.dump(config, stream=stream, indent=8)

    @classmethod
    def load(cls, config, name="config"):
        """Initialise SpectraConfig class from a config dictionary
        (classmethod).

        Args:
          config (dict): Dictionary to create spectra config out of.
          name (string, optional): Name to assign to the
            :class:`SpectraConfig`. If no name is supplied the default
            'spectra_config' will be used.

        Returns:
          (:class:`SpectraConfig`): A config object containing the
            spectra parameters from the config dictionary.

        Raises:
          KeyError: If the :obj:`config` dictionary does not start with
            the key 'parameters' as this suggests the dictionary has
            the wrong format.

        .. warning:: :obj:`config` must have valid format.

        Valid format is::

            {"parameters": {
                "<spectral_parameter>": {
                    "low": <low>,
                    "high": <high>.
                    "bins": <bins>}}}
        """
        main_key = "parameters"
        if not config.get(main_key):
            raise KeyError("Cannot read config dictionary. "
                           "Please check it has the correct form")
        parameters = collections.OrderedDict()
        for parameter in config[main_key]:
            parameters[parameter] = SpectraParameter(
                parameter, **config[main_key][parameter])
        return cls(name, parameters)

    @classmethod
    def load_from_file(cls, filename, name=None):
        """Initialise SpectraConfig class from a config file
        (classmethod).

        Args:
          filename (str): path to config file
          name (string, optional): Assign a name to the
            :class:`SpectraConfig` created. If no name is supplied, the
            default is 'spectra_config'. If a blank string is supplied,
            the name of the file will be used.

        Returns:
          (:class:`SpectraConfig`): A config object containing the
            parameters in the file.
        """
        with open(filename, 'r') as stream:
            config = yaml.load(stream)
        if not name:
            return cls.load(config)
        if name == "":
            name = filename[filename.rfind("/")+1:filename.rfind(".")]
        return cls.load(config, name)

    def get_dims(self):
        """Get list of dimension names.
        The _mc, _reco and _truth suffixes are removed.

        Returns:
          list: List of the dimensions names of the config.
        """
        dims = []
        for par in sorted(self._parameters.keys()):
            par = par.split('_')[:-1]
            dim = ""
            for entry in par:
                dim += entry+"_"
            dims.append(dim[:-1])
        return dims

    def get_dim(self, par):
        """Get the dimension of par.
        The _mc, _reco and _truth suffixes are removed.

        Args:
          par (string): Name of the parameter

        Returns:
          The dimension of par
        """
        dim = ""
        for entry in par.split('_')[:-1]:
            dim += entry+"_"
        return dim[:-1]

    def get_dim_type(self, dim):
        """Returns the type of the dimension i.e. mc, reco or truth.

        Args:
          dim (string): The name of the dimension

        Raises:
          IndexError: dim is not in the spectra.

        Returns:
          string: The type of the dimension (mc, reco or truth)
        """
        for par in sorted(self._parameters.keys()):
            par_split = par.split('_')[:-1]
            cur_dim = ""
            for entry in par_split:
                cur_dim += entry+"_"
            if cur_dim[:-1] == dim:
                return str(par.split('_')[-1])
        raise IndexError("No %s dimension in spectra" % dim)
