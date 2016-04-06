import yaml
from collections import OrderedDict


def ordered_load(stream):
    """ Loads a .yml file, retaining the ordering of the file's
    structure.

    Args:
      stream (file stream): Reference to readable .yml file

    Returns:
      (dict): Dictionary to create spectra config out of.
    """
    class OrderedLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, **kwargs):
    """ Dumps a .yml file, retaining the ordering of the dict
    structure.

    Args:
      data (dict): Data dictionary to be saved to .yml file
      stream (file stream, optional): Reference to pre-existing
        .yml file to write to.
      kwargs (float): Key word arguments to be passed to yaml.dump

    Returns:
      (dict): Dictionary to create spectra config out of.
    """
    class OrderedDumper(yaml.SafeDumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)

    return yaml.dump(data, stream, OrderedDumper, **kwargs)
