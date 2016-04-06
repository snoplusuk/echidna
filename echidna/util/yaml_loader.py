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
    class OrderedLoader(yaml.Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)
