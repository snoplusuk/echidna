class CompatibilityError(Exception):
    """ Exception raised when two :class: `spectra.Spectra` are not
    compatible.
    """
    pass

class LimitError(Exception):
    """ Exception raised when a limit has not been found.
    """
    pass
