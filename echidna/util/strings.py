import string
import random

def id_generator(size=4, chars=string.ascii_uppercase +\
                     string.ascii_lowercase + string.digits):
    """ Returns a random string

    Args:
      size (int, optional): Size of string
      chars (string, optional): Charachters to select from randomly.

    Returns:
      string: Random string.
    """
    return ''.join(random.choice(chars) for _ in range(size))
