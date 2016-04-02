""" Settings module for echidna

Use this module to store echidna-wide settings that can be set
according to the user's preference.

Attributes:
  user_profile (dict): Configure options to contol the behaviour of
    echidna for the user.

    .. note:: options include:

      * *mode* - Use the *cautious* mode to raise ``UserWarning`` as
        an exception. Otherwise default *normal* mode just logs these
        messages.
      * *log_level* - If logging is initiated with
        :func:`echidna.utilities.start_logging`, this controls the
        messages logged to the terminal. Select the level of messages
        to log to the terminal.

        * *debug* logs all messages
        * *info* (default) logs all messages of info-level and above.
        * *warning* only logs warning and error messages

"""
user_profile = {
    # Select the mode - "normal" or "cautious"
    "mode": "normal",
    # Select the logging level - "debug", "info" or "warning"
    "log_level": "info"
    }
