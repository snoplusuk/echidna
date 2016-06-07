""" Settings module for echidna

Use this module to store echidna-wide settings that can be set
according to the user's preference.

Attributes:
  user_profile (dict): Configure options to contol the behaviour of
    echidna for the user.

    .. note:: options include:

      * *hdf5_save_path*: Default directory to save hdf5 files.
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
      * *log_save_path*: Default directory to save logs.
      * *save_path*: Default directory to save other echidna output.


"""
import echidna


user_profile = {
    # hdf5 save path
    "hdf5_save_path": echidna.__echidna_base__ + "/output/hdf5/",
    # Select the mode - "normal" or "cautious"
    "mode": "normal",
    # Select the logging level - "debug", "info" or "warning"
    "log_level": "info",
    # Log save path
    "log_save_path": echidna.__echidna_base__ + "/output/logs/",
    # Save Path
    "save_path": echidna.__echidna_base__ + "/output/"
    }
