echidna
=======

echidna is a limit setting and spectrum fitting tool. It aims to create a fast flexible platform providing the tools for a variety of limit-setting and fitting based analyses.

Getting started with echidna
----------------------------

Whether you want to contribute to development or just use the latest release, this is the place to start.

Downloading echidna
-------------------

If you just want the latest release, there are two main options:

* [download](https://github.com/snoplusuk/echidna/releases/latest) the latest release from GitHub
* Clone the repository from the command line


        $ git clone -b "v0.1-alpha" git@github.com:snoplusuk/echidna.git

However if you plan on developing echidna, the best option is to fork the main
echidna repository and then clone your fork

    $ git clone git@github.com:yourusername/echidna.git

Software Requirements
---------------------

In order run echidna some extra python modules are required. The software and corresponding version numbers required listed in requirements.txt. To install all packages using pip run

    $ pip install -r requirements.txt

Note: you may need root access to install some modules.

Running echidna
---------------

You should now have a working copy of echidna. You can get started straight away by using some of the example scripts. Example scripts are located in `echidna/scripts/`. To run example scripts you must set your python path to

    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

from the echidna base directory. Details on how to run individual scripts can be found in their respective documentation.

Documentation
-------------

A copy of the latest version of the HTML documentation is available [here](echidna/blob/gh-pages/docs/html/docs/index.html)

But if you prefer you can build a copy of the Sphinx-based documentation locally. To build HTML documentation:

    $ cd docs
    $ make html

Output is placed in `docs/html/docs`. This documentation is built using the napoleon extension to sphinx, which allows the google style comments.

Note: After updating code be sure to run in the base directory
    
    $ sphinx-apidoc -f -H echidna -o docs echidna/
    $ cd docs
    $ make html

Testing
-------
To test the code is working as expected run

    $ python -m unittest discover echidna/test/

from the base directory.