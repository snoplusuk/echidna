echidna
=======

echidna is a limit setting and spectrum fitting tool. It aims to create a fast, flexible and user friendly platform providing the tools for a variety of limit-setting and fitting based analyses. For a more detailed description, installation guide and some tutorials, please see the [User manual](https://github.com/snoplusuk/echidna/wiki#user-manual).

The image below is taken from a presentation on echidna and aims to summarise the full package and workflow in a single image.

![overview](https://cloud.githubusercontent.com/assets/1931666/10453599/6673622a-71a5-11e5-971a-5be98a6bbe7a.png)

Downloading echidna
-------------------

If you just want the latest release, there are two main options:

* [download](https://github.com/snoplusuk/echidna/releases/latest) the latest release from GitHub
* Clone the repository from the command line


        $ git clone -b "v0.1-alpha" git@github.com:snoplusuk/echidna.git

However if you plan on developing echidna, the best option is to fork the main
echidna repository and then clone your fork

    $ git clone git@github.com:yourusername/echidna.git

Software requirements
---------------------

In order run echidna some extra python modules are required. The software and corresponding version numbers required listed in requirements.txt. To install all packages using pip run

    $ pip install -r requirements.txt

Note: you may need root access to install some modules. 

Getting started with echidna
----------------------------

Whether you want to contribute to development or just use the latest release, this is the place to start.

### Software structure

The diagram below shows the directory stucture for the echidna package, when if is first installed.

````
echidna (__echidna_base__)
├── docs
└── echidna (__echidna_home__)
    ├── calc
    ├── config
    ├── core
    ├── errors
    ├── limit
    ├── output
    ├── scripts
    ├── test
    └── util
```
The top level directory `__echidna_base__` contains the `docs` directory, which is populated when you build the documentation (see below), and the `echidna` directory containing the main echidna modules. Within this directory - referred to as `__echidna_home__` - the `calc` directory contains constants and the class for calculating expected rates for double beta processes. The `core` directory contains the core data structure and the code to configure/fill it, an example configuration file is included in the `config` directory. The `errors` directory contains custom error handling code, whilst the `limit` directory contains the code for limit setting and chi-squared calculation. The `output` directory handles saving to/loading from the HDF5 file format and scripts to create various plots. Inside the `scripts` directory are the scripts to actually run echidna. Finally the `test` directory contains the unittests and `util` just contains any useful functions that might be of use throughout echidna.

### Running echidna

You should now have a working copy of echidna. You can get started straight away by using some of the example scripts. Example scripts are located in `echidna/scripts/`. To run example scripts you must set your python path to

    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

from the echidna base directory. Details on how to run individual scripts can be found in their respective [documentation pages](https://snoplusuk.github.io/echidna/docs/echidna.scripts.html). However all should have command line help available too, so something like:

    $ python echidna/scripts/example_script -h (--help)

should explain most of what you need to know to run the script.

These scripts cover a large part of the full functionality of echidna and represent key tasks that that can be accomplished with echidna. However the scripts are also intended as a guide and so may need to be modified to accomplish a specific task.

Some simpler examples of echidna's functionality are available in the [getting started](https://github.com/snoplusuk/echidna/wiki/GettingStarted) section of the User guide.

Documentation
-------------

A copy of the latest version of the HTML documentation is available [here](https://snoplusuk.github.io/echidna/docs/index.html)

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
