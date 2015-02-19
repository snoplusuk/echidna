echidna
=======
A limit setting and spectrum fitting tool.

Documentation
-------------
Sphinx based documentation is available in the `docs` directory. To build HTML documentation,

    cd docs
    make html

Output is placed in `doc/_build/html`.
This documentation is built using the napoleon extension to sphinx, which allows the google style comments.

Note: After updating code be sure to run in the base directory
    
    sphinx-apidoc -f -H echidna -o docs echidna/
    cd docs
    make html

Software Requirements
---------------------

The software and corresponding version numbers required to run echidna are listed in requirements.txt. To install all packages using pip run

    sudo pip install -r requirements.txt

Running echidna
---------------

Example scripts are located in the echidna/scripts/ folder. To run example scripts you must set your python path to

    export PYTHONPATH=$PYTHONPATH:$(pwd)

from the echidna base directory. Details on how to run individual scripts can be found in their respective documentation.

Testing
-------
To test the code is working as expected run

    python -m unittest discover echidna/test/
