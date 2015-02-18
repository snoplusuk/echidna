echidna
=======
A limit setting and spectrum fitting tool.

Documentation
-------------
Sphinx based documentation is available in the `docs` directory. To build HTML documentation,

    cd doc
    make html

Output is placed in `doc/_build/html`.
This documentation is built using the napoleon extension to sphinx, which allows the google style comments.

Note: After updating code be sure to run in the base directory
    
    sphinx-apidoc -f -H echidna -o docs echidna/

Software Requirements
---------------------

The software and corresponding version numbers required to run echidna are listed in requirements.txt. To install a package using pip run

    sudo pip install software_name==version_number

where software_name and version_number are as listed in requirements.txt

Testing
-------
To test the code is working as expected run

    python -m unittest discover echidna/test/
