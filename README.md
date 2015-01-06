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

Testing
-------
To test the code is working as expected run

    python -m unittest discover echidna/test/
