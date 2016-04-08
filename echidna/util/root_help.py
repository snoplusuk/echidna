''' Helper script to allow :option:`--help` option to work with ROOT import.

Add to any module that imports ROOT or rat.
'''
import ROOT

import sys

_argv = sys.argv
sys.argv = []
ROOT.TObject
sys.argv = _argv
del _argv
