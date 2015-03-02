''' Helper script to allow --help option to work with ROOT import.

Add to any module that imports ROOT or rat.
'''
import sys
_argv = sys.argv
sys.argv = []
import ROOT
ROOT.TObject
sys.argv = _argv
del _argv
