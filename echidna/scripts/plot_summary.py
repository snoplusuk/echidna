""" Example script to plot the data stored in a floating background
  summary hdf5s.

This script:
  * Reads in a summary hdf5
  * Plots total test statistic, penalty term, best fit value and best fit value
    in terms of number of sigma away from the prior value all vs. signal scale.

Examples:
  To plot the summary hdf5 "file.hdf5"::

    $ python plot_summary -f /path/to/file.hdf5

  This will create the following plots ``./file_stat.png``,
  ``./file_best_fit.png``, ``./file_sigma_best_fit.png`` and
  ``./file_penalty_term.png``.
  To specify a save directory, include a -s flag followed by path to
  the required save destination.
  To plot to screen use the --graphical flag.
"""

from echidna.output import store
from echidna.output import plot_root as plot

import argparse
import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest="fname", type=str,
                        help="path to summary hdf5")
    parser.add_argument("-s", "--save_path", type=str, default="./",
                        help="Enter destination path for plots.")
    parser.add_argument("--graphical", dest="graphical", action="store_true",
                        help="Plot to screen")
    parser.set_defaults(graphical=False)
    args = parser.parse_args()
    s = store.load_summary(args.fname)
    base = args.save_path + args.fname.split('/')[-1].rstrip('.hdf5')
    can = ROOT.TCanvas()
    g_stat = plot.plot_stats_vs_scale(s, graphical=False)
    g_stat.GetYaxis().SetTitle("#Delta#chi^{2}")
    g_stat.SetTitle(base)
    g_stat.Draw("AP")
    can.Print(base+"_stat.png")
    can.Clear()
    g_best = plot.plot_best_fit_vs_scale(s, graphical=False)
    g_best.Draw("AP")
    g_best.SetTitle(base)
    can.Print(base+"_best_fit.png")
    can.Clear()
    g_best_sig = plot.plot_sigma_best_fit_vs_scale(s, graphical=False)
    g_best_sig.Draw("AP")
    g_best_sig.SetTitle(base)
    can.Print(base+"_sigma_best_fit.png")
    can.Clear()
    g_penalty = plot.plot_penalty_term_vs_scale(s, graphical=False)
    g_penalty.Draw("AP")
    g_penalty.SetTitle(base)
    can.Print(base+"_penalty_term.png")
