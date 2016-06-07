import ROOT
import array


def set_ticks(can):
    """ Display ticks on the upper and right axes.

    Args:
      can (ROOT.TCanvas): Canvas to set ticks on.
    """
    can.SetTickx(1)
    can.SetTicky(1)


def root_style(font=132):
    """ Sets the style for ROOT plots. The SNO+ standard style is adapted
    from a .C sent to collaboration.

    Args:
      font (int): Integer denoting the font style for plots. Default is 132.
        See https://root.cern.ch/root/html/TAttText.html for details.
    """
    hipStyle = ROOT.TStyle("clearRetro", "HIP plots style for publications")

    # use plain black on white colors
    hipStyle.SetFrameBorderMode(0)
    hipStyle.SetCanvasBorderMode(0)
    hipStyle.SetPadBorderMode(0)
    hipStyle.SetPadBorderSize(0)
    hipStyle.SetPadColor(0)
    hipStyle.SetCanvasColor(0)
    hipStyle.SetTitleColor(0)
    hipStyle.SetStatColor(0)

    # use bold lines
    hipStyle.SetHistLineWidth(2)
    hipStyle.SetLineWidth(2)

    # no title, stats box or fit as default
    hipStyle.SetOptTitle(0)
    hipStyle.SetOptStat(0)
    hipStyle.SetOptFit(0)

    # postscript dashes
    hipStyle.SetLineStyleString(2, "[12 12]")  # postscript dashes

    # text style and size
    hipStyle.SetTextFont(font)
    hipStyle.SetTextSize(0.24)
    hipStyle.SetLabelFont(font, "x")
    hipStyle.SetLabelFont(font, "y")
    hipStyle.SetLabelFont(font, "z")
    hipStyle.SetTitleFont(font, "x")
    hipStyle.SetTitleFont(font, "y")
    hipStyle.SetTitleFont(font, "z")
    hipStyle.SetLegendFont(font)
    hipStyle.SetLabelSize(0.04, "x")
    hipStyle.SetTitleSize(0.05, "x")
    hipStyle.SetTitleColor(1, "x")
    hipStyle.SetLabelSize(0.04, "y")
    hipStyle.SetTitleSize(0.05, "y")
    hipStyle.SetTitleColor(1, "y")
    hipStyle.SetLabelSize(0.04, "z")
    hipStyle.SetTitleSize(0.05, "z")
    hipStyle.SetTitleColor(1, "z")

    # AXIS OFFSETS
    hipStyle.SetTitleOffset(0.8, "x")
    hipStyle.SetTitleOffset(0.8, "y")
    hipStyle.SetTitleOffset(0.8, "z")

    # Legends
    hipStyle.SetLegendBorderSize(1)
    # graphs - set default marker to cross, rather than .
    hipStyle.SetMarkerStyle(2)  # cross +

    # Color palette
    NRGBs = 5
    NCont = 255
    stops = array.array('d', [0.00, 0.34, 0.61, 0.84, 1.00])
    red = array.array('d', [0.00, 0.00, 0.87, 1.00, 0.51])
    green = array.array('d', [0.00, 0.81, 1.00, 0.20, 0.00])
    blue = array.array('d', [0.51, 1.00, 0.12, 0.00, 0.00])
    ROOT.TColor.CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont)
    hipStyle.SetNumberContours(NCont)

    ROOT.gROOT.SetStyle("clearRetro")
