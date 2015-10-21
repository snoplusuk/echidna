""" Tutorial 1: Getting Started - full tutorial

Please see https://github.com/snoplusuk/echidna/wiki/GettingStarted for
a full explanation of the commands in this tutorial.

This script:
 * Creates :class:`echidna.core.spectra.Spectra` instance
 * Fills `Spectra`
 * Plots `Spectra`
 * Applies cuts and smears `Spectra`
 * Other `Spectra` manipulations e.g. `shrink_to_roi`, `rebin` and
   `scale`

Examples:
  To run type (from the base directory)::

    $ python echidna/scripts/tutorials/getting_started.py
"""


def main():
    """ Function to run tutorial script.
    """
    import matplotlib.pyplot as plt
    # # Tutorial 1: Getting started with echidna

    # This guide tutorial aims to get you started with some basic tasks you can
    # accomplish using echidna.

    # ## Spectra creation

    # The `Spectra` class is echidna's most fundamental class. It holds the
    # core data structure and provides much of the core functionality required.
    # Coincidentally, this guide will be centred around this class, how to
    # create it and then some manipulations of the class.

    # We'll begin with how to create an instance of the `Spectra` class. It is
    # part of the `echidna.core.spectra` module, so we will import this and
    # make a `Spectra` instance.

    # In[ ]:

    import echidna.core.spectra as spectra

    # Now we need a config file to create the spectrum from. There is an
    # example config file in `echidna/config`. If we look at the contents of
    # this yaml file, we see it tells the `Spectra` class to create a data
    # structure to hold two parameters:

    #  * `energy_mc`, with lower limit 0, upper limit 10 and 1000 bins
    #  * `radial_mc`, with lower limit 0, upper limit 15000 and 1500 bins

    # This config should be fine for us. We can load it using the
    # `load_from_file` method of the `SpectraConfig` class:

    # In[ ]:

    import echidna
    config = spectra.SpectraConfig.load_from_file(
        echidna.__echidna_base__ + "/echidna/config/spectra_example.yml")
    print config.get_pars()

    # Note we used the `__echidna_base__` member of the `echidna` module here.
    # This module has two special members for denoting the base directory (the
    # outermost directory of the git repository) and the home directory (the
    # `echidna` directory inside the base directory. The following lines show
    # the current location of these directories:

    # In[ ]:

    print echidna.__echidna_base__
    print echidna.__echidna_home__

    # Finally before creating the spectrum, we should define the number of
    # events it should represent:

    # In[ ]:

    num_decays = 1000

    # In[ ]:

    spectrum = spectra.Spectra("spectrum", num_decays, config)
    print spectrum

    # And there you have it, we've created a `Spectra` object.

    # ## Filling the spectrum

    # Ok, so we now have a spectrum, let's fill it with some events. We'll
    # generate random energies from a Gaussian distribution and random
    # positions from a Uniform distribution. Much of echidna is built using
    # the `numpy` and `SciPy` packages and we will use them here to generate
    # the random numbers. We'll also generate a third random number to
    # simulate some form rudimentary detector efficiency.

    # In[ ]:

    # Import numpy
    import numpy

    # In[ ]:

    # Generate random energies from a Gaussin with mean (mu) and sigma (sigma)
    mu = 2.5  # MeV
    sigma = 0.15  # MeV

    # Generate random radial position from a Uniform distribution
    outer_radius = 5997  # Radius of SNO+ AV

    # Detector efficiency
    efficiency = 0.9  # 90%

    for event in range(num_decays):
        energy = numpy.random.normal(mu, sigma)
        radius = numpy.random.uniform(high=outer_radius)
        event_detected = (numpy.random.uniform() < efficiency)
        if event_detected:  # Fill spectrum with values
            spectrum.fill(energy_mc=energy, radial_mc=radius)

    # This will have filled our `Spectra` class with the events. Make sure to
    # use the exact parameter names that were printed out above, as kewyord
    # arguments. To check we can now use the `sum` method. This returns the
    # total number of events stored in the spectrum at a given time - the
    # integral of the spectrum.

    # In[ ]:

    print spectrum.sum()

    # The value returned by `sum`, should roughly equal:

    # In[ ]:

    print num_decays * efficiency

    # We can also inspect the raw data structure. This is saved in the `_data`ashleyrback/getting_started
    # member of the `Spectra` class:

    # In[ ]:

    print spectrum._data

    # **Note: you probably won't see any entries in the above. For large
    # arrays, numpy only prints the first three and last three entries. Since
    # our energy range is in the middle, all our events are in the** `...`
    # **part at the  moment. But we will see entries printed out later when we
    # apply some cuts.**

    # ## Plotting

    # Another useful way to inspect the `Spectra` created is to plot it.
    # Support is available within echidna to plot using either `ROOT` or
    # `matplotlib` and there are some useful plotting functions available in
    # the `plot` and `plot_root` modules.

    # In[ ]:

    import echidna.output.plot as plot
    import echidna.output.plot_root as plot_root

    # To plot the projection of the spectrum on the `energy_mc` axis:

    # In[ ]:

    fig_1 = plot.plot_projection(spectrum, "energy_mc",
                                 fig_num=1, show_plot=False)
    plt.show()

    # and to plot the projection on the `radial_mc` axis, this time using root:

    # In[ ]:

    fig_2, can = plot_root.plot_projection(spectrum, "radial_mc",
                                           graphical=True, fig_num=2)

    # We can also project onto two dimensions and plot a surface:

    # In[ ]:

    fig_3 = plot.plot_surface(spectrum, "energy_mc", "radial_mc",
                              fig_num=3, show_plot=False)
    plt.show()

    # ## Convolution and cuts

    # The ability to smear the event, along a parameter axis, is built into
    # echidna in the `smear` module. There are three classes in the module that
    # allow us to create a smearer for different scenarios. There are two
    # smearers for energy-based parameters, `EnergySmearRes` and
    # `EnergySmearLY`, which allow smearing by energy resolution (e.g.
    # $\frac{5\%}{\sqrt{(E[MeV])}}$ and light yield (e.g. 200 NHit/Mev)
    # respectively. Then additionally the `RadialSmear` class handles smearing
    # along the axis of any radial based parameter.

    # We will go through an example of how to smear our spectrum by a fixed
    # energy resolution of 5%. There are two main smearing algorithms:
    # "weighted smear" and "random smear". The "random smear" algorithm takes
    # each event in each bin and randomly assigns it a new energy from the
    # Gaussian distribution for that bin - it is fast but not very accurate
    # for low statistics. The "weighted smear" algorithm is slower but much
    # more accurate, as re-weights each bin by taking into account all other
    # nearby bins within a pre-defined range. We will use the "weighted smear"
    # method in this example.

    # First to speed the smearing process, we will apply some loose cuts.
    # Although, fewer bins means faster smearing, you should be wary of cutting
    # the spectrum too tightly before smearing as you may end up cutting bins
    # that would have influenced the smearing. Cuts can be applied using the
    # `shrink` method. (Confusingly there is also a `cut` method which is
    # almost identical to the `shrink` method, but updates the number of
    # events the spectrum represents, after the cut is applied. Unless you are
    # sure this is what you want to do, it is probably better to use the
    # `shrink` method.) To shrink over multiple parameters, it is best to
    # construct a dictionary of `_low` and `_high` values for each parameter
    # and then pass this to the shrink method.

    # In[ ]:

    shrink_dict = {"energy_mc_low": mu - 5.*sigma,
                   "energy_mc_high": mu + 5.*sigma,
                   "radial_mc_low": 0.0,
                   "radial_mc_high": 3500}
    spectrum.shrink(**shrink_dict)

    # Using the `sum` method, we can check to see how many events were cut.

    # In[ ]:

    print spectrum.sum()

    # Import the smear class:

    # In[ ]:

    import echidna.core.smear as smear

    # and create the smearer object.

    # In[ ]:

    smearer = smear.EnergySmearRes()

    # By default the "weighted smear" method considers all bins within a
    # $\pm 5\sigma$ range. For the sake of speed, we will reduce this to 3
    # here. Also set the energy resolution - 0.05 for 5%.

    # In[ ]:

    smearer.set_num_sigma(3)
    smearer.set_resolution(0.05)

    # To smear our original spectrum and create the new `Spectra` object
    # `smeared_spectrum`:

    # In[ ]:

    smeared_spectrum = smearer.weighted_smear(spectrum)

    # this should hopefully only create a couple of seconds.

    # The following code shows how to make a simple script, using matplotlib,
    # to overlay the original and smeared spectra.

    # In[ ]:

    import numpy as np
    import matplotlib.pyplot as plt

    def overlay_spectra(original, smeared, dimension="energy_mc", fig_num=1):
        """ Overlay original and smeared spectra.

        Args:
          original (echidna.core.spectra.Spectra): Original spectrum.
          smeared (echidna.core.spectra.Spectra): Smeared spectrum.
          dimension (string, optional): Dimension to project onto.
            Default is "energy_mc".
          fignum (int, optional): Figure number, if producing multiple
            figures. Default is 1.

        Returns:
          matplotlib.figure.Figure: Figure showing overlaid spectra.
        """
        fig = plt.figure(num=fig_num)
        ax = fig.add_subplot(1, 1, 1)

        par = original.get_config().get_par(dimension)
        width = par.get_width()

        # Define array of bin boundaries (1 more than number of bins)
        bins = np.linspace(par._low, par._high, par._bins+1)
        # Define array of bin centres
        x = bins[:-1] + 0.5*width

        # Overlay two spectra using projection as weight
        ax.hist(x, bins, weights=original.project(dimension),
                histtype="stepfilled", color="RoyalBlue",
                alpha=0.5, label=original._name)
        ax.hist(x, bins, weights=smeared.project(dimension),
                histtype="stepfilled", color="Red",
                alpha=0.5, label=smeared._name)

        # Add label/style
        plt.legend(loc="upper right")
        plt.ylim(ymin=0.0)
        plt.xlabel(dimension + " [" + par.get_unit() + "]")
        plt.ylabel("Events per " + str(width) + " " + par.get_unit() + " bin")
        return fig

    # In[ ]:

    fig_4 = overlay_spectra(spectrum, smeared_spectrum, fig_num=4)
    plt.show()

    # ## Other spectra manipulations

    # We now have a nice smeared version of our original spectrum. To prepare
    # the spectrum for a final analysis there are a few final manipulations we
    # may wish to do.

    # ### Region of Interest (ROI)

    # There is a special version of the `shrink` method called `shrink_to_roi`
    # that can be used for ROI cuts. It saves some useful information about the
    # ROI in the `Spectra` class instance, including the efficiency i.e.
    # integral of spectrum after cut divided by integral of spectrum before
    # cut.

    # In[ ]:

    roi = (mu - 0.5*sigma, mu + 1.45*sigma)  # To get nice shape for rebinning
    smeared_spectrum.shrink_to_roi(roi[0], roi[1], "energy_mc")
    print smeared_spectrum.get_roi("energy_mc")

    # ### Rebin

    # Our spectrum is still quite finely binned, perhaps we want to bin it in
    # 50 keV bins instead of 10 keV bins. The `rebin` method can be used to
    # acheive this.

    # The `rebin` method requires us to specify the new shape (tuple) of the
    # data. With just two dimensions this is trivial, but with more dimensions,
    # it may be better to use a construct such as:

    # In[ ]:

    dimension = smeared_spectrum.get_config().get_pars().index("energy_mc")
    old_shape = smeared_spectrum._data.shape
    reduction_factor = 5  # how many bins to combine into a single bin
    new_shape = tuple([j / reduction_factor if i == dimension else j
                      for i, j in enumerate(old_shape)])
    print old_shape
    print new_shape

    # In[ ]:

    smeared_spectrum.rebin(new_shape)

    # ### Scaling

    # Finally, we "simulated" 1000 events, but we most likely want to scale
    # this down for to represent the number of events expected in our analysis.
    # The `Spectra` class has a `scale` method to accomplish this. Remember
    # that the `scale` method should always be supplied with the number of
    # events the full spectrum (i.e. before any cuts using `shrink` or
    # `shrink_to_roi`) should represent. Lets assume that our spectrum should
    # actually represent 104.25 events:

    # In[ ]:

    smeared_spectrum.scale(104.25)
    print smeared_spectrum.sum()

    # ## Putting it all together

    # After creating, filling, convolving and various other manipulations what
    # does our final spectrum look like?

    # In[ ]:

    print smeared_spectrum._data

    # In[ ]:

    fig_5 = plot.plot_projection(smeared_spectrum, "energy_mc",
                                 fig_num=5, show_plot=False)
    plt.show()


if __name__ == "__main__":
    main()  # run tutorial
