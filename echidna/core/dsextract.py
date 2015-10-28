import numpy
import rat
import echidna.calc.constants as const


def function_factory(dimension, **kwargs):
    '''Factory function that returns a dsextract class
    corresponding to the dimension (i.e. DS parameter)
    that is being extracted.

    Args:
      dimension (str): to extract from a RAT DS/ntuple file.
      kwargs (dict): to be passed to and checked by the extractor.

    Raises:
      IndexError: dimension is an unknown parameter

    Retuns:
      :class:`echidna.core.dsextract.Extractor`: Extractor object.
    '''
    kwdict = {}
    if "fv_radius" in kwargs:
        kwdict["fv_radius"] = kwargs["fv_radius"]
    if dimension == "energy_mc":
        return EnergyExtractMC(**kwdict)
    elif dimension == "energy_reco":
        return EnergyExtractReco(**kwdict)
    elif dimension == "energy_truth":
        return EnergyExtractTruth(**kwdict)
    elif dimension == "radial_mc":
        return RadialExtractMC(**kwdict)
    elif dimension == "radial_reco":
        return RadialExtractReco(**kwdict)
    elif dimension == "radial3_mc":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return Radial3ExtractMC(**kwdict)
    elif dimension == "radial3_reco":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return Radial3ExtractReco(**kwdict)
    elif dimension == "zradial3_mc":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return ZRadial3ExtractMC(**kwdict)
    elif dimension == "zradial3_reco":
        if "outer_radius" in kwargs:
            kwdict["outer_radius"] = kwargs["outer_radius"]
        return ZRadial3ExtractReco(**kwdict)
    else:
        raise IndexError("Unknown parameter: %s" % dimension)


class Extractor(object):
    '''Base class for extractor classes.

    Args:
      name (str): of the dimension
      fv_radius (float): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.

    Attributes:
      _name (str): of the dimension
      _fv_radius (float): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
    '''

    def __init__(self, name, fv_radius):
        '''Initialise the class
        '''
        self._name = name
        self._fv_radius = fv_radius


class EnergyExtractMC(Extractor):
    '''Quenched energy extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      reco_pos (bool, optional): If true then position cuts will be made on
        reconstructed position. If False (default) then MC position is used
        for cuts.

    Attributes:
      _reco_pos (bool): If true then position cuts will be made on
      reconstructed position. If False then MC position is used for cuts.
    '''

    def __init__(self, fv_radius=None, reco_pos=False):
        '''Initialise the class
        '''
        super(EnergyExtractMC, self).__init__("energy_mc", fv_radius)
        self._reco_pos = reco_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2))) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if entry.scintFit == 1:
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        return False  # Failed reconstruction

    def fv_cut_root(self, mc, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          mc (:class:`RAT.DS.MC`): Monte Carlo entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if ds.GetEVCount == 0:
            return False  # Failed reconstruction
        # Reco cut on MC value requires 1 ev else which position to use?
        if ds.GetEVCount() != 1:
            raise ValueError("Can only mix cuts for 1 mc = 1 ev")
        ev = ds.GetEV(0)
        if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, mc, ds):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`): entry
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            if self._fv_radius:
                return self.fv_cut_root(mc, ds)
            return True  # Passes particle count and no fid v cut
        return False  # Fails particle count

    def get_value_root(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): entry

        Returns:
          float: True quenched energy
        '''
        return mc.GetScintQuenchedEnergyDeposit()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            # MC has associated trigger or vice versa and no fid v applied
            return True
        return False  # No associated trigger with mc or vice versa

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True quenched energy
        '''
        return entry.mcEdepQuenched


class EnergyExtractReco(Extractor):
    '''Reconstructed energy extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      mc_pos (bool, optional): If true then  MC position is used for cuts.
        If False (default) then position cuts will be made on reconstructed
        position.

    Attributes:
      _mc_pos (bool): If true then MC position is used for cuts.
        If False then position cuts will be made on reconstructed position.
    '''

    def __init__(self, fv_radius=None, mc_pos=False):
        '''Initialise the class
        '''
        super(EnergyExtractReco, self).__init__("energy_reco", fv_radius)
        self._mc_pos = mc_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                  (entry.mcPosy)**2 +
                                  (entry.mcPosz)**2))) < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def fv_cut_root(self, ev, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          ev (:class:`RAT.DS.EV`): Event entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                    self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        mc = ds.GetMC()
        if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, ev, ds):
        '''Check whether energy of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsEnergy() \
                and ev.GetDefaultFitVertex().ValidEnergy():
            if self._fv_radius:
                return self.fv_cut_root(ev, ds)
            return True  # Passes fit checks and no fid v cut
        return False  # Fails fit checks

    def get_value_root(self, ev):
        '''Get energy value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed energy
        '''
        return ev.GetDefaultFitVertex().GetEnergy()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.scintFit == 1 and entry.energy > 0:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            return True  # Passes scintFit, has energy and no fid v cut
        return False  # Fails scintFit or no energy

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed energy
        '''
        return entry.energy


class EnergyExtractTruth(Extractor):
    '''True MC energy extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      reco_pos (bool, optional): If true then position cuts will be made on
        reconstructed position. If False (default) then MC position is used
        for cuts.

    Attributes:
      _reco_pos (bool): If true then position cuts will be made on
      reconstructed position. If False then MC position is used for cuts.
    '''

    def __init__(self, fv_radius=None, reco_pos=False):
        '''Initialise the class
        '''
        super(EnergyExtractTruth, self).__init__("energy_truth", fv_radius)
        self._reco_pos = reco_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2))) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if entry.scintFit == 1:
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        return False  # Failed reconstruction

    def fv_cut_root(self, mc, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          mc (:class:`RAT.DS.MC`): Monte Carlo entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if ds.GetEVCount == 0:
            return False  # Failed reconstruction
        # Reco cut on MC value requires 1 ev else which position to use?
        if ds.GetEVCount() != 1:
            raise ValueError("Can only mix cuts for 1 mc = 1 ev")
        ev = ds.GetEV(0)
        if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, mc, ds):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`): entry
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            if self._fv_radius:
                return self.fv_cut_root(mc, ds)
            return True  # Passes particle count and no fid v cut
        return False  # Fails particle count

    def get_value_root(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): entry

        Returns:
          float: True energy
        '''
        return mc.GetScintEnergyDeposit()

    def get_valid_ntuple(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            # MC has associated trigger or vice versa and no fid v applied
            return True
        return False  # No associated trigger with mc or vice versa

    def get_value_ntuple(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True energy
        '''
        return entry.mcEdep


class RadialExtractMC(Extractor):
    '''True radial extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      reco_pos (bool, optional): If true then position cuts will be made on
        reconstructed position. If False (default) then MC position is used
        for cuts.

    Attributes:
      _reco_pos (bool): If true then position cuts will be made on
      reconstructed position. If False then MC position is used for cuts.
    '''

    def __init__(self, fv_radius=None, reco_pos=False):
        '''Initialise the class
        '''
        super(RadialExtractMC, self).__init__("radial_mc", fv_radius)
        self._reco_pos = reco_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2))) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if entry.scintFit == 1:
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        return False  # Failed reconstruction

    def fv_cut_root(self, mc, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          mc (:class:`RAT.DS.MC`): Monte Carlo entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if ds.GetEVCount == 0:
            return False  # Failed reconstruction
        # Reco cut on MC value requires 1 ev else which position to use?
        if ds.GetEVCount() != 1:
            raise ValueError("Can only mix cuts for 1 mc = 1 ev")
        ev = ds.GetEV(0)
        if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, mc, ds):
        '''Check whether radius of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            if self._fv_radius:
                return self.fv_cut_root(mc, ds)
            return True  # Passes particle count and no fid v cut
        return False  # Fails particle count

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): event

        Returns:
          float: True radius
        '''
        return mc.GetMCParticle(0).GetPosition().Mag()

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            # MC has associated trigger or vice versa and no fid v applied
            return True
        return False  # No associated trigger with mc or vice versa

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True radius
        '''
        return numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                     (entry.mcPosy)**2 +
                                     (entry.mcPosz)**2))


class RadialExtractReco(Extractor):
    '''Reconstructed radial extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      mc_pos (bool, optional): If true then  MC position is used for cuts.
        If False (default) then position cuts will be made on reconstructed
        position.

    Attributes:
      _mc_pos (bool): If true then MC position is used for cuts.
        If False then position cuts will be made on reconstructed position.
    '''

    def __init__(self, fv_radius=None, mc_pos=False):
        '''Initialise the class
        '''
        super(RadialExtractReco, self).__init__("radial_reco", fv_radius)
        self._mc_pos = mc_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                  (entry.mcPosy)**2 +
                                  (entry.mcPosz)**2))) < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def fv_cut_root(self, ev, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          ev (:class:`RAT.DS.EV`): Event entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                    self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        mc = ds.GetMC()
        if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, ev, ds):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsPosition() \
                and ev.GetDefaultFitVertex().ValidPosition():
            if self._fv_radius:
                return self.fv_cut_root(ev, ds)
            return True
        return False

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed radius
        '''
        return ev.GetDefaultFitVertex().GetPosition().Mag()

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.scintFit == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            return True  # Passes scintFit and no fid v cut
        return False  # Fails scintFit

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed radius
        '''
        return numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2))


class Radial3ExtractMC(Extractor):
    ''' True :math:`(radius/outer\_radius)^3` radial extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`. If None then the av_radius in
        :class:`echidna.calc.constants` is used in the calculation.

    Attributes:
      _outer_radius (float): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`.
      _reco_pos (bool): If true then position cuts will be made on
      reconstructed position. If False then MC position is used for cuts.
    '''

    def __init__(self, fv_radius=None, outer_radius=None, reco_pos=False):
        '''Initialise the class
        '''
        super(Radial3ExtractMC, self).__init__("radial3_mc", fv_radius)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._reco_pos = reco_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2))) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if entry.scintFit == 1:
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        return False  # Failed reconstruction

    def fv_cut_root(self, mc, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          mc (:class:`RAT.DS.MC`): Monte Carlo entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if ds.GetEVCount == 0:
            return False  # Failed reconstruction
        # Reco cut on MC value requires 1 ev else which position to use?
        if ds.GetEVCount() != 1:
            raise ValueError("Can only mix cuts for 1 mc = 1 ev")
        ev = ds.GetEV(0)
        if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, mc, ds):
        '''Check whether radius of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            if self._fv_radius:
                return self.fv_cut_root(mc, ds)
            return True  # Passes particle count and no fid v cut
        return False  # Fails particle count

    def get_value_root(self, mc):
        '''Get radius value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): event

        Returns:
          float: True :math:`(radius/outer\_radius)^3`
        '''
        return (mc.GetMCParticle(0).GetPosition().Mag() /
                self._outer_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            # MC has associated trigger or vice versa and no fid v applied
            return True
        return False  # No associated trigger with mc or vice versa

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True :math:`(radius/outer\_radius)^3`
        '''
        return (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2)) /
                self._outer_radius) ** 3


class Radial3ExtractReco(Extractor):
    ''' Reconstructed :math:`(radius/outer_radius)^3` radial extraction
      methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`. If None then the av_radius in
        :class:`echidna.calc.constants` is used in the calculation.
      mc_pos (bool, optional): If true then  MC position is used for cuts.
        If False (default) then position cuts will be made on reconstructed
        position.

    Attributes:
      _outer_radius (float: The fixed radius used in calculating
        :math:`(radius/outer\_radius)^3`.
      _mc_pos (bool): If true then MC position is used for cuts.
        If False then position cuts will be made on reconstructed position.
    '''

    def __init__(self, fv_radius=None, outer_radius=None, mc_pos=False):
        '''Initialise the class
        '''
        super(Radial3ExtractReco, self).__init__("radial3_reco", fv_radius)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._mc_pos = mc_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                  (entry.mcPosy)**2 +
                                  (entry.mcPosz)**2))) < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def fv_cut_root(self, ev, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          ev (:class:`RAT.DS.EV`): Event entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                    self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        mc = ds.GetMC()
        if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, ev, ds):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsEnergy() \
                and ev.GetDefaultFitVertex().ValidEnergy():
            if self._fv_radius:
                return self.fv_cut_root(ev, ds)
            return True  # Passes fit checks and no fid v cut
        return False  # Fails fit checks

    def get_value_root(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed :math:`(radius/outer\_radius)^3`
        '''
        return (ev.GetDefaultFitVertex().GetPosition().Mag() /
                self._outer_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.scintFit == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            return True  # Passes scintFit and no fid v cut
        return False  # Fails scintFit

    def get_value_ntuple(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed :math:`(radius/outer\_radius)^3`
        '''
        return (numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                      (entry.posy)**2 +
                                      (entry.posz)**2)) /
                self._outer_radius) ** 3


class ZRadial3ExtractMC(Extractor):
    ''' True :math:`(z/\lvert z \rvert)(radius/outer\_radius)^3` radial
      extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`. If None then
        the av_radius in :class:`echidna.calc.constants` is used in the
        calculation.

    Attributes:
      _outer_radius (float): The fixed radius used in calculating
        :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`.
      _reco_pos (bool): If true then position cuts will be made on
      reconstructed position. If False then MC position is used for cuts.
    '''

    def __init__(self, fv_radius=None, outer_radius=None, reco_pos=False):
        '''Initialise the class
        '''
        super(ZRadial3ExtractMC, self).__init__("zradial3_mc", fv_radius)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._reco_pos = reco_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                      (entry.mcPosy)**2 +
                                      (entry.mcPosz)**2))) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if entry.scintFit == 1:
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        return False  # Failed reconstruction

    def fv_cut_root(self, mc, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          mc (:class:`RAT.DS.MC`): Monte Carlo entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._reco_pos:  # Default is to use MC info for the cut
            if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if ds.GetEVCount == 0:
            return False  # Failed reconstruction
        # Reco cut on MC value requires 1 ev else which position to use?
        if ds.GetEVCount() != 1:
            raise ValueError("Can only mix cuts for 1 mc = 1 ev")
        ev = ds.GetEV(0)
        if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, mc, ds):
        '''Check whether radius of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if mc.GetMCParticleCount > 0:
            if self._fv_radius:
                return self.fv_cut_root(mc, ds)
            return True  # Passes particle count and no fid v cut
        return False  # Fails particle count

    def get_value_root(self, mc):
        '''Get :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
          value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`): event

        Returns:
          float: True :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
        '''
        if mc.GetMCParticle(0).GetPosition()[2] < 0.:
            return -1. * (mc.GetMCParticle(0).GetPosition().Mag() /
                          self._outer_radius) ** 3
        else:
            return (mc.GetMCParticle(0).GetPosition().Mag() /
                    self._outer_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.mc == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            # MC has associated trigger or vice versa and no fid v applied
            return True
        return False  # No associated trigger with mc or vice versa

    def get_value_ntuple(self, entry):
        '''Get :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
          value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: True :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
        '''
        if entry.mcPosz < 0.:
            return -1. * (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                                (entry.mcPosy)**2 +
                                                (entry.mcPosz)**2)) /
                          self._outer_radius) ** 3
        else:
            return (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                          (entry.mcPosy)**2 +
                                          (entry.mcPosz)**2)) /
                    self._outer_radius) ** 3


class ZRadial3ExtractReco(Extractor):
    ''' Reconstructed :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
      radial extraction methods.

    Args:
      fv_radius (float, optional): Fiducial radius. Applies a cut to remove
        events which have a radial position greater than the radius of the
        fiducial volume. If None no cut is applied.
      outer_radius (float, optional): The fixed radius used in calculating
        :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`. If None then
        the av_radius in :class:`echidna.calc.constants` is used in the
        calculation.
      mc_pos (bool, optional): If true then  MC position is used for cuts.
        If False (default) then position cuts will be made on reconstructed
        position.

    Attributes:
      _outer_radius (float: The fixed radius used in calculating
        :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`.
      _mc_pos (bool): If true then MC position is used for cuts.
        If False then position cuts will be made on reconstructed position.
    '''

    def __init__(self, fv_radius=None, outer_radius=None, mc_pos=False):
        '''Initialise the class
        '''
        super(ZRadial3ExtractReco, self).__init__("zradial3_reco", fv_radius)
        if outer_radius:
            self._outer_radius = outer_radius
        else:
            self._outer_radius = const._av_radius
        self._mc_pos = mc_pos

    def fv_cut_ntuple(self, entry):
        """ Applies the fiducial volume (fv) cut to an ntuple entry.

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                     (entry.posy)**2 +
                                     (entry.posz)**2)) < self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        if (numpy.fabs(numpy.sqrt((entry.mcPosx)**2 +
                                  (entry.mcPosy)**2 +
                                  (entry.mcPosz)**2))) < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def fv_cut_root(self, ev, ds):
        """ Applies the fiducial volume (fv) cut to a root entry.

        Args:
          ev (:class:`RAT.DS.EV`): Event entry
          ds (:class:`RAT.DS`): Data Structure entry

        Raises:
          ValueError: If reconstruced position cut is used on an mc event
            which has multiple evs.

        Returns:
          bool: Indicates pass or fail of cut
        """
        if not self._mc_pos:  # Default is to use reco info for the cut
            if ev.GetDefaultFitVertex().GetPosition().Mag() < \
                    self._fv_radius:
                return True  # Passes fiducial volume cut
            return False  # Fails fiducial volume cut
        mc = ds.GetMC()
        if mc.GetMCParticle(0).GetPosition().Mag() < self._fv_radius:
            return True  # Passes fiducial volume cut
        return False  # Fails fiducial volume cut

    def get_valid_root(self, ev, ds):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`): event
          ds (:class:`RAT.DS`): Data Structure entry

        Returns:
          bool: Validity boolean
        '''
        if ev.DefaultFitVertexExists() and \
                ev.GetDefaultFitVertex().ContainsEnergy() \
                and ev.GetDefaultFitVertex().ValidEnergy():
            if self._fv_radius:
                return self.fv_cut_root(ev, ds)
            return True  # Passes fit checks and no fid v cut
        return False  # Fails fit checks

    def get_value_root(self, ev):
        '''Get :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3` value from
          a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`): event

        Returns:
          float: Reconstructed
            :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
        '''
        if ev.GetDefaultFitVertec().GetPositon()[2] < 0.:
            return 1. * (ev.GetDefaultFitVertex().GetPosition().Mag() /
                         self._outer_radius) ** 3
        else:
            return (ev.GetDefaultFitVertex().GetPosition().Mag() /
                    self._outer_radius) ** 3

    def get_valid_ntuple(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          bool: Validity boolean
        '''
        if entry.scintFit == 1:
            if self._fv_radius:
                return self.fv_cut_ntuple(entry)
            return True  # Passes scintFit and no fid v cut
        return False  # Fails scintFit

    def get_value_ntuple(self, entry):
        '''Get :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3` value from
          an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`): chain entry

        Returns:
          float: Reconstructed
            :math:``(z/\lvert z \rvert)(radius/outer\_radius)^3`
        '''
        if entry.posz < 0.:
            return -1. * (numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                                (entry.posy)**2 +
                                                (entry.posz)**2)) /
                          self._outer_radius) ** 3
        else:
            return (numpy.fabs(numpy.sqrt((entry.posx)**2 +
                                          (entry.posy)**2 +
                                          (entry.posz)**2)) /
                    self._outer_radius) ** 3
