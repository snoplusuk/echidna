import math
import rat


def function_factory(dimension):
    '''Factory function that returns a dsextract class
    corresponding to the dimension (i.e. DS parameter)
    that is being extracted.

    Args:
      dimension (str): to extract from a RAT DS/ntuple file.

    Retuns:
      Extractor object.
    '''

    if dimension=="energy":
        return EnergyExtract()
    elif dimension=="radial":
        return RadialExtract()
    else:
        raise IndexError("Unknown parameter: %s" % dimension)


class Extractor(object):
    '''Base class for extractor classes.

    Args:
      name (str): of the dimension

    Attributes:
      _name (str): of the dimension
    '''
    
    def __init__(self, name):
        '''Initialise the class
        '''
        self.name = name


class EnergyExtract(Extractor):
    '''Energy extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(EnergyExtract, self).__init__("energy")

    def ev_get_valid(self, ev):
        '''Check whether energy of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          Validity boolean
        '''
        if ev.DefaultFitVertexExists() and ev.GetDefaultFitVertex().ContainsEnergy() \
           and ev.GetDefaultFitVertex().ValidEnergy():
            return True
        return False
    
    def ev_get_value(self, ev):
        '''Get energy value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          Reconstructed energy
        '''
        return ev.GetDefaultFitVertex().GetEnergy()

    def mc_get_valid(self, mc):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          Validity boolean
        '''
        return True

    def mc_get_value(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          True quenched energy
        '''
        return mc.GetScintQuenchedEnergyDeposit()

    def truth_get_valid(self, mc):
        '''Check whether energy of a DS::MC is valid

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          Validity boolean
        '''
        return True

    def truth_get_value(self, mc):
        '''Get energy value from a DS::MC

        Args:
          mc (:class:`RAT.DS.MC`) entry

        Returns:
          True energy
        '''
        return mc.GetScintEnergyDeposit()

    def ntuple_ev_get_valid(self, entry):
        '''Check whether energy of an ntuple EV is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          Validity boolean
        '''
        return (entry.scintFit != 0 and entry.energy>0)

    def ntuple_ev_get_value(self, entry):
        '''Get energy value from an ntuple EV

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          Reconstructed energy
        '''
        return entry.energy

    def ntuple_mc_get_valid(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          Validity boolean
        '''
        return True

    def ntuple_mc_get_value(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          True quenched energy
        '''
        return entry.mcEdepQuenched

    def ntuple_truth_get_valid(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          Validity boolean
        '''
        return True

    def ntuple_truth_get_value(self, entry):
        '''Get energy value from an ntuple MC

        Args:
          entry (:class:`ROOT.TChain`) chain entry

        Returns:
          True quenched energy
        '''
        return entry.mcEdepQuenched


class RadialExtract(Extractor):
    '''Radial extraction methods.
    '''

    def __init__(self):
        '''Initialise the class
        '''
        super(RadialExtract, self).__init__("radial")

    def ev_get_valid(self, ev):
        '''Check whether radius of a DS::EV is valid

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          Validity boolean
        '''
        if ev.DefaultFitVertexExists() and ev.GetDefaultFitVertex().ContainsPosition() \
           and ev.GetDefaultFitVertex().ValidPosition():
            return True
        return False
    
    def ev_get_value(self, ev):
        '''Get radius value from a DS::EV

        Args:
          ev (:class:`RAT.DS.EV`) event

        Returns:
          Reconstructed radius
        '''
        return ev.GetDefaultFitVertex().GetPosition().Mag()

    def mc_get_valid(self, mc):
        '''Check whether radius of a DS::MC is valid

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          Validity boolean
        '''
        return True

    def mc_get_value(self, mc):
        '''Get radius value from a DS::MC

        Args:
          ev (:class:`RAT.DS.MC`) event

        Returns:
          True radius
        '''
        return mc.GetMCParticle(0).GetPosition().Mag()

    def ntuple_ev_get_valid(self, entry):
        '''Check whether radius of an ntuple EV is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          Validity boolean
        '''
        return entry.scintFit != 0

    def ntuple_ev_get_value(self, entry):
        '''Get radius value from an ntuple EV

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          Reconstructed radius
        '''
        return math.fabs(math.sqrt((entry.posx)**2 +
                                   (entry.posy)**2 +
                                   (entry.posz)**2))

    def ntuple_mc_get_valid(self, entry):
        '''Check whether energy of an ntuple MC is valid

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          Validity boolean
        '''
        return True

    def ntuple_mc_get_value(self, entry):
        '''Get radius value from an ntuple MC

        Args:
          ev (:class:`ROOT.TChain`) chain entry

        Returns:
          True radius
        '''
        return math.fabs(math.sqrt((entry.mcPosx)**2 +
                                   (entry.mcPosy)**2 +
                                   (entry.mcPosz)**2))
