import math
import rat

# Not currently used
_dimensions = ["energy", "radial"]

def function_factory(dimension):
    if dimension=="energy":
        return EnergyExtract()
    elif dimension=="radial":
        return RadialExtract()
    else:
        raise IndexError("Unknown parameter: %s" % dimension)


class EnergyExtract(object):

    def __init__(self):
        pass

    def ev_get_valid(self, ev):
        if ev.DefaultFitVertexExists() and ev.GetDefaultFitVertex().ContainsEnergy() \
           and ev.GetDefaultFitVertex().ValidEnergy():
            return True
        return False
    
    def ev_get_value(self, ev):
        return ev.GetDefaultFitVertex().GetEnergy()

    def mc_get_valid(self, mc):
        return True

    def mc_get_value(self, mc):
        return mc.GetScintQuenchedEnergyDeposit()

    def ntuple_ev_get_valid(self, entry):
        if entry.scintFit == 0:
            return False
        return True

    def ntuple_ev_get_value(self, entry):
        return entry.mcEdepQuenched


class RadialExtract(object):

    def __init__(self):
        pass

    def ev_get_valid(self, ev):
        if ev.DefaultFitVertexExists() and ev.GetDefaultFitVertex().ContainsPosition() \
           and ev.GetDefaultFitVertex().ValidPosition():
            return True
        return False
    
    def ev_get_value(self, ev):
        return ev.GetDefaultFitVertex().GetPosition().Mag()

    def mc_get_valid(self, mc):
        return True

    def mc_get_value(self, mc):
        return mc.GetMCParticle(0).GetPosition().Mag()

    def ntuple_ev_get_valid(self, entry):
        return True

    def ntuple_ev_get_value(self, entry):
        return math.fabs(math.sqrt((entry.posx)**2 +
                                   (entry.posy)**2 +
                                   (entry.posz)**2)

    def ntuple_mc_get_valid(self, entry):
        return True

    def ntuple_mc_get_value(self, entry):
        return math.fabs(math.sqrt((entry.mcPosx)**2
                                   (entry.mcPosy)**2 +
                                   (entry.mcPosz)**2)
