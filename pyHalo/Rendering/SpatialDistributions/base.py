class SpatialDistributionBase(object):

    @classmethod
    def from_Mhost(cls, *args, **kwargs):
        raise Exception('The spatial distribution model specified for subhalos must have a class method .from_Mhost')

    def draw(cls, *args, **kwargs):
        raise Exception('The spatial distribution model specified for subhalos must have a method draw()')
