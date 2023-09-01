class GalacticusParameters():
    """
    This class stores names of comonly used galacticus parametersr
    """

    #Names of galacticus parameters
    #Position of the subhalo relative to the top level host halo
    X = "positionOrbitalX"
    """The GALACTICUS output parameter for the X coordinate of the subhalo relative to the main halo"""
    Y = "positionOrbitalY" 
    """The GALACTICUS output parameter for the Y coordinate of the subhalo relative to the main halo"""
    Z = "positionOrbitalZ"
    """The GALACTICUS output parameter for the Z coordinate of the subhalo relative to the main halo"""

    #Position of the subhalo relative to it's host halo or subhalo (not top level)
    RELX = "satellitePositionX"
    """The GALACTICUS output parameter for the X coordinate of the subhalo relative to the halo / subhalo that hosts it."""
    RELY = "satellitePositionY"
    """The GALACTICUS output parameter for the Y coordinate of the subhalo relative to the halo / subhalo that hosts it."""
    RELZ = "satellitePositionZ"
    """The GALACTICUS output parameter for the Z coordinate of the subhalo relative to the halo / subhalo that hosts it."""

    MASS_BOUND = "satelliteBoundMass"
    """The GALACTICUS output parameter for the gravitationally bound mass contained within a subhalo"""
    MASS_BASIC = "basicMass"
    """The GALACTICUS output parameter for the mass at acrettion. Includes mass from substructure."""
    IS_ISOLATED = "nodeIsIsolated"
    """The GALACTICUS output parameter describing if the node is a halo / subhalo. 0 if subhalo 1 if halo"""
    HIERARCHYLEVEL = "nodeHierarchyLevel"
    """The GALACTICUS output parameter describing the level of substructure the current halo exists at. For the "main" halo
    this would be 0, for subhstrucure 1, for subs-substructure 2,..."""
    RVIR = 'darkMatterOnlyRadiusVirial'

    SPHERE_RADIUS = "spheroidRadius"
    """The GALACTICUS output parameter describing the sphereoid radius of a spheroid galaxy"""
    SPHERE_ANGLULARMOMENTUM = "spheroidAngularMomentum"
    """The GALACTICUS output parameter describing the angular momentum of a spheroid galaxy"""
    SPHERE_MASS_STELLAR = "spheroidMassStellar"
    """The GALACTICUS output parameter describing the stellar mass of a spheroid galaxy"""
    SPHERE_MASS_GAS = "spheroidMassGas"
    """The GALACTICUS output parameter describing the gas mass of a spheroid galaxy"""

    SCALE_RADIUS = "darkMatterProfileScaleRadius"
    """The GALACTICUS output parameter describing the scale radius of the halo / subhalo"""
    DENSITY_PROFILE_RADIUS = "densityProfileRadius"
    """The GALACTICUS output parameter describing the density profile radii of the halo / subhalo"""
    DENSITY_PROFILE = "densityProfile"

    Z_LASTISOLATED = "redshiftLastIsolated"

    TNFW_RADIUS_TRUNCATION = "radiusTidalTruncationNFW" #TODO: replace this with actual parameter
    TNFW_RHO_S = "densityNormalizationTidalTruncationNFW"
    


    DEF_TAB = [X,Y,Z,MASS_BOUND,MASS_BASIC,IS_ISOLATED,HIERARCHYLEVEL,RVIR]
    """Default galacticus parameters to include in tabulation"""
