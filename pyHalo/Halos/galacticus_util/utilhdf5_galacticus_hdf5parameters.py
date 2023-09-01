#!/usr/bin/env python
class GalacticusHDF5Parameters():
    """filename.py: Various constants used by utilhdf5 python code"""

    GGROUP_OUTPUT_PRIMARY = "Outputs"
    """
    The name of the primary output group in a GALACTICUS output file. 
    This is the primary output group that contains nodedata at different output times
    """

    GGROUP_OUTPUT_PREFIX = "Output"
    """The name of the prefix of the "OutputN" groups in GALACTICUS output file"""

    GGROUP_NODEDATA = "nodeData"
    """Name of group that contains nodedata datasets"""

    GPARAM_MERGERTREE_INDEX_START = "mergerTreeStartIndex"
    """GALACTICUS name for datasets that contain the start of merger trees"""
    GPARAM_MERGERTREE_INDEX = "mergerTreeIndex"
    """GALACTICUS name for datasets that contain the start of merger trees"""
    GPARAM_MERGERTREE_COUNT = "mergerTreeCount"
    """GALACTICUS name for datasets that contain the start of merger trees"""


    #Documentation
    DOCSTR_GFILE = """Data read from a galcticus file."""
    DOCSTR_NODOCUMENTATION = """No documentation provided"""

    #Custom properties
    CUSTOM_PREFIX = "custom_"

    PROPERTY_OUTPUTS_FIRST = CUSTOM_PREFIX + "property_firstoutput"
    """Key for getting the first output (highest redshift) output group"""
    PROPERTY_OUTPUTS_FINAL = CUSTOM_PREFIX + "property_finaloutput"
    """Key for getting the final output (lowest redshift) output group"""
    GETFINALOUTPUT = PROPERTY_OUTPUTS_FINAL
    """Key for getting the final output (lowest redshift) output group"""

    PROPERTY_KEYS_NODEDATA = CUSTOM_PREFIX + "keys_nodedata"
    """Key for getting nodedata keys"""

    PROPERTY_KEYS_TREEDATA = CUSTOM_PREFIX + "keys_tree"
    """Key for getting tree data keys"""

    PROPERTY_NODE_TREE = CUSTOM_PREFIX + "node_tree"
    """Key for getting the tree each node belongs to"""

    PROPERTY_NODE_TREE_OUTPUTORDER = CUSTOM_PREFIX + "node_tree_outputorder"
    """Key for getting the tree each node belongs to"""

    #KWARGS
    KWARG_NODEDATA_ALLOWED = "nodedata_allowed"

    #Prefixes
    PREFIX_TREE_DSET = "tree_"