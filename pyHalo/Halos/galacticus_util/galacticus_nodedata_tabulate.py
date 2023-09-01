from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_file import GalacticusFile
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_hdf5parameters import GalacticusHDF5Parameters as pnhdf
import numpy as np

def tabulate_node_data(galacticus_file:GalacticusFile,outputn:int = None)->dict[{str,np.ndarray}]:

    output_group = pnhdf.PROPERTY_OUTPUTS_FINAL
    if not outputn is None:
        output_group = f"{pnhdf.GGROUP_OUTPUT_PREFIX}{outputn}"

    tabulated = galacticus_file[pnhdf.GGROUP_OUTPUT_PRIMARY][output_group].tabulate_datasets()

    nodecount = np.sum(tabulated["tree_mergerTreeCount"])
 
    return {key:val for key,val in tabulated.items() if val.shape[0] == nodecount}
    