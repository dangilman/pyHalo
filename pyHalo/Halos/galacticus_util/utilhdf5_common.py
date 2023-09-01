#!/usr/bin/env python

"""filename.py: Imports modules used by utilhdf5 python files"""

from typing import Iterable, Any, Callable, Protocol
import h5py
import numpy as np
from pyHalo.Halos.galacticus_util.utilhdf5_galacticus_hdf5parameters import GalacticusHDF5Parameters

def ifisnone(arg:(Any | None),default:Any)->Any:
     """If arg is None returns default, else returns arg"""
     return default if arg is None else arg