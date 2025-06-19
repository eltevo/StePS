import numpy as np
import time
import sys
from scipy.spatial import Voronoi, ConvexHull

def find_nearest_id(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def get_periodic_distance_vec(Coords1,Coords2,Lbox):
    return np.mod(Coords1 - Coords2 + Lbox / 2, Lbox) - Lbox / 2

def get_periodic_distances(Coords1,Coords2,Lbox):
    return np.sqrt(np.sum(np.power(np.mod(Coords1 - Coords2 + Lbox / 2, Lbox) - Lbox / 2, 2), axis=1))

def voronoi_volumes(points, SILENT=False):
    """
    Function for calculating voronoi volumes
    Input:
        - points: array containing the coordinates of the input particles
    Returns:
        - vol: array containing the volumes of all cells
    """
    if SILENT==False:
        v_start = time.time()
        print("Calculating Voronoi tessellation...")
        sys.stdout.flush()
    v = Voronoi(points)
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s." % (v_end-v_start))
    if SILENT==False:
        v_start = time.time()
        print("Calculating Voronoi volumes...")
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume #NameError: name 'ConvexHull' is not defined
    if SILENT==False:
        v_end = time.time()
        print("...done in %.2f s.\n" % (v_end-v_start))
    return vol