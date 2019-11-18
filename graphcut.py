import maxflow
import numpy as np
from pyfigtree import figtree


def n_weight(d, sig):
    sig2 = 2 * sig * sig
    return np.exp(-np.square(np.sum(d, axis=2))/sig2)

def graphcut(image, fore, back, lam, sig):
    h, w, _ = image.shape
    graph = maxflow.GraphFloat(h * w)
    nodes = graph.add_grid_nodes((h, w))

    dx = image[:,1:] - image[:,:-1]
    dy = image[1:] - image[:-1]
