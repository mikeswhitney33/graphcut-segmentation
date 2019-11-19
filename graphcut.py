import maxflow
import numpy as np
from pyfigtree import figtree


def n_weight(d, sig):
    sig2 = 2 * sig * sig
    return np.exp(-np.square(np.sum(d, axis=2))/sig2)

def graphcut(image, fore, back, lam, sig):
    image = np.float64(image) / 255
    h, w, _ = image.shape
    graph = maxflow.GraphFloat(h * w)
    nodes = graph.add_grid_nodes((h, w))

    dx = image[:,1:] - image[:,:-1]
    dy = image[1:] - image[:-1]

    struct_left = [[0, 0, 0], [0, 0, lam], [0, 0, 0]]
    struct_down = [[0, 0, 0], [0, 0, 0], [0, lam, 0]]
    nx = n_weight(dx, sig)
    ny = n_weight(dy, sig)

    graph.add_grid_edges(nodes[:,1:], nx, struct_left, True)
    graph.add_grid_edges(nodes[1:], ny, struct_down, True)

    fseeds = image[fore]
    bseeds = image[back]
    flat_img = image.reshape(-1, 3)
    f_weights = figtree(fseeds, flat_img, np.ones(fseeds.shape[0]), sig, eval="direct").reshape(h, w)
    b_weights = figtree(bseeds, flat_img, np.ones(bseeds.shape[0]), sig, eval="direct").reshape(h, w)
    t_weights = f_weights + b_weights
    f = f_weights / t_weights
    b = b_weights / t_weights

    f[fore] = 1
    b[back] = 1

    graph.add_grid_tedges(nodes, b, f)
    graph.maxflow()
    return graph.get_grid_segments(nodes)
