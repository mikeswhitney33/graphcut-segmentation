exec env LD_LIBRARY_PATH=. python -x "$0" "$@"
import maxflow
import numpy as np
from pyfigtree import figtree


def graphcut(image, fore, back, lam, sig):
    h, w, c = image.shape
    graph = maxflow.GraphFloat(h * w)
    nodes = graph.add_grid_nodes((h, w))
