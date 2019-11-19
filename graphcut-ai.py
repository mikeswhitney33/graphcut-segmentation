import os
import torch


from PIL import Image
import numpy as np
import maxflow

from pyfigtree import figtree
from graphcut import n_weight

N = 1000000


class GraphcutAI(object):
    def __init__(self, root):
        self.im_path = os.path.join(root, "images")
        self.an_path = os.path.join(root, "images-labels")
        self.gt_path = os.path.join(root, "images-gt")
        impaths = [path.split(".")[0] for path in os.listdir(self.im_path)]
        self.states = [{"impath": path, "lam": 0.5, "sig": 0.5} for path in impaths]

        self.eps = 1.

    def getIMName(self, path):
        if os.path.exists("{}.png".format(path)):
            return "{}.png".format(path)
        elif os.path.exists("{}.jpg".format(path)):
            return "{}.jpg".format(path)
        else:
            return "{}.bmp".format(path)

    def getState(self):
        stateidx = np.random.choice(np.arange(len(self.states)))
        return self.getStateI(stateidx)

    def getStateI(self, idx):
        state = self.states[idx]
        sig = state['sig']
        imname = self.getIMName(os.path.join(self.im_path, state['impath']))
        image = np.float64(Image.open(imname).convert("RGB").resize((224, 224))) / 255

        anname = self.getIMName(os.path.join(self.an_path, "{}-anno".format(state['impath'])))
        ann = np.array(Image.open(anname).resize((224, 224), 0))

        fseeds = image[ann==1]
        bseeds = image[ann==2]

        flat_img = image.reshape(-1, 3)
        f_weights = figtree(fseeds, flat_img, np.ones(fseeds.shape[0]), sig, eval='direct').reshape(224, 224)
        b_weights = figtree(bseeds, flat_img, np.ones(bseeds.shape[0]), sig, eval='direct').reshape(224, 224)
        t_weights = f_weights + b_weights
        f = f_weights / t_weights
        b = b_weights / t_weights
        f[ann==1] = 1
        b[ann==2] = 1

        regions = np.array([f, b])

        dx = image[:,1:] - image[:,:-1]
        dy = image[1:] - image[:-1]

        boundaries = np.zeros((2, 224, 224))
        nx = n_weight(dx, sig)
        ny = n_weight(dy, sig)
        boundaries[0, :,1:] = nx
        boundaries[0, 1:] = ny
        boundaries = boundaries * state['lam']

        weights = np.concatenate([regions, boundaries])

        return idx, weights


    def getAction(self, state):
        if np.random.uniform() < self.eps:
            action = np.random.uniform(size=6)
        else:
            action = None
        self.eps *= 0.999
        return action

    def score(self, idx, state):
        _, h, w = state.shape
        graph = maxflow.GraphFloat(h * w)
        nodes = graph.add_grid_nodes((h, w))

        regions = state[:2]
        f = regions[0]
        b = regions[1]
        graph.add_grid_tedges(nodes, b, f)


        boundaries = state[2:]
        x = boundaries[0, :,1:]
        y = boundaries[1, 1:]
        graph.add_grid_edges(nodes[:,1:], x, [[0, 0, 0], [0, 0, 1], [0, 0, 0]], True)
        graph.add_grid_edges(nodes[1:], y, [[0, 0, 0], [0, 0, 0], [0, 1, 0]], True)

        graph.maxflow()
        seg = graph.get_grid_segments(nodes)

        gt_name = self.getIMName(os.path.join(self.gt_path, self.states[idx]['impath']))
        gt = np.array(Image.open(gt_name).resize((224, 224), 0)) == 255

        return (gt & seg).sum() / (gt | seg).sum()

    def move(self, sidx, action):
        if np.argmax(action[:3]) == 0:
            self.states[sidx]['lam'] = max(self.states[sidx]['lam'] - 0.001, 0)
        elif np.argmax(action[:3]) == 2:
            self.states[sidx]['lam'] += 0.001

        if np.argmax(action[3:]) == 0:
            self.states[sidx]['sig'] = max(self.states[sidx]['sig'] - 0.001, 0.001)
        elif np.argmax(action[3:]) == 2:
            self.states[sidx]['sig'] += 0.001

        _, state = self.getStateI(sidx)

        term = False
        if np.argmax(action[:3]) == 1 and np.argmax(action[3:]) == 1:
            term = True

        score = self.score(sidx, state)
        return state, term, score

    def getY(self, rj, state, terminal):
        return rj

    def step(self, yj, sj, aj):
        pass

class Transition(object):
    def __init__(self, st, at, rt, st1, terminal):
        self.st = st
        self.at = at
        self.rt = rt
        self.st1 = st1
        self.terminal = terminal


class ReplayMemory(object):
    def __init__(self, N):
        self.cap = N
        self.pos = 0
        self.memory = []

    def insert(self, item):
        if len(self.memory) >= self.cap:
            self.memory[self.pos] = item
            self.pos = (self.pos + 1) % self.cap
        else:
            self.memory.append(item)

    def batch(self, n):
        return np.random.choice(self.memory, min(n, len(self.memory)), replace=False)

    def __len__(self):
        return len(self.memory)


def run(ai, num_episodes, batch_size, gamma):
    D = ReplayMemory(N)
    for _ in range(num_episodes):
        sidx, st = ai.getState()
        at = ai.getAction(st)
        st1, termt, rt = ai.move(sidx, at)
        D.insert(Transition(st, at, rt, st1, termt))
        print(len(D))
        if len(D) > batch_size:
            batch = D.batch(batch_size)
            sj = [b.st for b in batch]
            aj = [b.at for b in batch]
            rj = [b.rt for b in batch]
            sj1 = [b.st1 for b in batch]
            termj = [b.terminal for b in batch]
            yj = ai.getY(rj, sj1, termj)
            ai.step(yj, sj, aj)



if __name__ == "__main__":
    run(GraphcutAI("/Data6/mike/vggiseg"), 100, 32, .99)
