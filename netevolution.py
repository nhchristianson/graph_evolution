import numpy as np
import scipy as sp
import networkx as nx
import numba as nb
from numba import jitclass, njit
from numba import int64, float64
from numba.types import UniTuple
from numba.typed import Dict

@njit()
def erdos_renyi(N, p):
    np.random.seed(1000)
    G = np.zeros((N, N), dtype=float64)
    edges = set()
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            if np.random.rand() < p:
                G[i, j] = 1.
                G[j, i] = 1.
                edges.add((i, j))
                edges.add((j, i))
    d = dict()
    disconnected = set()
    for i in np.arange(N):
        d[i] = np.where(G[i] != 0.)[0]
        if len(d[i]) == 0:
            disconnected.add(i)
    return G, d, edges

spec = [
    ('N', int64),
    ('payoffs', float64[::1]),
    ('G', float64[:, ::1]),
    ('d_nbhd', nb.typeof(Dict.empty(
                         key_type=int64,
                         value_type=int64[::1]))),
    ('strategies', float64[:, ::1])
]

@jitclass(spec)
class GraphEvo(object):
    def __init__(self, N, p, payoffs):
        self.N = N
        self.payoffs = payoffs
        self.G, self.d_nbhd, _ = erdos_renyi(N, p)
        self.strategies = np.random.uniform(0.0, 1.0, (N, 2))

    def stationary(self, strat1, strat2):
        # print(strat1, strat2)
        p1, q1 = strat1
        p2, q2 = strat2
        r1, r2 = p1 - q1, p2 - q2
        x = np.array([[1., 1.], [1.4, 2.4]])
        if np.abs(r1*r2) < 1:
            # print(r1, r2, p1, p2)
            s1 = (q2*r1 + q1)/(1 - r1*r2)
            s2 = (q1*r2 + q2)/(1 - r1*r2)
            # print(s1, s2)
            stat = [s1*s2, s1*(1 - s2), s2*(1 - s1), (1 - s1)*(1 - s2)]
            # print(stat)
            # print()
        else:
            print('no good')
        return np.array(stat)

    def calc_payoffs(self, new_strat, prev_strat,
                     strategies, neighbors, payoffs):
        n_nhbs = neighbors.shape[0]
        # print(n_nhbs)
        prev_payoffs = np.zeros(n_nhbs)
        new_payoffs = np.zeros(n_nhbs)
        for nhb in np.arange(n_nhbs):
            # print(nhb, neighbors[nhb])
            # print(prev_strat, new_strat)
            prev_stat = self.stationary(prev_strat, strategies[int(neighbors[nhb])])
            new_stat = self.stationary(new_strat, strategies[int(neighbors[nhb])])
            # print(prev_stat, new_stat)
            prev_payoffs[nhb] = np.dot(payoffs, prev_stat)
            new_payoffs[nhb] = np.dot(payoffs, new_stat)
        return prev_payoffs, new_payoffs

    """
    Just a regular old Monte Carlo step -

    """
    def mc_step(self):
        i = np.random.choice(self.N)
        new_strat = np.random.uniform(0.4, 0.6, 2)
        # calculate payoffs against neighbors
        prev_mean, new_mean = self.calc_payoffs(new_strat, self.strategies[i],
                                                self.strategies, self.d_nbhd[i],
                                                self.payoffs)
        if np.mean(new_mean) > np.mean(prev_mean):
            self.strategies[i] = new_strat


    """
    Monte Carlo step that also might change edges

    """
    def mc_step_edge(self, alpha):
        i = np.random.choice(self.N)
        # if this node has neighbors
        if self.d_nbhd[i].shape[0] > 0:
            new_strat = np.random.uniform(0.0, 1.0, 2)
            # nb_strats = self.strategies[self.d_nbhd[i]]
            # new_strat = np.array([np.mean(nb_strats[:, 0]),
            #                       np.mean(nb_strats[:, 1])])
            # calculate payoffs against neighbors
            prev_mean, new_mean = self.calc_payoffs(new_strat, self.strategies[i],
                                                    self.strategies, self.d_nbhd[i],
                                                    self.payoffs)
            if np.mean(new_mean) > np.mean(prev_mean):
                self.strategies[i] = new_strat
            #try changing edge dist by connecting to neighbors if there are some
            if len(self.d_nbhd[i]) < self.N - 1 and np.random.rand() < alpha:
                non_nbs = np.arange(self.N)
                non_nbs = np.delete(non_nbs, [i])
                for z in self.d_nbhd[i]:
                    non_nbs = non_nbs[non_nbs != z]
                j = np.random.choice(non_nbs)
                # print('got here')
                new_stat = self.stationary(self.strategies[i], self.strategies[j])
                # print('got here')
                new_payoff = np.dot(self.payoffs, new_stat)
                if new_payoff > np.min(new_mean):
                    ind = np.argmin(new_mean)
                    k = self.d_nbhd[i][ind]
                    self.G[i, k] = 0.
                    self.G[k, i] = 0.
                    self.G[i, j] = 1.
                    self.G[j, i] = 1.
                    self.d_nbhd[i] = np.where(self.G[i] != 0.)[0]
                    self.d_nbhd[k] = np.where(self.G[k] != 0.)[0]
                    self.d_nbhd[j] = np.where(self.G[j] != 0.)[0]
        # # if no neighbors
        elif np.random.rand() < alpha:
            j = np.random.choice(np.delete(np.arange(self.N), i))
            edges = np.where(self.G != 0)
            edge = np.random.choice(len(edges[0]))
            k, l = edges[0][edge], edges[1][edge]
            self.G[i, j] = 1.
            self.G[j, i] = 1.
            self.G[k, l] = 0.
            self.G[l, k] = 0.
            self.d_nbhd[i] = np.where(self.G[i] != 0.)[0]
            self.d_nbhd[j] = np.where(self.G[j] != 0.)[0]
            self.d_nbhd[k] = np.where(self.G[k] != 0.)[0]
            self.d_nbhd[l] = np.where(self.G[l] != 0.)[0]

    def run_mc_edge(self, alpha, N):
        for _ in range(N):
            self.mc_step_edge(alpha)




