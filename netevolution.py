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
    G = np.zeros((N, N), dtype=float64)
    for i in np.arange(N):
        for j in np.arange(i+1, N):
            if np.random.rand() < p:
                G[i, j] = 1.
                G[j, i] = 1.
    d = dict()
    disconnected = set()
    for i in np.arange(N):
        d[i] = np.where(G[i] != 0.)[0]
        if len(d[i]) == 0:
            disconnected.add(i)
    return G, d, disconnected

@njit()
def stationary(strat1, strat2):
    p1, q1 = strat1
    p2, q2 = strat2
    r1, r2 = p1 - q1, p2 - q2
    s1 = (q2*r1 + q1)/(1 - r1*r2)
    s2 = (q1*r2 + q2)/(1 - r1*r2)
    stat = [s1*s2, s1*(1 - s2), s2*(1 - s1), (1 - s1)*(1 - s2)]
    return np.array(stat)

@njit(parallel=True)
def calc_payoffs(new_strat, prev_strat, strategies, neighbors, payoffs):
    n_nhbs = neighbors.shape[0]
    prev_payoffs = np.zeros(n_nhbs)
    new_payoffs = np.zeros(n_nhbs)
    for nhb in nb.prange(n_nhbs):
        prev_stat = stationary(prev_strat, strategies[neighbors[nhb]])
        new_stat = stationary(new_strat, strategies[neighbors[nhb]])
        prev_payoffs[nhb] = np.dot(payoffs, prev_stat)
        new_payoffs[nhb] = np.dot(payoffs, new_stat)
    return prev_payoffs, new_payoffs


spec = [
    ('N', int64),
    ('payoffs', float64[:]),
    ('G', float64[:, ::1]),
    ('d_nbhd', nb.typeof(Dict.empty(
                         key_type=int64,
                         value_type=int64[::1]))),
    ('strategies', float64[:, :]),
    ('disconnected', nb.typeof({1}))
]

@jitclass(spec)
class GraphEvo(object):
    def __init__(self, N, p, payoffs):
        self.N = N
        self.payoffs = payoffs
        self.G, self.d_nbhd, self.disconnected = erdos_renyi(N, p)
        self.strategies = np.random.uniform(0.0, 1.0, (N, 2))

    """
    Just a regular old Monte Carlo step -

    """
    def mc_step(self):
        i = np.random.choice(self.N)
        new_strat = np.random.uniform(0.0, 1.0, 2)
        # calculate payoffs against neighbors
        prev_mean, new_mean = calc_payoffs(new_strat, self.strategies[i],
                                           self.strategies, self.d_nbhd[i],
                                           self.payoffs)
        if np.mean(new_mean) > np.mean(prev_mean):
            self.strategies[i] = new_strat


    """
    Monte Carlo step that also might change edges

    """
    def mc_step_edge(self, alpha):
        i = np.random.choice(self.N)
        new_strat = np.random.uniform(0.0, 1.0, 2)
        # calculate payoffs against neighbors
        if len(self.d_nbhd[i]) > 0:
            prev_mean, new_mean = calc_payoffs(new_strat, self.strategies[i],
                                               self.strategies, self.d_nbhd[i],
                                               self.payoffs)
            if np.mean(new_mean) > np.mean(prev_mean):
                self.strategies[i] = new_strat
        # try changing edge dist
        if np.random.rand() < alpha:
            # next_nbs.update(self.disconnected)
            j = np.random.choice(np.arange(self.N), i)
            # print('got here')
            new_stat = stationary(self.strategies[i], self.strategies[j])
            # print('got here')
            new_payoff = np.dot(self.payoffs, new_stat)
            if new_payoff > np.min(new_mean):
                ind = np.argmin(new_mean)
                k = self.d_nbhd[i][ind]
                self.G[i, k] = 0
                self.G[k, i] = 0
                self.G[i, j] = 1
                self.G[j, i] = 1
                self.d_nbhd[i][ind] = j
                self.d_nbhd[k] = np.array([x for x in self.d_nbhd[k] if x != i])
                if len(self.d_nbhd[k]) == 0:
                    self.disconnected.add(k)
                self.d_nbhd[j] = np.append(self.d_nbhd[j], [i])
                if j in self.disconnected:
                    self.disconnected.remove(j)


