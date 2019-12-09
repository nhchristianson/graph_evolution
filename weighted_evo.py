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
    for i in np.arange(N):
        d[i] = np.where(G[i] != 0.)[0]
    return G, d

"""
Ensures x and y are within eps of each other (to deal with float inconsistencies)
"""
@njit()
def within_eps(x, y, eps=1e-8):
    return np.abs(x - y) < eps

@njit(parallel=True)
def parallel_mc(N, p, payoffs, alpha, incr, n_steps, n_parallel):
    dev_Gs = np.zeros((n_parallel, 2, N, N))
    strategies = np.zeros((n_parallel, 100, N, 2))
    for p_copy in nb.prange(n_parallel):
        x = GraphEvo(N, p, payoffs)
        dev_Gs[p_copy, 0] = x.G
        for i in np.arange(100):
            for k in np.arange(n_steps/100):
                x.mc_step_incr_edge(alpha, incr)
            strategies[p_copy, i] = x.strategies
        dev_Gs[p_copy, 1] = x.G
        # print('copy done')
    return dev_Gs, strategies

spec = [
    ('N', int64),
    ('payoffs', float64[::1]),
    ('G', float64[:, ::1]),
    ('d_nbhd', nb.typeof(Dict.empty(
                         key_type=int64,
                         value_type=int64[::1]))),
    ('d_weights', nb.typeof(Dict.empty(
                         key_type=int64,
                         value_type=float64[::1]))),
    ('strategies', float64[:, ::1])
]

@jitclass(spec)
class GraphEvo(object):
    def __init__(self, N, p, payoffs):
        self.N = N
        self.payoffs = payoffs
        self.G, self.d_nbhd = erdos_renyi(N, p)
        d = dict()
        for i in range(self.N):
            d[i] = np.ones(self.d_nbhd[i].shape[0])
        self.d_weights = d
        self.strategies = np.random.uniform(0.0, 1.0, (N, 2))

    """
    Calculates the stationary distribution for two strategies

    """
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

    """
    Calculates the new and previous payoffs for a proposed new strategy

    """
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
    Monte Carlo step that also might change edges

    """
    def mc_step_edge(self, alpha):
        i = np.random.choice(self.N)
        # proposed new strategy
        new_strat = np.random.uniform(0.0, 1.0, 2)
        # if the node has no neighbors, just accept the change
        if self.d_nbhd[i].shape[0] == 0:
            self.strategies[i] = new_strat
        # otherwise, check if change increases payoff
        else:
            # calculate payoffs against neighbors
            prev_payoffs, new_payoffs = self.calc_payoffs(new_strat, self.strategies[i],
                                                    self.strategies, self.d_nbhd[i],
                                                    self.payoffs)
            # calculate weighted payoffs
            mean_prev = np.dot(prev_payoffs, self.d_weights[i]) / np.sum(self.d_weights[i])
            mean_new = np.dot(new_payoffs, self.d_weights[i]) / np.sum(self.d_weights[i])
            if mean_new > mean_prev:
                self.strategies[i] = new_strat
                chosen_payoffs = new_payoffs
            else:
                chosen_payoffs = prev_payoffs

        # Now consider changing the edge distribution
        choice = np.random.choice(3)
        # drop the lowest edge (unless no edges, do nothing)
        if choice == 0 and self.d_nbhd[i].shape[0] != 0:
            if self.d_nbhd[i].shape[0] == 1:
                if chosen_payoffs[0] < 0.:
                    self.G[i, self.d_nbhd[i][0]] = 0.
                    self.G[self.d_nbhd[i][0], i] = 0.
                    self.d_nbhd[i] = np.delete(self.d_nbhd[i], 0)
                    self.d_weights[i] = np.delete(self.d_weights[i], 0)
            else:
                j = np.argmin(chosen_payoffs)
                self.G[i, self.d_nbhd[i][j]] = 0.
                self.d_nbhd[i] = np.delete(self.d_nbhd[i], j)
                self.d_weights[i] = np.delete(self.d_weights[i], j)
                self.d_weights[i] = self.d_weights[i] / np.sum(self.d_weights[i])
        #reweight edges
        elif choice == 1 and self.d_nbhd[i].shape[0] != 0:
            j = np.random.choice(self.d_nbhd[i].shape[0])
            new_weights = self.d_weights[i].copy()
            new_weights[j] += 0.1
            new_weights = new_weights / np.sum(new_weights)
            mean_prev = np.dot(chosen_payoffs, self.d_weights[i])
            mean_new = np.dot(chosen_payoffs, new_weights)
            if mean_new > mean_prev:
                self.d_weights[i] = new_weights
        # add an edge (unless fully connected, do nothing)
        elif choice == 2 and self.d_nbhd[i].shape[0] != self.N - 1:
            if self.d_nbhd[i].shape[0] == 0:
                j = np.random.choice(np.delete(np.arange(self.N), i))
                ij_stat = self.stationary(self.strategies[i], self.strategies[j])
                ij_payoff = np.dot(self.payoffs, ij_stat)
                if ij_payoff > 0.:
                    self.G[i, j] = 1.
                    self.d_nbhd[i] = np.array([j], dtype=int64)
                    self.d_weights[i] = np.array([1], dtype=float64)
            else:
                avail = set(np.delete(np.arange(self.N), i))
                not_ns = avail.difference(set(self.d_nbhd[i]))
                nns = np.array(list(not_ns))
                j = np.random.choice(nns)
                ij_stat = self.stationary(self.strategies[i], self.strategies[j])
                new_weights = np.append(self.d_weights[i], [0.1])
                new_weights = new_weights / np.sum(new_weights)
                new_payoffs = np.append(chosen_payoffs, [ij_payoff])
                if np.dot(new_payoffs, new_weights) > np.dot(chosen_payoffs, self.d_weights[i]):
                    self.G[i, j] = 1.
                    self.d_nbhd[i] = np.append(self.d_nbhd[i], [j])
                    self.d_weights[i] = new_weights



    """
    Monte Carlo step that also might change edges

    """
    def mc_step_incr_edge(self, alpha, increment=0.1):
        i = np.random.choice(self.N)
        # proposed new strategy
        new_strat = 0.01*np.random.randn(2) + self.strategies[i]
        new_strat[new_strat > 0.99] = 0.99
        new_strat[new_strat < 0.01] = 0.01
        # if the node has no neighbors, just accept the change
        if self.d_nbhd[i].shape[0] == 0:
            self.strategies[i] = new_strat
            chosen_payoffs = np.zeros(0)
            chosen_mean = 0.
        # otherwise, check if change increases payoff
        else:
            # calculate payoffs against neighbors
            prev_payoffs, new_payoffs = self.calc_payoffs(new_strat, self.strategies[i],
                                                    self.strategies, self.d_nbhd[i],
                                                    self.payoffs)
            # calculate weighted payoffs
            mean_prev = np.dot(prev_payoffs, self.d_weights[i]) / np.sum(self.d_weights[i])
            mean_new = np.dot(new_payoffs, self.d_weights[i]) / np.sum(self.d_weights[i])
            if mean_new > mean_prev:
                self.strategies[i] = new_strat
                chosen_payoffs = new_payoffs
                chosen_mean = mean_new
            else:
                chosen_payoffs = prev_payoffs
                chosen_mean = mean_prev
        # consider modifying edge distribution
        choice = np.random.choice(4)
        # consider adding edge
        if choice == 0 and self.d_nbhd[i].shape[0] != self.N - 1:
            avail = set(np.delete(np.arange(self.N), i))
            not_ns = avail.difference(set(self.d_nbhd[i]))
            nns = np.array(list(not_ns))
            j = np.random.choice(nns)
            ij_stat = self.stationary(self.strategies[i], self.strategies[j])
            ij_payoff = np.dot(self.payoffs, ij_stat)
            prop_mean = ((np.dot(chosen_payoffs, self.d_weights[i]) + increment*ij_payoff)
                         / (np.sum(self.d_weights[i]) + increment))
            if prop_mean > chosen_mean:
                self.G[i, j] = increment
                self.G[j, i] = increment
                self.d_nbhd[i] = np.append(self.d_nbhd[i], [j])
                self.d_nbhd[j] = np.append(self.d_nbhd[j], [i])
                self.d_weights[i] = np.append(self.d_weights[i], [increment])
                self.d_weights[j] = np.append(self.d_weights[j], [increment])
        elif choice == 1 and self.d_nbhd[i].shape[0] != 0:
            k = np.random.choice(self.d_nbhd[i].shape[0])
            j = self.d_nbhd[i][k]
            prop_weights = np.delete(self.d_weights[i], k)
            prop_payoffs = np.delete(chosen_payoffs, k)
            if prop_weights.shape[0] == 0:
                prop_payoff = 0.
            else:
                prop_payoff = np.dot(prop_payoffs, prop_weights) / np.sum(prop_weights)
            if prop_payoff > chosen_mean:
                self.G[i, j] = 0.
                self.G[j, i] = 0.
                self.d_nbhd[i] = np.delete(self.d_nbhd[i], k)
                self.d_weights[i] = prop_weights
                k2 = np.where(self.d_nbhd[j] == i)[0][0]
                self.d_nbhd[j] = np.delete(self.d_nbhd[j], k2)
                self.d_weights[j] = np.delete(self.d_weights[j], k2)
        elif self.d_nbhd[i].shape[0] != 0:
            l = np.random.choice(self.d_nbhd[i].shape[0])
            k = self.d_nbhd[i][l]
            incr = increment if choice == 2 else -increment
            prop_weights = self.d_weights[i].copy()
            prop_weights[l] += incr
            prop_sum = np.sum(prop_weights)
            if prop_sum == 0.:
                prop_payoff = 0.
            else:
                prop_payoff = np.dot(chosen_payoffs, prop_weights) / np.sum(prop_weights)
            if prop_payoff > chosen_mean:
                if within_eps(prop_weights[l], 0.):
                    self.G[i, k] = 0.
                    self.G[k, i] = 0.
                    self.d_nbhd[i] = np.delete(self.d_nbhd[i], [l])
                    l2 = np.where(self.d_nbhd[k] == i)[0][0]
                    self.d_nbhd[k] = np.delete(self.d_nbhd[k], [l2])
                    self.d_weights[i] = np.delete(self.d_weights[i], l)
                    self.d_weights[k] = np.delete(self.d_weights[k], l2)
                else:
                    self.G[i, k] += incr
                    self.G[k, i] += incr
                    self.d_weights[i] = prop_weights
                    l2 = np.where(self.d_nbhd[k] == i)[0][0]
                    prop_k_weights = self.d_weights[k]
                    prop_k_weights[l2] += incr
                    self.d_weights[k] = prop_k_weights

    def run_mc(self, alpha, incr, n_steps):
        graphs = np.zeros((1000, self.N, self.N))
        strats = np.zeros((1000, self.N, 2))
        for i in range(1000):
            for n in np.arange(int(n_steps / 1000)):
                self.mc_step_incr_edge(alpha, incr)
            graphs[i] = self.G
            strats[i] = self.strategies
        return graphs, strats


