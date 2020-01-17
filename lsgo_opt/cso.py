# -*- coding: utf-8 -*-

import numpy as np
from lsgo_function import benchmark
from matplotlib import pyplot as plt


class CSO(object):
    def __init__(self, func_index, dim, pop=40, max_iter=150, lb=None, ub=None, phi=0.5):
        self.func_index = func_index
        self.func = benchmark.Benchmark(func_index)
        self.phi = phi
        self.pop = pop
        self.dim = dim
        self.max_iter = max_iter

        self.has_constraints = not (lb is None and ub is None)
        self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
        self.ub = np.ones(self.dim) if ub is None else np.array(ub)
        assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.dim))
        v_high = self.ub - self.lb
        self.V = np.random.uniform(low=-v_high, high=v_high, size=(self.pop, self.dim))
        self.Y = np.zeros([self.pop, ])
        self.Y = self.cal_y()

        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        self.gbest_y_hist = []

    def choose_group(self):
        self.group1 = np.random.choice(self.pop, size=int(self.pop / 2), replace=False)
        self.group2 = np.setdiff1d(np.arange(self.pop), self.group1)

    def update_V(self):
        r1 = np.random.rand(int(self.pop / 2), self.dim)
        r2 = np.random.rand(int(self.pop / 2), self.dim)
        r3 = np.random.rand(int(self.pop / 2), self.dim)
        self.V[self.lose_index] = r1 * self.V[self.lose_index] + r2 * (
                self.X[self.win_index] - self.X[self.lose_index]) + \
                                  r3 * self.phi * (self.x_mean - self.X[self.lose_index])

    def update_X(self):
        self.X[self.lose_index] = self.X[self.lose_index] + self.V[self.lose_index]

        if self.has_constraints:
            self.X[self.lose_index] = np.clip(self.X[self.lose_index], self.lb, self.ub)

    def cal_y(self):
        self.Y = np.zeros((self.pop, 1))
        for i in range(self.pop):
            self.Y[i, 0] = self.func.f4(self.X[i, :])
        return self.Y

    def update_Xw(self):
        self.win_index = np.where(self.Y[self.group1] > self.Y[self.group2], self.group2.reshape(-1, 1),
                                  self.group1.reshape(-1, 1)).reshape(1, -1)

    def update_Xl(self):
        self.lose_index = np.where(self.Y[self.group1] > self.Y[self.group2], self.group1.reshape(-1, 1),
                                   self.group2.reshape(-1, 1)).reshape(1, -1)

    def update_Xmean(self):
        self.x_mean = np.mean(self.X, axis=0)

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)
        self.record_value['V'].append(self.V)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iternum in range(self.max_iter):
            self.choose_group()
            self.cal_y()
            self.update_Xw()
            self.update_Xl()
            self.update_Xmean()
            self.update_V()
            self.update_X()
            self.gbest_y = self.Y.min()

            print("the number of %d is %.1f" % (iternum, self.gbest_y))
            self.gbest_y_hist.append(self.gbest_y)
        return self


if __name__ == "__main__":
    xmin = -100 * np.ones(1000)
    xmax = 100 * np.ones(1000)
    cso = CSO(func_index=4, dim=1000, lb=xmin, ub=xmax, pop=40, max_iter=1000)
    cso.run()
    plt.plot(cso.gbest_y_hist)
    plt.show()
