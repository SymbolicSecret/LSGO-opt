# -*- coding: utf-8 -*-

import numpy as np
from lsgo_function import benchmark
from matplotlib import pyplot as plt


class PSO(object):
    def __init__(self, func_index, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5):
        self.func_index = func_index
        self.func = benchmark.Benchmark(func_index)
        self.w = w
        self.cp, self.cg = c1, c2
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
        self.pbest_x = self.X.copy()
        self.pbest_y = self.Y.copy()
        self.gbest_x = np.zeros((1, self.dim))
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()

        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}

    def update_V(self):
        r1 = np.random.rand(self.pop, self.dim)
        r2 = np.random.rand(self.pop, self.dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V

        if self.has_constraints:
            self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        self.Y = np.zeros((self.pop, 1))
        for i in range(self.pop):
            self.Y[i, 0] = self.func.f4(self.X[i, :])
        return self.Y

    def update_pbest(self):
        self.pbest_x = np.where(self.pbest_y > self.Y, self.X, self.pbest_x)
        self.pbest_y = np.where(self.pbest_y > self.Y, self.Y, self.pbest_y)

    def update_gbest(self):
        if self.gbest_y > self.Y.min():
            self.gbest_x = self.X[self.Y.argmin(), :].copy()
            self.gbest_y = self.Y.min()

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['Y'].append(self.Y)
        self.record_value['V'].append(self.V)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iternum in range(self.max_iter):
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            print("the number of %d is %.1f" % (iternum, self.gbest_y))

            self.gbest_y_hist.append(self.gbest_y)
        return self


if __name__ == "__main__":
    xmin = -100 * np.ones(1000)
    xmax = 100 * np.ones(1000)
    pso = PSO(func_index=4, dim=1000, lb=xmin, ub=xmax, pop=40, max_iter=1000)
    pso.run()
    plt.plot(pso.gbest_y_hist)
    plt.show()
    np.savetxt('pso_hist.txt', pso.gbest_y_hist)
