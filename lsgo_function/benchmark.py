import numpy as np
import os


class Benchmark(object):
    def __init__(self, index):
        if index in [1, 4, 7, 8, 11, 12, 13, 14, 15]:
            self.ub = 100
            self.lb = -100
        elif index in [2, 5, 9]:
            self.ub = 5
            self.lb = -5
        elif index in [3, 6, 10]:
            self.ub = 32
            self.lb = -32
        else:
            raise ValueError("Error: The test function is not in this suite!")
        module_path = os.path.dirname(__file__)
        self.x_opt = np.loadtxt(module_path + '\cdatafiles\F%d-xopt.txt' % index)
        self.alpha = 10
        self.beta = 0.2
        self.m = 5
        self.s = None
        self.p = None
        self.r25 = None
        self.r50 = None
        self.r100 = None

        if index in range(4, 15):
            if index == 12 or index == 15:
                pass
            else:
                self.s = np.loadtxt(module_path + './cdatafiles/F%d-s.txt' % index)
                self.r25 = np.loadtxt(module_path + './cdatafiles/F%d-R25.txt' % index, delimiter=',')
                self.r50 = np.loadtxt(module_path + './cdatafiles/F%d-R50.txt' % index, delimiter=',')
                self.r100 = np.loadtxt(module_path + './cdatafiles/F%d-R100.txt' % index, delimiter=',')
                self.w = np.loadtxt(module_path + './cdatafiles/F%d-w.txt' % index)
                self.p = np.loadtxt(module_path + './cdatafiles/F%d-p.txt' % index)

    @staticmethod
    def _sphere(x):
        return np.dot(x, x)

    @staticmethod
    def _dlliptic(x):
        f = 1e6 ** np.linspace(0, 1, x.size)
        return np.dot(f, x ** 2)

    @staticmethod
    def _rastrigin(x):
        return np.dot(x, x) + 10 * x.size - 10 * np.cos(2 * np.pi * x).sum()

    @staticmethod
    def _ackley(x):
        return -20 * np.exp(-0.2 * np.sqrt(np.dot(x, x) / x.size)) - np.exp(
            (np.cos(2 * np.pi * x).sum()) / x.size) + 20 + np.e

    @staticmethod
    def _schwefel(x):
        f = 0
        for i in range(x.size):
            f = f + (np.sum(x[0:i + 1])) ** 2
        return f

    @staticmethod
    def _rosenbrock(x):
        a = x[0:x.size - 1] ** 2 - x[1:x.size]
        b = x - 1
        return 100 * np.dot(a, a) + np.dot(b, b)

    @staticmethod
    def _t_asy(x, beta):
        indx = (x > 0)
        power = 1 + beta * (np.linspace(0, 1, x.size)[indx]) * np.sqrt(x[indx])
        x_asy = x.copy()
        x_asy[indx] = x[indx] ** power
        return x_asy

    @staticmethod
    def _t_osz(x):
        indx = (x > 0)
        x_osz = x.copy()
        x_osz[indx] = np.log(x_osz[indx])
        x_osz[indx] = np.exp(x_osz[indx] + 0.049 * (np.sin(10 * x_osz[indx]) + np.sin(7.9 * x_osz[indx])))
        indx = (x < 0)
        x_osz[indx] = np.log(-1 * x_osz[indx])
        x_osz[indx] = -1 * np.exp(x_osz[indx] + 0.049 * (np.sin(5.5 * x_osz[indx]) + np.sin(3.1 * x_osz[indx])))
        return x_osz

    @staticmethod
    def _t_diag(x, alpha):
        temp = alpha ** (0.5 * np.linspace(0, 1, x.size))
        return np.diag(temp, 0)

    @staticmethod
    def _shift(x, x_opt):
        return x - x_opt

    def f1(self, x):
        return _ellipsis(_t_osz(_shift(x, self.x_opt)))

    def f2(self, x):
        mat = _t_diag(x, self.alpha)
        return _rastrigin(np.dot(mat, _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)))

    def f3(self, x):
        mat = _t_diag(x, self.alpha)
        return _ackley(np.dot(mat, _t_asy(_t_osz(shift(x, self.x_opt)), self.beta)))

    def f4(self, x):
        z = _t_osz(_shift(x, self.x_opt))
        count = 0
        fit = 0
        for i in range(self.s.size - 1):
            if self.s[i] == 25:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        fit += _elliptic(z[self.p[count:]])
        return fit

    # 7-nonseparable, 1-separable shifted and rotated rastrigin's function
    def f5(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size - 1):
            if self.s[i] == 25:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        fit += _rastrigin(z[self.p[count:]])
        return fit

    # 7-nonseparable, 1-separable shifted and rotated ackley's function
    def f6(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size - 1):
            if self.s[i] == 25:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        fit += _ackley(z[self.p[count:]])
        return fit

    # 7-nonseparable, 1-separable shifted and rotated schwefel's function
    def f7(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size - 1):
            if self.s[i] == 25:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        fit += _schwefel(z[self.p[count:]])
        return fit

    ''' partially additice separable functions 2'''

    # 20-nonseparable shifted and rotated elliptic function
    def f8(self, x):
        z = _t_osz(_shift(x, self.x_opt))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _elliptic(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        return fit

    # 20-nonseparable shifted and rotated rastrigin's function
    def f9(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _rastrigin(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        return fit

        # 20-nonseparable shifted and rotated ackley's function

    def f10(self, x):
        z = np.dot(_t_diag(x, self.alpha), _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta))
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _ackley(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        return fit

    # 20-nonseparable shifted schwefel's function
    def f11(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25
            elif self.s[i] == 50:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50
            else:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100
        return fit

    ''' overlapping functions'''

    # shifted rosenbrock's function
    def f12(self, x):
        return _rosenbrock(_shifet(x, self.x_opt))

    # shifted schwefel's function with conforming overlapping subcomponents
    def f13(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        count = 0
        fit = 0
        for i in range(self.s.size):
            if self.s[i] == 25:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_25, z[self.p[count:count + 25]]))
                count += 25 - self.m
            elif self.s[i] == 50:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_50, z[self.p[count:count + 50]]))
                count += 50 - self.m
            else:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_100, z[self.p[count:count + 100]]))
                count += 100 - self.m
        return fit

    # shifted schwefel's function with confllicting overlapping subcomponents
    def f14(self, x):
        c = np.cumsum(self.s)
        count = 0
        fit = 0
        for i in range(self.s.size):
            z = x[self.p[count:count + s[i]]] - self.x_opt[c[i - 1]:c[i]]
            z = _t_asy(_t_osz(z), self.beta)
            if self.s[i] == 25:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_25, z))
                count += 25 - self.m
            elif self.s[i] == 50:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_50, z))
                count += 50 - self.m
            else:
                fit += self.w[i] * _schwefel(np.dot(self.rota_mat_100, z))
                count += 100 - self.m
        return fit

    ''' fully non-separable functions'''

    # shifted schwefel's function
    def f15(self, x):
        z = _t_asy(_t_osz(_shift(x, self.x_opt)), self.beta)
        return _schwefel(z)
