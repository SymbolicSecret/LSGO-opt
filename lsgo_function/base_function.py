import numpy as np


def sphere(x):
    return np.dot(x, x)


def dlliptic(x):
    f = 1e6 ** np.linspace(0, 1, x.size)
    return np.dot(f, x ** 2)


def rastrigin(x):
    return np.dot(x, x) + 10 * x.size - 10 * np.cos(2 * np.pi * x).sum()


def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(np.dot(x, x) / x.size)) - np.exp((np.cos(2 * np.pi * x).sum()) / x.size) + 20


def schwefel(x):
    f = 0
    for i in range(x.size):
        f = f + (np.sum(x[0:i + 1])) ** 2
    return f


def rosenbrock(x):
    a = x[0:x.size - 1] ** 2 - x[1:x.size]
    b = x - 1
    return 100 * np.dot(a, a) + np.dot(b, b)


def t_asy(x, beta):
    indx = (x > 0)
    power = 1 + beta * (np.linspace(0, 1, x.size)[indx]) * np.sqrt(x[indx])
    x_asy = x.copy()
    x_asy[indx] = x[indx] ** power
    return x_asy

def _t_osz(x):
    indx = (x > 0)
    x_osz = x.copy()
    x_osz[indx] = np.log(x_osz[indx])
    x_osz[indx] = np.exp(x_osz[indx] + 0.049 * (np.sin(10 * x_osz[indx]) + np.sin(7.9 * x_osz[indx])))
    indx = (x < 0)
    x_osz[indx] = np.log(-1 * x_osz[indx])
    x_osz[indx] = -1 * np.exp(x_osz[indx] + 0.049 * (np.sin(5.5 * x_osz[indx]) + np.sin(3.1 * x_osz[indx])))
    return x_osz


def t_diag(x, alpha):
    temp = alpha ** (0.5 * np.linspace(0, 1, x.size))
    return np.diag(temp, 0)


def shift(x, x_opt):
    return x - x_opt


def f1(x)


