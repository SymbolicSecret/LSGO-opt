import numpy as np


def sphere(x):
    x = np.array(x)
    return np.sum(x ** 2)


def dlliptic(x):
    x = np.array(x)
    Dim = x.size
    f = 0
    for i in range(Dim):
        temp = 10 ** (6 * i / (Dim - 1))
        f = f + temp * x[i] ** 2
    return f


def rastrigin(x):
    x = np.array(x)
    Dim = x.size
    f = x ** 2 - 10 * np.cos(2 * np.pi * x) + 10
    return np.sum(f)


def ackley(x):
    x = np.array(x)
    Dim = x.size
    f = -20 * np.exp(-0.2 * np.sqrt(1 / Dim * np.sum(x))) - np.exp(1 / Dim * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e
    return f


def schwefel(x):
    x = np.array(x)
    Dim = x.size
    f = 0
    for i in range(Dim):
        f = f + (np.sum(x[:i + 1])) ** 2
    return f


def rosenbrock(x):
    x = np.array(x)
    Dim = x.size
    f = 0
    for i in range(Dim - 1):
        f = f + 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
    return f
