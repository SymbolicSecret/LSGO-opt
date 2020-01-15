import numpy as np
from lsgo_function import benchmark

function_index = 1
f = benchmark.Benchmark(function_index)
xmin = f.lb
xmax = f.ub
print(xmax)
# 初始化参数
