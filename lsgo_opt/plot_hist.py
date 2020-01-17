# -*- coding: UTF-8 -*-

"""
Created on 2020-01-17
Updated on 2020-01-17
Author: CodingLittlePig
GitHub: https://github.com/CodingLittlePig/LSGO-opt
"""

import numpy as np
import matplotlib.pyplot as plt
pso_hist = np.loadtxt('pso_hist.txt')
cso_hist = np.loadtxt('cso_hist.txt')

plt.plot(cso_hist, color='green', label='cso')
plt.plot(pso_hist, color='red', label='pso')
plt.show()
