# https://blog.csdn.net/liuyunduo/article/details/84098884
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

plt.subplots()[0].canvas.mpl_connect('key_press_event',
                    lambda e: e.key != 'escape' or plt.close())

def spiky(r, h):
    return (h - r)**3

def poly6(r, h):
    return (h**2 - r**2)**3

def dspiky(r, h):
    return -3 * (h - r)**2

def dpoly6(r, h):
    return -6 * r * (h**2 - r**2)**2

np_h = 1
np_r = np.linspace(0, 1, 50)
plt.plot(np_r, spiky(np_r, np_h), label='spiky')
plt.plot(np_r, poly6(np_r, np_h), label='poly6')
plt.legend()
plt.show()
