import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

dt = 0.4
totime = 10
steps = int(totime / dt)
w = 0.8
#         from: pos  vel
A = ti.Matrix([[ +0,  +1],   # to: pos
               [ -w,  +0]])  # to: vel
x0 = ti.Vector([ +1,  +0])
k = 0.5

I = ti.Matrix([[1, 0], [0, 1]])

def inverse(m):
    # Sadly, ti.Matrix.inverse doesn't work in Python-scope..
    a, b, c, d = m.entries
    det = a * d - b * c
    return ti.Matrix([
            [d, -b],
            [-c, a]]) / det

def analytic():
    return np.cos(w * T)

def ode10():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x[0])
        new_x = x + dt * A @ x
        x = new_x
    return np.array(xs)

def ode01():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x[0])
        # new_x = x + dt * A @ new_x
        # (I - dt * A) @ new_x = x
        new_x = inverse(I - dt * A) @ x
        x = new_x
    return np.array(xs)

def ode11():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x[0])
        # new_x = x + dt * A @ (x * k + new_x * (1 - k))
        # new_x = (I + dt * A * k) @ x + (dt * A * (1 - k)) @ new_x
        # (I - dt * A * (1 - k)) @ new_x = (I + dt * A * k) @ x
        print(inverse(I - dt * A * (1 - k)) @ (I - dt * A * (1 - k)))
        new_x = inverse(I - dt * A * (1 - k)) @ (I + dt * A * k) @ x
        x = new_x
    return np.array(xs)

T = np.linspace(0, totime, steps)
plt.subplots()[0].canvas.mpl_connect('key_press_event',
                    lambda e: e.key != 'escape' or plt.close())
plt.title(f'dt = {dt}')
plt.plot(T, analytic(), label='analytic')
plt.plot(T, ode10(), label='explicit')
plt.plot(T, ode01(), label='implicit')
plt.plot(T, ode11(), label='semi-implicit')
plt.ylim(-2, 2)
plt.xlabel('value')
plt.ylabel('time')
plt.legend()
plt.show()
