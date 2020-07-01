import matplotlib.pyplot as plt
import numpy as np

dt = 1.5
totime = 6
steps = int(totime / dt)
A = -1
x0 = 1
k = 0.5

def analytic():
    return np.exp(A * T)

def ode10():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x)
        new_x = x + dt * A * x
        x = new_x
    return np.array(xs)

def ode01():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x)
        # new_x = x + dt * A * new_x
        # (1 - dt * A) * new_x = x
        new_x = x / (1 - dt * A)
        x = new_x
    return np.array(xs)

def ode11():
    xs = []
    x = x0
    for i in range(steps):
        xs.append(x)
        # new_x = x + dt * A * (x * k + new_x * (1 - k))
        # (1 - dt * A * (1 - k)) * new_x = (1 + dt * A * k) * x
        new_x = x * (1 + dt * A * k) / (1 - dt * A * (1 - k))
        x = new_x
    return np.array(xs)

T = np.linspace(0, steps * dt, steps)
plt.subplots()[0].canvas.mpl_connect('key_press_event',
                    lambda e: e.key != 'escape' or plt.close())
plt.title(f'dt = {dt}')
plt.plot(T, analytic(), label='analytic')
plt.plot(T, ode10(), label='explicit')
plt.plot(T, ode01(), label='implicit')
plt.plot(T, ode11(), label='semi-implicit')
plt.ylim(-0.51, 1.01)
plt.xlabel('value')
plt.ylabel('time')
plt.legend()
plt.show()
