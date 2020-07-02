import taichi as ti
import matplotlib.pyplot as plt
import numpy as np
ti.init()

dt = 0.6
totime = 15
steps = int(totime / dt)
k = 0.5

pos = ti.var(ti.f32, 2)
vel = ti.var(ti.f32, 2)
######################

@ti.kernel
def ode20():
    x1, v1 = pos[0], vel[0]
    x2, v2 = pos[1], vel[1]
    nx1 = x1 + v1 * dt
    nx2 = x2 + v2 * dt
    a = x2 - x1
    nv1 = v1 + a * dt
    nv2 = v2 - a * dt
    pos[0], vel[0] = nx1, nv1
    pos[1], vel[1] = nx2, nv2

@ti.kernel
def ode21():
    x1, v1 = pos[0], vel[0]
    x2, v2 = pos[1], vel[1]
    nx1 = x1 + v1 * dt
    nx2 = x2 + v2 * dt
    a = nx2 - nx1
    nv1 = v1 + a * dt
    nv2 = v2 - a * dt
    pos[0], vel[0] = nx1, nv1
    pos[1], vel[1] = nx2, nv2

@ti.kernel
def ode02():
    # F: x1 x2 v1 v2
    A = ti.Matrix([
        [+0,+0,+1,+0],  # T: x1
        [+0,+0,+0,+1],  # T: x2
        [-1,+1,+0,+0],  # T: v1
        [+1,-1,+0,+0],  # T: v2
        ])
    I = ti.Matrix.identity(ti.f32, 4)

    x1, v1 = pos[0], vel[0]
    x2, v2 = pos[1], vel[1]
    X = ti.Vector([x1, x2, v1, v2])
    M = I - dt * A
    NX = M.inverse() @ X
    nx1, nx2, nv1, nv2 = NX
    pos[0], vel[0] = nx1, nv1
    pos[1], vel[1] = nx2, nv2

@ti.kernel
def ode22():
    # F: x1 x2 v1 v2
    A = ti.Matrix([
        [+0,+0,+1,+0],  # T: x1
        [+0,+0,+0,+1],  # T: x2
        [-1,+1,+0,+0],  # T: v1
        [+1,-1,+0,+0],  # T: v2
        ])
    I = ti.Matrix.identity(ti.f32, 4)

    x1, v1 = pos[0], vel[0]
    x2, v2 = pos[1], vel[1]
    X = ti.Vector([x1, x2, v1, v2])
    M = I - dt * A * (1 - k)
    N = I + dt * A * k
    NX = M.inverse() @ N @ X
    nx1, nx2, nv1, nv2 = NX
    pos[0], vel[0] = nx1, nv1
    pos[1], vel[1] = nx2, nv2


######################
plt.subplots()[0].canvas.mpl_connect('key_press_event',
                    lambda e: e.key != 'escape' or plt.close())

######################
pos[0], pos[1] = -1, 1
vel[0], vel[1] = 1, 0

X1, X2 = [], []
for i in range(steps):
    ode20()
    X1.append(pos[0])
    X2.append(pos[1])

X1 = np.array(X1)
X2 = np.array(X2)
plt.plot(X1, X2, label='explicit')

######################
pos[0], pos[1] = -1, 1
vel[0], vel[1] = 1, 0

X1, X2 = [], []
for i in range(steps):
    ode21()
    X1.append(pos[0])
    X2.append(pos[1])

X1 = np.array(X1)
X2 = np.array(X2)
plt.plot(X1, X2, label='forward-explicit')

######################
pos[0], pos[1] = -1, 1
vel[0], vel[1] = 1, 0

X1, X2 = [], []
for i in range(steps):
    ode02()
    X1.append(pos[0])
    X2.append(pos[1])

X1 = np.array(X1)
X2 = np.array(X2)
plt.plot(X1, X2, label='implicit')

######################
pos[0], pos[1] = -1, 1
vel[0], vel[1] = 1, 0

X1, X2 = [], []
for i in range(steps):
    ode22()
    X1.append(pos[0])
    X2.append(pos[1])

X1 = np.array(X1)
X2 = np.array(X2)
plt.plot(X1, X2, label='semi-implicit')

######################
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'dt = {dt}')
plt.xlim(-1, 9)
plt.ylim(-1, 9)
plt.legend()
plt.show()
